# LeRobot 0.6.0 GR00T N1.7 学習・推論 実装知見

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0
(torch 2.11.0+cu130) + 16軸ヒューマノイド + HSB カメラの実機構築 (2026-08) で
確認した内容。データセットは LeRobotDataset v3 (40 エピソード / 17,681 フレーム /
30fps / 640×360、単一カメラ front + state/action 16 次元)。

## 1. モデル構成と lerobot 0.6.0 の対応状況

- lerobot 0.6.0 の `--policy.type=groot` は **N1.7 専用** (N1.5 サポートは打切り)。
- 構成: バックボーン `nvidia/Cosmos-Reason2-2B` (Qwen3-VL 系 VLM) + projector +
  DiT アクションヘッド (flow-matching/diffusion 系)。
- パラメータ実測 (学習ログ `ot_train.py:410-411`):
  total **3,144,016,000 (3B)** / 学習対象 **1,620,515,968 (1.6B)**。
  学習対象の内訳は既定の `tune_projector=True` + `tune_diffusion_model=True` +
  `tune_vlln=True` (VLM 本体は `tune_llm=False` / `tune_visual=False` で凍結)。

### 主要な既定値 (学習ログの config ダンプで確認)

| 設定 | 既定値 | 備考 |
|---|---|---|
| `chunk_size` / `n_action_steps` | 40 / 40 | RTC の切替頻度計算に効く (§5) |
| `image_size` | [256, 256] | 入力画像は内部でこのサイズに処理される → データセットは 640×360 で十分 |
| `max_state_dim` / `max_action_dim` | 132 / 132 | 実次元 (16/16) はパディングされる。次元指定は不要 |
| `normalization_mapping` | 全て IDENTITY | 正規化はポリシー内部処理。lerobot 側の normalizer は素通し |
| `embodiment_tag` | `new_embodiment` | データセットのカメラ名・次元に自動適応 → **カメラ rename 不要** |
| `use_bf16` | True | AMP 指定不要 (`--policy.use_amp` は付けない) |
| `use_flash_attention` | False | **flash-attn のインストール不要**。Jetson でそのまま動く |
| `rtc_ramp_rate` | None (フィールドあり) | RTC ネイティブ対応の証跡 (`configuration_groot.py:320`) |

## 2. 依存パッケージ

lerobot 0.6.0 の extras 定義は `pip install "lerobot[groot]"` で、内訳は
transformers / peft / diffusers / timm / dm-tree。**decord は
`platform_machine == x86_64/AMD64` マーカー付き**なので aarch64 (Jetson) では
インストールされない (そもそも不要 — 動画は `--dataset.video_backend=pyav` で読む)。

実機で動作確認済みのバージョン (Thor の lerobot060-venv 実測):

| パッケージ | バージョン | 備考 |
|---|---|---|
| peft | 0.20.0 | |
| diffusers | 0.35.2 | |
| timm | 1.0.28 | |
| dm-tree | 0.1.10 | import 名は `tree` |
| transformers | 5.5.4 | SmolVLA 移行時に導入済みのものと共用 |
| accelerate | 1.14.0 | |
| torch / torchvision | 2.11.0+cu130 / 0.26.0 | **CUDA ビルドを壊さないこと** |

- Jetson では pip の依存解決が CUDA ビルドの torch/numpy をダウングレードする
  事故が起きうる。追加パッケージのインストール後は必ず
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  で無傷を確認する。壊れそうなら extras を使わず上記パッケージを個別に入れる。
- flash-attn は不要 (§1)。ビルドに長時間かかるだけなので入れないこと。

## 3. 起動の落とし穴 (ハマり所)

### 3.1 `--policy.path=nvidia/GR00T-N1.7-3B` は ParsingError

NVIDIA 公式リポジトリの生 config.json は lerobot の CLI パーサ (draccus) が
解釈できるフォーマットではないため、`--policy.path` に指定すると ParsingError
になる。

```bash
# NG (ParsingError)
lerobot-train --policy.path=nvidia/GR00T-N1.7-3B ...

# OK (正解)
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B ...
```

一方、**lerobot が保存したチェックポイント**
(`<output_dir>/checkpoints/<step>/pretrained_model`) は lerobot 形式の
config.json を持つので `--policy.path` で指定できる (推論・継続学習ともに)。
「HF の生モデル = type+base_model_path、自分のチェックポイント = path」と覚える。

### 3.2 バックボーン `nvidia/Cosmos-Reason2-2B` はゲート付き

`base_model_path` は `nvidia/GR00T-N1.7-3B` だが、その内部でバックボーン
`nvidia/Cosmos-Reason2-2B` のダウンロードが走り、これが**ゲート付きモデル**。
事前に (1) HF の Web でライセンス同意、(2) `hf auth login` (トークン保存) の
両方が必要。未認証だと初回学習がダウンロード段階で失敗する。

### 3.3 HF ダウンロードの hf-xet 無限ハング

CDN 側の接続断の後、hf-xet バックエンドは**エラーを出さずに futex 待ちで無限に
ハング**する (プロセスは Sleeping、数分間ゼロバイト)。lerobot-train 内部の
from_pretrained ダウンロードも同様に固まる。

- 対策: `HF_HUB_DISABLE_XET=1` を付けて通常 HTTP 経路にする
  (10 秒 read timeout で失敗が顕在化し、リトライ・レジュームが効く)。
- 診断: キャッシュの `.incomplete` ファイルの mtime が止まっている /
  `ss -tnp | grep <pid>` で CLOSE-WAIT ソケット。
- 学習済み・キャッシュ済みなら推論は `HF_HUB_OFFLINE=1` で起動するのが確実。

## 4. 学習実測データ (Jetson AGX Thor)

条件: batch 4, 640×360 データセット (40ep/17,681 フレーム), pyav, num_workers=0,
save_freq=5000。

| 項目 | 実測値 |
|---|---|
| スループット | **1.08〜1.12 step/s** (進捗バー実測。updt_s 0.872〜0.878 + data_s 約0.05) |
| 15K steps 所要 | **約 3.7h** (5000 steps ≈ 78分) |
| メモリ (mem_gb) | **35.90 で完全に平坦** (増加しない) |
| loss 推移 | 10K: 0.040 → 15K: **0.027** (5K/10K/15K の実機評価でも段階的に改善) |

### Thor 固有の注意 (ACT 学習で切り分け済み・GR00T にも適用)

- **`PYTORCH_CUDA_ALLOC_CONF` は Thor (iGPU/CUDA13) では一切設定しない**
  (デフォルトアロケータが正解。障害の実測・機序・診断は thor-platform-skills の
  `reference/reference.md` §2〜§3 に集約)。
- 空きメモリは事前に確認 (3B は **40GB 以上を推奨**。実測ピーク 35.9GB)。
  強制終了した CUDA プロセスの GPU メモリ (nvmap) はリークすることがあり、
  `free -h` で戻っていなければ再起動。
- **同じ学習の二重起動はチェックポイント破損リスク** → 起動前に
  `pgrep -f "lerobot-(record|train|rollout)"` で確認。
- CUDA プロセスは Ctrl+C で正常終了を待つ (SIGKILL はメモリリークの元)。

## 5. 推論: sync vs RTC (実機評価の結論 = sync 採用)

GR00T は `rtc_ramp_rate` を持つ RTC (Real-Time Chunking) ネイティブ対応
ポリシーだが、**実機評価の結果 sync を採用**した。

| | sync | RTC (chunk40 + queue_threshold30) | RTC (QT=5) |
|---|---|---|---|
| 動きの質 | **滑らか・確実、把持成立** | チャンク切替 (0.33秒ごと) でカクつき | QT=30 比で改善するもまだカクつく |
| 制御ループ | 約 4Hz (下記) | 30Hz 追従を狙えるが不連続 | 同左 |
| 再生速度 | 実演の約 1/7 スローモーション | 実時間寄り | 実時間寄り |
| 判定 | **採用** | 不採用 | 不採用 |

- **sync が 4Hz になる理由**: 毎ティックの前処理 (VLM 画像処理 + トークン化) が
  CPU 約 240ms かかりこれが律速。チャンク推論自体はキュー化済みなので、
  「滑らかさ」の正体はこのスローモーション再生。
  リアルタイム化したければこの前処理の最適化 (タスク文トークン化のキャッシュ等)
  が必要 — 未着手。
- **queue_threshold (QT) の意味**: キュー残がこの値を下回ると次チャンクに
  切り替える。chunk40 + QT30 なら 10 アクション (30fpsで 0.33秒) ごとに切替。
  **QT が小さいほど切替が減る** (カクつき対策の第一ノブ)。
- rollout 中の警告は基本無害:
  - 「Record loop is running slower」→ sync のスローモーションの現れ。無害。
  - RTC の「Indexes diff is not equal to real delay」(on_queue.py) → 実測ログに
    毎秒出るが動作は継続する。
- **教訓: RTC の合否はポリシー×chunk×threshold 依存で、実機評価が必須**。
  同一ロボット・同一データセットでも SmolVLA は RTC (chunk50+threshold30) で
  問題なし、GR00T はカクつき、ACT はそもそも RTC 非対応 (inference_delay 対応は
  flow-matching 系のみ)。実例スクリプト `/home/jetson/RS/run_infer06_groot.sh` は
  既定 sync のまま `INFER=rtc QT=<n>` で切替比較できる形にしてある。

## 6. 学習中の中間チェックポイント評価 (SIGSTOP 方式)

学習プロセスを止めずに GPU を推論に明け渡すテクニック (実例:
`/home/jetson/RS/run_infer06_groot_early.sh`):

1. `kill -STOP <lerobot-train の PID>` で一時停止 (進捗は失われない)。
2. **実行中の GPU カーネルが掃けるまで 3 秒待つ**。
3. 中間チェックポイント
   (`<output_dir>/checkpoints/005000/pretrained_model` 等) で lerobot-rollout。
4. `kill -CONT <PID>` で再開。シェルスクリプトでは `trap ... EXIT` に入れて
   Ctrl+C・異常終了でも必ず再開させる。

unified memory の Thor では学習 (35.9GB) + 推論モデルの同時常駐が可能なため
成立する。5K/10K/15K の段階評価で「学習を続ける価値があるか」を早期判断できた。

## 7. 診断コマンド集

| 症状 | 診断コマンド | 対処 |
|---|---|---|
| 初回学習がダウンロードで失敗 | `hf auth whoami` | Cosmos-Reason2-2B のライセンス同意 + `hf auth login` (§3.2) |
| ダウンロードが無言で固まる | `.incomplete` の mtime / `ss -tnp` で CLOSE-WAIT | `HF_HUB_DISABLE_XET=1` で再実行 (§3.3) |
| 起動即 ParsingError | コマンドラインの `--policy.path` を確認 | `--policy.type=groot --policy.base_model_path=...` に直す (§3.1) |
| 学習が失速 / step/s 低下 | `free -h` (MemAvailable) | アロケータ環境変数を外す・nvmap リークなら再起動 (§4) |
| 学習ログが出ない/進まない | `pgrep -af lerobot-train` / ログの `ot_train.py:606` 行 | 二重起動していないか確認 (§4) |
| 推論起動が遅い/ネットで止まる | 環境変数確認 | キャッシュ済みなら `HF_HUB_OFFLINE=1` |
| 推論の動きがカクつく (RTC) | `QT` を下げて比較 (`INFER=rtc QT=5`) | 改善不足なら sync に切替 (§5) |
| 推論がスローモーション (sync) | 仕様 (約 4Hz、CPU 前処理 240ms 律速) | 許容するか前処理最適化 (§5)。学習の質の問題ではない |
| チェックポイントが無い | `ls <output_dir>/checkpoints/` | `save_freq` と総 steps を確認 (5000 の倍数 + last) |

## 8. 3ポリシー比較の位置づけ (同一データセット・同一ロボット、参考)

| ポリシー | 学習実測 (Thor) | 推論方式 | 実機評価 |
|---|---|---|---|
| ACT (chunk50) | 1080p batch2 30K で loss 0.143 / 640×360 batch8 15K で loss 0.114 | sync (RTC 非対応) | 完走・動作成立 |
| SmolVLA (expert のみ 100M) | batch8 で 4.2 step/s、20K ≈ 80分 | **RTC OK** (chunk50+QT30) | 問題なし |
| **GR00T N1.7 (1.6B 学習)** | **batch4 で 1.08〜1.12 step/s、15K ≈ 3.7h、loss 0.027** | **sync 採用** (RTC はカクつき) | 滑らか・把持成立 (4Hz スロー) |

GR00T は 3 ポリシー中で最も重いが、new_embodiment の自動適応により
データセット側の加工 (rename 等) が一切不要という運用上の利点がある。
