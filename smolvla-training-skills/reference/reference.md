# LeRobot 0.6.0 SmolVLA 学習・推論の実装知見

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0 venv
(torch 2.11.0+cu130) + 16 軸ヒューマノイド (rs_follower) + HSB カメラの実機構築
(2026-08) で確認した内容。データセットは LeRobotDataset v3
(640×360, 40 エピソード, `/home/jetson/RS/humanoid_test060_640`)。

## 1. 依存パッケージ (検証済みの版数)

lerobot 0.6.0 venv に追加したのは 2 つだけ (どちらも torch 無傷を確認して導入):

| パッケージ | 版数 (実測) | 用途 |
|---|---|---|
| transformers | 5.5.4 | SmolVLM2 バックボーンのロード |
| num2words | 0.5.14 | SmolVLA の言語前処理 |

前提として入っている主要パッケージ (参考):

| パッケージ | 版数 (実測) |
|---|---|
| lerobot | 0.6.0 |
| torch / torchvision | 2.11.0 (+cu130) / 0.26.0 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |

- **導入直後に必ず `import torch; torch.cuda.is_available()` を確認する**。
  Jetson の CUDA ビルド torch は pip の依存解決で CPU 版に差し替えられたり
  numpy をダウングレードされたりしやすい (本環境では無傷だったが確認は必須)。
- flash-attn は不要。torchcodec は本プラットフォームに無いため
  `--dataset.video_backend=pyav` を明示する (無指定でも pyav にフォールバックする
  WARNING が出るだけだが、明示が安全)。

## 2. smolvla_base の構造と rename_map

`lerobot/smolvla_base` (873MB) のファインチューンは `--policy.path=lerobot/smolvla_base`
で開始する (`--policy.type=smolvla` はスクラッチ初期化になるので使わない)。
チェックポイント config.json の実値 (本環境で 20K 学習後に確認):

| 項目 | 値 | 意味 |
|---|---|---|
| vlm_model_name | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | バックボーン VLM |
| num_vlm_layers | 16 | ロード時に「Reducing the number of VLM layers to 16」と出る (正常) |
| train_expert_only | True | 学習対象は action expert 約 100M パラメータのみ (VLM は凍結) |
| freeze_vision_encoder | True | vision encoder も凍結 |
| chunk_size / n_action_steps | 50 / 50 | アクションチャンク長 |
| num_steps | 10 | flow-matching の積分ステップ数 |
| max_state_dim / max_action_dim | 32 | **これ未満の次元はパディングで自動吸収** (16 軸 OK) |
| resize_imgs_with_padding | [512, 512] | 画像はポリシー側でリサイズ (高解像度データセット不要) |
| optimizer_lr / warmup / decay | 1e-4 / 1000 / 30000 | 既定スケジューラ。20K 学習では lr は 2.5e-6 まで低下 (実測) |

### rename_map (最大のハマり所)

- smolvla_base の入力特徴量は **`observation.images.camera1` / `camera2` / `camera3`
  の 3 カメラ固定**。データセットのカメラ名 (例: `observation.images.front`) とは
  一致しないため、そのままでは特徴量検証を通らない。
- 対処は `--rename_map='{"observation.images.front": "observation.images.camera1"}'`。
  **学習と推論の両方に、同一の値で**付ける (チェックポイントの train_config.json に
  rename_map は保存されるが、rollout 側では引数として再度渡す)。
- **データセット側がポリシー入力の部分集合なら検証が通る**: camera2/camera3 を
  持たないデータセット (camera1 のみ) で学習・推論とも動作することを実測確認。
  カメラが 2 台以上あるなら camera2/camera3 に追加マップすればよい。
- 状態 16 次元 / 行動 16 次元は max_state_dim/max_action_dim=32 へのパディングで
  自動吸収されるため、次元適応の作業・引数は一切不要
  (config.json の input_features に基底モデル由来の shape が残っていても実害なし)。

## 3. 学習の実測データ (Thor, batch8)

コマンドは SKILL.md / `run_train06_smolvla.sh` のとおり
(AMP なし・num_workers=0・save_freq=5000)。実測 (2026-08-13, 20000 steps):

| 項目 | 実測値 |
|---|---|
| スループット | 4.05〜4.25 step/s (立ち上がり数秒後から安定) |
| 20K steps 所要 | **1:19:37 (≈80 分)** |
| loss 推移 | step200: 2.168 → step400: 0.642 → step20K: **0.034** |
| grad norm | 8.98 (序盤) → 0.57〜0.59 (終盤) |
| updt_s / data_s | 0.155〜0.170 / 0.081〜0.082 |
| smp/s | 32〜34 |
| mem_gb (torch) | **2.35 で完全平坦** |
| データセット周回 | 40ep × 640×360 で 20K steps ≈ 9.05 epoch |

- 速さの理由は train_expert_only=True (100M のみ学習) + VLM 16 層 + 640×360 入力。
- **Thor では `PYTORCH_CUDA_ALLOC_CONF` を一切設定しない** (デフォルトアロケータが
  正解。障害の実測・機序・診断は thor-platform-skills の `reference/reference.md`
  §2〜§3 に集約)。SmolVLA でも同様に未設定で mem 平坦を確認。
- 空きメモリの事前チェックは 16GB 以上を推奨 (実走時の空きは 102GB だった)。
- 多重起動禁止: 同一 output_dir への 2 本目の学習はチェックポイント破損リスク。
  起動前に `pgrep -f "[l]erobot-(record|train|rollout)"` を確認する。

### HF ダウンロードの hf-xet ハング (初回の smolvla_base 取得で遭遇しうる)

- CDN 側の接続断後、hf-xet バックエンドは**エラーを出さず futex 待ちで無限ハング**する
  (数分間ゼロバイト、プロセスは Sleeping)。
- 診断: `.incomplete` ファイルの mtime と `~/.cache/huggingface/xet` の成長停止、
  `ss -tnp | grep <pid>` で CLOSE-WAIT ソケット。
- 対処: `HF_HUB_DISABLE_XET=1` で通常 HTTP 経路に切り替える (10 秒 read timeout で
  失敗が顕在化し、リトライ・レジュームが効くようになる)。

## 4. 推論と RTC (Real-Time Chunking)

- SmolVLA は flow-matching 系で **inference_delay に対応しており RTC が使える**
  (RTC 対応は pi0/SmolVLA/GR00T 等のみ。**ACT は非対応** — `--inference.type=rtc` を
  付けても使えないので sync + `--policy.n_action_steps` 調整で代替する)。
- 検証済み設定: `--inference.type=rtc --inference.queue_threshold=30` + chunk50
  (学習時の chunk_size のまま)。fps=30 の 60 秒実走で**滑らかに動作** — カクつきが
  出た GR00T (chunk40 + threshold30 = 0.33 秒ごとのチャンク切替) と異なり、
  chunk50 + threshold30 は切替頻度が半分で不連続が目立たない。
- **queue_threshold の意味**: キュー残がこの値を下回ると次チャンクの計算を開始する。
  小さいほどチャンク切替が減る。**RTC の合否はポリシー×chunk×threshold の組に依存し、
  実機評価が必須** (本環境の教訓)。
- `Indexes diff is not equal to real delay. indexes_diff=8, real_delay=9` の WARNING は
  成功した実走で毎秒出続けた。**無害**。
- 推論起動時に「Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights」と出るが、
  smolvla_base ダウンロード済み環境なら `HF_HUB_OFFLINE=1` でキャッシュから解決される
  (実測確認)。
- `--task` にはデータセット収録時と同じタスク文を渡す (SmolVLA は言語条件付き)。
- ヘッドレス環境では `--display_data=false` 必須 (rerun のチャネル詰まりで
  ループがブロックする)。

### 学習を止めずに途中チェックポイントを実機評価する (SIGSTOP 方式)

実例 `run_infer06_smolvla_paused.sh` の要点:

```bash
TRAIN_PID=$(pgrep -f "bin/lerobot-train" | head -1)
kill -STOP "$TRAIN_PID"; sleep 3          # GPU カーネルが掃けるのを待つ
trap 'kill -CONT '"$TRAIN_PID"' 2>/dev/null' EXIT   # Ctrl+C でも自動再開
# ... lerobot-rollout を実行 ...
```

- 学習進捗は失われない。unified memory の Thor では停止中の学習メモリと
  推論メモリが同居できることを実測確認 (60 秒推論 → 学習自動再開まで完走)。

## 5. 5 ポリシー実機比較 (本環境, 同一データセット 640×360)

SmolVLA を推奨する根拠。loss はポリシー間で定義が異なるため**横比較不可**
(収束の目安としてのみ読む):

| ポリシー | 学習実測 (Thor) | 所要 | 最終 loss | RTC | 実機評価 |
|---|---|---|---|---|---|
| **SmolVLA** | batch8, 4.2 step/s | 20K ≈ 80 分 | 0.034 | **対応・良好** | **RTC で滑らか。5 ポリシー中の推奨** |
| ACT | batch2 (1080p), 3.1 step/s | 30K | 0.143 | 非対応 | 動作良好。反応性は `--policy.n_action_steps=30` で改善 |
| GR00T N1.7 | batch4, 1.12 step/s | 15K ≈ 3.7h | 0.027 | 対応だがカクつき | sync 採用 (前処理 CPU 240ms 律速で 4Hz スロー再生、滑らか・確実) |
| VLA-JEPA (v2: chunk30) | batch4, 1.94 s/step | 25K ≈ 13.5h | 0.141 | 非対応 (sync のみ) | sync + K=8 平均で動作良好 (方向一致 0.93)。デフォルト chunk7 はタスク不成立 |
| FastWAM | batch4, 3.04 s/step | 15K ≈ 14.2h | 0.17 | 非対応 (**指定禁止**: 起動するが静かに壊れる) | sync + denoise 3 でタスク成功。オフライン指標 (方向一致 0.87〜0.89) は最良 |

- SmolVLA の位置づけ: **学習コスト最小 (GR00T の約 1/3、FastWAM・VLA-JEPA の
  約 1/10 の時間) で、実機のリアルタイム性 (RTC) を唯一問題なく満たした**。
- GR00T は品質は高いがリアルタイム化には毎ティック前処理の最適化が必要。
  RTC を活かすなら SmolVLA、が本環境の結論。

## 6. 診断コマンド集

```bash
# 学習ログからメトリクス行を抽出 (進捗バーの \r を改行に直す)
tr '\r' '\n' < train_smolvla.log | grep "loss:" | tail -5

# 学習プロセスの多重起動チェック (起動前に必ず)
pgrep -af "[l]erobot-(record|train|rollout)"

# 空きメモリ (16GB 未満なら NvMap リーク疑い → 再起動)
awk '/MemAvailable/ {printf "%d GB\n", $2/1048576}' /proc/meminfo
free -h

# Thor の禁止環境変数が漏れていないか
env | grep PYTORCH_CUDA_ALLOC_CONF   # 何も出ないのが正解

# チェックポイントの確認 (config.json + *.safetensors が揃っていること)
ls <output_dir>/checkpoints/last/pretrained_model/
python -c "
import json; c = json.load(open('<output_dir>/checkpoints/last/pretrained_model/config.json'))
print(c['type'], 'chunk:', c['chunk_size'], 'expert_only:', c['train_expert_only'])"

# 学習時の rename_map がチェックポイントに残っているか (推論側と一致させる)
python -c "
import json; t = json.load(open('<output_dir>/checkpoints/last/pretrained_model/train_config.json'))
print(t['rename_map'])"

# HF ダウンロードのハング診断 (hf-xet)
ls -l --time-style=full-iso ~/.cache/huggingface/hub/models--lerobot--smolvla_base/blobs/*.incomplete
ss -tnp | grep "$(pgrep -f lerobot-train | head -1)"   # CLOSE-WAIT ならハング
```
