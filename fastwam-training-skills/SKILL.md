---
name: fastwam-training-skills
description: LeRobot 0.6.0 の FastWAM ポリシー (--policy.type=fastwam、Wan2.2-TI2V-5B video diffusion + action expert 1B の MoT 共拡散) の学習・推論を Jetson AGX Thor 上で実行するスキル。必須引数 action_dim/proprio_dim と cross-embodiment ロード、base 非互換 config の無警告ランダム初期化の罠、RTC の「静かな誤動作」(禁止)、denoise ステップ削減 (10→3) の無料高速化を含む。40エピソード・15K steps の学習完走と実機評価 (タスク成功、6ポリシー中最良クラス) の実測 (2026-08-14〜15) に基づく。
---

# 概要

LeRobot 0.6.0 組み込みの FastWAM ポリシー (`--policy.type=fastwam`) を、
自前の LeRobotDataset v3 データセットで学習し、実機で推論するまでを実施する。
前提確認 → バックボーン事前取得 → 学習起動 → 監視 → オフライン予測評価 →
実機推論、の順で進める。

FastWAM は Wan2.2-TI2V-5B (video diffusion) + action expert 1B の
MoT (Mixture of Transformers) 共拡散モデルで、`--policy.type=fastwam` を指定する
だけで公開チェックポイント `lerobot/fastwam_base` (12.04GB、apache-2.0) からの
ファインチューンになる。Jetson AGX Thor + 16軸ヒューマノイド (rs_follower) +
40エピソードの実機検証 (2026-08-14〜15) で、**オフライン指標・実機とも
6ポリシー (ACT/SmolVLA/GR00T/VLA-JEPA/FastWAM/LingBot-VA) 中最良クラス**
(高モーション窓の移動方向一致率 0.87-0.89、実機タスク成功) を確認した。

# 実装前に必ず参照する

- 実装知見 (罠の機構・実測データ・診断コマンド): `./reference/reference.md`
- プラットフォーム共通の掟 (`PYTORCH_CUDA_ALLOC_CONF` 禁止・`kill -9` 禁止・
  nohup 作法・`HF_HUB_DISABLE_XET=1`・preflight): thor-platform-skills
- 実機検証済みスクリプトの実例:
  `/home/jetson/RS/run_train06_fastwam.sh` (学習) /
  `/home/jetson/RS/run_train06_fastwam_overnight.sh` (夜間ランチャー) /
  `/home/jetson/RS/run_infer06_fastwam.sh` (実機推論)

# 前提知識 (作業前に必ず理解すること)

1. **構成**: FastWAM は lerobot 0.6.0 組み込みポリシー (プラグイン不要)。
   学習対象は video expert 5B + action expert 1B = 6.02B (bf16)。
   `--policy.type=fastwam` だけで config の `__post_init__` が
   `pretrained_path=lerobot/fastwam_base` を自動設定し、ファインチューンになる。
   **追加 pip 依存はゼロ** (transformers 5.5.4 / diffusers 0.35.2 の既存 venv で
   そのまま動いた)。ダウンロードは計約 26GB (reference.md §1)。
2. **必須引数: `--policy.action_dim=16 --policy.proprio_dim=16`** (16軸機の場合)。
   データセットから自動適応**しない**。action_dim 漏れは validate_features で
   即エラー、proprio_dim 漏れは config 検証を素通りして**最初の forward で
   実行時エラー**になる。指定すると次元不一致テンソル (action encoder/head +
   proprio encoder の6個) のみ再初期化される cross-embodiment ロードが働く
   (正常確認の方法は reference.md §2)。
3. **罠 (最重要)**: hidden_dim 等のアーキテクチャ系 config を base 非互換に
   変えると、**警告なしで pretrained_path が設定されず 5B がランダム初期化**の
   スクラッチ学習になる (reference.md §3)。action_dim/proprio_dim は互換判定
   キー外なので 16 にしても安全 (実行確認済み)。
4. **rename 不要・task 必須**: カメラ名は `observation.images.*` を sorted で
   自動採用。task はプロンプトテンプレートに埋め込まれ UMT5-xxl でエンコード
   されるため、データセットに task が無いと KeyError で落ちる。
5. **RTC 禁止**: `predict_action_chunk` が `**_` で kwargs を無言破棄するため、
   `--inference.type=rtc` は**エラーなく起動するが遅延補償が壊れる「静かな
   誤動作」**になる (reference.md §5)。推論は **sync のみ**。chunk=32
   (30fps で 1.07秒分)。代わりに **denoise ステップ削減
   (`--policy.num_inference_steps` 10→3) が品質劣化なしの無料高速化**になる
   (552→322ms、reference.md §8)。
6. **LIBERO 用ハック (`toggle_action_dimensions`) は設定禁止**: VLA-JEPA の
   グリッパーハックと同種の機構が存在するが、デフォルト `[]` で無効かつ
   `fastwam_base` の postprocessor JSON にも不在を実物確認済み。触らなければ
   発火しない。ただし `--policy.path` で第三者のチェックポイントから学習する
   場合は JSON の検査が必要 (reference.md §4)。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot 0.6.0 venv と CUDA torch (追加 pip 依存はなし)
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"

# データセット存在確認 (LeRobotDataset v3、task 必須)
cat <dataset_root>/meta/info.json | grep -o '"total_episodes": [0-9]*'
```

- 空きメモリ: フル 6B 学習は **75GB 以上** (実測ピーク mem_gb 68.2)。
  足りなければ再起動、それでも無理なら `FREEZE_VIDEO=true` (45GB、
  action expert 1B のみ学習) にフォールバック。
- ディスク: HF キャッシュ約 26GB + チェックポイント (1個約 34GB =
  pretrained_model 12GB + training_state 22GB。save_freq=2500 の 15K 学習で
  **6個 計202GB**、`last` は最新への symlink) の空きを確認する。
- 二重起動禁止・アロケータ・kill の掟は thor-platform-skills の preflight に従う。

## Step 2: バックボーンの事前取得 (計約26GB)

**`--include` で絞ることが重要** — Wan-AI リポジトリの `transformer/` (約20GB) は
学習経路では不要 (5B の重みは fastwam_base 側から来る。reference.md §1):

```bash
export HF_HUB_DISABLE_XET=1   # hf-xet の無限ハング回避 (thor-platform-skills)
<venv>/bin/hf download lerobot/fastwam_base                                    # 12.04GB
<venv>/bin/hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --include "text_encoder/*" "vae/*" "*.json"                                # 11.4GB + 2.8GB
<venv>/bin/hf download google/umt5-xxl --include "spiece.model" "tokenizer*" "*.json"  # 25MB
```

3リポジトリとも公開・**非ゲート** (GR00T のような HF ライセンス同意は不要)。

## Step 3: 学習起動

```bash
# HF_HUB_OFFLINE は設定しない (初回はバックボーンをダウンロードする)
export HF_HUB_DISABLE_XET=1

lerobot-train \
    --policy.type=fastwam \
    --policy.action_dim=16 \
    --policy.proprio_dim=16 \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=<repo_id> \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=<output_dir> \
    --wandb.enable=false \
    --steps=15000 \
    --batch_size=4 \
    --num_workers=2 \
    --save_freq=2500
```

- `--policy.toggle_action_dimensions` は**絶対に設定しない** (LIBERO 用ハック)。
- `--num_workers=2`: FastWAM は1サンプルにつき動画9フレームをデコードするため
  dataloader が重く、0 より 2 を推奨 (実測は 2 で data_s:0.016 と余裕)。
- メモリ不足時は `--policy.freeze_video_expert=true` (1B のみ学習、45GB)。
  docstring 推奨では `lambda_video=0` 併用だが loss dict の CLI 上書きは未検証
  (書式は reference.md §10)。
- 実例: `/home/jetson/RS/run_train06_fastwam.sh` (preflight 75GB/45GB、
  既存出力の自動バックアップ、FREEZE_VIDEO 環境変数対応)。

### 夜間学習の運用

**ユーザーのターミナルから** nohup 起動する (作法は thor-platform-skills):

```bash
nohup sh /home/jetson/RS/run_train06_fastwam_overnight.sh > train_fastwam.log 2>&1 &
```

overnight ラッパーは「進行中ダウンロードを待つ (30分無応答で引き継ぐ) →
バックボーン3点を `--include` 付きリトライループで確実にキャッシュ (60秒間隔・
最大2時間) → フル 6B 学習 → **序盤30分以内のクラッシュなら FREEZE_VIDEO=true で
自動フォールバック**」という構成。実例:
`/home/jetson/RS/run_train06_fastwam_overnight.sh`
(2026-08-14 の実走ではフル 6B が問題なく完走し、フォールバックは不発)。

## Step 4: 学習監視

```bash
# ログには tqdm のヌル文字が混ざるので tr -d '\0' を挟む
tail -c 2000 train_fastwam.log | tr -d '\0' | tail -5
grep -a -o "loss:[0-9.]*" train_fastwam.log | tail -3

# base からのファインチューンになっているか (最重要、reference.md §3)
grep -a -o "'pretrained_path': '[^']*'" train_fastwam.log
# → 'pretrained_path': 'lerobot/fastwam_base' が出ること
```

期待値 (Thor 実測、batch4・640×360・16軸・40ep、2026-08-14〜15):

| 項目 | 実測値 |
|---|---|
| 起動 (重みロード〜学習開始) | 約1.5時間 |
| スループット | 3.04 s/step (updt_s 3.016) |
| 15K steps 所要 | 学習 12.7h + 起動 ≈ **14.2h** (実効 3.4 s/step) |
| メモリ (mem_gb) | 68.2 GB で完全平坦 |
| loss | 0.993 (step200) → 0.168〜0.182 (15K、**まだ下降中**) |
| チェックポイント | save_freq=2500 → 002500 … 015000 の6個 (各約34GB、計202GB。`last` は 015000 への symlink) |

- ロード完了時に WARNING `Missing key(s) when loading model: {...action_encoder/
  head/proprio_encoder の6テンソル}` が出るのが**正常** (16軸への
  cross-embodiment 適応の証拠。reference.md §2)。
- loss は 15K 時点でまだ下降中だった → **resume で総ステップ延長の価値あり**
  (手順は thor-platform-skills Step 3)。
- 失速・メモリ異常の一次対応は thor-platform-skills Step 7。

## Step 5: オフライン予測評価 (実機に載せる前に必ず)

`tools/eval_fastwam_offline.py` (rollout と同一経路でロード) で、均一5点 +
高モーション5窓の予測 vs 正解を評価する。VLA-JEPA で実機暴走バグを検出した
評価セットの FastWAM 版で、postprocessor に Toggle step が無いことも assert する。

期待値 (15K チェックポイント、denoise 10 ステップ):

| 指標 | 実測 | 判定基準 |
|---|---|---|
| MAE | 2.0〜2.6 | ホールド基準を下回ること |
| 高モーション窓の移動方向一致率 | **0.87〜0.89** | 0.5 (ランダム) 近傍なら実機は期待できない |
| 予測移動量 | 正解の8割超 | ほぼ静止予測なら局所解落ち |
| 定数出力 | なし | どこかの次元が定数なら postprocessor 混入を疑う |
| チャンク推論 (32アクション) | 552ms | — |

このあと `--policy.num_inference_steps` を 10→3 に落として再評価し、劣化が
ないことを確認してから実機の既定にする (実測で MAE 2.79→2.52 とむしろ改善。
reference.md §8)。

## Step 6: 実機推論

**sync のみ** (`--inference.type=rtc` は禁止 — 起動するが静かに壊れる)。
カメラ解像度・露光・タスク文・初期姿勢は学習時と一致させる:

```bash
export HF_HUB_OFFLINE=1

lerobot-rollout \
    --strategy.type=base \
    --policy.path=<checkpoint>/pretrained_model \
    --policy.num_inference_steps=3 \
    --robot.type=<robot_type> \
    --robot.cameras='<学習データと同じ解像度のカメラ設定>' \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="<学習データセットの single_task と同一文字列>" \
    --display_data=false
```

- 短時間 (20秒) から始め、**いつでも Ctrl+C できる状態**で。障害物を排除する。
- `--policy.num_inference_steps=3` は再学習不要の CLI 上書き
  (チャンク推論 552→322ms、品質劣化なし)。
- 推論前チェックの実例 (`/home/jetson/RS/run_infer06_fastwam.sh`、INFER_STEPS=3
  既定): ロボットバス (can0) up / カメラリンク / ポリシーの config.json と
  .safetensors 存在 / 学習プロセス非実行 / 空きメモリ **35GB 以上**
  (Wan2.2 5B + UMT5 のロードに必要)。
- 実機評価 (2026-08-15): **問題なく動作、タスク成功**。制御ループ律速の
  スロー再生 (6〜7Hz — 生ログ未保存のため**要再測定**) は他ポリシーの sync と
  同傾向だが、チャンク32 (1.07秒分) を 322ms で生成するため切替時の停滞は
  短い (reference.md §9)。
- 途中チェックポイントの試走は `--policy.path` を
  `checkpoints/012500/pretrained_model` 等に差し替える。

## Step 7: トラブル対処と知見の記録

| 症状 | 原因 | 対処 |
|---|---|---|
| 起動時「action feature shape must be (7,), got (16,)」 | `--policy.action_dim` 漏れ | `--policy.action_dim=16` を指定 |
| 最初の forward で「proprio last dim must be 8, got 16」 | `--policy.proprio_dim` 漏れ | `--policy.proprio_dim=16` を指定 |
| loss が高いまま・ロード系 WARNING が出ない | base 非互換 config で 5B が無警告ランダム初期化 | ログの pretrained_path を確認し config 変更を戻す (reference.md §3) |
| RTC を指定したい | predict_action_chunk が kwargs を無言破棄 | **禁止**。sync のみ。RTC が要るなら SmolVLA / pi0 系へ |
| 推論チャンクが遅い | denoise 10 ステップ | `--policy.num_inference_steps=3` (劣化なし、床 ~290ms = 5B プリフィル) |
| 学習がメモリ不足で落ちる | フル 6B (preflight 75GB) | `FREEZE_VIDEO=true` で 1B のみ学習 (45GB) |
| ダウンロードが 26GB を大幅超過 | Wan-AI の transformer/ (20GB) を掴んだ | `--include "text_encoder/*" "vae/*" "*.json"` で絞る |
| ディスク枯渇 | チェックポイント 6個 202GB | 実機確認後に中間チェックポイントを整理 (reference.md §10) |
| 他人のチェックポイントから学習したい | toggle step 焼き込みの可能性 | policy_postprocessor.json を検査 (reference.md §4) |
| DL 無言停止・失速・kill 後のメモリ未回収 | プラットフォーム共通 | thor-platform-skills Step 7 |

- 新たな知見 (resume 延長の結果、freeze コースの実測、別ロボットでの検証等) は
  `./reference/reference.md` に追記する。
- 実機に載せる前のオフライン予測評価 (Step 5) を必ず挟むこと。
