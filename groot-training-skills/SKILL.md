---
name: groot-training-skills
description: NVIDIA GR00T N1.7 (3B VLA) を LeRobot 0.6.0 で独自データセット (LeRobotDataset v3) にファインチューンし、実機自律動作 (lerobot-rollout) まで実施するスキル。Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) での実機構築・6ポリシー比較評価 (2026-08) で検証済みの手順と実測値に基づく。
---

# 概要

GR00T N1.7 (`nvidia/GR00T-N1.7-3B`) を独自のロボットデータセット
(LeRobotDataset v3) にファインチューンし、学習の監視 → 中間チェックポイントの
段階評価 → 実機推論 (sync) までをコーディングエージェントが自走するためのスキル。
本スキルは Jetson AGX Thor + lerobot 0.6.0 venv (torch 2.11.0+cu130) +
16軸ヒューマノイド (両腕 7DOF+グリッパ) + HSB カメラの実機構築 (2026-08) で
15K ステップ完走 (loss 0.027)・実機での把持成立まで検証した知見に基づく。

# 実装前に必ず参照する

- 実装知見 (起動の落とし穴・実測データ表・sync/RTC 実機評価・診断コマンド):
  `./reference/reference.md`

# 前提知識 (作業前に必ず理解すること)

1. **lerobot 0.6.0 の GR00T は N1.7 専用** (N1.5 サポートは打切り)。
   `--policy.type=groot` + `--policy.base_model_path=nvidia/GR00T-N1.7-3B` で起動する。
2. **`--policy.path=nvidia/GR00T-N1.7-3B` は不可**: NVIDIA 公式リポジトリの生
   config.json は draccus が解釈できず ParsingError になる。`--policy.path` が
   使えるのは lerobot が保存したチェックポイントのみ (詳細は reference.md)。
3. **バックボーン `nvidia/Cosmos-Reason2-2B` はゲート付き**: HF の Web で
   ライセンス同意 + `hf auth login` を済ませないとダウンロードで止まる。
4. **カメラの rename は不要**: `embodiment_tag=new_embodiment` (既定) が
   データセットのカメラ名・state/action 次元に自動適応する
   (SmolVLA のような `--rename_map` は要らない)。
5. **推論は sync を採用する**: GR00T は RTC ネイティブ対応 (`rtc_ramp_rate` あり)
   だが、実機評価ではチャンク切替のカクつきが解消できず sync の方が滑らかで確実
   だった。sync は毎ティックの前処理 (CPU 約240ms) 律速で制御ループ約 4Hz の
   スローモーション再生になるが、把持は成立する。RTC の合否はポリシー×chunk×
   threshold 依存なので、使うなら必ず実機評価する。
6. **Thor では `PYTORCH_CUDA_ALLOC_CONF` を設定しない** (メモリリーク/激遅化の
   実測あり。デフォルトアロケータが最適)。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot 0.6.0 venv と CUDA torch
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
# → 0.6.0 / 2.11.0+cu130 / True (実例 venv: /home/jetson/camera/lerobot060-venv)

# GR00T の追加依存 (未インストールなら reference.md の「依存パッケージ」参照)
<venv>/bin/python -c "import peft, diffusers, timm, tree; print('groot deps OK')"

# ゲート付きバックボーンの認証 (初回ダウンロード前に必須)
<venv>/bin/hf auth whoami   # 未ログインなら: hf auth login
# + https://huggingface.co/nvidia/Cosmos-Reason2-2B でライセンス同意しておく

# データセット (LeRobotDataset v3)
cat <dataset_root>/meta/info.json | grep -E "total_episodes|total_frames|fps"

# 空きメモリ (3B モデルは 40GB 以上を推奨) と二重起動の防止
awk '/MemAvailable/ {printf "%d GB\n", $2/1048576}' /proc/meminfo
pgrep -f "lerobot-(record|train|rollout)" && echo "NG: 別プロセス実行中"
```

- 学習データセットは**低解像度版 (640×360) を推奨**。GR00T は内部で画像を
  256×256 に処理するため高解像度は動画デコードの無駄になる。1080p 収録済みなら
  縮小変換ツールで別データセットを作る
  (実例: `/home/jetson/RS/convert_dataset_resolution.py`、元データ無変更・
  フレーム数検証付き。631MB→158MB)。

## Step 2: 学習起動

環境非依存の形 (数時間コースなので nohup + ログ推奨):

```bash
export HF_HUB_OFFLINE=0   # 初回はバックボーンのダウンロードが走る
lerobot-train \
    --policy.type=groot \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=<repo_id> \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=<output_dir> \
    --job_name=<job_name> \
    --wandb.enable=false \
    --steps=15000 \
    --batch_size=4 \
    --num_workers=0 \
    --save_freq=5000
```

- 実例: `/home/jetson/RS/run_train06_groot.sh`
  (事前チェック・既存出力の日時バックアップ・環境変数 STEPS/BATCH 対応込み)。
- 学習対象は projector + DiT (diffusion head) + vlln の約 1.6B (VLM 本体は凍結、
  既定のまま)。flash-attn 不要 (`use_flash_attention=False` 既定で Jetson で動く)。
- AMP は指定しない (GR00T は `use_bf16=True` 既定で内部 bf16。ACT レシピの
  `--policy.use_amp=true` を流用しないこと)。

## Step 3: 監視

```bash
tail -f <log> | grep --line-buffered "ot_train.py:606"
# step:10K ... loss:0.040 ... updt_s:0.878 ... mem_gb:35.90 のような行が約3分ごと
```

| 監視項目 | 正常値 (Thor, batch 4, 640×360, 40ep) | 異常時 |
|---|---|---|
| step/s (進捗バー) | 1.08〜1.12 step/s | 大幅低下→ `free -h` でメモリ枯渇確認 |
| mem_gb | 35.9 で平坦 | 単調増加→アロケータ設定を疑う (Thor は未設定が正解) |
| loss | 10K で 0.04 前後 → 15K で 0.03 弱 | 下がらない→データセット/タスク文を確認 |
| 所要時間 | 15K steps ≈ 3.7h (5K ≈ 78分) | — |

- 初回のダウンロードが数分間ゼロバイトで固まる場合は hf-xet ハング
  → `HF_HUB_DISABLE_XET=1` を付けて再実行 (reference.md 参照)。

## Step 4: 中間チェックポイントの段階評価 (推奨)

`save_freq=5000` で 5K/10K/15K のチェックポイントができる。学習を止めずに
**SIGSTOP で一時停止 → 中間チェックポイントで実機推論 → SIGCONT で再開**すると、
改善傾向を早期に確認できる (実測でも 5K→10K→15K と段階的に改善)。

実例: `/home/jetson/RS/run_infer06_groot_early.sh`
(`sh run_infer06_groot_early.sh 60 010000` = 010000 で 60秒。trap で Ctrl+C でも
学習を自動再開。SIGSTOP 後は GPU カーネルが掃けるまで 3 秒待つ)。

## Step 5: 実機推論 (sync)

環境非依存の形:

```bash
export HF_HUB_OFFLINE=1   # 学習済みならオフラインで起動が速く確実
lerobot-rollout \
    --strategy.type=base \
    --inference.type=sync \
    --policy.path=<output_dir>/checkpoints/last/pretrained_model \
    --robot.type=<robot_type> \
    --robot.cameras='<学習時と同じカメラ構成 (解像度も一致させる)>' \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="<学習時と同じタスク文>" \
    --display_data=false
```

- 実例: `/home/jetson/RS/run_infer06_groot.sh`
  (can0/カメラリンク/チェックポイント存在の事前チェック、`INFER=sync|rtc` と
  `QT=<queue_threshold>` の切替ノブ、`--return_to_initial_position=true` 付き)。
- **カメラ解像度は学習データセットと一致させる** (実例では
  `{type: hsb, camera_mode: 1, width: 640, height: 360, ...}` で 640×360 に縮小出力)。
- sync の挙動 (仕様として理解しておく): 毎ティックの前処理 (VLM 画像処理+
  トークン化、CPU 約240ms) 律速で制御ループは約 4Hz = 実演の約 1/7 の
  スローモーション再生。ただし動きは滑らかで確実、把持成立。
  「Record loop is running slower」警告は rollout では無害。
- RTC を試す場合は `INFER=rtc QT=5` 等で**必ず実機評価**する
  (実測では chunk40 + threshold30 で 0.33 秒ごとの切替カクつき、QT=5 でも
  改善不足で sync を採用。判断材料は reference.md の比較表)。

## Step 6: 継続学習 (resume でステップ延長)

チェックポイントから総ステップ数を延長して再開できる:

```bash
lerobot-train \
    --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --steps=<延長後の総ステップ数>
```

- 実例: `/home/jetson/RS/run_train06_groot_resume.sh`
  (延長後の総ステップ数を引数で受け、現チェックポイント以下なら拒否。
  二重起動・空きメモリの preflight 付き)。実測の 15K 完走も 10K から
  この方式で延長した。
- **`--steps` を延長すると lr スケジュール (cosine) は新しい総数に引き伸ばされて
  再計算される**ため、resume 後も lr が生きて改善が続く (reference.md §4)。
- 実測 (2026-08-15): 15K→20K の +5K resume で loss 0.027→**0.023**
  (1.06〜1.08 step/s、+5K ≈ 78分)。

## Step 7: 知見の記録

- 新たな知見 (別ロボット・別データセットでの実測、RTC が通る条件の発見など) は
  `./reference/reference.md` に追記する。
- 学習ログ (`ot_train.py:606` 行) と rollout の体感評価 (滑らかさ・把持成否) を
  セットで残すこと。RTC/sync の判断はログだけでは下せない。
