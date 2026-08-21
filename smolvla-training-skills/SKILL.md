---
name: smolvla-training-skills
description: LeRobot 0.6.0 の SmolVLA を lerobot/smolvla_base からファインチューンし、RTC (Real-Time Chunking) で実機推論するまでをコーディングエージェントが実行するスキル。Jetson AGX Thor (JetPack 7 / CUDA 13) + LeRobotDataset v3 の実機構築 (2026-08) で検証済みの手順・実測値・落とし穴に基づく。
---

# 概要 (このスキルでできること)

収録済みの LeRobotDataset v3 を入力に、SmolVLA (`lerobot/smolvla_base`, 873MB) の
ファインチューンを起動・監視し、学習済みチェックポイントを RTC 推論
(`lerobot-rollout --inference.type=rtc`) で実機に流すまでを一気通貫で実施する。
SmolVLA は本環境の 6 ポリシー比較 (ACT / SmolVLA / GR00T N1.7 / VLA-JEPA / FastWAM / LingBot-VA) で
**学習が最速 (batch8 で 4.2 step/s、20K steps ≈ 80 分) かつ実機で RTC が滑らかに
動作した唯一のポリシー**であり、実機評価での推奨ポリシーである。
Jetson AGX Thor (unified memory 122GB) + lerobot 0.6.0 venv (torch 2.11.0+cu130) +
16 軸ヒューマノイド (rs_follower) + HSB カメラで実機検証済み。

# 実行前に必ず参照する

- 実装知見 (依存パッケージ・rename_map の理屈・実測データ表・診断コマンド):
  `./reference/reference.md`
- 実例スクリプト (このプロジェクトの検証済み実装):
  - 学習: `/home/jetson/RS/run_train06_smolvla.sh`
  - 推論 (RTC): `/home/jetson/RS/run_infer06_smolvla.sh`
  - 学習中の安全な推論 (SIGSTOP/SIGCONT 方式): `/home/jetson/RS/run_infer06_smolvla_paused.sh`
  - データセット縮小変換: `/home/jetson/RS/convert_dataset_resolution.py`

# 前提知識 (実行前に必ず理解すること)

1. **smolvla_base は 3 カメラ (camera1/2/3) 前提**: ポリシーの入力特徴量は
   `observation.images.camera1/2/3` 固定。データセットのカメラ名が違う場合は
   **`--rename_map='{"observation.images.front": "observation.images.camera1"}'` を
   学習・推論の両方に**付ける。データセット側の特徴量がポリシー入力の
   **部分集合**なら検証が通る (camera1 のみ供給で学習・推論とも動作を実測確認)。
2. **学習対象は action expert のみ**: `train_expert_only=True` (smolvla_base の既定)
   で学習されるのは約 100M パラメータのみ。VLM (SmolVLM2-500M、16 層に削減) と
   vision encoder は凍結。これが batch8 で 4.2 step/s・torch メモリ 2.35GB 平坦の理由。
3. **状態・行動次元はパディングで自動吸収**: `max_state_dim=32` / `max_action_dim=32`
   に満たない次元はゼロパディングされるため、16 軸ロボットでも次元適応の作業は不要
   (GR00T のような embodiment 適応も、VLA-JEPA のような次元上書きも意識しなくてよい)。
4. **RTC 対応**: SmolVLA は flow-matching 系で `inference_delay` に対応しており
   RTC が使える (ACT は非対応、GR00T は対応だが実機でカクつき sync 採用)。
   `queue_threshold` がチャンク切替頻度を決める。chunk50 + threshold30 で滑らかを実測。
5. **Thor では `PYTORCH_CUDA_ALLOC_CONF` を設定しない** (iGPU/CUDA13 での実測:
   `expandable_segments:True` はドライバ側リーク、`max_split_size_mb` は激遅化)。
6. **依存追加は transformers + num2words のみ**: 導入後は必ず torch が無傷か確認する
   (Jetson の CUDA ビルド torch は pip の依存解決で壊れやすい)。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot 0.6.0 venv と CUDA torch の確認
<venv>/bin/python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available())"
# 期待値: 2.11.0 True (Thor / cu130)

# SmolVLA の依存 (未導入なら)
<venv>/bin/pip install transformers num2words   # 検証済み: transformers 5.5.4 / num2words 0.5.14
<venv>/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # torch 無傷確認

# データセット (LeRobotDataset v3) の確認
ls <dataset_root>/meta/info.json   # 存在すること
grep -o '"total_episodes": [0-9]*' <dataset_root>/meta/info.json
```

- **多重起動チェック**: `pgrep -f "[l]erobot-(record|train|rollout)"` が空であること。
  同じ学習を 2 本起動するとチェックポイント破損リスクがある。
- **空きメモリチェック**: `awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo`
  が 16 以上 (GB)。少なければ NvMap リークの可能性 → 再起動。
- カメラ解像度はデータセット準拠で高解像度は不要 (ポリシー側で 512×512 パディング付き
  リサイズ)。1080p 収録データは `convert_dataset_resolution.py` で 640×360 に縮小すると
  読み込みが軽い (631MB→158MB の実績。実例: `/home/jetson/RS/humanoid_test060_640`)。

## Step 2: 学習起動

環境非依存の形 (venv の `lerobot-train` を使用):

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --rename_map='{"observation.images.front": "observation.images.camera1"}' \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=local/<name> \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=<output_dir> \
    --wandb.enable=false \
    --steps=20000 \
    --batch_size=8 \
    --num_workers=0 \
    --save_freq=5000
```

実例: `sh /home/jetson/RS/run_train06_smolvla.sh` (STEPS/BATCH/DATASET_ROOT 等を
環境変数で上書き可。既存 output_dir の日時バックアップ・事前チェック込み)。

- `--policy.type=smolvla` ではなく **`--policy.path=lerobot/smolvla_base`**
  (ファインチューンなので事前学習済み重みから開始する)。
- **初回は `HF_HUB_OFFLINE` を設定しない** (smolvla_base 873MB のダウンロードが走る)。
  ダウンロードが無言で止まる場合は hf-xet のハング → `HF_HUB_DISABLE_XET=1` を付けて
  リトライ (詳細は reference.md)。
- 長時間になるので `nohup ... > train_smolvla.log 2>&1 &` で起動し、起動前に
  必ず `pgrep -f lerobot-train` で多重起動がないことを確認する。

## Step 3: 学習監視

```bash
# 進捗バー (\r) を改行に直してメトリクス行を抽出
tr '\r' '\n' < train_smolvla.log | grep "loss:" | tail -3
```

正常時の基準値 (Thor, batch8, 640×360, 40ep で実測):

| 項目 | 基準値 (実測) |
|---|---|
| スループット | 4.05〜4.25 step/s (立ち上がり後) |
| 20K steps 所要 | 1 時間 20 分 (実測 1:19:37) |
| loss | step200: 2.168 → step400: 0.642 → step20K: **0.034** |
| updt_s / data_s | 0.155〜0.170 / 0.081 |
| mem_gb (torch) | 2.35 で**平坦** (増え続けるなら異常) |

- step/s が大きく落ちる・mem_gb が増え続ける → `free -h` を確認。
  `PYTORCH_CUDA_ALLOC_CONF` が設定されていないか最初に疑う (Thor の既知障害)。
- チェックポイントは `<output_dir>/checkpoints/{005000,010000,...,last}/pretrained_model`。

## Step 4: 実機推論 (RTC)

環境非依存の形 (venv の `lerobot-rollout` を使用):

```bash
lerobot-rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.queue_threshold=30 \
    --policy.path=<output_dir>/checkpoints/last/pretrained_model \
    --rename_map='{"observation.images.front": "observation.images.camera1"}' \
    --robot.type=<robot> \
    --robot.cameras='{...}' \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="<データセットと同じタスク文>" \
    --display_data=false
```

実例: `sh /home/jetson/RS/run_infer06_smolvla.sh 60`
(can0/カメラリンク/ポリシーファイルの事前チェック、`HF_HUB_OFFLINE=1`、
`--return_to_initial_position=true` 込み)。

- **rename_map を推論でも忘れない** (学習と同一の値。付け忘れが最頻のミス)。
- `HF_HUB_OFFLINE=1` で可 (VLM 重みのロードは smolvla_base 取得時のキャッシュで解決される)。
- ヘッドレスでは `--display_data=false` 必須。
- ログに `Indexes diff is not equal to real delay` の WARNING が毎秒出るが**無害**
  (滑らかに動作した 60 秒実走で終始出続けたことを確認済み)。
- 学習を止めずに途中チェックポイントを試すには SIGSTOP/SIGCONT 方式が安全
  (実例: `run_infer06_smolvla_paused.sh` — 学習プロセスを一時停止→推論→trap で自動再開)。
- 安全: ポリシーはアームを自律で動かす。周囲の安全確保と Ctrl+C 即応体制を必ず取る。

## Step 5: トラブル対処

| 症状 | 原因 | 対処 |
|---|---|---|
| 特徴量検証エラーで学習が始まらない | カメラ名がポリシー入力 (camera1/2/3) と不一致 | `--rename_map` でデータセット側のカメラ名を `camera1` にマップ |
| ダウンロードが無言で止まる (数分ゼロバイト) | hf-xet が CDN 接続断後に futex 待ちで無限ハング | プロセス停止 → `HF_HUB_DISABLE_XET=1` を付けて再実行 |
| 学習が step 途中から激遅化 / メモリ枯渇 | Thor で `PYTORCH_CUDA_ALLOC_CONF` が設定されている | unset する (デフォルトアロケータが正解)。枯渇後は再起動 |
| transformers 導入後に torch が壊れた | pip の依存解決による差し替え | torch スタックを再インストール。以後は導入直後の torch 確認を必須化 |
| 推論でロボットが動かない/エラー | 推論側の rename_map 忘れ | 学習と同一の `--rename_map` を付ける |
| `Indexes diff is not equal to real delay` WARNING | RTC の内部整合ログ | 無害。対処不要 |
| RTC がカクつく | queue_threshold とチャンク長の相性 | `--inference.queue_threshold` を下げて切替頻度を減らす (合否はポリシー×chunk×threshold 依存、実機評価が必須) |

## Step 6: 知見の記録

- 新たな実測値 (別バッチサイズのスループット、別ロボットでの RTC 評価など) は
  `./reference/reference.md` の表に追記する。
- 学習・推論コマンドを変更した場合は実例スクリプトと本書の整合を保つ。
