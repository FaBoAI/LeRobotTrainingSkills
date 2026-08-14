---
name: act-training-skills
description: LeRobot 0.6.0 の ACT ポリシーを Jetson 上で学習し、lerobot-rollout による実機自律動作(推論)まで実施するスキル。Jetson AGX Thor (JetPack 7 / CUDA 13) + LeRobotDataset v3 の実機構築 (2026-08) で 30K steps 学習 → 実機タスク成功まで検証済みの手順と実測値に基づく。
---

# 概要

実機収録した LeRobotDataset v3 から ACT (Action Chunking Transformer) ポリシーを
学習し、チェックポイント管理(継続学習)・学習監視・`lerobot-rollout` による
実機自律動作までを一貫して実行する。FaBo レシピ(AMP + `use_vae=false` +
`chunk_size=50`)と、データセットの低解像度化(640×360)によるスループット
約8.4倍の高速化パイプラインを含む。本スキルは Jetson AGX Thor
(JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0 venv
(torch 2.11.0+cu130) での実機検証(1080p 30K steps loss 0.143 /
640×360 15K steps loss 0.114、いずれも実機タスク成功)に基づく。

# 実装前に必ず参照する

- 実装知見(実測スループット・Thor アロケータ問題・RTC 非対応の詳細・落とし穴): `./reference/reference.md`
- 実機検証済みスクリプト(このプロジェクトの実例):
  - 学習: `/home/jetson/RS/run_train06.sh`(env 上書き対応・preflight 付き)
  - 継続学習: `/home/jetson/RS/run_train06_resume.sh`
  - 推論: `/home/jetson/RS/run_infer06.sh`(事前チェック付き)
  - 低解像度化: `/home/jetson/RS/convert_dataset_resolution.py`

# 前提知識(実行前に必ず理解すること)

1. **学習レシピ (FaBo)**: `--policy.type=act` + `--policy.use_amp=true` +
   `--policy.use_vae=false` + `--policy.chunk_size=50`。Jetson では
   `--num_workers=0`、`--dataset.video_backend=pyav` を使う。
2. **Thor では `PYTORCH_CUDA_ALLOC_CONF` を一切設定しない**(最重要)。
   FaBo の Orin 向けレシピにある同設定は Thor (iGPU / CUDA 13) では
   ドライバ側メモリリークまたは激遅化を起こす(実測 2026-08-12)。
   デフォルトアロケータが最適。詳細は `reference.md`。
3. **ACT は RTC (`--inference.type=rtc`) 非対応**(実測で TypeError)。
   反応性を上げたい場合は sync のまま `--policy.n_action_steps` を CLI で
   小さく上書きする(再学習不要)。`chunk_size` 自体の変更は要再学習。
4. **入力解像度がスループットを支配する**: 1080p batch2 で 3.1 step/s に対し
   640×360 batch8 で 6.3 step/s(サンプル/秒換算で約8.4倍)。精度も
   640×360 15K steps (loss 0.114) で実機タスク成功しており、まず低解像度で
   回すのが効率的。
5. **同一学習の二重起動は厳禁**(チェックポイント破損リスク)。起動前に
   `pgrep -f lerobot-train` で確認する。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot venv と CUDA の確認 (期待: 0.6.0 / 2.11.0+cu130 / True)
<lerobot_venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"

# データセットの存在確認 (LeRobotDataset v3: meta/info.json が必須)
grep -o '"total_episodes": [0-9]*' <dataset_root>/meta/info.json

# 空きメモリ preflight (8GB 未満なら NvMap リークの疑い → 再起動)
awk '/MemAvailable/ {printf "%d GB\n", $2/1048576}' /proc/meminfo

# 二重起動チェック (何も出ないこと)
pgrep -af lerobot-train
```

実例: この環境の venv は `/home/jetson/camera/lerobot060-venv`、
データセットは `/home/jetson/RS/humanoid_test060`(30 エピソード / 13181 フレーム @30fps)。

## Step 2: (推奨) データセットの低解像度化

1080p のまま学習も可能だが、640×360 に縮小すると学習が約8.4倍速くなり、
実機性能も確認済み。変換ツールは元データセットと同一コーデック設定
(libsvtav1 / CRF / GOP)で再エンコードし、フレーム数検証まで行う。

```bash
python3 convert_dataset_resolution.py \
    --src <dataset_root> \
    --dst <dataset_root>_640 \
    --width 640 --height 360
```

実例: `/home/jetson/RS/convert_dataset_resolution.py`
(`humanoid_test060` 631MB → `humanoid_test060_640` 158MB、変換時点の全30ep読込検証済み。
その後 10ep 追加収録され、現在の `humanoid_test060_640` は 40ep / 17,681 フレーム)。
変換後は LeRobotDataset として読み込めることを必ず確認する。
**推論時はカメラ設定の width/height をこの解像度に一致させる**こと(Step 6)。

## Step 3: 学習起動

環境非依存の形(FaBo レシピ):

```bash
export HF_HUB_OFFLINE=1     # ローカルデータセットのみ使用
# PYTORCH_CUDA_ALLOC_CONF は Thor では設定しない

lerobot-train \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=<repo_id> \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.use_vae=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.push_to_hub=false \
    --output_dir=<output_dir> \
    --job_name=<job_name> \
    --wandb.enable=false \
    --steps=<steps> \
    --batch_size=<batch> \
    --num_workers=0 \
    --save_freq=5000
```

実例(env 上書き対応・データセット存在/メモリ preflight・既存出力の
日時付きバックアップ込み):

```bash
# 既定: humanoid_test060 (1080p) を 10000 steps, batch 2
sh /home/jetson/RS/run_train06.sh

# 640x360 版で高速学習
DATASET_ROOT=/home/jetson/RS/humanoid_test060_640 \
REPO_ID=local/humanoid_test060_640 \
OUTPUT_DIR=/home/jetson/RS/outputs/train/act_humanoid_test060_640 \
STEPS=15000 BATCH=8 sh /home/jetson/RS/run_train06.sh
```

長時間学習は nohup でバックグラウンド起動する
(`nohup sh run_train06.sh > train.log 2>&1 &`)。
**起動前に必ず `pgrep -f lerobot-train`** — 同じ学習を2本起動すると
チェックポイントが破損しうる。

## Step 4: 学習監視

```bash
# 進捗行 (200 step ごとの INFO 行) を確認
grep -oE 'step:[0-9K]+ .*mem_gb:[0-9.]+' train.log | tail -3

# メモリの平坦性を確認 (Thor で最重要の健全性指標)
free -h
```

正常時の目安(Thor 実測):

| 入力解像度 | batch | step/s | mem_gb (torch) | メモリ推移 |
|---|---|---|---|---|
| 1920×1080 | 2 | 約3.1 | 4.7 | 完全平坦 (±0.1GB) |
| 640×360 | 8 | 約6.3 | 1.38 | 完全平坦 |

異常のサイン(いずれも `reference.md` に詳細):

- **step/s が徐々に低下 + `free -h` の空きが減り続ける** → アロケータ設定を疑う
  (`PYTORCH_CUDA_ALLOC_CONF` が設定されていないか env を確認)
- **数十 step で突然 3.5s/step 級に劣化** → `max_split_size_mb` が設定されている
- 停止は **Ctrl+C で正常終了を待つ**(SIGKILL は GPU メモリリークの元)

## Step 5: 継続学習 (resume)

チェックポイントから再開して総ステップ数を延長する。
**`--steps` は「追加分」ではなく「延長後の総ステップ数」**を指定する。

```bash
lerobot-train \
    --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --steps=<総ステップ数>
```

実例: `sh /home/jetson/RS/run_train06_resume.sh 30000`(15K → 30K に延長)。
チェックポイントは `<output_dir>/checkpoints/<step>/`(save_freq ごと)と
`checkpoints/last/`(シンボリックリンク)に置かれる。

## Step 6: 推論(実機 rollout)

**ポリシーがアームを自律で動かす。** 周囲の障害物を除去し、いつでも Ctrl+C
できる状態で、まず短時間(20秒)から試す。
**カメラ位置・照明・物の配置・初期姿勢・task 文字列は学習時と一致させる**こと。

事前チェック(実例スクリプトが自動実行する項目):

```bash
cat /sys/class/net/can0/operstate          # "up" (フォロワー CAN)
cat /sys/class/net/mgbe0_0/carrier         # "1" (HSB カメラリンク)
ls <policy_path>/config.json <policy_path>/*.safetensors   # ポリシー実体
awk '/MemAvailable/ {printf "%d GB\n", $2/1048576}' /proc/meminfo   # >= 8GB
```

環境非依存の形:

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=<output_dir>/checkpoints/last/pretrained_model \
    --policy.n_action_steps=30 \
    --robot.type=<robot_type> \
    --robot.cameras='<学習時と同じ解像度になるカメラ設定>' \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="<データセットの single_task と同一文字列>" \
    --display_data=false \
    --return_to_initial_position=true
```

- **RTC は使わない**(ACT 非対応、実測 TypeError)。`--policy.n_action_steps=30`
  の CLI 上書き(再学習不要)で再推論間隔が 1.7s → 1.0s に縮まる。
- ヘッドレス環境では `--display_data=false` 必須(rerun のチャネル詰まりで
  制御ループがブロックする)。
- 640×360 学習モデルの場合はカメラ設定に `width: 640, height: 360` を入れて
  学習時解像度と一致させる(HSB プラグインはワーカー側で INTER_AREA 縮小)。

実例: `sh /home/jetson/RS/run_infer06.sh 20`
(rs_follower + HSB カメラ。can0/カメラリンク/ポリシー/メモリの事前チェック、
`return_to_initial_position=true`、task="Put the object on the table")。

## Step 7: トラブル対処

| 症状 | 原因 | 対処 |
|---|---|---|
| 約1000 step で失速、空きメモリ枯渇 | `expandable_segments:True` によるドライバ側リーク | env から `PYTORCH_CUDA_ALLOC_CONF` を外す。リーク分は **再起動でのみ回収** |
| 数十 step で 3.5s/step に劣化 | `max_split_size_mb` によるキャッシュ迂回 | 同上(デフォルトアロケータに戻す) |
| 起動時 preflight で空きメモリ < 8GB | 強制終了した CUDA プロセスの NvMap リーク | `sudo reboot` |
| 推論でカメラ TimeoutError | HSB カメラのリンク断(発熱) | カメラ冷却 + `sudo ip link set mgbe0_0 down && sleep 2 && sudo ip link set mgbe0_0 up` |
| `--inference.type=rtc` で TypeError | ACT は inference_delay 非対応 | sync + `--policy.n_action_steps` 短縮を使う |
| 推論の動きが学習時と別物 | 観測条件の不一致(解像度・露出・配置・task 文字列) | 学習時条件に揃える。解像度はカメラ config で明示 |

詳細な診断コマンド・実測データは `./reference/reference.md`。

# 実績(このレシピの到達点)

| モデル | 学習 | 最終 loss | 実機結果 |
|---|---|---|---|
| ACT 1080p | 30K steps (batch 2, 15K+resume15K) | 0.143 | タスク成功 |
| ACT 640×360 | 15K steps (batch 8) | 0.114 | タスク成功 |
