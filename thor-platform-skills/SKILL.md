---
name: thor-platform-skills
description: Jetson AGX Thor (JetPack 7 / CUDA 13 / unified memory 122GB) 上で LeRobot 0.6.0 のポリシー学習を安全に運用するためのプラットフォーム共通スキル。CUDA アロケータの掟・メモリリーク対処・夜間学習の起動作法・HF ダウンロードのハング回避・データセット解像度設計・rerun 可視化を、実機構築 (2026-08) の実測に基づき手順化する。ポリシー固有の設定 (ACT/SmolVLA/GR00T/VLA-JEPA の引数・依存) は各ポリシー別スキルを参照。
---

# 概要

Jetson AGX Thor で LeRobot 0.6.0 の学習ジョブを回すときの、**ポリシーに依存しない
運用手順**をまとめたスキル。Thor は iGPU + unified memory (122GB) という
デスクトップ GPU と異なる構成のため、x86/dGPU 向けの定番チューニング
(`PYTORCH_CUDA_ALLOC_CONF` 等) がそのまま**障害の原因になる**。本スキルは
「環境確認 → 学習前 preflight → 起動 (夜間学習含む) → 監視 → トラブル対処」の
順で、エージェントが学習運用を代行するための確定手順と実測データを提供する。

# 実装前に必ず参照する

- 実装知見 (アロケータ障害の実測・メモリリークの機序・診断コマンド): `./reference/reference.md`
- ポリシー固有の学習/推論引数・追加依存: 各ポリシー別スキル (act- / smolvla- / groot- / vlajepa- 等)
- カメラ (HSB) の物理運用: 姉妹リポジトリ LeRobotPluginSkills の hsb-camera-skills

# 前提知識 (作業前に必ず理解すること — Thor の絶対の掟)

1. **`PYTORCH_CUDA_ALLOC_CONF` を設定しない (最重要)**。Orin 向け FaBo レシピの
   `expandable_segments:True` は Thor (iGPU/CUDA13) でドライバ側メモリリーク
   (~100MB/step、プロセス kill でも回収不能 = 要再起動) を起こし、
   `max_split_size_mb:128` は cudaMalloc 連発で激遅化する。
   **デフォルトアロケータが正解** (実測で完治確認済み)。詳細は `reference.md` §2。
2. **CUDA プロセスを `kill -9` しない**。iGPU では GPU メモリが NvMap 経由で
   ホストメモリと共有されており、強制終了すると NvMap リークが残り
   **再起動でしか回収できない**。停止は Ctrl+C 1回 → 正常終了を待つ。
3. **夜間学習はユーザーのターミナルから nohup で起動する**。エージェント
   (Claude Code 等) のセッションから起動したプロセスはセッション終了と共に死ぬ。
   エージェントの仕事はコマンドの準備と検証まで。起動自体はユーザーに依頼する。
4. **HF ダウンロードには `HF_HUB_DISABLE_XET=1`**。hf-xet バックエンドは CDN 側の
   接続断後に**エラーを出さず無限ハング**する (futex 待ち)。学習済みキャッシュで
   回すジョブは `HF_HUB_OFFLINE=1` でハブアクセス自体を封じる。
5. **学習と推論でカメラの解像度・露光設定を必ず一致させる**。低解像度データセットで
   学習したら、推論時のカメラ設定にも同じ width/height (と露光) を指定する。

# ワークフロー

## Step 1: 環境確認

```bash
# venv の lerobot / torch / CUDA を確認 (実例: /home/jetson/camera/lerobot060-venv)
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
# 期待値 (本プロジェクト): 0.6.0 2.11.0+cu130 True
```

- torch が `+cu130` 付きで CUDA 有効であることを必ず確認する。Jetson の CUDA torch は
  PyPI の依存解決で簡単に CPU 版へ差し替えられてしまうため、
  **プラグインやポリシーのパッケージ追加は常に `pip install -e . --no-deps`**。
- 依存のピン留め (datasets<5.0.0 / av>=15,<16 / rerun-sdk==0.32.2 等) と
  インストール済み実測バージョン表は `reference.md` §1。

## Step 2: 学習前 preflight

学習スクリプトの冒頭に以下の3チェックを入れる (3つ全て入った実例:
`/home/jetson/RS/run_train06_smolvla.sh`。なお `run_train06.sh` は
データセット・メモリの2チェックのみで、二重起動チェックは未実装)。

```bash
# 1) データセット存在確認 (LeRobotDataset v3)
[ -f "$DATASET_ROOT/meta/info.json" ] || { echo "NG: データセットなし"; exit 1; }

# 2) 二重起動チェック (ブラケット法で pgrep の自己マッチを回避)
if pgrep -f "[l]erobot-(record|train|rollout)" >/dev/null; then
    echo "NG: 別の lerobot プロセスが実行中"; exit 1
fi

# 3) 空きメモリチェック (NvMap リーク検知。少なければ再起動しかない)
AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$AVAIL_GB" -lt 8 ]; then
    echo "NG: 空きメモリ ${AVAIL_GB}GB。NvMap リークの可能性 → sudo reboot 推奨"; exit 1
fi
```

正確な preflight スクリプト全文と閾値の根拠 (小規模ポリシー 8GB / 2B級バックボーン
40GB) は `reference.md` §3。二重学習は速度半減に加え、同一 output_dir への同時
チェックポイント書込で**破損リスク**があるため必ず弾く。

## Step 3: 学習の起動

### 短時間 (対話中) の学習

```bash
export HF_HUB_OFFLINE=1     # キャッシュ済みならハブアクセスを封じる
# PYTORCH_CUDA_ALLOC_CONF は設定しない (unset であることを確認)
<venv>/bin/lerobot-train --dataset.root=... --dataset.video_backend=pyav ...
```

実例: `/home/jetson/RS/run_train06.sh` (環境変数 DATASET_ROOT/REPO_ID/OUTPUT_DIR/
STEPS/BATCH で上書き可能、既存 output_dir は日時付きバックアップへ退避)。

### 夜間学習 (nohup — 必ずユーザーのターミナルから)

```bash
# ユーザーに以下をそのまま実行してもらう (エージェントのセッションからは起動しない)
nohup sh /path/to/run_train.sh > /path/to/train.log 2>&1 &
```

実例: `/home/jetson/RS/run_train06_vlajepa_overnight.sh`。バックボーンの
ダウンロードが未完の場合に「リトライループでダウンロード完了を待ってから
学習を自動開始する」ラッパーの設計は `reference.md` §5 (hf-xet ハング対策込み)。

### レジューム (チェックポイントから総ステップ数を延長)

```bash
<venv>/bin/lerobot-train \
    --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=<新しい総ステップ数>
```

実例: `/home/jetson/RS/run_train06_resume.sh`。

## Step 4: 監視

```bash
tail -f train.log            # 進捗 (step/s, loss, mem_gb)
free -h                      # メモリ平坦性 (available が減り続けたら異常)
ps -o vsz,rss,cmd -p $(pgrep -f "[l]erobot-train")   # VSZ 異常肥大の検知
```

- **謎の失速はまず `free -h`**。step/s が落ちる原因の大半はメモリ枯渇
  (アロケータ設定ミス or NvMap リーク) で、GPU 演算自体の問題ではない。
- 健全な学習はメモリ完全平坦 (実測 ±0.1GB / 10分)。**VSZ が数百 GB 規模に
  膨れていたら expandable_segments が有効になっている**サイン (即中止して
  環境変数を外す。リーク分の回収は再起動のみ)。
- 判定基準の実測値 (ACT 1080p: 3.0 step/s、640×360: 6.3 step/s 等) は
  `reference.md` §6 の表と照合する。

## Step 5: データセット解像度の設計 (学習前に検討)

- 1080p のまま学習すると画像デコードと前処理が支配的になる。**640×360 への縮小で
  ACT の学習スループットは実測 8.4倍** (samples/s 比、`reference.md` §6)。
- 既存データセットは ffmpeg で縮小変換できる (元データセット無変更・
  コーデック設定維持・フレーム数検証・info.json 書換):

```bash
/usr/bin/python3 convert_dataset_resolution.py \
    --src <元データセット> --dst <変換先> --width 640 --height 360
```

実例: `/home/jetson/RS/convert_dataset_resolution.py` (631MB→158MB、全30ep
読込検証済み)。変換の設計要点は `reference.md` §6。

- **推論時はカメラ設定を学習解像度に一致させる**。実例 (HSB カメラ):
  `--robot.cameras="{ front: {type: hsb, camera_mode: 1, width: 640, height: 360, exposure: 1000, analog_gain: 6}}"`
  — 収録時と同じ露光・ゲインにすること (輝度分布が変わると方策が崩れる)。

## Step 6: rerun によるデータセット可視化 (ヘッドレス Jetson → PC ブラウザ)

```bash
<venv>/bin/lerobot-dataset-viz \
    --repo-id local/<name> --root <dataset_root> --episode-index 0 \
    --mode distant --web-port 9090 --grpc-port 9876 \
    --num-workers 0 --batch-size 1 --display-compressed-images
```

実例: `/home/jetson/RS/run_viz06.sh [ep]`。閲覧は PC のブラウザで
**`?url=` パラメータ必須**:

```
http://<JetsonのIP>:9090/?url=rerun%2Bhttp%3A%2F%2F<JetsonのIP>%3A9876%2Fproxy
```

- lerobot 0.6.0 の viz 制約により **rerun-sdk==0.32.2** (<0.34) に固定する。
- ブラウザから gRPC プロキシへ接続するには lerobot_dataset_viz.py への
  **CORS パッチが必要** (`rr.serve_grpc(cors_allow_origin=[...])`)。パッチ内容は
  `reference.md` §7。
- 逆に**収録・rollout をヘッドレスで回すときは `--display_data=false` 必須**
  (rerun のチャネル詰まりで制御ループがブロックする)。

## Step 7: トラブル対処 (症状 → 一次対応)

| 症状 | まず見る | 対処 |
|---|---|---|
| step/s が徐々に低下・失速 | `free -h` | available 枯渇なら学習を Ctrl+C。`PYTORCH_CUDA_ALLOC_CONF` が設定されていないか確認。リーク分が戻らなければ再起動 |
| 突然 3秒/step 級に激遅化 | 環境変数 | `max_split_size_mb` 設定の混入を疑う → unset して再実行 |
| kill 後もメモリが戻らない | `free -h` | NvMap リーク = **再起動のみ**。以後 kill -9 を使わない |
| HF ダウンロードが無言で止まる | `.incomplete` の mtime / `ss -tnp` の CLOSE-WAIT | `HF_HUB_DISABLE_XET=1` を付けて再実行 (リジューム可能)。診断詳細は `reference.md` §5 |
| 学習が起動済みか不明 | `pgrep -af "[l]erobot-train"` | 二重起動は即中止 (チェックポイント破損リスク) |
| エージェントがプロセスを止められない | — | kill 系は権限分類器にブロックされることがある → 停止コマンドを提示してユーザーに実行を依頼 |

# 知見の記録

- 新たな実測 (別ポリシーのスループット、新しいハマり所) は
  `./reference/reference.md` の該当表に追記する。
- ポリシー固有の発見はポリシー別スキルへ、プラットフォーム共通 (アロケータ・
  メモリ・I/O・可視化) は本スキルへ、と分離を維持する。
