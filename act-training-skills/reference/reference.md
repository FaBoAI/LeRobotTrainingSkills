# LeRobot 0.6.0 ACT 学習・推論 実装知見

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) +
lerobot 0.6.0 venv (torch 2.11.0+cu130) + LeRobotDataset v3 の
実機構築 (2026-08) で確認した内容。

## 1. 学習レシピ (FaBo) の要点

FaBo の Jetson 向け ACT レシピ (pak.fabo.io/lerobot/train) をベースに、
Thor 向けに補正した確定形:

| フラグ | 値 | 理由 |
|---|---|---|
| `--policy.type` | `act` | ACT ポリシー |
| `--policy.use_amp` | `true` | 混合精度。Thor で安定動作を確認 |
| `--policy.use_vae` | `false` | FaBo レシピ。VAE なしで実機タスク成功 |
| `--policy.chunk_size` | `50` | アクションチャンク長。**推論時に変更不可(要再学習)** |
| `--policy.n_action_steps` | `50`(学習時) | 推論時に CLI で小さく上書き可(§5) |
| `--dataset.video_backend` | `pyav` | Jetson で安定。データセット読み単体ではリークなし(切り分け済み) |
| `--num_workers` | `0` | Jetson (iGPU) 向け |
| `--batch_size` | 2 (1080p) / 8 (640×360) | 実測値は §2 |
| `--save_freq` | `5000` | 5K ごとにチェックポイント |
| `--policy.push_to_hub` | `false` | ローカル運用 |
| env `HF_HUB_OFFLINE` | `1` | ローカルデータセットのみ使用 |
| env `PYTORCH_CUDA_ALLOC_CONF` | **設定しない** | Thor では致命的(§3)。FaBo レシピの同設定は Orin 向け |

- pin_memory は iGPU では `False` が適切(本環境ではパッチ済み)。
- accelerate 導入済み環境で検証(単一 GPU 実行)。

## 2. 実測スループット (Jetson AGX Thor)

データセット: カメラ1系統 + 16軸 state/action @30fps。1080p の実測は
30 エピソード / 13,181 フレーム時点、640×360 の実測は 10ep 追加収録後の
40 エピソード / 17,681 フレーム版 (`humanoid_test060_640`) で行った。
ログの INFO 行(`updt_s`=更新時間, `data_s`=データ読み時間, `mem_gb`=torch 確保分)より。

| 入力解像度 | batch | step/s | smp/s | updt_s | data_s | mem_gb | 所要 (換算) |
|---|---|---|---|---|---|---|---|
| 1920×1080 | 2 | 3.1 | 6 | 0.26 | 0.06 | 4.70 | 30K steps ≈ 2.7h |
| 640×360 | 8 | 6.3 | 50 | 0.085 | 0.074 | 1.38 | 15K steps ≈ 40分 |

- サンプル/秒換算で 640×360 は 1080p の**約8.4倍**。バッチも 2→8 に増やせる
  (mem_gb 1.38 と軽い)ため、まず 640×360 で回すのが効率的。
- 640×360 では `data_s`(0.074)が `updt_s`(0.085)に迫る =
  `num_workers=0` のデータ読みが相対的に効いてくる領域。
- loss 推移の実測: 1080p は 15K 時点 0.19 → 30K で 0.143。
  640×360 は 15K で 0.114。**どちらも実機タスク成功**しており、
  「30K/loss 0.14 台」「15K/loss 0.11 台」が本タスク規模の完了目安。

## 3. Thor の CUDA アロケータ問題(最重要の落とし穴)

**`PYTORCH_CUDA_ALLOC_CONF` は Thor (iGPU / CUDA 13) では一切設定しないこと**
(2026-08-12 の ACT 学習で切り分け・完治確認)。要点のみ:

- `expandable_segments:True` → ドライバ側リーク ~100MB/step(kill でも回収不能 =
  再起動のみ、「step 1013 の壁」)。`max_split_size_mb:128` → cudaMalloc 連発で
  step 68 から 3.5s/step に激遅化。**デフォルトアロケータで完治**
  (メモリ完全平坦 ±0.1GB・3.0 step/s 安定)。
- 運用原則: CUDA プロセスは Ctrl+C で正常終了を待つ / 突然の失速はまず `free -h` /
  終了後もメモリが戻らない = NvMap リーク = 再起動。

障害の機序・実測の詳細・診断コマンドは**プラットフォーム共通知見として
thor-platform-skills の `reference/reference.md` §2〜§3 に集約**している
(本件はポリシー非依存のため、詳細はそちらを正とする)。

## 4. データセット低解像度化パイプライン

高速化のためデータセット動画を縮小変換する
(実例: `/home/jetson/RS/convert_dataset_resolution.py`)。

- **元データセットと同一のコーデック設定で再エンコード**する:
  info.json の `features.<key>.info` から `video.crf` / `video.g` /
  `video.pix_fmt` を読み、libsvtav1 + lanczos scale で変換。
- data/(parquet)と meta/ はそのままコピー — 状態/行動・エピソード構造・
  タイムスタンプは完全保持。
- **フレーム数検証必須**: 変換前後で ffprobe のパケット数が一致しなければ中断。
- 最後に info.json の `shape` / `video.height` / `video.width` を更新。
- 実測: 1080p 631MB → 640×360 158MB(30ep、全エピソード読込検証済み)。

### 推論側の解像度一致

LeRobot はカメラフレームをリサイズしないため、**縮小データセットで学習した
モデルの推論では、カメラ設定の width/height を学習解像度に一致させる**。
HSB カメラプラグインは config の width/height がモード解像度と異なると
ワーカー側で cv2 INTER_AREA 縮小して出力する(例:
`{type: hsb, camera_mode: 1, width: 640, height: 360}`)。

## 5. 推論 (lerobot-rollout)

### RTC は ACT 非対応(確定事項)

- `--inference.type=rtc` は **ACT では動かない**: RTC は
  `inference_delay` 引数をポリシーに渡すが、ACT はこれを受けず
  **実測で TypeError** になる(2026-08-13)。inference_delay 対応は
  pi0 / SmolVLA / GR00T 等の flow-matching 系のみ。
- ACT の反応性向上は **sync + `--policy.n_action_steps` の CLI 上書き**で行う:

| 方法 | 再学習 | 効果 (fps=30, chunk_size=50) |
|---|---|---|
| `--policy.n_action_steps=30` を CLI 指定 | **不要** | 再推論間隔 1.7s → 1.0s |
| `--policy.chunk_size` の変更 | **必要** | チャンク長自体が変わる |

- n_action_steps はチャンク先頭から実行するステップ数なので、chunk_size 以下で
  あれば学習済み重みのまま安全に短縮できる。

### 実測レイテンシ (1080p + AMP, Thor)

| 処理 | 実測 |
|---|---|
| チャンク推論(50 アクション生成) | 60ms |
| キューからのアクション取り出し | 3.8ms |

fps=30(33ms/tick)に対しチャンク推論 60ms は 2 tick 分だが、キュー方式なので
制御ループは破綻しない。ACT は VLA 系(GR00T sync は前処理律速で 4Hz)と違い
**リアルタイム 30Hz 制御が素直に成立する**。

### rollout の確定パラメータ(実例 run_infer06.sh)

| 項目 | 値 | 理由 |
|---|---|---|
| `--strategy.type` | `base` | 標準 rollout |
| `--fps` | `30` | データセット収録 fps と一致 |
| `--task` | 学習データセットの single_task と同一文字列 | 観測条件の一部 |
| `--duration` | まず `20`(秒) | FaBo 推奨: 短時間で挙動確認してから延長 |
| `--display_data` | `false` | ヘッドレスでは必須(rerun のチャネル詰まりでループがブロック) |
| `--return_to_initial_position` | `true` | 終了時に初期姿勢へ復帰 |

- ロボット側に initial_position を設定しておくと connect() 時に初期位置へ
  ブロッキング移動し、アーム静止後に推論が始まる(値はデータセット開始姿勢の
  平均にする。本環境では rs_follower の YAML に設定済み)。
- **観測条件の一致が精度の前提**: カメラ位置・照明・露出
  (本環境の屋内実測: exposure=1000, analog_gain=6)・物の配置・初期姿勢・
  解像度・task 文字列を学習時と揃える。

### 事前チェック(スクリプトに組み込むべき項目)

```bash
cat /sys/class/net/can0/operstate      # up (フォロワー CAN)
cat /sys/class/net/mgbe0_0/carrier     # 1 (HSB カメラリンク。0 なら mgbe bounce)
test -f "$POLICY_PATH/config.json"     # ポリシー config
find "$POLICY_PATH" -maxdepth 1 -name '*.safetensors' | grep -q .   # モデル実体
awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo        # >= 8GB
```

## 6. 運用の落とし穴一覧

| ハマり所 | 症状 | 対処 |
|---|---|---|
| 同一学習の二重起動 | チェックポイント破損リスク。ログに2本の進捗バーが交互に出る | nohup 起動前に `pgrep -f lerobot-train` |
| `--resume` の `--steps` を追加分と誤解 | 期待より早く終了 | `--steps` は**延長後の総ステップ数**(15K→30K なら 30000) |
| 既存 output_dir への上書き | 前回の学習結果を失う | 実行前に日時付きで `mv`(実例スクリプトは自動バックアップ) |
| 強制終了した CUDA プロセス | 空きメモリが戻らない(NvMap リーク) | preflight で MemAvailable < 8GB を検出したら `sudo reboot` |
| ヘッドレスで `--display_data=true` | rerun のチャネル詰まりで収録/推論ループがブロック | `false` 固定 |
| resume スクリプトの出力先固定 | 別モデルの checkpoint を延長してしまう | `--config_path` が意図した output_dir を指すか確認 |
| チェックポイント途中の推論 | `checkpoints/last` は最新 save_freq 地点 | 学習完走を待つか、`checkpoints/<step>/pretrained_model` を明示指定 |

## 7. チェックポイントのレイアウト (lerobot 0.6.0)

```
<output_dir>/checkpoints/
├── 005000/
│   └── pretrained_model/     # config.json + model.safetensors + train_config.json
├── 010000/
├── 015000/
└── last -> 015000            # 最新へのリンク
```

- 推論の `--policy.path` は `.../pretrained_model` まで指定する。
- resume の `--config_path` は
  `.../checkpoints/last/pretrained_model/train_config.json`。

## 8. 監視ログの読み方

lerobot-train の INFO 行(200 step ごと):

```
step:15K smpl:120K ep:271 epch:6.79 loss:0.114 grdn:6.920 lr:1.0e-05 updt_s:0.085 data_s:0.074 smp/s:50 mem_gb:1.38
```

| フィールド | 意味 | 見るポイント |
|---|---|---|
| `loss` | 平滑化済み L1 系 loss | §2 の完了目安と比較 |
| `updt_s` / `data_s` | 1 step の更新時間 / データ読み時間 | data_s 支配なら解像度かデコードがボトルネック |
| `smp/s` | サンプルスループット | 解像度間の比較はこの値で行う |
| `mem_gb` | torch の確保メモリ | **増え続けたら異常**(§3)。ただしドライバ側リークはここに出ない → `free -h` 併用 |
| `epch` | サンプル数換算のエポック | 30ep 規模なら数エポックで十分な実績 |
