# Jetson AGX Thor / LeRobot 0.6.0 学習運用の実装知見

Jetson AGX Thor (JetPack 7 / L4T R38.2 / CUDA 13.0, unified memory 122GB) +
LeRobot 0.6.0 の実機構築 (2026-08) で確認した内容。ポリシー非依存の
プラットフォーム知見のみを扱う (ポリシー固有はポリシー別スキル参照)。

## 1. lerobot060-venv の構築と依存ピン留め

### なぜ `--no-deps` が必須か

Jetson の torch は CUDA 対応の専用ビルド (`2.11.0+cu130`)。pip の依存解決に
任せると、プラグインやポリシー追加のインストール時に numpy / torch が PyPI 版
(CPU torch) へダウングレード・差し替えられ、CUDA スタックが壊れる。

```bash
# プラグイン・自作パッケージの追加は常に:
pip install -e . --no-deps
# 追加依存が必要なときも個別に (依存ツリーごと入れない):
pip install <pkg>==<version>
```

### lerobot 0.6.0 の宣言ピン (dist METADATA より) と実測インストール

| パッケージ | lerobot 0.6.0 の要求 | 実測 (lerobot060-venv) | 備考 |
|---|---|---|---|
| torch | >=2.7,<2.12.0 | **2.11.0+cu130** | Jetson CUDA ビルド。差替厳禁 |
| torchvision | >=0.22,<0.27 | 0.26.0 | |
| numpy | >=2.0,<2.3 | 2.5.2 | torch 側都合で宣言範囲外だが動作検証済み。`pip check` の警告は既知 |
| datasets | **>=4.7,<5.0.0** (dataset extra) | 4.8.5 | 5.x は不可 |
| av | **>=15,<16** (av-dep extra) | 15.1.0 | video_backend=pyav 用 |
| rerun-sdk | >=0.24,**<0.34** (viz extra) | **0.32.2** | 可視化用。0.34 以上は不可 |
| accelerate | >=1.14,<2.0 | 1.14.0 | lerobot-train が使用 |
| huggingface_hub / hf-xet | — | 1.27.0 / 1.6.0 | hf-xet はハング問題あり (§5) |

- venv は Python 3.12。ポリシー別の追加依存 (transformers, peft, diffusers,
  qwen-vl-utils 等) は各ポリシー別スキルに記載 — いずれも torch を巻き込まない
  ことを確認してから入れる。
- torchcodec は aarch64 でも dataset extra の宣言に入っているが、本構成では
  **`--dataset.video_backend=pyav`** を常用しており不要。
- 実例 venv: `/home/jetson/camera/lerobot060-venv`。動作確認:

```bash
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
# → 0.6.0 2.11.0+cu130 True
```

## 2. CUDA アロケータの掟 (最重要)

**Thor (iGPU / CUDA 13) では `PYTORCH_CUDA_ALLOC_CONF` を一切設定しない。**
FaBo の Orin 向けレシピにある2設定が、それぞれ別種の障害を起こした
(2026-08-12 に切り分け・完治確認)。

| 設定 | 障害 | 実測 |
|---|---|---|
| `expandable_segments:True` | **ドライバ側メモリリーク** ~100MB/step。RSS にも torch の mem_gb にも見えない。プロセス kill でも回収されず**再起動のみ** | 約1000 step (実測 step 1013) で 122GB の unified memory が枯渇 (119GB 消費) し失速。**VSZ 300GB が兆候** |
| `max_split_size_mb:128` | 128MB 超の確保がキャッシュを迂回して **cudaMalloc 連発** | step 68 から 3.5s/step に劣化 (正常時 0.33s/step) |
| (未設定 = デフォルト) | **完治** | 10分監視でメモリ完全平坦 (±0.1GB)・3.00 step/s 安定。上記2つの「壁」を両方通過 |

- 切り分けで判明した無関係要因: データセット読み (pyav) 単体はリークなし。
  pin_memory も無関係 (iGPU では False が適切)。
- 機序の推定: Thor は iGPU で GPU メモリ = ホストメモリ (unified)。
  expandable_segments の仮想メモリ操作 (cuMemMap 系) が CUDA 13 + Tegra
  ドライバでページを返却しない。ユーザー空間からは不可視のため、
  **`free -h` の available 減少と `ps` の VSZ 異常だけが観測手段**。
- 診断コマンド:

```bash
env | grep PYTORCH_CUDA_ALLOC_CONF        # 何も出ないのが正解
free -h                                    # available が平坦か
ps -o vsz,rss,cmd -p $(pgrep -f "[l]erobot-train")   # VSZ が数百GBなら即中止
```

## 3. メモリ運用 (NvMap リークと preflight)

- **NvMap リーク**: CUDA プロセスを `kill -9` (SIGKILL) すると、iGPU の GPU
  メモリ (NvMap 経由でホストと共有) が解放されずに残る。プロセスは消えても
  `free -h` の available が戻らない。**回収手段は再起動のみ**。
- したがって学習の停止は **Ctrl+C 1回 → 正常終了を待つ** (SIGINT なら CUDA
  リソースのクリーンアップが走る)。連打や SIGKILL は使わない。
- **謎の失速はまず `free -h`**。step/s 低下の主因はメモリ枯渇であり、
  スワップ/リクレイムに入った時点で学習速度は崩壊する。
- 学習前 preflight (実例: `/home/jetson/RS/run_train06.sh` 冒頭):

```bash
AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$AVAIL_GB" -lt 8 ]; then
    echo "NG: 空きメモリ ${AVAIL_GB}GB しかありません。NvMap リークの可能性 → sudo reboot 推奨"
    exit 1
fi
```

- 閾値の実績: ACT 級 (mem_gb 1.4〜4.7) は **8GB**、2B バックボーンの VLA 学習
  (mem_gb 27+) は **40GB** を下限にした (実例: run_train06_vlajepa.sh)。

## 4. 学習ジョブの運用

### 二重起動の禁止

同じ学習を2本起動すると (a) スループット半減 (b) 同一 output_dir への同時
チェックポイント書込で**破損リスク**。起動前チェックを必ず入れる:

```bash
if pgrep -f "[l]erobot-(record|train|rollout)" >/dev/null; then
    echo "NG: 別の lerobot プロセスが実行中"; exit 1
fi
```

- **ブラケット法 `[l]erobot`**: `pgrep -f lerobot-train` はこのチェックを含む
  シェルスクリプト自身 (のコマンドライン) にマッチして誤検知することがある。
  先頭1文字を文字クラスにするとパターン文字列自身にはマッチしなくなる。
  `pkill` でも同様に使う。

### 夜間学習の起動作法

- **エージェント (Claude Code 等) のセッションから起動したプロセスは
  セッション終了で死ぬ**。夜間学習は必ず**ユーザーのターミナルから nohup** で:

```bash
nohup sh /path/to/run_train.sh > /path/to/train.log 2>&1 &
```

- ログは nohup のリダイレクト先に集約し、進捗確認は `tail -5 train.log` /
  追跡は `tail -f`。lerobot-train のログには `step / loss / updt_s / data_s /
  smp/s / mem_gb` が出るので、mem_gb の平坦性と step/s をここで監視できる。
- 実例: `/home/jetson/RS/run_train06_vlajepa_overnight.sh` (30K steps ≈ 15.7h の
  一晩ジョブ。ヘッダに起動コマンドをコメントで明記しユーザーがコピペできる形に
  しておくと運用が安定する)。
- 補足 (エージェント運用): kill 系コマンドは権限分類器にブロックされる場合が
  ある。プロセス停止が必要なときは停止コマンドを提示してユーザーに実行して
  もらう。

### シェルスクリプトは sh (dash) で動く前提で書く

運用スクリプトは `nohup sh run_train.sh` のように **`sh` (Ubuntu では dash) で
実行される**。shebang が `#!/bin/bash` でも `sh` 起動では無視されるため、
bash 専用構文を書くと実行時に落ちる。既知の非互換:

| bash 専用構文 | dash での症状 | 代替 |
|---|---|---|
| `$((10#$VAR))` (基数指定) | 「arithmetic expression: expecting EOF」で即死 (2026-08-15 実測) | 先頭ゼロ除去は `sed 's/^0*//'` (全ゼロは空になるので `${VAR:-0}` で補う。実例: run_train06_groot_resume.sh のチェックポイント番号 `015000`→`15000`) |
| `source file` | 「source: not found」 | `. file` |
| 配列 (`arr=(a b)` / `${arr[0]}`) | 「Syntax error: "(" unexpected」 | スペース区切り文字列 + for ループ / `set --` の位置パラメータ |

## 5. HF ダウンロードの hf-xet ハング

### 症状 (2026-08-13 実測)

- CDN 側の接続断 (CloudFront リセット) の後、**hf-xet バックエンドはエラーを
  出さず futex 待ちで無限ハング**する。数分間ゼロバイトのままプロセスは
  Sleeping。lerobot-train 内部の `from_pretrained` ダウンロードも CLOSE-WAIT
  ソケットを掴んだまま固まった。

### 診断

```bash
ls -la --time-style=full-iso ~/.cache/huggingface/hub/**/*.incomplete  # mtime が止まっている
du -s ~/.cache/huggingface/xet          # xet キャッシュが成長していない
ss -tnp | grep "pid=<PID>"              # CLOSE-WAIT ソケットの確認
```

### 対策

```bash
export HF_HUB_DISABLE_XET=1   # 通常 HTTP 経路に切替
```

通常 HTTP 経路は 10秒 read timeout で失敗が**顕在化**する (= リトライ・
レジューム可能になる)。加えて、キャッシュ済みで回すジョブは
`HF_HUB_OFFLINE=1` でハブアクセス自体を封じるのが最も確実。

### 「ダウンロード完了待ち → 学習自動開始」ラッパーの設計

夜間学習でバックボーンのダウンロードが残っている場合のラッパー
(実例: `/home/jetson/RS/run_train06_vlajepa_overnight.sh`):

1. **既存ダウンロードとの競合回避**: 進行中のダウンロードプロセスがあれば待つ
   (上限付き。30分無応答なら停止して引き継ぐ)。
2. **リトライループ**: 必要リポジトリごとに `hf download <repo>` を成功するまで
   60秒間隔でリトライ (上限 2時間)。hf download は**続きからレジューム**される
   ので、切断のたびに進む。キャッシュ済みなら即終了するため冪等。
3. 全リポジトリ完了後に `exec sh run_train.sh` で学習開始。

```bash
for REPO in "<org/model1>" "<org/model2>"; do
    N=0
    until "$VENV/bin/hf" download "$REPO" >/dev/null 2>&1; do
        N=$((N + 1)); [ "$N" -ge 120 ] && exit 1
        sleep 60      # 続きから再開される
    done
done
exec sh /path/to/run_train.sh
```

## 6. データセット解像度の設計

### 実測スループット (ACT / LeRobotDataset v3 / video_backend=pyav / num_workers=0)

| 入力解像度 | batch | step/s | samples/s | updt_s | data_s | mem_gb |
|---|---|---|---|---|---|---|
| 1920×1080 | 2 | 3.0〜3.1 | 6 | 0.26 | 0.06 | 4.7 |
| 640×360 | 8 | 6.3 | 50〜51 | 0.085 | 0.074 | 1.38 |

**samples/s 比で約8.4倍**。1080p では画像デコード・前処理・転送が支配的で、
batch を上げてもスループットが伸びない。学習品質も維持された
(1080p 30K steps: loss 0.143 / 640×360 15K steps: loss 0.114。
注: 1080p の実測は 30ep/13,181 フレーム時点、640×360 の実測は
10ep 追加収録後の 40ep/17,681 フレーム版で行った)。
データセットサイズは変換時点 (30ep) で 631MB → 158MB。

### 既存データセットの ffmpeg 変換 (実例: `/home/jetson/RS/convert_dataset_resolution.py`)

元データセット無変更で縮小版を新規作成する。設計要点:

1. **コーデック設定を info.json から引き継ぐ**: `libsvtav1` +
   元の `video.crf` (実例 30) / `video.g` (GOP、実例 2) / `video.pix_fmt`
   (yuv420p) を使う。GOP を維持しないとランダムアクセス読みの性能特性が変わる。
   スケールは `scale=W:H:flags=lanczos`、`-an` (音声なし)、`-preset 10`。
2. **フレーム数検証**: 変換ごとに ffprobe (`-count_packets`) で src/dst の
   フレーム数一致を検証し、不一致なら即中断 (タイムスタンプ整合が壊れるため)。
3. **動画以外は丸ごとコピー**: `data/` (parquet) と `meta/` はコピーのみ。
   状態/行動・エピソード構造・タイムスタンプは完全保持。
4. **meta/info.json の書換**: video フィーチャの `shape` を `[H, W, C]` に、
   `info` 内の `video.height` / `video.width` を新解像度に更新。
5. 変換後に LeRobotDataset として全エピソード読込検証を行う (実績: 30ep 全数)。
6. スクリプトは ffmpeg/ffprobe だけに依存するのでシステム python
   (`/usr/bin/python3`) で実行できる (venv 不要)。

### 学習と推論の設定一致 (必須)

- **カメラの解像度・露光設定は収録時と推論時で必ず一致させる**。低解像度
  データセットで学習した場合、推論のカメラ config にも同じ width/height を指定
  (プラグイン側で縮小出力)。露光・ゲイン (実例 HSB: exposure=1000,
  analog_gain=6) も収録時と揃える — 輝度分布のずれは方策の入力分布ずれになる。
- LeRobot はカメラ画像をリサイズしないまま features を確定するため、
  解像度不一致は形状エラーか無言の性能劣化として現れる。

## 7. rerun 可視化 (lerobot-dataset-viz、ヘッドレス運用)

- **バージョン固定**: lerobot 0.6.0 の viz 制約は rerun-sdk <0.34。
  実績は **rerun-sdk==0.32.2**。
- ヘッドレス Jetson では `--mode distant` で gRPC サーバ (実例 9876) +
  Web ビューア (実例 9090) を立て、PC のブラウザから閲覧する。

### CORS パッチ (必須)

ブラウザ上の Web ビューアが Jetson の gRPC プロキシへ接続するには、
`lerobot/scripts/lerobot_dataset_viz.py` の `mode == "distant"` ブロックで
`rr.serve_grpc()` に `cors_allow_origin` を渡す (pak.fabo.io/lerobot/cors 方式):

```python
server_uri = rr.serve_grpc(
    grpc_port=grpc_port,
    cors_allow_origin=[
        f"http://192.168.128.100:{web_port}",  # LAN
        f"http://192.168.55.1:{web_port}",     # USB (l4tbr0)
        f"http://localhost:{web_port}",
    ],
)
rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)
```

(オリジンは自環境のビューア URL に合わせて列挙する。パッチ適用先は
site-packages 直編集になるので、venv 再構築時の再適用を忘れない。)

### 閲覧 URL — `?url=` パラメータ必須

```
http://<JetsonのIP>:9090/?url=rerun%2Bhttp%3A%2F%2F<JetsonのIP>%3A9876%2Fproxy
```

(= `?url=rerun+http://<IP>:9876/proxy` の URL エンコード。素の `:9090` を開いても
データソースに接続されず何も表示されない。)

実例: `/home/jetson/RS/run_viz06.sh [ep]` — `--num-workers 0 --batch-size 1
--display-compressed-images` で安定。

### 収録・rollout 側の注意

- ヘッドレスでの `lerobot-record` / `lerobot-rollout` は **`--display_data=false`
  必須**。rerun のチャネル詰まりで制御ループがブロックし、収録が止まる。

## 8. torch.compile / triton を使うポリシーの必須設定 (Thor = sm_110a)

**`torch.compile` / `flex_attention` を使う全ポリシーに影響する普遍知見**
(2026-08-16 に LingBot-VA 学習で発見・解決。実例は lingbotva-training-skills)。

### 機構

- Thor の GPU は **sm_110a** (arch 110)。triton (実測 3.6.0) は
  **arch ≥ 100 では同梱の `ptxas-blackwell` を使う**:
  `triton/backends/nvidia/compiler.py:35` —
  `return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas`。
- ところが同梱の `ptxas-blackwell` は **CUDA 12.9 ビルド (PTX 8.8 世代) で
  sm_110a を扱えず**、triton カーネルのアセンブルが失敗する。
  flex_attention / torch.compile の初回コンパイルで顕在化する。

### 対策 (compile 系ポリシーの学習スクリプトに必ず入れる)

```bash
# システム CUDA 13 の ptxas に向ける (PTX 9.0 で生成されるようになる。実測で解決)
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

- **罠: `TRITON_PTXAS_PATH` は別ノブ** (`triton/knobs.py:491-492` で別変数)。
  こちらは arch<100 用で、**Thor (arch≥100) では設定しても効かない**。
  名前が似ているので混同しないこと。
- venv 再構築や triton 更新後も環境変数方式なので消えないが、学習スクリプトの
  冒頭で export + 該当ポリシーのパッチ類の自己検証とセットにしておくと安全
  (実例: `/home/jetson/RS/run_train06_lingbotva.sh`)。

### 診断

```bash
# 同梱 ptxas-blackwell の世代 (CUDA 12.9 なら本問題に該当)
<venv>/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell --version | tail -1
# システム ptxas (CUDA 13.0 = 対策に使う方)
/usr/local/cuda/bin/ptxas --version | tail -1
```

## 9. 律速切り分けの定石: 成分の単体実測 (py-spy は使えない)

- **py-spy は ptrace 制限で他人のプロセスに attach できない** (Thor の本環境の
  実測) → 実行中の学習をプロファイラで覗く手が使えない。
- 定石 = **学習1ステップを構成する成分を、単体スクリプトで個別に実測して
  犯人を絞る**。切り分け実績 (LingBot-VA の 18s/step、2026-08-16):

| 成分 | 単体実測 | 判定 |
|---|---|---|
| 動画デコード (pyav、1サンプル) | 0.06s | シロ |
| flex ブロックマスク構築+再コンパイル | 15.7s/回 | 主犯1 (毎ステップ発火していた) |
| UMT5-XXL の CPU テキストエンコード | 15〜32s/回 | 主犯2 (毎ステップ同一文を再計算) |

- → 2つのメモ化パッチで **17.7s/step → 1.8s/step** (詳細は
  lingbotva-training-skills reference.md §3)。
- 着眼点: single-task データセットでは「**毎ステップ同じ入力を再計算して
  いないか**」(テキストエンコード・マスク構築・トークン化) をまず疑う。
  GR00T sync の毎ティック前処理 240ms (タスク文トークン化含む) も同族の症状。
- なお「謎の失速」(徐々に遅くなる) はまず `free -h` (§2〜§3 のメモリ問題) —
  本節の対象は「最初から一定して遅い」場合の切り分け。
