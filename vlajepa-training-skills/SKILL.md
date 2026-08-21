---
name: vlajepa-training-skills
description: LeRobot 0.6.0 の VLA-JEPA ポリシー (--policy.type=vla_jepa、Qwen3-VL-2B + V-JEPA2 バックボーン) の学習・推論を Jetson AGX Thor 上で実行するスキル。実機で発見した2つのバグ (LIBERO 用グリッパーハックによる実機暴走・state のラベルリーク) の検出と修正手順、flow-matching ノイズ対策 (K サンプル平均パッチ) を含む。デフォルトレシピ (chunk_size=7) では実機タスク不成立、修正した v2 レシピ (chunk_size=30) で成立 (2026-08-16 実機確認) — その実証記録に基づく。
---

# 概要

LeRobot 0.6.0 組み込みの VLA-JEPA ポリシー (`--policy.type=vla_jepa`) を、
自前の LeRobotDataset v3 データセットで学習し、実機で推論するまでを実施する。
学習起動 → 夜間運用 → 監視 → **チェックポイントの検査・修正 (必須)** →
オフライン予測評価 → 実機推論、の順で進める。

**重要な前置き**: 本スキルは Jetson AGX Thor + 16軸ヒューマノイド (rs_follower) +
40エピソードの実機検証 (2026-08-13〜16) の記録。**デフォルトレシピ (chunk_size=7)
では実機タスクは成立しなかった** (同じデータで ACT / SmolVLA / GR00T / FastWAM は
動作) が、**v2 レシピ (chunk_size=30 + state リーク修正 + グリッパーハック無効化)
で成立した** (2026-08-16 実機確認)。同一データ・同一バックボーンで不成立→成立に
転じたことで、**chunk 長が支配的要因**であることを実証済み。その過程で発見した
**実機を暴走させる致命バグ (グリッパーハック) と修正パッチ、学習時ラベルリーク、
ノイズ対策**は VLA-JEPA を使う全ケースに適用できる。学習は v2 レシピを使うこと。

# 実装前に必ず参照する

- 実装知見 (バグの機構・修正パッチ・実測データ・敗因分析): `./reference/reference.md`
- 実機検証済みスクリプトの実例:
  `/home/jetson/RS/run_train06_vlajepa.sh` (学習) /
  `/home/jetson/RS/run_train06_vlajepa_overnight.sh` (夜間ランチャー) /
  `/home/jetson/RS/run_infer06_vlajepa.sh` (実機推論。v2 チェックポイント +
  `VLAJEPA_SAMPLES=8` 既定に更新済み) /
  `/home/jetson/RS/run_train06_vlajepa_v2.sh` (成立レシピ v2、下記)

# 前提知識 (作業前に必ず理解すること)

1. **構成**: VLA-JEPA は lerobot 0.6.0 組み込みポリシー (プラグイン不要)。
   バックボーンは Qwen3-VL-2B-Instruct + facebook/vjepa2-vitl-fpc64-256
   (どちらも公開・ゲートなし、計約6GB)。追加依存は
   `qwen-vl-utils>=0.0.11,<0.1.0` のみ (torch スタック無傷)。
   `action_dim`/`state_dim` はデータセット実次元で自動上書きされ、
   カメラ名の rename も不要 — **データセット側の追加設定はゼロで学習が始まる**。
2. **バグ1 (致命的・実機暴走)**: postprocessor に LIBERO 用グリッパーハック
   (`vla_jepa_pre_snap_gripper` + `vla_jepa_binarize_gripper`、`gripper_dim=6` 固定)
   がデフォルトで保存される。dim6 がグリッパーでないロボット (7軸以上の多軸機)
   では **action dim6 が毎 tick ±1.0 (物理レンジ外) に上書きされ実機が暴走する**。
   今後の学習には必ず `--policy.pre_snap_gripper_action=false
   --policy.binarize_gripper_action=false` を付け、既存チェックポイントは
   Step 4 の JSON 修正を行う。**rollout 時のフラグ指定では直らない**
   (postprocessor はチェックポイントの JSON からロードされる)。
3. **バグ2 (ラベルリーク)**: configuration の `observation_delta_indices` = [0..7]
   が state にも適用され、学習時の state に **t+7 の未来値**が入る
   (state ≈ 直前 action のデータセットでは正解の一部が入力に漏れる)。
4. **RTC 非対応**: VLA-JEPA は `inference_delay` を持たない → 推論は **sync のみ**。
   デフォルト chunk_size=7 では 7 フレーム (30fps で 0.23秒) ごとに再推論が走る
   (v2 の chunk_size=30 なら 30 フレームごと → デューティ比が大きく改善)。
5. **ノイズ**: flow-matching のサンプリング分散でチャンク内が振動する
   (積分ステップを増やしても直らない)。対策は K サンプル平均パッチ
   (環境変数 `VLAJEPA_SAMPLES`、reference.md 参照)。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot 0.6.0 venv と CUDA torch
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"

# 追加依存 (これだけ。torch/numpy に触れないことを pip の出力で確認)
<venv>/bin/pip install "qwen-vl-utils>=0.0.11,<0.1.0"

# データセット存在確認 (LeRobotDataset v3)
cat <dataset_root>/meta/info.json | grep -o '"total_episodes": [0-9]*'
```

- 空きメモリ: 学習前に **40GB 以上** (`awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo`)。
  足りなければ再起動 (Thor は unified memory 122GB)。
- 二重起動禁止: `pgrep -f "[l]erobot-(record|train|rollout)"` が空であること
  (同じ学習を2本起動するとチェックポイント破損リスク)。
- バックボーンのダウンロード: 初回は約6GB。CDN 接続断で **hf-xet バックエンドが
  エラーなしで無限ハング**する既知問題があるため、`HF_HUB_DISABLE_XET=1` を推奨
  (診断法は reference.md)。事前取得は:

```bash
HF_HUB_DISABLE_XET=1 <venv>/bin/hf download Qwen/Qwen3-VL-2B-Instruct
HF_HUB_DISABLE_XET=1 <venv>/bin/hf download facebook/vjepa2-vitl-fpc64-256
```

## Step 2: 学習起動

**必ずグリッパーハック無効化フラグを付ける** (バグ1回避)。
フラグの効果は v2 学習 (2026-08-16 完走) で検証済み — フラグ付きで学習した
チェックポイントの `policy_postprocessor.json` にハックが焼き込まれていないことを
実物確認した (フラグ**なし**で実行した初回 30K 学習は焼き込まれ、Step 4 の
JSON 修正が必要になった)。以下はベースの形:

```bash
lerobot-train \
    --policy.type=vla_jepa \
    --policy.pre_snap_gripper_action=false \
    --policy.binarize_gripper_action=false \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=<repo_id> \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=<output_dir> \
    --wandb.enable=false \
    --steps=30000 \
    --batch_size=4 \
    --num_workers=0 \
    --save_freq=5000
```

- **成立レシピ v2** (デフォルトの敗因対策。**このレシピで学習・実機とも成立を
  確認済み 2026-08-16**。実例 `/home/jetson/RS/run_train06_vlajepa_v2.sh`):
  上記に加えて
  (a) `--policy.chunk_size=30 --policy.n_action_steps=30`、
  (b) state リーク修正パッチ (reference.md §3) を先に適用 — v2 スクリプトは
  起動時に grep でパッチの存在を自己検証し、未適用なら中断する、
  (c) グリッパーハック無効化フラグ2つ、
  (d) `--policy.scheduler_decay_steps` を `--steps` に一致させる。
  本番実測: chunk30 で 1.94 s/step (chunk7 の 1.89 s/step 比 +3%)、
  25K steps ≈ **13.5 時間**で完走、loss 1.0台 → **0.141**。
  オフライン評価・実機結果の詳細は reference.md §7。
- **Thor では `PYTORCH_CUDA_ALLOC_CONF` を一切設定しない**
  (expandable_segments はドライバ側リーク、max_split_size_mb は激遅化の実測あり)。
- `HF_HUB_OFFLINE` は学習では設定しない (初回はバックボーンをダウンロードする)。
- 実例: `/home/jetson/RS/run_train06_vlajepa.sh`
  (データセット存在・空きメモリ・二重起動の事前チェック、既存出力の自動バックアップ付き)。
  **注意: このスクリプトは検証時のままでグリッパーハック無効化フラグが入っていない** —
  再利用するときは上記2フラグを追記すること。

### 夜間学習の運用

SSH やエージェントセッションを閉じても継続するよう、**ユーザーのターミナルから**
nohup 起動する:

```bash
nohup sh run_train06_vlajepa_overnight.sh > train_vlajepa.log 2>&1 &
```

overnight ラッパーは「進行中のダウンロードを待つ (30分無応答で引き継ぐ) →
`hf download` をリトライループで完了させる (60秒間隔・最大2時間) → 学習を exec」
という構成。実例: `/home/jetson/RS/run_train06_vlajepa_overnight.sh`。

## Step 3: 学習監視

```bash
# 進捗とメトリクス (200 step ごとに INFO 行)
tail -5 train_vlajepa.log
grep -a -o "loss:[0-9.]*" train_vlajepa.log | tail -3
```

期待値 (Thor 実測、batch4・640×360・16軸・40ep):

| 項目 | v1 (chunk7・30K、不成立) | v2 (chunk30・25K、成立レシピ) |
|---|---|---|
| スループット | 1.89 s/step (0.53 step/s) | 1.94 s/step |
| 所要時間 | 約 15.8 時間 (30K) | 約 13.5 時間 (25K) |
| loss | 1.355 (step200) → 0.302 (完走) | 1.0台 (序盤) → 0.141 (完走) |
| メモリ (mem_gb) | 27.4 GB でほぼ平坦 | — (chunk7 実測が目安) |
| チェックポイント | save_freq=5000 で `005000` … `030000`, `last` | 同様に `005000` … `025000`, `last` |

- メモリが平坦でない / 突然の失速 → まず `free -h`。Thor のアロケータ問題は
  reference.md §9。
- 学習開始前にダウンロードで止まっている場合 → hf-xet ハングを疑う
  (プロセスは Sleeping、数分間ゼロバイト。reference.md §8 の診断コマンド)。

## Step 4: チェックポイントの検査と修正 (実機推論の前に必須)

グリッパーハック無効化フラグ**なし**で学習したチェックポイントは、
postprocessor にハックが焼き込まれている。**必ず検査する**:

```bash
python3 -c "
import json
p = '<checkpoint>/pretrained_model/policy_postprocessor.json'
print([s['registry_name'] for s in json.load(open(p))['steps']])
"
# NG: [... 'vla_jepa_pre_snap_gripper', ..., 'vla_jepa_binarize_gripper', ...] が含まれる
# OK: ['vla_jepa_clip_actions', 'unnormalizer_processor', 'device_processor']
```

含まれていたら該当2ステップを JSON から削除する (検証済みの修正。
unnormalizer は `state_file` をファイル名で参照するため、ステップ削除で壊れない):

```bash
python3 - <<'EOF'
import json, shutil
p = "<checkpoint>/pretrained_model/policy_postprocessor.json"
shutil.copy(p, p + ".bak.gripper")          # 退避
d = json.load(open(p))
d["steps"] = [s for s in d["steps"] if s["registry_name"]
              not in ("vla_jepa_pre_snap_gripper", "vla_jepa_binarize_gripper")]
json.dump(d, open(p, "w"), indent=2)
print("removed:", len(json.load(open(p + ".bak.gripper"))["steps"]) - len(d["steps"]), "steps")
EOF
```

チェックポイントごと (5000, 10000, …, last) に必要。修正効果の実測:
全体 MAE 12.0 → 6.7、dim6 の毎 tick -1.0 固定が解消 (reference.md §2)。

## Step 5: オフライン予測評価 (実機に載せる前に必ず)

実機暴走バグ (バグ1) は**オフライン評価で発見された**。データセットの
1エピソードに対しポリシーの予測を正解 action と比較し、以下を確認する:

1. **全 action 次元が物理レンジ内か** — どこかの次元が定数 (±1.0 等) に
   張り付いていたら postprocessor の混入を疑う (バグ1 は dim6 が 35/35 step
   すべて -1.0 だった)。
2. **次元ごとの MAE** — ホールド基準 (直前値を出し続ける) と比較する。
   ホールド基準を大きく下回れないなら実機タスクは期待できない。
3. **高モーション区間の移動方向一致率** — 0.5 (ランダム) 前後なら追従できていない。
   実測の判定例: v1 (chunk7) は 0.40 で実機不成立、v2 (chunk30) は **0.93** で成立。
4. **チャンク内振動** — 大きければ K サンプル平均 (`VLAJEPA_SAMPLES`) を上げる。
   v1 実測: K=1: MAE 6.5/振動 9.0 → K=8: 3.7/3.7 → K=32: 2.8/1.8。
   v2 実測: K=8 で MAE 1.40・振動 0.26 (GT と同値) — **K=8 で十分** (reference.md §4)。

## Step 6: 実機推論

**sync のみ** (RTC 非対応)。カメラ解像度・タスク文・初期姿勢は学習時と一致させる。
v2 チェックポイント + sync + K=8 で実機動作良好を確認済み (2026-08-16):

```bash
export HF_HUB_OFFLINE=1
export VLAJEPA_SAMPLES=8   # K サンプル平均 (要 action_head.py パッチ、reference.md §4)。v2 実機検証値

lerobot-rollout \
    --strategy.type=base \
    --policy.path=<checkpoint>/pretrained_model \
    --robot.type=<robot_type> \
    --robot.cameras='<学習データと同じ解像度のカメラ設定>' \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="<学習データセットの single_task と同一文字列>" \
    --display_data=false
```

- 短時間 (20秒) から始め、**いつでも Ctrl+C できる状態**で。障害物を排除する。
- 推論前チェックの実例 (`/home/jetson/RS/run_infer06_vlajepa.sh`):
  ロボットバス (can0) up / カメラリンク / ポリシーの config.json と
  .safetensors 存在 / 学習プロセス非実行 / 空きメモリ **16GB 以上**
  (Qwen2B + JEPA のロードに必要)。
- レイテンシ実測 (v2・chunk30): チャンク推論 145ms (K=1) / 244ms (K=8) で
  **30 アクション生成** = 6ポリシー中最良のデューティ比。v1 (chunk7) は
  7 アクションごとに再推論が入りスロー再生気味だった (GR00T sync と同傾向)。
- 途中チェックポイントの試走は `--policy.path` を `checkpoints/020000/pretrained_model`
  等に差し替える (Step 4 の修正を忘れずに)。

## Step 7: トラブル対処と知見の記録

| 症状 | 原因 | 対処 |
|---|---|---|
| 実機が特定関節でリミットに突っ込む | バグ1 (postprocessor のグリッパーハック) | Step 4 の JSON 修正。rollout フラグでは直らない |
| 実機がプルプル振動する | flow-matching サンプリング分散 | `VLAJEPA_SAMPLES=8` (要パッチ、reference.md §4) |
| ダウンロードが無言で止まる | hf-xet ハング | `HF_HUB_DISABLE_XET=1` で再実行 (レジューム可) |
| 学習が step 途中から激遅化/メモリ枯渇 | Thor で `PYTORCH_CUDA_ALLOC_CONF` を設定した | 変数を外してデフォルトアロケータで再実行 |
| RTC を指定したい | VLA-JEPA は inference_delay 非対応 | sync のみ。RTC が要るなら SmolVLA / pi0 系へ |
| 動きはするがタスク不成立 | デフォルト chunk_size=7 (静止区間支配の損失) | reference.md §7 の v2 レシピに切り替える (**不成立→成立を実証済み**) |

- 新たな知見 (別データセット/別ロボットでの検証、レシピのさらなる改良等) は
  `./reference/reference.md` に追記する。
- 実機に載せる前のオフライン予測評価 (Step 5) を必ず挟むこと —
  今回の致命バグはこれで検出できた。
