---
name: lingbotva-training-skills
description: LeRobot 0.6.0 の LingBot-VA ポリシー (Wan2.2 系 5.09B 自己回帰 video-action 世界モデル) の学習を Jetson AGX Thor 上で実行するスキル。--policy.type 指定は 5.09B transformer が無警告ランダム初期化になる罠のため、事前学習重みを注入する変換スクリプト + --policy.path が必須。Thor 3大障害 (triton 同梱 ptxas の sm_110a 非対応・flex ブロックマスクの毎ステップ再構築・UMT5-XXL の CPU テキストエンコード) の対策で 17.7s/step → 1.8s/step (10倍速)。58エピソード・30K steps を完走したが、オフライン評価が関門不合格 (方向一致 0.66 で頭打ち) のため実機は見送り (2026-08-16〜17) — 現行レシピでは不成立という失敗記録込み。
---

# 概要

LeRobot 0.6.0 組み込みの LingBot-VA ポリシー (Wan2.2 系 5.09B 自己回帰
video-action 世界モデル) を、自前の LeRobotDataset v3 データセットで学習する。
ベース変換 (必須) → 高速化パッチの確認 → 学習起動 → 監視 → resume →
**オフライン予測評価 (関門)**、の順で進める。

**重要な前置き — 本スキルは成功レシピではない**: Jetson AGX Thor + 16軸
ヒューマノイド (rs_follower) + 58エピソードの実機検証 (2026-08-16〜17) で、
学習自体は成立させた (30K steps 完走、3対策で 17.7s/step → 1.8s/step の10倍速)
が、**オフライン評価が関門不合格 (方向一致 5K 0.60 → 15K 0.68 → 30K 0.66 で
頭打ち、MAE 13.5) となり実機テストは見送った**。同じデータで
ACT / SmolVLA / GR00T / VLA-JEPA v2 / FastWAM は動作している。
本スキルの価値は (a) **変換必須という罠の回避手順**、(b) **Thor 3大障害の対策**
(torch.compile / flex_attention を使う全ポリシーに再利用可)、(c) **オフライン
関門で実機前に止める判断の記録**、の3点にある。

# 実装前に必ず参照する

- 実装知見 (罠の機構・パッチ実物・実測データ・不合格判定の根拠): `./reference/reference.md`
- プラットフォーム共通の掟 (`PYTORCH_CUDA_ALLOC_CONF` 禁止・`kill -9` 禁止・
  nohup 作法・`HF_HUB_DISABLE_XET=1`・triton ptxas 問題・律速切り分けの定石):
  thor-platform-skills (特に reference.md の torch.compile / triton 節)
- 実機検証済みスクリプトの実例:
  `tools/convert_lingbot_va_base.py` (ベース変換、本スキル同梱) /
  `/home/jetson/RS/run_train06_lingbotva.sh` (学習) /
  `/home/jetson/RS/run_train06_lingbotva_resume.sh` (resume)

# 前提知識 (作業前に必ず理解すること)

1. **構成**: LingBot-VA は lerobot 0.6.0 組み込みポリシー (プラグイン不要)。
   Wan2.2 系 transformer **5.09B** (実測 num_learnable_params=5,088,872,670) が
   学習対象で、映像と action を自己回帰チャンクで共同生成する世界モデル。
   凍結部 (VAE + UMT5-XXL text encoder + tokenizer、約20GB) は
   `wan_pretrained_path` (`robbyant/lingbot-va-base`) から遅延ロードされ、
   チェックポイントには含まれない。
2. **変換必須 (最重要の罠)**: `--policy.type=lingbot_va` で学習すると、事前学習
   ロードは凍結部のみで **5.09B transformer は警告なしでランダム初期化**になる。
   `tools/convert_lingbot_va_base.py` で `robbyant/lingbot-va-base` の
   `transformer/` シャードを注入した LeRobot 形式ベースを作り、
   **学習は必ず `--policy.path=<変換済みベース>`** で行う (reference.md §2)。
3. **変換時に焼き込む config 4点** (すべて必須、reference.md §2):
   `obs_cam_keys` (モデルがバッチから直接参照するカメラキー列。デフォルトは
   LIBERO の2カメラで、rename では吸収しない) /
   `used_action_channel_ids` (固定30次元 action 空間への scatter。データセット
   次元に自動適応**しない**) / `attn_mode="flex"` (学習は flex 必須。デフォルトは
   推論用の "torch") / `normalization_mapping["ACTION"]=QUANTILES` (事前学習の
   [-1,1] quantile 空間に合わせる。**デフォルト全 IDENTITY のままは罠**)。
4. **Thor 3大障害** (対策なしでは 17.7s/step、対策後 1.8s/step。reference.md §3):
   (1) triton 同梱 `ptxas-blackwell` が sm_110a 非対応 →
   `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas`、
   (2) 毎ステップ乱数 (chunk 1-4 × window 4-64 = 244通り) のブロックマスク
   再構築+再コンパイル ~15s → `utils.py` へのメモ化パッチ、
   (3) UMT5-XXL (11GB) の CPU テキストエンコードが毎ステップ 15〜32s →
   `modeling_lingbot_va.py` へのメモ化パッチ。
   **パッチは site-packages 直編集なので venv 再構築で消える** — 学習スクリプトは
   3点とも起動時に自己検証する。
5. **state 入力は完全未使用・RTC 非対応・task 必須**。チャンクは
   `frame_chunk_size(4) × action_per_frame(4)` = **16 アクション** (30fps で
   0.53秒分)。チェックポイントは **1個 約29GB** (training_state 込み) —
   ディスクフルの実績あり (Step 8)。
6. **結論 (2026-08-17)**: 現行レシピ (chunk16・30K・58ep) では**タスク不成立**。
   実機に載せる前に必ずオフライン関門 (Step 7) を通し、方向一致がプラトーなら
   そこで止める。再挑戦案は末尾の「再挑戦するなら」。

# ワークフロー

## Step 1: 前提確認

```bash
# lerobot 0.6.0 venv と CUDA torch (追加 pip 依存はなし。flex の compile は torch 同梱の triton を使用)
<venv>/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"

# データセット存在確認 (LeRobotDataset v3、task 必須)
cat <dataset_root>/meta/info.json | grep -o '"total_episodes": [0-9]*'
```

- 空きメモリ: **70GB 以上** (実測 mem_gb 48.8 + 凍結部 UMT5-XXL の CPU 常駐)。
- ディスク: 変換済みベース 10.2GB + HF キャッシュ約20GB + チェックポイント
  1個約29GB × 保存数。**save_freq=2500 で 15K resume すると6個 +174GB** —
  資金計画を先に立てる (Step 8)。
- 二重起動禁止・アロケータ・kill の掟は thor-platform-skills に従う。

## Step 2: Thor 3大対策の適用確認

学習前に必ず3点を確認する (実例スクリプトは3点とも自己検証して未適用なら中断):

```bash
# (1) ptxas: 環境変数 (学習スクリプト内で export する)
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas

# (2) ブロックマスクのメモ化パッチ (venv 再構築で消える)
grep -q "_mask_cache" <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/utils.py \
    || echo "NG: マスクキャッシュパッチ未適用 → reference.md §3.2"

# (3) UMT5 エンコードのメモ化パッチ (同上)
grep -q "_t5_embed_cache" <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/modeling_lingbot_va.py \
    || echo "NG: T5 キャッシュパッチ未適用 → reference.md §3.3"
```

- パッチの内容と再適用手順は reference.md §3。適用済み venv では元ファイルが
  `utils.py.bak.maskcache` / `modeling_lingbot_va.py.bak.t5cache` として同
  ディレクトリに退避されている (diff で差分を確認できる)。
- `TRITON_PTXAS_PATH` は **arch≥100 では効かない別ノブ** — 似た名前に注意
  (機構は thor-platform-skills reference.md 参照)。

## Step 3: ベース変換 (初回のみ、必須)

```bash
<venv>/bin/python tools/convert_lingbot_va_base.py
# → /home/jetson/RS/outputs/lingbot_va_base_lerobot (model.safetensors 10.18GB)
```

変換スクリプトがやること (reference.md §2):

1. config を構築して4点を焼き込む (obs_cam_keys / used_action_channel_ids /
   attn_mode=flex / ACTION=QUANTILES)。**自前環境ではスクリプト冒頭の定数
   (OUT_DIR / DS_ROOT / REPO とこの4点) を自分のデータセットに合わせて書き換える**。
2. `robbyant/lingbot-va-base` の `transformer/` シャードのみダウンロードし、
   **strict にキー・形状を検証して注入** (不一致なら中断)。余りの旧 Conv3d 版
   `patch_embedding.{weight,bias}` 2キーだけは既知の名残として破棄する
   (forward は `patch_embedding_mlp` のみ使用)。
3. bf16 化して `save_pretrained` (processor 込み)。

## Step 4: 学習起動

**必ず `--policy.path` を使う** (`--policy.type` は 5.09B ランダム初期化の罠)。
実例 `/home/jetson/RS/run_train06_lingbotva.sh` は事前チェック (データセット・
変換済みベース・二重起動・空きメモリ 70GB) + パッチ自己検証3点つき:

```bash
export HF_HUB_DISABLE_XET=1
# HF_HUB_OFFLINE は設定しない (凍結部 VAE/UMT5 を wan_pretrained_path から遅延ロード)
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas

lerobot-train \
    --policy.path=<変換済みベース> \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.root=<dataset_root> \
    --dataset.repo_id=<repo_id> \
    --dataset.video_backend=pyav \
    --output_dir=<output_dir> \
    --wandb.enable=false \
    --steps=15000 \
    --batch_size=4 \
    --num_workers=2 \
    --save_freq=2500
```

- 夜間学習は**ユーザーのターミナルから nohup** (thor-platform-skills の作法):

```bash
nohup sh /home/jetson/RS/run_train06_lingbotva.sh > /home/jetson/RS/train_lingbotva.log 2>&1 &
```

- scheduler は warmup(1000)+constant なので、VLA-JEPA のような
  `scheduler_decay_steps` の合わせ込みは不要 (lr は 1e-5 のまま張り付く)。
- **初回 step は ~52s、序盤はマスク warmup で遅い** — 乱数 (chunk×window) の
  新キーごとに再構築+再コンパイル ~15s が入り、244通り×15s ≈ 61分が学習全体に
  分散する。失速と誤認して止めないこと (数百 step で 1.8s/step に収束する)。

## Step 5: 学習監視

```bash
tail -c 2000 train_lingbotva.log | tr -d '\0' | tail -3
tr -d '\0' < train_lingbotva.log | grep -a -o "loss:[0-9.]*" | tail -3
```

期待値 (Thor 実測、batch4・640×360・16軸・58ep、2026-08-16〜17):

| 項目 | 実測値 |
|---|---|
| スループット | 1.79〜1.84 s/step (updt_s 1.77、data_s 0.02) |
| 15K steps 所要 | **約7.5時間** (19:47 開始 → 翌 03:20 完走) |
| メモリ (mem_gb) | 48.8 GB で完全平坦 |
| loss | 1.208 (step200) → 0.285 (6K) → 0.251 (15K) |
| grad norm | 17.4 (step200) → 2.3〜2.6 で安定 |
| チェックポイント | save_freq=2500 → 002500 〜 015000 の6個 (各約29GB) |

- 1 step が 15s 前後で張り付く → パッチ未適用か venv 再構築で消えた
  (Step 2 の grep で確認)。序盤の散発的な遅い step はマスク warmup (正常)。
- 失速・メモリ異常の一次対応は thor-platform-skills Step 7。

## Step 6: resume (総ステップ延長)

実例 `/home/jetson/RS/run_train06_lingbotva_resume.sh` (config_path + ptxas env +
パッチ自己検証つき。引数は**延長後の総ステップ数**):

```bash
nohup sh /home/jetson/RS/run_train06_lingbotva_resume.sh 30000 > /home/jetson/RS/train_lingbotva_resume.log 2>&1 &
```

- 実測: 15K→30K の +15K が **7.7時間** (1.84s/step)。loss は 0.251 → 0.243 と
  ほぼ動かず = **プラトー** (reference.md §6)。
- resume では `HF_HUB_OFFLINE=1` にできる (凍結部はキャッシュ済み)。
- **ディスク残量に注意**: 実績として resume 中にディスクフルで1回クラッシュした
  (save_freq=2500 の中間チェックポイントが約29GB ずつ積み上がる)。

## Step 7: オフライン予測評価 (関門 — 実機の前に必ず、そして止まる勇気)

teacher-forcing でデータセットのエピソードに対する予測 vs 正解を評価する
(プロトコルは他ポリシーと同一: 均一5点 + 高モーション5窓。判定指標は
vlajepa-training-skills Step 5 と同じ4点セット)。

**本件の実測 (不合格の記録、reference.md §6)**:

| チェックポイント | 高モーション方向一致率 | MAE | 判定 |
|---|---|---|---|
| 5K | 0.60 | — | 学習不足 (単調改善中) |
| 15K | 0.68 | 13.3 | 改善中 → +15K resume を実施 |
| 30K | **0.66** | **13.5** | **プラトー確定 → 不合格・実機見送り** |

- 症状: `right_shoulder_yaw` が正解と**真逆**に動く等、方向の系統誤りが継続。
  比較: 実機成立ポリシーは方向一致 0.87〜0.93 (FastWAM / VLA-JEPA v2)。
- レイテンシも非実用: **チャンク生成 6.5〜7s / 16 アクション** (30fps 実時間
  0.53秒分の生成に約13倍かかる。チャンク内の中間 tick は 2ms)。
- **判断基準**: 「5K→15K で単調改善なら resume、resume 後に改善が止まったら
  そこで打ち切り」— 15K→30K で 0.68→0.66 と頭打ちを確認して実機を見送った。
  loss (0.25 前後) だけ見ていると「まだ下がりそう」に見えるので注意。
- 評価スクリプトは当時のもの (`eval_lingbotva_offline.py`) が保存されていない。
  再評価するときは vlajepa / fastwam の `tools/eval_*_offline.py` を LingBot-VA
  のロード経路 (`--policy.path` 相当) に合わせて作り直す。

## Step 8: チェックポイント掃除 (完了後に必ず)

チェックポイント1個 約29GB (pretrained_model 9.5GB + training_state 約19.5GB)。
**完了した学習は `last/pretrained_model` のみ残し、training_state と中間
チェックポイントを消す** — 本件では完了済み学習の大掃除で **597GB を解放**した
(ディスクフルでの学習クラッシュ実績があるため、学習前の空き確認とセットで運用)。

```bash
# 完了確認後 (resume の予定がなくなってから):
du -sh <output_dir>/checkpoints/*/
# last の実体だけ残して整理する。training_state を消すと resume は不可能になる点に注意
```

## Step 9: トラブル対処と知見の記録

| 症状 | 原因 | 対処 |
|---|---|---|
| loss が高いまま・事前学習の効果が見えない | `--policy.type` 起動で 5.09B が無警告ランダム初期化 | 変換ベース + `--policy.path` に切り替える (Step 3) |
| flex_attention の compile が ptxas エラー | triton 同梱 ptxas-blackwell が sm_110a 非対応 | `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` (thor-platform-skills) |
| 毎 step 15s 超で張り付く | メモ化パッチが venv 再構築で消えた | Step 2 の grep → reference.md §3 で再適用 |
| 序盤だけ散発的に ~15s の step | マスク warmup (244通りの新キー) | 正常。放置すれば収束 (計 ~61分が分散) |
| 学習がディスクフルで死ぬ | チェックポイント 29GB × save 数 | Step 8。save_freq を粗くする / 中間を都度消す |
| 方向一致がプラトー | 現行レシピの限界 (本件 30K で確定) | 実機に載せない。「再挑戦するなら」参照 |
| RTC を指定したい | 自己回帰チャンク生成で inference_delay 非対応 | 不可。そもそもチャンク生成 6.5s は sync でも実機非実用 |
| DL 無言停止・失速・kill 後のメモリ未回収 | プラットフォーム共通 | thor-platform-skills Step 7 |

# 再挑戦するなら (未検証)

- **`action_per_frame=8` (チャンク 16→32 アクション)**: 実機成立ポリシーの
  chunk は 30〜50 (ACT 50 / SmolVLA 50 / GR00T 40 / FastWAM 32 / VLA-JEPA v2 30)
  に対し LingBot-VA は 16 と短く、VLA-JEPA v1 (chunk7) を敗因とした「静止区間
  支配」仮説の圏内。ただし**事前学習は action_per_frame=4 のトークン配置**で
  行われており、変更は事前学習配置から外れる**五分五分の賭け** (未検証)。
- 検証時も本ワークフローの関門 (Step 7) は同じ: 5K 時点の方向一致が 0.60 台
  なら早期に見切る。

# 知見の記録

- 新たな実測 (action_per_frame=8 の検証、別データセットでの再現、パッチの
  改良等) は `./reference/reference.md` に追記する。
- Thor 3大障害のうち ptxas 問題と律速切り分けの定石は**ポリシー非依存**なので
  thor-platform-skills 側 (reference.md) にも反映済み — 更新時は両方を保つ。
