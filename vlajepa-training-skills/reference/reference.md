# LeRobot 0.6.0 VLA-JEPA 学習・推論の実装知見

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0 venv
(torch 2.11.0+cu130) + 16軸ヒューマノイド (rs_follower、両腕 7DOF+グリッパ) +
自前データセット (LeRobotDataset v3、640×360、40エピソード) の実機検証
(2026-08-13〜14) で確認した内容。**2つのバグの機構と修正、ノイズ対策パッチ、
実機タスク不成立の敗因分析**を含む。

## 1. 構成と依存関係

| 項目 | 内容 |
|---|---|
| ポリシー指定 | `--policy.type=vla_jepa` (lerobot 0.6.0 組み込み、プラグイン不要) |
| VLM バックボーン | `Qwen/Qwen3-VL-2B-Instruct` (公開・ゲートなし) |
| 動画エンコーダ | `facebook/vjepa2-vitl-fpc64-256` (公開・ゲートなし) |
| ダウンロード量 | 計約 6GB |
| 追加依存 | `qwen-vl-utils>=0.0.11,<0.1.0` のみ (実測 0.0.14 で動作、torch/numpy 無傷) |
| アクションヘッド | flow-matching (velocity 予測を積分) |

- **action_dim / state_dim はデータセット実次元で自動上書き**される:
  `modeling_vla_jepa.py` が `dataset_meta` の features から
  `config.state_dim` / `config.action_dim` を設定する (デフォルト 7/8 のままでも
  16軸データセットで 16/16 になる)。カメラ名の rename も不要。
- config のデフォルト (configuration_vla_jepa.py で確認):

| フィールド | デフォルト | 備考 |
|---|---|---|
| `chunk_size` / `n_action_steps` | 7 / 7 | 30fps で 0.23 秒先まで。§7 の敗因 |
| `num_video_frames` | 8 | `observation_delta_indices` = [0..7] の源 (§3) |
| `pre_snap_gripper_action` | **True** | LIBERO 用ハック (§2)。**false 推奨** |
| `binarize_gripper_action` | **True** | 同上。**false 推奨** |
| `gripper_dim` / `gripper_threshold` | 6 / 0.5 | dim6 固定がバグの根 (§2) |

## 2. バグ1 (致命的): LIBERO 用グリッパーハックによる実機暴走

### 機構

`processor_vla_jepa.py` は postprocessor に2つのステップをデフォルトで積む:

1. `vla_jepa_pre_snap_gripper` — unnormalize **前**に
   `action[..., 6] = (action[..., 6] >= 0.5)` で {0, 1} にスナップ
2. `vla_jepa_binarize_gripper` — unnormalize **後**に
   `action[..., 6] = 1.0 - 2.0 * (action[..., 6] > 0.5)` で **{+1.0, -1.0}** に強制

どちらも `action.shape[-1] > gripper_dim (=6)` のとき無条件に発動する。つまり
**7次元以上の action を持つロボットすべてで dim6 が ±1.0 に上書きされる**。
LIBERO (7次元、dim6=グリッパー) では正しいが、それ以外では物理単位の関節指令が
±1.0 に化ける。

### 実機での症状と実証

- 16軸機では dim6 = left_shoulder_pitch (物理レンジ [43.7, 100])。
  **毎 tick 必ず -1.0 (レンジ外) に上書き**され、左肩がリミットに突っ込み続けた
  (初回実機推論が「めちゃくちゃな動き」になった直接原因)。
- オフライン予測評価 (データセット ep0 に対する予測 vs 正解) で
  dim6 が **35/35 ステップすべて -1.0** であることを実証して特定した。

### なぜ rollout 時のフラグでは直らないか

postprocessor はチェックポイントの `policy_postprocessor.json` から
そのままロードされる。**`--policy.binarize_gripper_action=false` を rollout に
付けても無効** (config は変わるが、保存済み JSON のステップ列が使われる)。

### 修正

- **今後の学習** (焼き込ませない):
  `--policy.pre_snap_gripper_action=false --policy.binarize_gripper_action=false`
- **既存チェックポイント**: `policy_postprocessor.json` から該当2ステップを削除
  (検証済み)。削除が安全な理由: unnormalizer ステップは
  `"state_file": "policy_postprocessor_step_2_unnormalizer_processor.safetensors"`
  と**ファイル名で**統計を参照しており、ステップの並び順に依存しない。

```
修正前 steps: [vla_jepa_clip_actions, vla_jepa_pre_snap_gripper,
               unnormalizer_processor, vla_jepa_binarize_gripper, device_processor]
修正後 steps: [vla_jepa_clip_actions, unnormalizer_processor, device_processor]
```

(スニペットは SKILL.md Step 4。元 JSON は `.bak.gripper` として退避する)

- 修正効果の実測: dim6 正常化、全体 MAE **12.0 → 6.7**。

## 3. バグ2: observation_delta_indices による state のラベルリーク

### 機構

- `configuration_vla_jepa.py` の `observation_delta_indices` プロパティは
  `list(range(num_video_frames))` = **[0..7]** を返す。これは動画フレーム
  [t..t+7] を読むためのものだが、LeRobot のデータローダは delta indices を
  **state を含む全 observation キーに適用**する。
- `modeling_vla_jepa.py` の入力変換は時系列 state を `state = state[:, -1, :]`
  で潰す — つまり**学習時にモデルへ入る state は t+7 の未来値**。
- 推論時の state は単フレーム (現在値 t) なので、train/inference の不一致 +
  「state ≈ 直前の action」なデータセットでは正解の一部が入力に漏れる
  (ラベルリーク)。

### 実測された影響と修正案

- 評価の結果、**今回のモデルは state をほぼ無視していた** — 値崩壊 (バグ1) の
  原因ではなかった。ただし再挑戦時 (特に state 依存を高める設定) には修正すべき。
- 最小修正案: `modeling_vla_jepa.py` の `state = state[:, -1, :]` を
  `state = state[:, 0, :]` (時刻 t の state) にする。**この修正での再学習は未実施**
  — 適用したら結果をここに追記すること。

## 4. チャンク内振動 (プルプル) と K サンプル平均パッチ

### 現象と原因切り分け

- 実機でプルプル振動する見込みの予測ノイズ: チャンク内で ±10 の振動、
  ホールド基準 MAE 1.1 に対し 6.7 (グリッパーハック修正後・K=1)。
- **積分ステップ増 (num_inference_timesteps 4→16) は無効果** → 積分精度ではなく
  **flow-matching の初期ノイズ由来のサンプリング分散**が正体。

### パッチ (検証済み)

site-packages の `lerobot/policies/vla_jepa/action_head.py` の `predict_action`
に、環境変数 `VLAJEPA_SAMPLES` で **K 回をバッチ生成して平均**する処理を追加
(元ファイルは `.bak.samples` として退避):

- `conditioning_tokens` (と state) を `repeat_interleave(k, dim=0)` で K 倍に複製
  → 積分ループを K 並列で実行 → `view(-1, k, ...).mean(dim=1)` で平均。
- **Qwen のトークン計算は1回だけ共有**され、K 倍になるのはアクションヘッドの
  flow-matching 積分のみ → 低コストで済む。

### 実測 (オフライン、640×360・16軸)

| K | MAE | チャンク内振動 | チャンク推論時間 |
|---|---|---|---|
| 1 (無効) | 6.5 | 9.0 | 143 ms |
| 8 | 3.7 | 3.7 | 206 ms |
| 32 | 2.8 | 1.8 | 446 ms |

- 推論スクリプトの既定は K=32 (実例: `/home/jetson/RS/run_infer06_vlajepa.sh`)。
- **限界**: K を上げてもホールド基準 (MAE 0.89) を下回れず、方向一致率も 0.5 前後
  のまま — ノイズは消せるがモデル自体の追従精度は改善しない (§7)。
- **注意**: site-packages への直パッチなので lerobot の再インストールで消える。
  依存する前に確認する:

```bash
grep -n "VLAJEPA_SAMPLES" <venv>/lib/python3.12/site-packages/lerobot/policies/vla_jepa/action_head.py
```

## 5. RTC 非対応 (sync 推論のみ)

- VLA-JEPA は `inference_delay` を持たない (2026-08-13 コード確認) →
  `--inference.type=rtc` は不可。**sync のみ**。
- chunk_size=7 なので 7 フレーム (30fps で 0.23 秒) ごとに再推論が走り、
  毎チャンク Qwen3-VL の前処理 (画像処理+トークン化) が入るため、
  実効制御はスロー再生気味になる (GR00T sync の 4Hz 化と同傾向)。

## 6. 実測データ (学習)

環境: Jetson AGX Thor、batch4、640×360 1カメラ、16軸、40エピソード、
`--num_workers=0 --dataset.video_backend=pyav`。

| 項目 | 実測値 |
|---|---|
| スループット | 1.89 s/step (0.53 step/s) |
| メモリ (mem_gb) | 27.4 GB (30K steps 通して平坦) |
| 30K steps 所要 | 約 15.8 時間 (21:02 開始 → 翌 12:49 完走) |
| loss | 1.355 (step200、初回ログ) → 0.302 (30K) |
| grad norm (step200) | 7.217 |
| チェックポイント | save_freq=5000 → 005000/010000/…/030000/last |

## 7. 最終評価 (2026-08-14): 実機タスク不成立と敗因分析

### 結果

- 40エピソード・30K steps では**実機タスク不成立**。
- 高モーション区間の移動方向一致率は 0.5 前後 (ランダム同等) のまま
  (保存済みの K サンプル平均評価では 0.47〜0.59。「0.40」という単発値も
  記録されたが生ログ未保存のため**要再測定**)。
- 把持動作を予測できず (正解はグリッパー軸が大きく閉じる区間で予測はほぼ静止。
  具体値 -62.7 / +0.3 は単発評価の記録で生ログ未保存 — **要再測定**)。
- **同じデータセットで ACT / SmolVLA / GR00T は動作した** → データではなく
  レシピ/モデル側の問題。

### 敗因の見立て

デフォルト `chunk_size=7` は 30fps で **0.23 秒先までしか予測しない**。
実演データの大半が静止〜低速区間のため、損失が静止区間に支配され、
「直前値を出せば損失が下がる」局所解に落ちる (ホールド基準 MAE を下回れない・
方向一致率がランダム以下、という観測と整合)。
比較: 動作した ACT は chunk 50、GR00T は chunk 40、SmolVLA は chunk 50。

### 再挑戦レシピ (未検証 — 実施したら結果を追記)

1. `--policy.chunk_size=30 --policy.n_action_steps=30` (1秒先まで予測させる)
2. state リーク修正 (§3) を先に適用
3. `--policy.pre_snap_gripper_action=false --policy.binarize_gripper_action=false`
   (§2、必須)

### 教訓

- **実機に載せる前にオフライン予測評価を必ず挟む**。今回の致命バグ (§2) は
  「データセット ep0 への予測 vs 正解」の次元別比較で発見できた。
  loss が順調に下がっても (1.355→0.302) 実機で動く保証はない。
- 評価指標は「次元別 MAE + 物理レンジ内チェック + ホールド基準比 +
  高モーション区間の方向一致率」のセットが有効だった。

## 8. 診断コマンド集

```bash
# postprocessor にグリッパーハックが焼き込まれていないか (最重要)
python3 -c "
import json
p = '<checkpoint>/pretrained_model/policy_postprocessor.json'
print([s['registry_name'] for s in json.load(open(p))['steps']])"

# K サンプル平均パッチの生存確認 (lerobot 再インストールで消える)
grep -n "VLAJEPA_SAMPLES" <venv>/lib/python3.12/site-packages/lerobot/policies/vla_jepa/action_head.py

# 学習の loss 推移 (200 step ごとの INFO 行から)
grep -a -o "loss:[0-9.]*" train_vlajepa.log | tail -5

# メモリ平坦性の確認 (Thor アロケータ問題の検知)
grep -a -o "mem_gb:[0-9.]*" train_vlajepa.log | sort | uniq -c

# hf-xet ハングの診断 (ダウンロードが無言で止まったとき)
#   .incomplete の mtime が数分前で止まっている / ソケットが CLOSE-WAIT なら該当
find ~/.cache/huggingface -name "*.incomplete" -mmin +5 2>/dev/null
ss -tnp | grep "pid=<ダウンロードプロセスのPID>"
# → 対策: HF_HUB_DISABLE_XET=1 で再実行 (通常 HTTP 経路、レジューム可)
```

## 9. 運用ノウハウ (Jetson AGX Thor)

- **`PYTORCH_CUDA_ALLOC_CONF` は Thor (iGPU/CUDA13) では一切設定しない**
  (デフォルトアロケータが正解。障害の実測・機序・診断は thor-platform-skills の
  `reference/reference.md` §2〜§3 に集約)。VLA-JEPA でも未設定で
  30K steps メモリ平坦 (27.4GB) を確認済み。
- **二重学習の禁止**: 同一 output_dir へ2本起動するとチェックポイント破損リスク。
  nohup 起動前に `pgrep -f lerobot-train` を確認する。
- **夜間ランチャーの設計** (実例: `run_train06_vlajepa_overnight.sh`):
  (1) 進行中の snapshot_download がいれば待つ (30分無応答なら pkill して引き継ぐ)
  → (2) バックボーン2つを `hf download` のリトライループ (60秒間隔、最大2時間)
  で確実にキャッシュ → (3) 学習スクリプトを exec。エージェントのセッションが
  閉じても継続するよう、**ユーザーのターミナルから nohup で起動**する。
- **推論時は `HF_HUB_OFFLINE=1`** (キャッシュ済みバックボーンのみ使用、
  起動時の不要なネットワークアクセスを防ぐ)。空きメモリ 16GB 以上が必要。
- 学習と推論の同時実行は非推奨 (メモリ/速度とも)。
- CUDA プロセスは Ctrl+C で正常終了を待つ。kill 後にメモリが戻らなければ
  nvmap リーク = 再起動。

## 付属ツール (tools/)

実機を動かす前のオフライン予測評価スクリプト (このスキルの致命バグ発見・平滑化定量化に実際に使ったもの。パスは環境に合わせて先頭の定数を書き換える):

- `tools/eval_vlajepa_offline.py` — チェックポイントを rollout と同一経路でロードし、
  データセットのエピソードに対する予測アクションと正解を比較 (全体/関節別 MAE、
  正規化空間チェック、postprocessor ステップ列の確認)。**gripper ハックによる
  dim6=-1.0 の焼き込みはこのスクリプトが検出した**
- `tools/eval_vlajepa_smooth.py` — 積分ステップ数 × K サンプル平均のマトリクス評価
  (MAE / チャンク内振動 / 境界ジャンプ / レイテンシ)。`VLAJEPA_SAMPLES` の
  適正値決定に使用
