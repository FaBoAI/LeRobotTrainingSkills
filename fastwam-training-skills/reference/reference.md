# LeRobot 0.6.0 FastWAM 学習・推論の実装知見

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0 venv
(torch 2.11.0+cu130) + 16軸ヒューマノイド (rs_follower、両腕 7DOF+グリッパ) +
自前データセット (LeRobotDataset v3、640×360、40エピソード) の実機検証
(2026-08-14〜15) で確認した内容。**必須引数の機構、無警告ランダム初期化の罠、
RTC の静かな誤動作、denoise ステップ削減、実機タスク成功の記録**を含む。
コードの行番号は lerobot 0.6.0 の site-packages
(`lerobot/policies/fastwam/`) に対するもの。

## 1. 構成と依存関係

| 項目 | 内容 |
|---|---|
| ポリシー指定 | `--policy.type=fastwam` (lerobot 0.6.0 組み込み、プラグイン不要) |
| アーキテクチャ | MoT (Mixture of Transformers): WanVideoDiT (video expert) + ActionDiT (action expert) の共拡散 |
| 学習対象パラメータ | video expert 5.000B + action expert 1.021B = **6.02B** (bf16 12GB、メタデバイス実測) |
| 事前学習チェックポイント | `lerobot/fastwam_base` (12.04GB、apache-2.0、**非ゲート**) — 自動でファインチューン元になる (§3) |
| 凍結コンポーネント | UMT5-xxl text encoder (bf16 約11.4GB) + Wan VAE (2.8GB) — optimizer には入らない |
| 追加 pip 依存 | **なし** (fastwam extra のピン transformers>=5.4,<5.6 / diffusers>=0.27.2,<0.36 に対し、venv の 5.5.4 / 0.35.2 が範囲内でそのまま動作) |
| attention | SDPA 固定 (flash-attn 不要・無効) |
| アクション表現 | 連続 flow matching (FAST 等の離散トークナイザなし) |

### ダウンロード物 (初回計約26GB、全て非ゲート)

| リポジトリ | 取得物 | サイズ |
|---|---|---|
| `lerobot/fastwam_base` | model.safetensors + config + processor JSON | 12.04GB |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `text_encoder/*` (UMT5-xxl 3シャード) | 11.4GB |
| 同上 | `vae/*` | 2.8GB |
| `google/umt5-xxl` | トークナイザのみ (`spiece.model` + `tokenizer*`) | 25MB |

- **Wan-AI リポジトリの `transformer/` (約20GB) は不要**。video expert 5B の
  重みは policy checkpoint (`fastwam_base`) からしか来ない
  (modeling_fastwam.py:244-291 — WanVideoDiT を空で作り base の safetensors から
  埋める。Wan-AI の DiT シャードを読む `from_wan22_pretrained` は base 作成用の
  オフライン経路で、lerobot-train からは呼ばれない)。
  `hf download --include` / `allow_patterns` で必ず絞ること。
- umt5-xxl の 52GB の重み .bin もダウンロードされない (トークナイザのみ)。

### config デフォルト (configuration_fastwam.py で確認)

| フィールド | デフォルト | 備考 |
|---|---|---|
| `action_dim` / `proprio_dim` | **7 / 8** | 自動適応しない → 16軸機は両方 16 を明示 (§2) |
| `action_horizon` / `n_action_steps` | 32 / 32 | 30fps で 1.07 秒分。queue 全消費後に再推論 (完全 open-loop) |
| `n_obs_steps` | 1 | 入力は現在フレームのみ |
| `num_video_frames` / `action_video_freq_ratio` | 33 / 4 | 33フレーム窓を4間引き → 9フレームを video 教師にロード (§6) |
| `image_size` | (224, 448) | N カメラで幅分割。1カメラなら 224×448 (§6) |
| `num_inference_steps` | 10 | denoise ステップ。**3 に減らして劣化なし** (§8) |
| `inference_seed` | 42 | 毎チャンク同一初期ノイズ (決定的) |
| `torch_dtype` / `fp32_attention` | bfloat16 / True | |
| `freeze_video_expert` | False | true で学習対象 1.02B に (§10) |
| `toggle_action_dimensions` | **[]** | LIBERO 用ハック。**設定禁止** (§4) |
| optimizer / scheduler | AdamW lr=1e-4, wd=1e-2, grad_clip=10 / なし (定数LR) | |

## 2. 必須引数 action_dim/proprio_dim と cross-embodiment ロード

### 機構: データセットから自動適応しない

make_policy はデータセットから output_features を上書きするが、
`config.action_dim` (デフォルト7) / `config.proprio_dim` (デフォルト8) は
変更されない。`set_dataset_feature_metadata` (configuration_fastwam.py:288-317)
はカメラキーの再構築のみで、state 特徴量の shape も dataset ではなく
`self.proprio_dim` から作る。

### 指定漏れの症状 (16軸データセットの場合)

- **action_dim 漏れ**: validate_features が
  「action feature shape must be (7,), got (16,)」で**即エラー** (fail-fast)。
- **proprio_dim 漏れ** (action_dim=16 だけ指定): config 検証を**素通り**し、
  最初の forward で「proprio last dim must be 8, got 16」の
  **実行時エラー** (wan/modular.py:1185-1187)。学習開始まで気付けないので
  必ず両方指定する。

### 指定時: cross-embodiment ロード

互換判定キー `_FASTWAM_*_COMPAT_KEYS` (configuration_fastwam.py:36-57) に
action_dim/proprio_dim は**含まれない**ため、16 を指定しても
`fastwam_base` の自動ロード (§3) は有効のまま。shape が合わない
action encoder/head + proprio encoder (計6テンソル) だけが再初期化され、
残り全部 (5B video expert 含む) がロードされる (modeling_fastwam.py:90-137)。

### ログでの正常確認 (2026-08-14 実走で確認)

実走ログに出たのは次の WARNING (ロード完了時):

```
WARNING ... utils.py:91 Missing key(s) when loading model:
{'model.mot.mixtures.action.action_encoder.weight', 'model.proprio_encoder.weight',
 'model.mot.mixtures.action.head.weight', 'model.mot.mixtures.action.action_encoder.bias',
 'model.mot.mixtures.action.head.bias', 'model.proprio_encoder.bias'}
```

fastwam_base はこれら action/proprio 層のキー自体を含まないため、
missing-key 経路 (strict=False) で再初期化された。コード上は checkpoint に
同名キーが shape 違いで存在する場合の「FastWAM cross-embodiment load:
reinitializing ...」警告 (modeling_fastwam.py:126) もある。
**どちらの形でも「再初期化されたのが action encoder/head + proprio encoder の
6テンソルだけ」なら正常**。それ以外のキー (video expert 等) が混ざっていたら
config の互換性を疑う (§3)。

## 3. 罠 (最重要): base 非互換 config は無警告で 5B がランダム初期化

- `FastWAMConfig.__post_init__` (configuration_fastwam.py:263-268) は
  pretrained_path 未指定かつ **config が base 互換のときだけ**
  `pretrained_path = 'lerobot/fastwam_base'` を自動設定する。
  非互換 (hidden_dim 等のアーキテクチャキーを変えた) 場合は
  **警告なしに return** し、6B 全体がランダム初期化のスクラッチ学習になる。
  §1 の通り 5B の重みは fastwam_base からしか来ないので、これは致命的。
- action_dim / proprio_dim は互換判定キー外なので 16 でも安全
  (draccus parse で pretrained_path 自動設定を実行確認済み)。
- **診断**: 学習ログの config ダンプに
  `'pretrained_path': 'lerobot/fastwam_base'` があること +
  §2 のロード WARNING が出ること。両方とも無い + 序盤 loss が高止まり、なら
  ランダム初期化を疑う。

```bash
grep -a -o "'pretrained_path': '[^']*'" train_fastwam.log
```

- 意図的にスクラッチ学習したい場合のみ `--policy.base_model_id=""`。

## 4. LIBERO 用ハック (toggle_action_dimensions) — デフォルト無効・設定禁止

- VLA-JEPA の実機暴走バグ (グリッパーハック) と同種の機構が
  `FastWAMActionToggleProcessorStep` (processor_fastwam.py:44-73、登録名
  `fastwam_action_toggle_processor`) として存在する:
  指定 dim に `sign(-(2v-1))` を適用し ±1 に強制する LIBERO 用トグル。
- ただし VLA-JEPA と違い **config でゲートされている**:
  `toggle_action_dimensions` (デフォルト `[]`) が空なら postprocessor に
  step 自体が追加されない (processor_fastwam.py:126-129)。さらに
  **`lerobot/fastwam_base` の policy_postprocessor.json は
  unnormalizer + device の2ステップのみで toggle 不在を Hub 実物で確認済み**。
  つまり fastwam_base からのファインチューンでは発火しない。
- **`--policy.toggle_action_dimensions` は絶対に設定しない**。
- **例外的に要注意**: `--policy.path` で第三者のチェックポイント (LIBERO 調整済み
  等) から学習する場合。pretrained_path があると processor はその repo の JSON
  からロードされ、train 時に差し替わるのは normalizer/unnormalizer の
  stats・device・rename だけで toggle step は素通しになる —
  **VLA-JEPA の焼き込みバグと同型の経路**。学習前に必ず検査する:

```bash
python3 -c "
import json
p = '<checkpoint>/pretrained_model/policy_postprocessor.json'
print([s['registry_name'] for s in json.load(open(p))['steps']])
"
# OK: ['unnormalizer_processor', 'device_processor']
# NG: 'fastwam_action_toggle_processor' が含まれる → VLA-JEPA スキル Step 4 と同様に JSON から削除
```

## 5. RTC 非対応 — 「静かな誤動作」なので禁止

- fastwam ディレクトリ全体で `inference_delay` の grep は 0 ヒット。
  `predict_action_chunk` のシグネチャは `(self, batch, **_: Any)`
  (modeling_fastwam.py:206) で、RTC エンジンが渡す
  `inference_delay` / `prev_chunk_left_over` (rollout/inference/rtc.py:307-309)
  を**無言で破棄**する。
- 結果、`--inference.type=rtc` は**エラーなく起動する**が、遅延補償・inpainting
  ガイダンスが完全無効の素朴な非同期チャンク差し替えに退化し、チャンク境界で
  アクション不連続が出うる。ACT (TypeError で落ちる) や VLA-JEPA と違い
  **壊れたことに気付けない**ため、**rtc 指定を禁止**と明記する。推論は sync のみ。
- sync の実態: `select_action` は 32 アクションの deque を消費し尽くしてから
  再推論する完全 open-loop (modeling_fastwam.py:236-242)。
- rollout の `use_torch_compile` は FastWAM (VAE・動的分岐込み) では未検証 —
  使わない。

## 6. 入力形式と既知の軽微問題

- **画像**: 推論入力は現在フレーム1枚のみ (n_obs_steps=1)。モデル内で
  bilinear+antialias により 224×448 へ自動リサイズ (preprocessor に resize step
  は無い)。640×360 (16:9) → 224×448 (2:1) の**アスペクト歪みがある**が、
  実機動作に問題なし (実測)。正規化は VISUAL=IDENTITY で [0,1] のまま通し、
  VAE 境界で [-1,1] 化。STATE/ACTION は MEAN_STD で、stats は必ずデータセットの
  ものに上書きされる (7次元 stats の混入はない)。
- **学習時の video 教師**: 33フレーム窓を4間引きした9フレームをロードする。
  1サンプルにつき動画9フレームのデコードが走るため dataloader が重い —
  `--num_workers=2` を採用 (実測 data_s:0.016 で余裕)。
- **state の未来スタックはリークではない**: `observation_delta_indices` =
  [0,4,...,32] が state にも適用され [B,9,16] でロードされるが、モデルは
  `proprio[:, 0, :]` (現在フレームのみ) を使う (wan/modular.py:1189)。
  MoT の attention mask も action トークンには先頭フレームの video トークン
  しか見せない。未来フレームは video (世界モデル) 教師として**意図的**に
  使われる — VLA-JEPA のラベルリーク (バグ2) は再現しない。
- **既知の軽微問題 (image_is_pad)**: build_inputs は `image_is_pad` キーを
  期待する (wan/modular.py:1151) が、LeRobot が生成するのは
  `observation.images.front_is_pad` で変換コードが無い → エピソード末尾の
  video loss がパディングフレーム (最終フレーム繰り返し) を教師にする。
  **action 側は `action_is_pad` で正しくマスク**されるため実害は小さい
  (リークではなく末尾のみの品質問題)。
- **task 必須**: 無いと KeyError「FastWAM training requires a `task`/`prompt`」。
  task は prompt_template「A video recorded from a robot's point of view
  executing the following instruction: {task}」に埋められ、凍結 UMT5-xxl で
  毎 step オンザフライにエンコードされる (max 128 トークン)。
- **rename 不要**: `observation.images.` prefix の任意キーを sorted で拾う。
  preprocessor の rename_map も空 (fastwam_base の JSON 実物で確認)。

## 7. 実測データ (学習)

環境: Jetson AGX Thor、batch4、640×360 1カメラ、16軸、40エピソード (17,681
フレーム ≈ 4,420 steps/epoch)、`--num_workers=2 --dataset.video_backend=pyav`、
フル 6B (FREEZE_VIDEO=false)。2026-08-14 17:18 開始 → 08-15 07:30 完走。

| 項目 | 実測値 |
|---|---|
| 起動 (重みロード〜学習開始) | 約1.5時間 (17:18 → 18:47) |
| スループット | 3.04 s/step (INFO の updt_s 3.016、data_s 0.016) |
| 15K steps 所要 | 学習 12.7h + 起動 ≈ **14.2h** (実効 3.4 s/step) |
| メモリ (mem_gb) | **68.2 GB で完全平坦** (68.20〜68.22。開始時空き 89GB、preflight 閾値 75GB) |
| loss | 0.993 (step200) → 0.168〜0.182 (15K 付近。**まだ下降中**) |
| grad norm (15K) | 1.660 |
| epoch | 15K steps ≈ 3.39 epoch |
| チェックポイント | save_freq=2500 → 002500/…/015000 の**6個** (`last` は 015000 への symlink)、**各約34GB (pretrained_model 12GB + training_state 22GB)・計202GB** (du 実測) |

- フル 6B が unified memory 122GB に問題なく収まった (フォールバック不発)。
  `FREEZE_VIDEO=true` (action expert 1B のみ、preflight 45GB) は保険として
  overnight ラッパーに組み込み済みだが未使用。
- loss が 15K 時点で下降中 → resume による総ステップ延長の価値あり
  (手順は thor-platform-skills Step 3)。

## 8. オフライン評価と denoise ステップ削減 (推論の無料高速化)

### 15K チェックポイントの評価 (tools/eval_fastwam_offline.py、2026-08-15)

均一5点 + 高モーション5窓 (32フレーム窓の総移動量トップ) で予測 vs 正解:

| 指標 | 実測 |
|---|---|
| MAE | 2.0〜2.6 |
| 高モーション窓の移動方向一致率 | **0.87〜0.89** (実機成立組トップ級。VLA-JEPA v2 は 0.93、v1 (chunk7) は 0.5 前後 = ランダム同等、LingBot-VA は 0.66 で不合格) |
| 予測移動量 | 正解の8割超 (静止予測への局所解落ちなし) |
| 定数出力 (VLA-JEPA dim6 型) | なし (全16関節が可変) |
| チャンク推論 (32アクション、denoise 10) | 552 ms |

### denoise ステップ削減 (`--policy.num_inference_steps`、再学習不要)

推論は「初期フレーム latent を 5B に1回プリフィル (KV cache) → action latent
のみ denoise」の高速パスなので、denoise ステップ削減がほぼ線形に効く:

| num_inference_steps | MAE | 方向一致 | チャンク推論 |
|---|---|---|---|
| 10 (デフォルト) | 2.79 | 0.95 | 552 ms |
| **3 (採用)** | **2.52** | **0.98** | **322 ms** |

- 10→3 で**品質劣化なし** (むしろ微改善)。2 でも劣化なしだったが余裕を見て 3 を
  既定にした (`run_infer06_fastwam.sh` の INFER_STEPS=3)。
- 床は **~290ms = 5B プリフィル**で、これ以上はステップ削減では縮まない。
- `inference_seed=42` 固定のため毎チャンク同一初期ノイズ (決定的)。
  VLA-JEPA で必要だった K サンプル平均のようなノイズ対策は不要だった。

## 9. 実機評価 (2026-08-15): タスク成功

- **問題なく動作し、タスク成功** (ユーザー確認)。同一タスク・同一 Thor の
  6ポリシー比較で、**オフライン指標・実機とも最良クラス**。
- 制御ループ律速のスロー再生 (6〜7Hz — セッション記録のみで生ログ未保存、
  **要再測定**) は他ポリシーの sync と同傾向 (GR00T sync の 4Hz 化と同系)。ただし chunk=32 (30fps で 1.07秒分) を
  322ms で生成するため、チャンク切替時の停滞は短い。
- 推論の preflight (実例 `run_infer06_fastwam.sh`): can0 up / カメラリンク
  (mgbe0_0 carrier) / config.json + .safetensors 存在 / 学習プロセス非実行 /
  空きメモリ **35GB 以上** (Wan2.2 5B + UMT5 のロードに必要)。
- `HF_HUB_OFFLINE=1` で起動 (キャッシュ済みバックボーンのみ使用)。

## 10. 運用ノウハウ

- **ディスク管理**: HF キャッシュ 26GB + チェックポイント 202GB
  (6個。`last` は 015000 への symlink であり実体を持たない)。
  実機で採用チェックポイントを確定したら、**中間チェックポイント
  (002500〜012500) を整理して採用分だけ残す**ことを推奨 (1個約34GB なので
  効果が大きい)。resume 延長の予定がなければ、採用分も training_state
  (22GB、optimizer 状態) を消して pretrained_model (12GB) だけ残せる。
- **freeze コース** (`--policy.freeze_video_expert=true`): 学習対象が
  action expert 1.02B に減り preflight 45GB で回る。docstring 推奨では
  `lambda_video=0` も併用するが、loss は dict フィールドで**ドット記法不可**
  (`--policy.loss.lambda_video=0` は unrecognized arguments エラーを実測) —
  JSON 丸ごと `--policy.loss='{"lambda_video": 0.0, "lambda_action": 1.0}'` の
  形になる (この形での学習は未実施。freeze のみなら video loss は計算されるが
  凍結側に勾配が流れないだけで動作する)。
- Thor 共通の掟 (`PYTORCH_CUDA_ALLOC_CONF` 禁止・`kill -9` 禁止・nohup 作法・
  `HF_HUB_DISABLE_XET=1`・hf-xet ハング診断) は **thor-platform-skills の
  reference.md に集約** — 本スキルでは繰り返さない。FastWAM でも未設定
  デフォルトアロケータで 15K steps メモリ完全平坦 (68.2GB) を確認済み。
- 夜間ランチャーの FastWAM 固有部 (実例 `run_train06_fastwam_overnight.sh`):
  バックボーン3点を `--include` 付き `hf download` のリトライループ (60秒間隔・
  最大2時間) で揃える → フル 6B 学習 → **序盤30分以内の異常終了なら
  FREEZE_VIDEO=true・別 output_dir で自動フォールバック**。

## 付属ツール (tools/)

実機を動かす前のオフライン予測評価スクリプト (このスキルの評価数値の実測に
使ったもの。パスは環境に合わせて先頭の定数を書き換える):

- `tools/eval_fastwam_offline.py` — チェックポイントを rollout と同一経路で
  ロードし、均一5点 + 高モーション5窓で予測 vs 正解を比較 (MAE / 移動方向一致率 /
  予測移動量 / 定数出力チェック / チャンク推論レイテンシ)。postprocessor に
  LIBERO toggle step が焼き込まれていないことも assert する (§4)。
  `--policy.num_inference_steps` の適正値決定 (§8) にも使用。
