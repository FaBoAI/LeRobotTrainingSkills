# LeRobot 0.6.0 LingBot-VA 学習の実装知見 (オフライン関門不合格の記録込み)

Jetson AGX Thor (JetPack 7 / CUDA 13, unified memory 122GB) + lerobot 0.6.0 venv
(torch 2.11.0+cu130, triton 3.6.0) + 16軸ヒューマノイド (rs_follower、両腕
7DOF+グリッパ) + 自前データセット (LeRobotDataset v3、640×360、58エピソード/
25,780フレーム) の実機検証 (2026-08-16〜17) で確認した内容。
**変換必須の機構、Thor 3大障害の対策パッチ (17.7s/step → 1.8s/step)、
30K steps 完走とオフライン関門不合格 (実機見送り) の記録**を含む。
コードの行番号は lerobot 0.6.0 の site-packages
(`lerobot/policies/lingbot_va/`) に対するもの (パッチ適用後は多少ずれる)。

## 1. 構成と依存関係

| 項目 | 内容 |
|---|---|
| ポリシー指定 | **`--policy.path=<変換済みベース>`** (§2。`--policy.type=lingbot_va` はランダム初期化の罠) |
| アーキテクチャ | Wan2.2 系 transformer による自己回帰 video-action 世界モデル (映像 latent と action を同一系列で共同生成、flow matching) |
| 学習対象パラメータ | **5.09B** (ログ実測 num_learnable_params=5,088,872,670。bf16 で model.safetensors 10.18GB) |
| 凍結コンポーネント | Wan VAE + **UMT5-XXL text encoder (約11GB、`text_encoder_device="cpu"` 既定)** + tokenizer — 計約20GB。チェックポイント非同梱で `wan_pretrained_path`=`robbyant/lingbot-va-base` から遅延ロード |
| チャンク構造 | `frame_chunk_size=4` × `action_per_frame=4` = **チャンク16アクション** (`chunk_size`/`n_action_steps` は読み取り専用プロパティで直接指定不可) |
| action 空間 | **固定30次元** (`action_dim=30`)。実次元は `used_action_channel_ids` で 30次元空間へ scatter (§2) |
| attention | 学習 = `attn_mode="flex"` (flex_attention + torch.compile、triton 必須)。推論 = "torch" SDPA / "flashattn" (config コメントに明記) |
| 追加 pip 依存 | なし (既存 lerobot060-venv で動作。compile は torch 同梱の triton 3.6.0) |
| RTC | 非対応 (自己回帰チャンク生成 + KV フィードバック。sync のみ) |
| state 入力 | **完全未使用** (`observation.state` は input_features に載るがモデルは参照しない) |
| task | **必須** (UMT5-XXL でエンコードされ cross-attention に入る) |

## 2. 変換必須: `--policy.type` の無警告ランダム初期化と config 焼き込み

### 機構

- `--policy.type=lingbot_va` の学習起動では、事前学習からロードされるのは
  **凍結部 (VAE / UMT5 / tokenizer) のみ**。5.09B transformer は
  **警告ひとつ出ずにランダム初期化**され、気づかずに一晩を溶かす。
  (FastWAM の「base 非互換 config で無警告ランダム初期化」と同族だが、
  こちらは type 指定というだけで必ず発動する分たちが悪い。)
- 対策 = `tools/convert_lingbot_va_base.py`:
  `robbyant/lingbot-va-base` の `transformer/` シャード
  (`diffusion_pytorch_model.safetensors.index.json` 経由) をポリシーに
  **strict 注入** (missing / unexpected / 形状不一致のいずれかで即中断) して
  `save_pretrained` → 学習は `--policy.path=<出力先>`。
- **既知の余りキー**: シャードに残る旧 Conv3d 版 `patch_embedding.{weight,bias}`
  2キーは学習初期化の名残で、lerobot 実装の forward は `patch_embedding_mlp`
  のみ使用 (utils.py:911, 1063 で確認) → 変換時に破棄して問題ない。
  それ以外のキー不一致は握りつぶさない (strict のまま調査する)。

### 変換時に焼き込む config 4点 (すべて必須)

| 設定 | デフォルト (罠) | 焼き込み値 (16軸・1カメラの実例) | 理由 |
|---|---|---|---|
| `obs_cam_keys` | `["observation.images.image", "observation.images.image2"]` (LIBERO 2カメラ) | `["observation.images.front"]` | モデルが**バッチからこのキーを直接参照** (`_extract_raw_obs`)。SmolVLA のような rename_map での吸収はしない。カメラ latent は width 方向連結なので順序も意味を持つ |
| `used_action_channel_ids` | `list(range(7))` (LIBERO 7DoF) | `list(range(16))` | 固定30次元 action 空間への scatter 位置。**データセット実次元への自動適応なし** (VLA-JEPA と違う)。漏れると 7 次元しか学習されない |
| `attn_mode` | `"torch"` | `"flex"` | 学習は flex 必須 (config コメント: "flex" = training only)。flex は triton compile を伴う → §3.1 |
| `normalization_mapping["ACTION"]` | `IDENTITY` (全種 IDENTITY) | `QUANTILES` | 事前学習は [-1,1] quantile action 空間。IDENTITY のままだと物理単位の生値が 5B に入り、事前学習分布から完全に外れる |

- 焼き込み済み config の実物: `/home/jetson/RS/outputs/lingbot_va_base_lerobot/config.json`
  (`"attn_mode": "flex"`, `"used_action_channel_ids": [0..15]`,
  `"normalization_mapping": {"ACTION": "QUANTILES", ...}` を確認済み)。
- quantile 統計は変換時ではなく学習時に dataset stats から
  processor (`policy_preprocessor/postprocessor.json`) に入る。

## 3. Thor 3大障害と対策 (17.7s/step → 1.8s/step、10倍速)

対策前の初回本番は **17.7s/step** (15K ≈ 3日超ペース)。以下3点で
**1.8s/step** になった (2026-08-16 実測)。(1) はポリシー非依存の普遍知見
(thor-platform-skills reference.md にも記載)、(2)(3) は LingBot-VA 固有。

### 3.1 triton 同梱 ptxas-blackwell の sm_110a 非対応

- triton 3.6.0 は **arch ≥ 100 では同梱の `ptxas-blackwell` を使う**
  (`triton/backends/nvidia/compiler.py:35` —
  `return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas`)。
  同梱版は CUDA 12.9 ビルド (PTX 8.8 世代) で **Thor の sm_110a を扱えず**、
  flex_attention の compile が失敗する。
- 対策 (学習スクリプトで export):

```bash
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas   # CUDA 13.0 (PTX 9.0 生成)
```

- **`TRITON_PTXAS_PATH` は arch<100 用の別ノブで Thor では効かない**
  (`triton/knobs.py:491-492` で別変数)。名前が似ているので混同注意。
- 詳細と診断コマンドは thor-platform-skills reference.md の
  「torch.compile / triton を使うポリシーの必須設定」節。

### 3.2 ブロックマスクの毎ステップ再構築 (律速その1)

- 学習の forward は**毎ステップ乱数**で `chunk_size` = randint(1,5) (=1〜4)、
  `window_size` = randint(4,65) (=4〜64) を引く
  (modeling_lingbot_va.py:317-318) → 組合せ **4×61 = 244通り**。
- 素の実装は毎ステップ `FlexAttnFunc.init_mask` でブロックマスクを再構築 +
  `create_block_mask` を再コンパイルし、**単体実測 ~15.7s/回**。
- 対策 = `utils.py` の `FlexAttnFunc` に**キー付きメモ化パッチ**
  (元ファイルは `utils.py.bak.maskcache` として退避):
  - クラス辞書 `_mask_cache` / `_cross_mask_cache` を追加。
  - `init_mask` 冒頭でキーを作り、ヒットしたら即 return:

```python
shape_key = (tuple(latent_shape), tuple(action_shape), int(padded_length),
             tuple(patch_size), str(device))
mask_key = shape_key + (int(chunk_size), int(window_size))   # self-attn は乱数込み
cached = FlexAttnFunc._mask_cache.get(mask_key)
cached_cross = FlexAttnFunc._cross_mask_cache.get(shape_key) # cross は shape のみ
if cached is not None and cached_cross is not None:
    FlexAttnFunc.attention_mask = cached
    FlexAttnFunc.cross_attention_mask = cached_cross
    return
# ... 従来の構築処理の末尾で両キャッシュに格納 ...
```

- 実測: **同一キーは 0.000s**。キャッシュミス (新しい chunk×window) のときだけ
  ~15s かかり、244通り × 15s ≈ **計61分の warmup が学習序盤に分散**する
  (初回 step 52s、数百 step で 1.8s/step に収束)。バッチ形状が変わると
  shape_key ごと別キャッシュになる点に注意。

### 3.3 UMT5-XXL の CPU テキストエンコード (律速その2・主犯)

- `text_encoder_device="cpu"` 既定 (VRAM ~11GB 節約) のため、UMT5-XXL の
  エンコードは CPU 実行で **1回 15〜32秒** (単体実測)。素の実装は
  **毎ステップ同一のタスク文を再計算**していた (single-task データセット)。
- 対策 = `modeling_lingbot_va.py` の `_get_t5_prompt_embeds` に
  **プロンプト別メモ化パッチ** (元ファイルは
  `modeling_lingbot_va.py.bak.t5cache` として退避)。要点:

```python
cache_key = (tuple(prompt), int(max_sequence_length))
cached = self._t5_embed_cache.get(cache_key)
if cached is not None:
    return cached
# ... 従来のエンコード処理 ...
if len(self._t5_embed_cache) < 64:   # マルチタスクでも暴走しない上限
    self._t5_embed_cache[cache_key] = result
```

- single-task なら2ステップ目以降エンコードコストはゼロ。マルチタスクでも
  タスク文の種類数が 64 以下ならフルヒットする。

### パッチ運用 (共通)

- **site-packages 直編集なので lerobot 再インストール / venv 再構築で消える**。
  学習スクリプト (`run_train06_lingbotva.sh` / 同 `_resume.sh`) は起動時に
  3点 (ptxas env は export、パッチ2点は grep) を自己検証して未適用なら中断する。

```bash
grep -q "_mask_cache"     <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/utils.py
grep -q "_t5_embed_cache" <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/modeling_lingbot_va.py
```

## 4. 律速切り分けの定石 (py-spy は使えない)

- **py-spy は ptrace 制限で他プロセスに attach できない** (Thor の本環境) →
  プロファイラに頼らず**成分の単体実測**で犯人を絞るのが定石
  (thor-platform-skills reference.md にも一般形を記載)。
- 本件の切り分け実績 (18s/step の内訳):

| 成分 | 単体実測 | 判定 |
|---|---|---|
| 動画デコード (pyav) | 0.06s/サンプル | シロ |
| flex ブロックマスク構築+コンパイル | 15.7s/回 (新キー時) | 主犯1 → §3.2 |
| UMT5-XXL CPU テキストエンコード | 15〜32s/回 (毎ステップ発火) | 主犯2 → §3.3 |

- 教訓: single-task データセットでは「**毎ステップ同じ入力を再計算していないか**」
  (テキストエンコード・マスク構築・トークン化) をまず疑う。

## 5. 実測データ (学習)

環境: Jetson AGX Thor、batch4、640×360 1カメラ、16軸、58ep/25,780フレーム、
`--num_workers=2 --dataset.video_backend=pyav`、パッチ3点適用済み。

| 項目 | 本番 (0→15K) | resume (15K→30K) |
|---|---|---|
| 実施 | 2026-08-16 19:47 → 08-17 03:20 | 2026-08-17 12:02 → 19:47 |
| スループット | 1.79〜1.80 s/step (updt_s 1.77) | 1.84 s/step (updt_s 1.81) |
| 所要時間 | **約7.5時間** | **約7.7時間** (計 30K ≈ 15.3h) |
| メモリ (mem_gb) | 48.77 で完全平坦 | 48.76 で完全平坦 |
| loss | 1.208 (200) → 0.362 (2K) → 0.285 (6K) → 0.251 (15K) | 0.248 (15K) → **0.243 (30K)** |
| grad norm | 17.4 (200) → 2.3〜2.6 | 2.3〜2.7 |
| エポック | 2.33 (15K) | **4.65 (30K)** |
| チェックポイント | save_freq=2500 → 002500/…/015000 (6個) | save_freq=2500 → 017500/…/030000 (6個) |

- **loss は 15K→30K でほぼ動かず (0.251→0.243) = プラトー**。grad norm も
  平坦で、これ以上回しても改善しない形 (§6 のオフライン指標と整合)。
- 初回 step 52s (compile)、序盤にマスク warmup (§3.2) が分散する以外は完全平坦。
- 注: メモリ (記録) 上は「55ep/24,430フレーム」とされていたが、学習ログの
  一次記録は `dataset.num_episodes=58 / num_frames=25780` (両ランとも同一)。
  本書は後者に従う。

## 6. オフライン評価と不合格判定 (実機見送りの根拠)

teacher-forcing で ep0 に対する予測 vs 正解を評価 (プロトコルは他ポリシーと
同一: 均一5点 + 高モーション5窓、判定4点セットは vlajepa-training-skills 参照)。

| 指標 | 5K | 15K | 30K | 実機成立ライン (参考) |
|---|---|---|---|---|
| 高モーション方向一致率 | 0.60 | 0.68 | **0.66** | FastWAM 0.87-0.89 / VLA-JEPA v2 0.93 |
| MAE | — | 13.3 | **13.5** | FastWAM 2.0-2.6 / VLA-JEPA v2 1.40 |
| チャンク生成時間 | — | 6.4s | 6.5〜7s / 16アクション | — |

- **5K→15K は単調改善 (0.60→0.68) = 学習不足と診断して +15K resume。
  15K→30K で 0.68→0.66 と頭打ち = プラトー確定 → 不合格・実機見送り**
  (2026-08-17 夜)。
- 質的症状: `right_shoulder_yaw` が正解と**真逆**に動く等、方向の系統誤りが
  30K でも継続。ノイズではなくモデルの追従自体が成立していない
  (VLA-JEPA v1 の敗因と同じ見え方)。
- レイテンシも単独で失格級: チャンク生成 **6.5〜7s / 16アクション** =
  30fps 実時間 0.53秒分の生成に約13倍。中間 tick (キュー取り出し) は 2ms で
  問題ないが、チャンク境界ごとに 6.5s 停止する sync 制御は実機非実用
  (GR00T sync の 4Hz スローどころではない)。
- **loss との乖離に注意**: loss 0.24 は数字だけ見れば FastWAM (0.17) と
  大差ないが、方向一致は 0.66 vs 0.89 と決定的に違う。**関門はオフライン指標**
  (loss では実機の成否を判定できない — 5ポリシー通しての教訓と同じ)。
- 当時の評価スクリプト (`eval_lingbotva_offline.py`) は未保存。再評価時は
  vlajepa / fastwam の `tools/eval_*_offline.py` を流用して再構成する
  (ロードは学習と同じ `--policy.path` 経路、チャンクは 16 アクション単位)。

## 7. 敗因の考察と再挑戦案 (未検証)

- **チャンク長仮説**: LingBot-VA のチャンクは 16 アクション (0.53秒) で、
  実機成立ポリシー (ACT 50 / SmolVLA 50 / GR00T 40 / FastWAM 32 /
  VLA-JEPA v2 30) より短い。VLA-JEPA が chunk 7→30 で不成立→成立に転じた
  実証があり、同じ「静止区間支配」の圏内の可能性がある。
- **再挑戦案 = `action_per_frame=8`** (frame_chunk_size 4 のまま chunk 32):
  ただし**事前学習は action_per_frame=4 のトークン配置**で行われており、
  変更は事前学習の系列配置から外れる。**五分五分の賭け (未検証)** —
  試すなら 5K 時点の方向一致で早期見切りする (0.60 台なら望み薄)。
- 他の可能性 (未検証): 58ep では 5.09B に対しデータ不足 /
  quantile 正規化と実ロボットの action 分布の相性。いずれも切り分けていない。

## 8. ディスク運用 (チェックポイント 29GB の現実)

- チェックポイント **1個 約29GB** = pretrained_model 9.5GB (5.09B bf16) +
  training_state 約19.5GB (optimizer 状態)。save_freq=2500 の resume で
  6個 +174GB。
- **実績: resume 中にディスクフルで1回クラッシュ** → 完了済み学習
  (FastWAM 等含む) の中間チェックポイント・training_state の大掃除で
  **597GB 解放**して再開した。
- 定石: **完了した学習は `last/pretrained_model` のみ残す** (training_state を
  消すと resume 不可になるので、延長の予定がなくなってから)。学習前 preflight に
  ディスク残量確認を足す価値がある (メモリ 70GB チェックだけでは防げなかった)。

## 9. 診断コマンド集

```bash
# 変換ベースが事前学習入りか (--policy.type の罠検知): config と重みの存在
ls -la <変換済みベース>/model.safetensors        # 10.18GB あること
python3 -c "import json; c=json.load(open('<変換済みベース>/config.json'));
print(c['attn_mode'], c['used_action_channel_ids'], c['normalization_mapping']['ACTION'], c['obs_cam_keys'])"
# → flex [0..15] QUANTILES ['observation.images.front'] (自環境の値) であること

# 高速化パッチ2点の生存確認 (venv 再構築で消える)
grep -n "_mask_cache" <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/utils.py | head -1
grep -n "_t5_embed_cache" <venv>/lib/python3.12/site-packages/lerobot/policies/lingbot_va/modeling_lingbot_va.py | head -1

# ptxas: 同梱 ptxas-blackwell の世代 (CUDA 12.9 なら sm_110a 非対応で該当)
<venv>/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell --version | tail -1
/usr/local/cuda/bin/ptxas --version | tail -1     # CUDA 13.0 であること

# 学習の実勢 (ログには tqdm のヌル文字が混ざるので tr -d '\0')
tr -d '\0' < train_lingbotva.log | grep -a -o "loss:[0-9.]*" | tail -3
tr -d '\0' < train_lingbotva.log | grep -a -o "updt_s:[0-9.]*" | tail -3   # 1.8 前後が正常
tr -d '\0' < train_lingbotva.log | grep -a -o "mem_gb:[0-9.]*" | sort -u   # 48.8 で平坦

# ディスク (チェックポイント積み上がりの監視)
du -sh <output_dir>/checkpoints/*/ ; df -h <output_dir> | tail -1
```

## 付属ツール (tools/)

- `tools/convert_lingbot_va_base.py` — `robbyant/lingbot-va-base` の
  `transformer/` シャードを strict 検証つきで注入し LeRobot 形式ベースを作る
  (§2)。**変換必須**の根拠と config 焼き込み4点はスクリプト内コメントにも記載。
  自環境では冒頭の定数 (OUT_DIR / DS_ROOT / REPO) と config 4点を
  自分のデータセットに合わせて書き換えてから venv の python で実行する。
