# LeRobotTrainingSkills

LeRobot ポリシー学習のための **Agent Skills** です。

実機収録した LeRobotDataset v3 から各種ポリシー(ACT / SmolVLA / GR00T N1.7 / VLA-JEPA / FastWAM / LingBot-VA)を [LeRobot](https://github.com/huggingface/lerobot) 0.6.0 で学習し、`lerobot-rollout` による実機自律動作(推論)まで実施するためのワークフロー・実装知見・実測データをスキル形式でまとめています。Claude Code などのコーディングエージェントに読み込ませて使います。

テレオペレーター/カメラプラグイン開発の姉妹リポジトリ [LeRobotPluginSkills](https://github.com/FaBoAI/LeRobotPluginSkills) と対になります(データ収録 → **本リポジトリで学習・推論**)。

## スキル一覧

### [thor-platform-skills](./thor-platform-skills/) — Jetson AGX Thor プラットフォーム共通

ポリシーに依存しない**学習運用の共通手順**(全ポリシースキルの前提)。

| テーマ | 要点 |
|---|---|
| Thor の絶対の掟 | `PYTORCH_CUDA_ALLOC_CONF` 禁止(expandable_segments はドライバ側リーク・max_split_size_mb は激遅化、実測)/ CUDA プロセスの `kill -9` 禁止(NvMap リークは再起動でしか回収不能)/ 夜間学習はユーザーのターミナルから nohup / `HF_HUB_DISABLE_XET=1`(hf-xet の無限ハング回避) |
| 学習前 preflight | データセット存在・二重起動(ブラケット法 pgrep)・空きメモリの3チェック |
| 解像度設計 | 1080p → 640×360 変換で学習スループット実測 **8.4倍**(`convert_dataset_resolution.py`)。学習と推論のカメラ設定は必ず一致させる |
| rerun 可視化 | ヘッドレス Jetson → PC ブラウザ(`?url=` 必須・CORS パッチ・rerun-sdk 0.32.2 固定) |

### [act-training-skills](./act-training-skills/) — ACT

FaBo レシピ(AMP + `use_vae=false` + `chunk_size=50`)による**最軽量・最短経路**の学習。RTC 非対応(実測 TypeError)のため、sync + `--policy.n_action_steps=30` の CLI 上書き(再学習不要)で再推論間隔を 1.7s → 1.0s に短縮する。低解像度化パイプラインと resume(総ステップ延長)の手順込み。

### [smolvla-training-skills](./smolvla-training-skills/) — SmolVLA

`lerobot/smolvla_base`(873MB)からのファインチューン。smolvla_base は 3 カメラ(camera1/2/3)前提のため **`--rename_map` を学習・推論の両方に**付ける(付け忘れが最頻のミス)。`train_expert_only=True` で学習対象は約 100M のみ → 6 ポリシー中**学習が最速で、実機で RTC が滑らかに動作した唯一のポリシー = 推奨**。

### [groot-training-skills](./groot-training-skills/) — GR00T N1.7(3B VLA)

`--policy.type=groot` + `--policy.base_model_path=nvidia/GR00T-N1.7-3B` でファインチューン(`--policy.path=nvidia/...` は ParsingError の罠)。バックボーン Cosmos-Reason2-2B は**ゲート付き**(HF ライセンス同意 + `hf auth login` 必須)。`new_embodiment` がカメラ名・次元に自動適応するため rename 不要。RTC ネイティブ対応だが実機ではカクつき、**sync を採用**(4Hz スロー再生だが滑らか・把持成立)。SIGSTOP/SIGCONT による中間チェックポイントの段階評価手順込み。

### [vlajepa-training-skills](./vlajepa-training-skills/) — VLA-JEPA

`--policy.type=vla_jepa`(Qwen3-VL-2B + V-JEPA2)。**デフォルトレシピ(`chunk_size=7`)の罠と、それを修正して実機タスクを成立させた v2 レシピ(`chunk_size=30`、2026-08-16 実機確認)**。同一データ・同一バックボーンで不成立→成立に転じた実証記録(chunk 長が支配的要因)。**実機を暴走させる LIBERO 用グリッパーハック(postprocessor 焼き込み)の検出・修正手順**、state のラベルリーク、flow-matching ノイズの K サンプル平均パッチ(`VLAJEPA_SAMPLES`)、実機前のオフライン予測評価(致命バグはこれで発見)を含む。

### [fastwam-training-skills](./fastwam-training-skills/) — FastWAM(Wan2.2 5B + action expert 1B)

`--policy.type=fastwam` だけで公開 `lerobot/fastwam_base`(12GB)から MoT 共拡散モデル(計 6B)をファインチューン。**`--policy.action_dim=16 --policy.proprio_dim=16` の明示指定が必須**(データセットから自動適応しない)。base 非互換 config だと**警告なしで 5B がランダム初期化**になる罠、RTC は kwargs を無言破棄する「静かな誤動作」のため禁止(sync のみ)、**denoise ステップ 10→3 で品質劣化なしの推論高速化**(552→322ms)。オフライン指標・実機とも最良クラス。

### [lingbotva-training-skills](./lingbotva-training-skills/) — LingBot-VA(Wan2.2 系 5.09B 世界モデル)【不合格の記録】

`--policy.type=lingbot_va` は 5.09B transformer が**無警告ランダム初期化**になるため、同梱の変換スクリプトで事前学習重みを注入した `--policy.path` 経由の学習が**必須**(config 4点の焼き込み込み)。**Thor 3大障害**(triton 同梱 ptxas の sm_110a 非対応 / flex ブロックマスクの毎ステップ再構築 / UMT5-XXL の CPU テキストエンコード)を対策して 17.7s/step → 1.8s/step の**10倍高速化**で 30K steps を完走したが、**オフライン評価が関門不合格(方向一致 0.66 で頭打ち・チャンク生成 6.5〜7s)となり実機は見送り**。現行レシピでは不成立という失敗記録と、compile 系全ポリシーに再利用できる Thor 高速化知見を残すスキル。

## 6ポリシー実機比較 + 参考(DreamZero)

同一タスク("Put the object on the table")・同一 Jetson AGX Thor での実測。データセットは `local/humanoid_test060_640`(両腕16軸ヒューマノイド、640×360@30fps)で、ACT〜FastWAM は 40エピソード/17,681フレーム版、LingBot-VA のみ追加収録後の 58エピソード/25,780フレーム版を使用:

| ポリシー | 学習実測 | 推論方式 | 実機/オフライン評価 |
|---|---|---|---|
| ACT | 15K steps・loss 0.114 | sync + `n_action_steps=30` | ◎ タスク成功 |
| SmolVLA | 20K ≈ 80分・batch8 4.2 step/s・loss 0.034 | RTC(`queue_threshold=30`) | ◎ 滑らか・**推奨**(RTC 成功は唯一) |
| GR00T N1.7 | 15K ≈ 3.7h・batch4 1.1 step/s・loss 0.027 | sync(RTC はカクつき) | ○ 4Hz スロー再生だが確実 |
| VLA-JEPA | 25K ≈ 13.5h・batch4 1.94s/step・loss 0.141(v2: chunk30) | sync + K=8 平均(RTC 非対応) | ◎ v2 レシピで動作良好(デフォルトレシピ chunk7 は×) |
| FastWAM | 15K ≈ 14.2h・batch4 3.04s/step・loss 0.17 | sync + denoise 3(RTC 非対応) | ◎ 実機動作良好・方向一致 0.89 |
| LingBot-VA | 30K ≈ 15.3h・batch4 1.8s/step・loss 0.24 プラトー | sync のみ(RTC 非対応)だが実機投入せず | **× オフライン関門不合格**(方向一致 5K 0.60 → 15K 0.68 → 30K 0.66 で頭打ち)→ 実機見送り |

### 学習実測の詳細比較

| ポリシー | 学習対象パラメータ | batch | 実測速度 | steps・所要時間 | 最終 loss | データセット | 特記事項 |
|---|---|---|---|---|---|---|---|
| ACT | 約34M(スクラッチ、`use_vae=false` 時の実測 learnable_params=34,208,592) | 8 | 6.3 step/s | 15K ≈ 40分 | 0.114 | 40ep/17,681f | 最軽量・最短経路 |
| SmolVLA | 450M 中 expert 約100M | 8 | 4.2 step/s | 20K ≈ 80分 | 0.034 | 40ep/17,681f | `train_expert_only=True`・rename_map 必須 |
| GR00T N1.7 | 3B 中 1.6B | 4 | 1.08〜1.12 step/s | 15K ≈ 3.7h | 0.027(+5K resume で 0.023) | 40ep/17,681f | バックボーンはゲート付き(HF 同意必須) |
| VLA-JEPA v2 | Qwen3-VL-2B + V-JEPA2 | 4 | 1.94 s/step | 25K ≈ 13.5h | 0.141 | 40ep/17,681f | v2 レシピ(chunk30)で成立。v1(chunk7)は不成立 |
| FastWAM | 6.02B フル(Wan2.2 5B + action expert 1B) | 4 | 3.04 s/step | 15K ≈ 14.2h(起動 1.5h 込み) | 0.17 | 40ep/17,681f | action_dim/proprio_dim の明示指定必須 |
| LingBot-VA | 5.09B フル(Wan2.2 系 transformer) | 4 | 1.8 s/step(パッチ後。**パッチ前 17.7s**) | 30K = 7.5h + resume 7.7h | 0.24 プラトー | 58ep/25,780f | 変換必須・Thor 3大障害対策・オフライン不合格(MAE 13.5、チャンク生成 6.5〜7s/16act) |

### 参考: DreamZero(NVIDIA GEAR Lab)— LeRobot 未統合のため本リポジトリでは実行不可

Wan2.1-I2V-14B ベースの world-action モデル(FastWAM / LingBot-VA と同系統の 14B 版)。映像と 24 アクションチャンクを flow matching で共同生成する。**比較表には載せていない = 本リポジトリの学習・推論パイプラインでは動かない**:

- **LeRobot ポリシーとしては未統合**(2026-08-17 に GitHub main の `policies/` 一覧で不在を確認。「LeRobot 0.6.0 に統合済み」と書く記事は誤り)。`--policy.type` に相当する指定は存在しない。
- LeRobot 形式データを GEAR 形式に変換して学習した公開事例はある(DreamZero-SO101: SO-101 データ 715ep、LoRA rank4 108M、2×H100 で 127時間/72K steps、推論 600ms/チャンク on H100)。つまり「LeRobot でデータ収録 → GEAR 側で学習・推論」という**別ライン**であり、`lerobot-train` / `lerobot-rollout` は使えない。
- Thor 単機では学習非現実的(2×H100 127h 規模)で、推論 600ms/チャンク(H100)も実機制御には遅い。
- 出典: <https://vizuara-ai-lab.github.io/dreamzero-so101/paper.html>

教訓: **RTC の合否はポリシー × chunk × queue_threshold 依存で、実機評価が必須**(ログの loss だけでは判断できない)。また**実機に載せる前のオフライン予測評価(方向一致率)が関門として機能する** — LingBot-VA は loss だけ見れば FastWAM と大差なかった(0.24 vs 0.17)が、方向一致 0.66 vs 0.89 の差で実機見送りを判断できた。

## 各スキルの構成

```
<skill-name>/
├── SKILL.md              # エージェント向けワークフロー(前提確認→学習起動→監視→推論→トラブル対処)
└── reference/
    └── reference.md      # 実装知見(落とし穴・実測データ表・診断コマンド)
```

## 検証済み環境

- プラットフォーム: NVIDIA **Jetson AGX Thor**(JetPack 7 / CUDA 13、unified memory 122GB、Docker レス)
- LeRobot **0.6.0** venv(torch **2.11.0+cu130**。CUDA torch を守るためパッケージ追加は `--no-deps` 運用)
- データセット: **LeRobotDataset v3**(実機収録、`--dataset.video_backend=pyav`)
- ロボット: 両腕 7DOF+グリッパ = **16軸ヒューマノイド**(rs_follower、CAN)
- カメラ: Holoscan Sensor Bridge + VB1940(姉妹リポジトリの hsb-camera-skills プラグイン、640×360 縮小出力)
- 6ポリシーすべて学習完走(2026-08)。うち5ポリシー(ACT/SmolVLA/GR00T/VLA-JEPA/FastWAM)は実機 rollout まで検証済み。LingBot-VA はオフライン評価不合格のため実機は見送り(その記録もスキル化)

## 知見の追記

- ポリシー固有の発見(別バッチサイズのスループット、RTC 評価、新しい落とし穴)は各スキルの `reference/reference.md` に追記してください。
- プラットフォーム共通の発見(アロケータ・メモリ・I/O・可視化)は thor-platform-skills 側へ。ポリシー別/共通の分離を維持します。
