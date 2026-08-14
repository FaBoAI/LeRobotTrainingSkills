# LeRobotTrainingSkills

LeRobot ポリシー学習のための **Agent Skills** です。

実機収録した LeRobotDataset v3 から各種ポリシー(ACT / SmolVLA / GR00T N1.7 / VLA-JEPA)を [LeRobot](https://github.com/huggingface/lerobot) 0.6.0 で学習し、`lerobot-rollout` による実機自律動作(推論)まで実施するためのワークフロー・実装知見・実測データをスキル形式でまとめています。Claude Code などのコーディングエージェントに読み込ませて使います。

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

`lerobot/smolvla_base`(873MB)からのファインチューン。smolvla_base は 3 カメラ(camera1/2/3)前提のため **`--rename_map` を学習・推論の両方に**付ける(付け忘れが最頻のミス)。`train_expert_only=True` で学習対象は約 100M のみ → 4 ポリシー中**学習が最速で、実機で RTC が滑らかに動作した唯一のポリシー = 推奨**。

### [groot-training-skills](./groot-training-skills/) — GR00T N1.7(3B VLA)

`--policy.type=groot` + `--policy.base_model_path=nvidia/GR00T-N1.7-3B` でファインチューン(`--policy.path=nvidia/...` は ParsingError の罠)。バックボーン Cosmos-Reason2-2B は**ゲート付き**(HF ライセンス同意 + `hf auth login` 必須)。`new_embodiment` がカメラ名・次元に自動適応するため rename 不要。RTC ネイティブ対応だが実機ではカクつき、**sync を採用**(4Hz スロー再生だが滑らか・把持成立)。SIGSTOP/SIGCONT による中間チェックポイントの段階評価手順込み。

### [vlajepa-training-skills](./vlajepa-training-skills/) — VLA-JEPA

`--policy.type=vla_jepa`(Qwen3-VL-2B + V-JEPA2)。**実機を暴走させる LIBERO 用グリッパーハック(postprocessor 焼き込み)の検出・修正手順**、state のラベルリーク、flow-matching ノイズの K サンプル平均パッチ(`VLAJEPA_SAMPLES`)、実機前のオフライン予測評価(致命バグはこれで発見)を含む。デフォルトレシピではタスク不成立だった記録と、再挑戦レシピ(`chunk_size=30`)まで。

## 4ポリシー実機比較

同一データセット(`local/humanoid_test060_640`: 両腕16軸ヒューマノイド、40エピソード、640×360@30fps)・同一 Jetson AGX Thor での実測:

| ポリシー | 学習実測 | 推論方式 | 実機評価 |
|---|---|---|---|
| ACT | 15K steps・loss 0.114 | sync + `n_action_steps=30` | ◎ タスク成功 |
| SmolVLA | 20K ≈ 80分・batch8 4.2 step/s・loss 0.034 | RTC(`queue_threshold=30`) | ◎ 滑らか・**推奨** |
| GR00T N1.7 | 15K ≈ 3.7h・batch4 1.1 step/s・loss 0.027 | sync(RTC はカクつき) | ○ 4Hz スロー再生だが確実 |
| VLA-JEPA | 30K ≈ 15.8h・batch4 0.53 step/s・loss 0.302 | sync + K=32 平均(RTC 非対応) | × 不成立(chunk=7 が主因の見立て) |

教訓: **RTC の合否はポリシー × chunk × queue_threshold 依存で、実機評価が必須**(ログの loss だけでは判断できない)。

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
- 4ポリシーすべて学習完走 + 実機 rollout まで検証済み(2026-08)

## 知見の追記

- ポリシー固有の発見(別バッチサイズのスループット、RTC 評価、新しい落とし穴)は各スキルの `reference/reference.md` に追記してください。
- プラットフォーム共通の発見(アロケータ・メモリ・I/O・可視化)は thor-platform-skills 側へ。ポリシー別/共通の分離を維持します。
