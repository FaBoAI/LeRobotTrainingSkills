"""robbyant/lingbot-va-base (diffusers 形式) を LeRobot 形式チェックポイントに変換する。

lerobot 0.6.0 の lingbot_va は --policy.type だと 5B transformer が無警告ランダム初期化になる
(事前学習ロードは凍結部 VAE/UMT5/tokenizer のみ)。本スクリプトで transformer/ シャードを
ポリシーに注入して save_pretrained し、学習は --policy.path=<出力先> で行う。

実行: /home/jetson/camera/lerobot060-venv/bin/python /home/jetson/RS/convert_lingbot_va_base.py
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

OUT_DIR = Path("/home/jetson/RS/outputs/lingbot_va_base_lerobot")
DS_ROOT = "/home/jetson/RS/humanoid_test060_640"
REPO = "robbyant/lingbot-va-base"

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.configs.types import NormalizationMode

print("=== 1/4: config 構築 (16軸・front カメラ・flex attention・ACTION=QUANTILES) ===")
cfg = LingBotVAConfig(
    obs_cam_keys=["observation.images.front"],
    used_action_channel_ids=list(range(16)),
    attn_mode="flex",
    device="cpu",  # 変換は CPU で十分 (学習時に cuda 指定)
)
# 事前学習モデルは [-1,1] quantile アクション空間 → QUANTILES 正規化に合わせる
cfg.normalization_mapping["ACTION"] = NormalizationMode.QUANTILES

meta = LeRobotDatasetMetadata("local/humanoid_test060_640", root=DS_ROOT)
policy = make_policy(cfg, ds_meta=meta)  # pretrained_path なし → transformer はランダム初期化
print("policy 構築完了 (transformer params:",
      sum(p.numel() for p in policy.transformer.parameters()) / 1e9, "B)")

print("=== 2/4: 事前学習 transformer シャードのロード ===")
local = Path(snapshot_download(REPO, allow_patterns=["transformer/*"]))
idx = json.load(open(local / "transformer/diffusion_pytorch_model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
sd = {}
for s in shards:
    sd.update(load_file(local / "transformer" / s))
print(f"シャード {len(shards)} 個 / テンソル {len(sd)} 個ロード")

print("=== 3/4: 重み注入 (strict) ===")
# 既知の無害な余りキー: 旧 Conv3d 版 patch_embedding (リリースに残る学習初期化の名残)。
# lerobot 実装の forward は patch_embedding_mlp のみを使用 (utils.py:911,1063 で確認)。
KNOWN_UNUSED = {"patch_embedding.weight", "patch_embedding.bias"}
for k in KNOWN_UNUSED & set(sd):
    sd.pop(k)
    print(f"  既知の未使用キーを破棄: {k}")
model_sd = policy.transformer.state_dict()
missing = [k for k in model_sd if k not in sd]
unexpected = [k for k in sd if k not in model_sd]
if missing or unexpected:
    print("!! キー不一致 — 変換中断")
    print("missing (モデル側にあるが重みなし):", missing[:10], f"... 計{len(missing)}")
    print("unexpected (重み側にあるがモデルになし):", unexpected[:10], f"... 計{len(unexpected)}")
    raise SystemExit(1)
shape_ng = [k for k in model_sd if model_sd[k].shape != sd[k].shape]
if shape_ng:
    print("!! 形状不一致:", [(k, tuple(model_sd[k].shape), tuple(sd[k].shape)) for k in shape_ng[:5]])
    raise SystemExit(1)
policy.transformer.load_state_dict(sd, strict=True)
policy.transformer = policy.transformer.to(torch.bfloat16)
print("注入完了 (全キー一致・全形状一致)")

print("=== 4/4: LeRobot 形式で保存 ===")
OUT_DIR.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(OUT_DIR)
# processor も保存 (学習時は dataset stats + config.normalization_mapping で上書きされるが、
# --policy.path 経路がロードできる形を揃えておく)
try:
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    pre.save_pretrained(OUT_DIR)
    post.save_pretrained(OUT_DIR)
    print("processor 保存 OK")
except Exception as e:
    print(f"processor 保存スキップ ({type(e).__name__}: {e}) — 学習時に既定生成される")

print("\n保存先:", OUT_DIR)
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name, f"{f.stat().st_size/1e9:.2f}GB" if f.stat().st_size > 1e8 else "")
print("\n次: sh /home/jetson/RS/run_train06_lingbotva.sh (STEPS=20 で煙試験から)")
