"""FastWAM チェックポイントのオフライン予測評価 (rollout と同一経路)。
均一5点 + 高モーション5窓で、予測 vs 正解の MAE・方向一致・定数出力チェック・レイテンシを測る。"""

import json
import time

import numpy as np
import torch

CKPT = "/home/jetson/RS/outputs/train/fastwam_humanoid_test060_640/checkpoints/last/pretrained_model"
DS_ROOT = "/home/jetson/RS/humanoid_test060_640"
TASK = "Put the object on the table"
DEVICE = "cuda"
CHUNK = 32

from lerobot.policies import get_policy_class
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.datasets.lerobot_dataset import LeRobotDataset

info = json.load(open(f"{DS_ROOT}/meta/info.json"))
motor_names = info["features"]["action"]["names"]

print("Loading policy ...")
policy = get_policy_class("fastwam").from_pretrained(CKPT).to(DEVICE)
policy.eval()
print("action_dim =", policy.config.action_dim, "proprio_dim =", policy.config.proprio_dim,
      "n_action_steps =", policy.config.n_action_steps)
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                     preprocessor_overrides={"device_processor": {"device": DEVICE}})
steps_names = [type(s).__name__ for s in post.steps]
print("postprocessor:", steps_names)
assert not any("Toggle" in n for n in steps_names), "LIBERO toggle ステップが焼き込まれている!"

ds = LeRobotDataset("local/humanoid_test060_640", root=DS_ROOT)
ep = ds.meta.episodes[0]
f0, f1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
L = f1 - f0
acts = np.stack([ds[i]["action"].numpy() for i in range(f0, f1)])

# 均一5点 + 高モーション5窓 (32フレーム窓の総移動量トップ、重複回避)
uniform = [f0 + 5, f0 + L // 4, f0 + L // 2, f0 + 3 * L // 4, f0 + L - CHUNK - 2]
win = np.array([np.abs(np.diff(acts[i:i + CHUNK], axis=0)).sum() for i in range(L - CHUNK)])
hi, used = [], set()
for i in np.argsort(win)[::-1]:
    if all(abs(i - j) > CHUNK for j in used):
        hi.append(f0 + int(i)); used.add(int(i))
    if len(hi) == 5:
        break
print("均一:", uniform, "\n高モーション:", hi, [f"{win[i-f0]:.0f}" for i in hi])


def predict_chunk(t):
    policy.reset()
    it = ds[t]
    obs = {"observation.images.front": (it["observation.images.front"] * 255).round().clamp(0, 255)
           .to(torch.uint8).permute(1, 2, 0).numpy().copy(),
           "observation.state": it["observation.state"].numpy().astype(np.float32).copy()}
    preds = []
    with torch.inference_mode():
        for k in range(CHUNK):
            o = prepare_observation_for_inference(dict(obs), torch.device(DEVICE), TASK, "")
            if k == 0:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            a = policy.select_action(pre(o))
            if k == 0:
                torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
            preds.append(post(a).squeeze(0).float().cpu().numpy().copy())
    return np.stack(preds), obs["observation.state"], lat


_ = predict_chunk(f0 + 5)  # warmup
torch.manual_seed(0)

def evaluate(ts, label):
    maes, lats, agrees, moves_gt, moves_pr = [], [], [], [], []
    samples = []
    for t in ts:
        p, st, lat = predict_chunk(t)
        gt = acts[t - f0:t - f0 + CHUNK]
        maes.append(np.abs(p - gt).mean()); lats.append(lat)
        dg, dp = gt[-1] - st, p[-1] - st
        m = np.abs(dg) > 1.0
        if m.sum():
            agrees.append(float((np.sign(dg[m]) == np.sign(dp[m])).mean()))
            moves_gt.append(np.abs(dg[m]).mean()); moves_pr.append(np.abs(dp[m]).mean())
        samples.append((t, gt, p))
    print(f"\n== {label} ==")
    print(f"MAE={np.mean(maes):.2f}  1チャンク推論={np.mean(lats):.0f}ms")
    if agrees:
        print(f"移動方向一致率={np.mean(agrees):.2f}  GT移動量={np.mean(moves_gt):.2f} vs 予測移動量={np.mean(moves_pr):.2f}")
    return samples


samples_u = evaluate(uniform, "均一5点")
samples_h = evaluate(hi, "高モーション5窓")

# 定数出力チェック (VLA-JEPA dim6=-1.0 型の検出)
allp = np.concatenate([p for _, _, p in samples_u + samples_h])
for j in range(16):
    u = np.unique(allp[:, j].round(2))
    if len(u) <= 2:
        print(f"!! dim{j} ({motor_names[j]}) がほぼ定数: {u}")
print("\n定数チェック完了 (警告なければ全関節が可変)")

# 高モーション窓の代表1つ: 最も動いた3関節の GT vs 予測 (8ステップ間引き)
t, gt, p = samples_h[0]
top = np.argsort(-np.abs(gt[-1] - gt[0]))[:3]
print(f"\n代表窓 t={t} (8ステップ間引き表示):")
for j in top:
    print(f"  {motor_names[j]}:")
    print("    GT  :", " ".join(f"{v:7.2f}" for v in gt[::8, j]), f"→ {gt[-1, j]:7.2f}")
    print("    PRED:", " ".join(f"{v:7.2f}" for v in p[::8, j]), f"→ {p[-1, j]:7.2f}")
