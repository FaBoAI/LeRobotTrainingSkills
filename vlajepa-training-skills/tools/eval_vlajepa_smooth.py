"""VLA-JEPA 推論時平滑化の実験: num_inference_timesteps x Kサンプル平均のマトリクス評価."""

import json
import time

import numpy as np
import torch

CKPT = "/home/jetson/RS/outputs/train/vlajepa_humanoid_test060_640/checkpoints/last/pretrained_model"
DS_ROOT = "/home/jetson/RS/humanoid_test060_640"
REPO_ID = "local/humanoid_test060_640"
TASK = "Put the object on the table"
DEVICE = "cuda"
CHUNK = 7

from lerobot.policies import get_policy_class
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.datasets.lerobot_dataset import LeRobotDataset

info = json.load(open(f"{DS_ROOT}/meta/info.json"))
motor_names = info["features"]["action"]["names"]
robot_type = info.get("robot_type") or ""

print("Loading policy ...")
policy = get_policy_class("vla_jepa").from_pretrained(CKPT).to(DEVICE)
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)
print("postprocessor steps:", [type(s).__name__ for s in postprocessor.steps])
assert not any("Gripper" in type(s).__name__ for s in postprocessor.steps), "gripper 修正が反映されていない"

head = policy.model.action_model
orig_predict = head.predict_action


def make_avg_predict(k):
    def predict_avg(conditioning_tokens, state=None):
        ct = conditioning_tokens.repeat_interleave(k, dim=0)
        st = state.repeat_interleave(k, dim=0) if state is not None else None
        out = orig_predict(ct, st)                    # [B*k, T, A]
        out = out.view(-1, k, *out.shape[1:]).mean(1)  # [B, T, A]
        return out
    return predict_avg


ds = LeRobotDataset(REPO_ID, root=DS_ROOT)
ep = ds.meta.episodes[0]
ep_from, ep_to = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
ep_len = ep_to - ep_from
timesteps = sorted(set([
    ep_from + 5, ep_from + ep_len // 4, ep_from + ep_len // 2,
    ep_from + (3 * ep_len) // 4, ep_from + ep_len - CHUNK * 3,
]))
# チャンク境界の連続性測定用: 連続する2チャンク (t, t+7)
boundary_ts = [ep_from + ep_len // 3, ep_from + ep_len // 2, ep_from + (2 * ep_len) // 3]


def frame_to_obs(item):
    img = item["observation.images.front"]
    img_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return {
        "observation.images.front": img_u8.copy(),
        "observation.state": item["observation.state"].numpy().astype(np.float32).copy(),
    }


def predict_chunk(t):
    """1チャンク推論 (rollout と同じ select_action 経路) → [7,16] raw + latency ms"""
    policy.reset()
    obs = frame_to_obs(ds[t])
    preds = []
    with torch.inference_mode():
        for k in range(CHUNK):
            observation = prepare_observation_for_inference(dict(obs), torch.device(DEVICE), TASK, robot_type)
            observation = preprocessor(observation)
            if k == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            action = policy.select_action(observation)
            if k == 0:
                torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) * 1000.0
            preds.append(postprocessor(action).squeeze(0).float().cpu().numpy().copy())
    return np.stack(preds), lat


VARIANTS = [(4, 1), (16, 1), (4, 8), (16, 8), (4, 32)]
gt_all = np.concatenate([np.stack([ds[t + k]["action"].numpy() for k in range(CHUNK)]) for t in timesteps])
gt_jerk = float(np.mean([np.abs(np.diff(np.stack([ds[t + k]["action"].numpy() for k in range(CHUNK)]), axis=0)).mean()
                         for t in timesteps]))
states = np.concatenate([np.tile(frame_to_obs(ds[t])["observation.state"], (CHUNK, 1)) for t in timesteps])

print(f"\nGT within-chunk jerk (mean |Δ|/step): {gt_jerk:.3f}")
print(f"{'steps':>5} {'K':>3} | {'MAE':>6} {'jerk':>6} {'boundary':>8} {'sign':>5} {'lat_ms':>7}")

results = {}
for n_steps, k_avg in VARIANTS:
    head.num_inference_timesteps = n_steps
    head.predict_action = make_avg_predict(k_avg) if k_avg > 1 else orig_predict

    torch.manual_seed(0)
    _ = predict_chunk(timesteps[0])  # warmup (グラフ/キャッシュ)

    torch.manual_seed(0)
    preds, lats = [], []
    for t in timesteps:
        p, lat = predict_chunk(t)
        preds.append(p)
        lats.append(lat)
    pred_all = np.concatenate(preds)

    mae = float(np.abs(pred_all - gt_all).mean())
    jerk = float(np.mean([np.abs(np.diff(p, axis=0)).mean() for p in preds]))

    # チャンク境界ジャンプ: 連続2チャンクの継ぎ目 |chunk2[0]-chunk1[-1]| と GT の同区間差
    jumps, gt_jumps = [], []
    for bt in boundary_ts:
        c1, _ = predict_chunk(bt)
        c2, _ = predict_chunk(bt + CHUNK)
        jumps.append(np.abs(c2[0] - c1[-1]).mean())
        g1 = ds[bt + CHUNK - 1]["action"].numpy()
        g2 = ds[bt + CHUNK]["action"].numpy()
        gt_jumps.append(np.abs(g2 - g1).mean())
    boundary = float(np.mean(jumps))

    dg, dp = gt_all - states, pred_all - states
    m = np.abs(dg) > 0.5
    sign = float((np.sign(dg[m]) == np.sign(dp[m])).mean()) if m.sum() else float("nan")

    lat = float(np.mean(lats))
    results[f"s{n_steps}_k{k_avg}"] = dict(mae=mae, jerk=jerk, boundary=boundary, sign=sign, lat_ms=lat)
    print(f"{n_steps:>5} {k_avg:>3} | {mae:6.2f} {jerk:6.2f} {boundary:8.2f} {sign:5.2f} {lat:7.1f}")

print(f"\n(参考) GT boundary jump: {float(np.mean(gt_jumps)):.3f}, GT jerk: {gt_jerk:.3f}")
print("hold-state baseline MAE:", float(np.abs(states - gt_all).mean()))

with open("/tmp/claude-1000/-home-jetson-camera/079a4ad8-aa41-4b68-9957-42a467f68dbe/scratchpad/smooth_summary.json", "w") as f:
    json.dump(results, f, indent=1)
print("summary written")
