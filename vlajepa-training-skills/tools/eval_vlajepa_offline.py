"""Offline prediction eval for VLA-JEPA checkpoint, mimicking lerobot-rollout sync path."""

import json
import time
import copy as copy_mod

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
policy_class = get_policy_class("vla_jepa")
policy = policy_class.from_pretrained(CKPT)
policy = policy.to(DEVICE)
policy.eval()
print("policy loaded. use_amp =", policy.config.use_amp)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)
print("postprocessor steps:", [type(s).__name__ for s in postprocessor.steps])

# Build an alternative postprocessor without the gripper hack steps (diagnostic only)
post_nogrip = None
try:
    post_nogrip = copy_mod.copy(postprocessor)
    kept = [s for s in postprocessor.steps if "Gripper" not in type(s).__name__]
    post_nogrip.steps = kept
    print("no-gripper postprocessor steps:", [type(s).__name__ for s in kept])
except Exception as e:
    print("could not build no-gripper postprocessor:", e)

ds = LeRobotDataset(REPO_ID, root=DS_ROOT)
# episode 0 boundaries
ep_from, ep_to = None, None
try:
    ep = ds.meta.episodes[0]
    ep_from = int(ep["dataset_from_index"]) if "dataset_from_index" in ep else 0
    ep_to = int(ep["dataset_to_index"]) if "dataset_to_index" in ep else None
except Exception as e:
    print("meta.episodes access failed:", e)
if ep_from is None or ep_to is None:
    # fallback: scan
    ep_from = 0
    ep_to = 0
    i = 0
    while i < len(ds):
        if int(ds[i]["episode_index"]) != 0:
            break
        i += 1
    ep_to = i
ep_len = ep_to - ep_from
print(f"episode 0: frames [{ep_from}, {ep_to}) len={ep_len}")

timesteps = sorted(set([
    ep_from + 5,
    ep_from + ep_len // 4,
    ep_from + ep_len // 2,
    ep_from + (3 * ep_len) // 4,
    ep_from + ep_len - CHUNK - 8,  # near end but leaves room for t+7 leak test
]))
print("eval timesteps (global idx):", timesteps)


def frame_to_obs(item):
    img = item["observation.images.front"]  # float32 CHW [0,1]
    img_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    state = item["observation.state"].numpy().astype(np.float32)
    return {"observation.images.front": img_u8.copy(), "observation.state": state.copy()}


def run_chunk(obs_np, postproc):
    """Mimic sync.py get_action loop: preprocess once per tick, select_action, postprocess."""
    policy.reset()
    preds = []
    raw_norm = []
    t0 = None
    with torch.inference_mode():
        for k in range(CHUNK):
            observation = prepare_observation_for_inference(dict(obs_np), torch.device(DEVICE), TASK, robot_type)
            observation = preprocessor(observation)
            if k == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            action = policy.select_action(observation)  # [B,16] normalized
            if k == 0:
                torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) * 1000.0
            raw_norm.append(action.squeeze(0).float().cpu().numpy().copy())
            out = postproc(action)
            preds.append(out.squeeze(0).float().cpu().numpy().copy())
    return np.stack(preds), np.stack(raw_norm), lat


# warmup
print("warmup ...")
_ = run_chunk(frame_to_obs(ds[ep_from + 5]), postprocessor)

results = []
lat_list = []
for t in timesteps:
    item = ds[t]
    obs = frame_to_obs(item)
    gt = np.stack([ds[t + k]["action"].numpy() for k in range(CHUNK)])  # [7,16] raw

    pred, raw_norm, lat = run_chunk(obs, postprocessor)
    lat_list.append(lat)

    # diagnostic 1: no-gripper postprocessor
    pred_ng = None
    if post_nogrip is not None:
        pred_ng, _, _ = run_chunk(obs, post_nogrip)

    # diagnostic 2: training-like leak state = state(t+7)
    pred_leak = None
    if t + CHUNK < ep_to:
        obs_leak = dict(obs)
        obs_leak["observation.state"] = ds[t + CHUNK]["observation.state"].numpy().astype(np.float32).copy()
        pp = post_nogrip if post_nogrip is not None else postprocessor
        pred_leak, _, _ = run_chunk(obs_leak, pp)

    results.append(dict(t=t, gt=gt, pred=pred, raw_norm=raw_norm, pred_ng=pred_ng, pred_leak=pred_leak,
                        state=obs["observation.state"]))
    print(f"t={t} done, first-chunk latency {lat:.1f} ms")

# ---- metrics ----
all_gt = np.concatenate([r["gt"] for r in results])          # [5*7,16]
all_pred = np.concatenate([r["pred"] for r in results])
mae_overall = float(np.abs(all_pred - all_gt).mean())
per_joint = np.abs(all_pred - all_gt).mean(axis=0)           # [16]

mae_ng = per_joint_ng = None
if results[0]["pred_ng"] is not None:
    all_ng = np.concatenate([r["pred_ng"] for r in results])
    mae_ng = float(np.abs(all_ng - all_gt).mean())
    per_joint_ng = np.abs(all_ng - all_gt).mean(axis=0)

leak_pairs = [(r["gt"], r["pred_leak"]) for r in results if r["pred_leak"] is not None]
mae_leak = None
if leak_pairs:
    lg = np.concatenate([p[0] for p in leak_pairs])
    lp = np.concatenate([p[1] for p in leak_pairs])
    mae_leak = float(np.abs(lp - lg).mean())
    per_joint_leak = np.abs(lp - lg).mean(axis=0)

all_raw = np.concatenate([r["raw_norm"] for r in results])

# baseline: predict "hold current state" (constant state(t) repeated 7x)
hold = np.concatenate([np.tile(r["state"], (CHUNK, 1)) for r in results])
mae_hold = float(np.abs(hold - all_gt).mean())

print("\n=== RESULTS ===")
print("MAE overall (rollout path, raw units):", mae_overall)
if mae_ng is not None:
    print("MAE without gripper postproc steps:", mae_ng)
if mae_leak is not None:
    print("MAE with leaked state(t+7) (no-gripper post):", mae_leak)
print("MAE of trivial 'hold state(t)' baseline:", mae_hold)
print("mean chunk latency ms:", np.mean(lat_list))

print("\nPer-joint MAE (rollout path | no-gripper | leak-state):")
for j, name in enumerate(motor_names):
    ng = per_joint_ng[j] if per_joint_ng is not None else float("nan")
    lk = per_joint_leak[j] if mae_leak is not None else float("nan")
    print(f"  {j:2d} {name:28s} {per_joint[j]:8.3f} | {ng:8.3f} | {lk:8.3f}")

print("\nGT action stats (eval window): min %.3f max %.3f std %.3f" % (all_gt.min(), all_gt.max(), all_gt.std()))
print("Pred (rollout path) stats: min %.3f max %.3f std %.3f" % (all_pred.min(), all_pred.max(), all_pred.std()))
print("Pred normalized (pre-postproc) stats: min %.3f max %.3f std %.3f" % (all_raw.min(), all_raw.max(), all_raw.std()))
print("Pred dim6 unique values (rollout path):", np.unique(all_pred[:, 6].round(3)))

# constancy check: std of predictions across the 5 timesteps (per joint, averaged over chunk steps)
pred_by_t = np.stack([r["pred"] for r in results])  # [5,7,16]
across_t_std = pred_by_t.std(axis=0).mean()
gt_by_t = np.stack([r["gt"] for r in results])
gt_across_t_std = gt_by_t.std(axis=0).mean()
print("Pred std across timesteps (mean): %.4f  (GT: %.4f)" % (across_t_std, gt_across_t_std))

# sign agreement on deltas from state(t)
def sign_agree(pred_all):
    s = np.concatenate([np.tile(r["state"], (CHUNK, 1)) for r in results])
    dg = all_gt - s
    dp = pred_all - s
    m = np.abs(dg) > 0.5
    if m.sum() == 0:
        return float("nan")
    return float((np.sign(dg[m]) == np.sign(dp[m])).mean())

print("Sign agreement of (pred-state) vs (gt-state), rollout path: %.3f" % sign_agree(all_pred))
if mae_ng is not None:
    print("Sign agreement, no-gripper: %.3f" % sign_agree(all_ng))

print("\n=== SAMPLES (2 timesteps, first 3 joints x 7 steps) ===")
for r in [results[0], results[len(results) // 2]]:
    print(f"-- t={r['t']} --")
    for j in range(3):
        gt_row = " ".join(f"{v:7.2f}" for v in r["gt"][:, j])
        pr_row = " ".join(f"{v:7.2f}" for v in r["pred"][:, j])
        print(f"  {motor_names[j]}:")
        print(f"    GT  : {gt_row}")
        print(f"    PRED: {pr_row}")
print("\nDim6 (%s) sample t=%d  GT: %s  PRED: %s" % (
    motor_names[6], results[0]["t"],
    " ".join(f"{v:.2f}" for v in results[0]["gt"][:, 6]),
    " ".join(f"{v:.2f}" for v in results[0]["pred"][:, 6])))

# dump machine-readable summary
summary = dict(
    mae_overall=mae_overall, mae_no_gripper=mae_ng, mae_leak_state=mae_leak, mae_hold_baseline=mae_hold,
    per_joint={motor_names[j]: float(per_joint[j]) for j in range(16)},
    per_joint_no_gripper={motor_names[j]: float(per_joint_ng[j]) for j in range(16)} if per_joint_ng is not None else None,
    per_joint_leak={motor_names[j]: float(per_joint_leak[j]) for j in range(16)} if mae_leak is not None else None,
    latency_ms=[float(x) for x in lat_list],
    gt_min=float(all_gt.min()), gt_max=float(all_gt.max()), gt_std=float(all_gt.std()),
    pred_min=float(all_pred.min()), pred_max=float(all_pred.max()), pred_std=float(all_pred.std()),
    raw_norm_min=float(all_raw.min()), raw_norm_max=float(all_raw.max()),
)
with open("/tmp/claude-1000/-home-jetson-camera/079a4ad8-aa41-4b68-9957-42a467f68dbe/scratchpad/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("\nsummary written")
