"""Small multi-seed smoke test for null_space_probe conditions (seeds 0-4)."""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import redundant_arm_calibration as exp

TARGET_CONDITIONS = [
    "null_space",
    "null_space_probe",
    "null_space_probe_recovery",
    "null_space_probe_posture",
]
N_SEEDS_SMOKE = 5

results = {c: [] for c in TARGET_CONDITIONS}

for cond in TARGET_CONDITIONS:
    print(f"Running '{cond}'", end="", flush=True)
    for seed in range(N_SEEDS_SMOKE):
        r = exp.run_one(seed, cond)
        results[cond].append(r)
        print(".", end="", flush=True)
    print()

CHANGE_STEP = exp.CHANGE_STEP
N_STEPS = exp.N_STEPS

print(f"\n{'Condition':35s}  {'RMSE@150':>9}  {'TaskErr@200':>11}  {'EEhold':>8}  {'probe_w':>8}  {'probe_g':>8}  {'Prank':>6}  {'failTask':>8}")
summary = {}
for cond in TARGET_CONDITIONS:
    rmse_at_ch = np.array([r["rmse_at_change"] for r in results[cond]])
    task_final = np.array([r["task_err_final"] for r in results[cond]])
    ee_hold = np.array([r["ee_hold_mean"] for r in results[cond]])
    prank = np.array([r["p_theta_rank_at_change"] for r in results[cond]])
    probe_w = np.array([r["probe_weight_mean_phase1"] for r in results[cond]])
    probe_g = np.array([r["probe_gain_max_phase1"] for r in results[cond]])
    fail_task = float(np.mean(task_final > exp.TASK_ERR_FAIL))

    summary[cond] = {
        "n_seeds": N_SEEDS_SMOKE,
        "rmse_at_change": {"median": float(np.median(rmse_at_ch)), "values": rmse_at_ch.tolist()},
        "task_err_final": {"median": float(np.median(task_final)), "values": task_final.tolist()},
        "ee_hold_mean": {"median": float(np.median(ee_hold)), "values": ee_hold.tolist()},
        "probe_weight_mean_phase1": {"median": float(np.median(probe_w)), "values": probe_w.tolist()},
        "probe_gain_max_phase1": {"median": float(np.median(probe_g)), "values": probe_g.tolist()},
        "p_theta_rank_at_change": {"median": float(np.median(prank)), "values": prank.tolist()},
        "task_failure_rate": fail_task,
    }

    print(
        f"  {cond:35s}  {np.median(rmse_at_ch):>9.4f}  {np.median(task_final):>11.4f}  "
        f"{np.median(ee_hold):>8.4f}  {np.median(probe_w):>8.4f}  {np.median(probe_g):>8.4f}  "
        f"{np.median(prank):>6.1f}  {fail_task:>8.2f}"
    )

out = project_root / "results" / "multiseed_smoke.json"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {out}")
