"""Phase 7: late-only rollout-risk probing check.

This is a focused follow-up to Phase 5.  The full Phase-1 rollout-risk penalty
was negative, plausibly because it competes with early identifiability.  Here
the penalty is enabled only near the end of Phase 1, after the baseline probe
has already gathered most calibration information.

Question:
    Is late-only rollout-risk probing positive enough to add to the RA-L paper?

Output:
    results/phase7_late_rollout_risk_probe.json
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (
    CHANGE_STEP,
    N_SEEDS,
    PARAM_RMSE_FAIL,
    PROBE_ROLLOUT_K,
    PROBE_ROLLOUT_LATE_START,
    PROBE_ROLLOUT_PENALTY,
    TASK_ERR_FAIL,
    _summarize,
    run_one,
)


PROBES = ["null_space_probe", "null_space_probe_rollout_risk_late"]
CONTROLLERS = ["plain", "posture"]
HORIZONS = [50, 80, 100, 120]


def run_cell(probe, controller, horizon):
    runs = []
    for seed in range(N_SEEDS):
        r = run_one(
            seed,
            probe,
            phase2_horizon_override=horizon,
            phase2_controller_override=controller,
        )
        runs.append(
            {
                "seed": seed,
                "task_failed": bool(r["task_err_final"] > TASK_ERR_FAIL),
                "rmse_failed": bool(r["rmse_at_change"] > PARAM_RMSE_FAIL),
                "rmse_at_change": float(r["rmse_at_change"]),
                "task_err_final": float(r["task_err_final"]),
                "ee_hold_mean": float(r["ee_hold_mean"]),
                "q_norm_change": float(np.linalg.norm(r["q_change"])),
            }
        )
        print(".", end="", flush=True)
    return runs


def summarize_runs(runs):
    task_failed = np.array([r["task_failed"] for r in runs], dtype=bool)
    rmse_failed = np.array([r["rmse_failed"] for r in runs], dtype=bool)
    rmse = np.array([r["rmse_at_change"] for r in runs], dtype=float)
    task_err = np.array([r["task_err_final"] for r in runs], dtype=float)
    ee_hold = np.array([r["ee_hold_mean"] for r in runs], dtype=float)
    q_norm = np.array([r["q_norm_change"] for r in runs], dtype=float)
    return {
        "task_failure_rate": float(np.mean(task_failed)),
        "rmse_failure_rate": float(np.mean(rmse_failed)),
        "n_failed": int(np.sum(task_failed)),
        "failed_seeds": [r["seed"] for r in runs if r["task_failed"]],
        "rmse_at_change": _summarize(rmse),
        "task_err_final": _summarize(task_err),
        "ee_hold_err_phase1": _summarize(ee_hold),
        "q_norm_change": _summarize(q_norm),
        "per_seed": runs,
    }


def main():
    out = {
        "settings": {
            "probes": PROBES,
            "controllers": CONTROLLERS,
            "horizons": HORIZONS,
            "n_seeds": N_SEEDS,
            "change_step": CHANGE_STEP,
            "late_start": PROBE_ROLLOUT_LATE_START,
            "probe_rollout_k": int(PROBE_ROLLOUT_K),
            "probe_rollout_penalty": float(PROBE_ROLLOUT_PENALTY),
            "task_err_fail": TASK_ERR_FAIL,
            "param_rmse_fail": PARAM_RMSE_FAIL,
        },
        "results": {probe: {ctrl: {} for ctrl in CONTROLLERS} for probe in PROBES},
    }

    for probe in PROBES:
        for ctrl in CONTROLLERS:
            for horizon in HORIZONS:
                print(f"  {probe:38s} / {ctrl:7s} / H={horizon:3d} ", end="", flush=True)
                runs = run_cell(probe, ctrl, horizon)
                summary = summarize_runs(runs)
                out["results"][probe][ctrl][str(horizon)] = summary
                print(
                    f"  failTask={summary['task_failure_rate']:.2f}  "
                    f"RMSE={summary['rmse_at_change']['median']:.4f}  "
                    f"EEhold={summary['ee_hold_err_phase1']['median']:.4f}  "
                    f"qNorm={summary['q_norm_change']['median']:.3f}"
                )

    print("\n--- late-risk minus baseline failTask ---")
    for ctrl in CONTROLLERS:
        print(f"  controller={ctrl}")
        for horizon in HORIZONS:
            base = out["results"]["null_space_probe"][ctrl][str(horizon)]["task_failure_rate"]
            late = out["results"]["null_space_probe_rollout_risk_late"][ctrl][str(horizon)][
                "task_failure_rate"
            ]
            print(f"    H={horizon:3d}: baseline={base:.2f}  late={late:.2f}  delta={late - base:+.2f}")

    out_path = project_root / "results" / "phase7_late_rollout_risk_probe.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
