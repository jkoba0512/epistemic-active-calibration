"""Phase 2 horizon sweep.

Measures task failure rate as a function of Phase 2 horizon for plain and
posture controllers.  Phase 1 is identical to null_space_probe across all runs.

Horizons : [50, 80, 100, 120, 150, 180, 200]
Controllers: plain, posture
Seeds     : N = 50

Outputs:
    results/phase2_horizon_sweep.json
    results/phase2_horizon_sweep.png
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (
    run_one,
    N_SEEDS,
    CHANGE_STEP,
    TASK_ERR_FAIL,
    PARAM_RMSE_FAIL,
    _summarize,
)

HORIZONS = [50, 80, 100, 120, 150, 180, 200]
CONTROLLERS = ["plain", "posture"]
BASE_CONDITION = "null_space_probe"


def main():
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "settings": {
            "base_condition": BASE_CONDITION,
            "horizons": HORIZONS,
            "controllers": CONTROLLERS,
            "n_seeds": N_SEEDS,
            "change_step": CHANGE_STEP,
            "task_err_fail": TASK_ERR_FAIL,
        },
        "results": {ctrl: {} for ctrl in CONTROLLERS},
    }

    for ctrl in CONTROLLERS:
        for horizon in HORIZONS:
            key = str(horizon)
            print(f"  {ctrl:7s} / {horizon:3d} steps ", end="", flush=True)

            per_seed = []
            for seed in range(N_SEEDS):
                r = run_one(
                    seed,
                    BASE_CONDITION,
                    phase2_horizon_override=horizon,
                    phase2_controller_override=ctrl,
                )
                task_failed = bool(r["task_err_final"] > TASK_ERR_FAIL)

                # first step in Phase 2 where task_err drops below threshold
                task_arr = np.array(r["task_err"])
                phase2_task = task_arr[CHANGE_STEP:]
                below = np.where(phase2_task < TASK_ERR_FAIL)[0]
                first_success_step = int(below[0] + CHANGE_STEP) if len(below) > 0 else None

                per_seed.append({
                    "seed": seed,
                    "task_failed": task_failed,
                    "rmse_at_change": float(r["rmse_at_change"]),
                    "task_err_final": float(r["task_err_final"]),
                    "ee_hold_mean": float(r["ee_hold_mean"]),
                    "first_success_step": first_success_step,
                })
                print(".", end="", flush=True)

            task_failed_arr = np.array([s["task_failed"] for s in per_seed])
            task_err_arr = np.array([s["task_err_final"] for s in per_seed])
            rmse_arr = np.array([s["rmse_at_change"] for s in per_seed])
            eehold_arr = np.array([s["ee_hold_mean"] for s in per_seed])
            success_steps = [s["first_success_step"] for s in per_seed
                             if s["first_success_step"] is not None]

            fail_task = float(np.mean(task_failed_arr))
            failed_seeds = [s["seed"] for s in per_seed if s["task_failed"]]

            print(
                f"  failTask={fail_task:.2f}  "
                f"TaskErr_med={np.median(task_err_arr):.4f}  "
                f"RMSE_med={np.median(rmse_arr):.4f}"
            )

            output["results"][ctrl][key] = {
                "horizon": horizon,
                "controller": ctrl,
                "task_failure_rate": fail_task,
                "n_failed": int(np.sum(task_failed_arr)),
                "n_success": int(np.sum(~task_failed_arr)),
                "failed_seeds": failed_seeds,
                "task_err_final": _summarize(task_err_arr),
                "rmse_at_change": _summarize(rmse_arr),
                "ee_hold_err_phase1": _summarize(eehold_arr),
                "time_to_success": _summarize(np.array(success_steps)) if success_steps else None,
                "per_seed": per_seed,
            }

    out_path = results_dir / "phase2_horizon_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n--- failTask vs horizon ---")
    print(f"{'horizon':>8s}  {'plain':>8s}  {'posture':>8s}  {'diff':>8s}")
    for h in HORIZONS:
        p = output["results"]["plain"][str(h)]["task_failure_rate"]
        q = output["results"]["posture"][str(h)]["task_failure_rate"]
        print(f"{h:8d}  {p:8.2f}  {q:8.2f}  {q - p:+8.2f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(
            "Phase 2 horizon sweep  (Phase 1 = null_space_probe, N=50)\n"
            "Does posture controller shift the convergence curve or only remove the slow tail?",
            fontsize=10,
        )

        horizons_arr = np.array(HORIZONS)
        colors = {"plain": "C0", "posture": "C3"}

        # Panel A: failTask vs horizon
        ax = axes[0]
        for ctrl in CONTROLLERS:
            rates = [output["results"][ctrl][str(h)]["task_failure_rate"] for h in HORIZONS]
            ax.plot(horizons_arr, rates, "o-", color=colors[ctrl], label=ctrl, linewidth=2)
        ax.set_xlabel("Phase 2 horizon (steps)")
        ax.set_ylabel("Task failure rate")
        ax.set_title("(A) Failure rate vs horizon")
        ax.set_ylim(-0.02, 0.6)
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel B: median task error vs horizon
        ax = axes[1]
        for ctrl in CONTROLLERS:
            medians = [output["results"][ctrl][str(h)]["task_err_final"]["median"]
                       for h in HORIZONS]
            ax.plot(horizons_arr, medians, "o-", color=colors[ctrl], label=ctrl, linewidth=2)
        ax.axhline(TASK_ERR_FAIL, color="red", linestyle="--", linewidth=0.8,
                   label=f"fail threshold ({TASK_ERR_FAIL})")
        ax.set_xlabel("Phase 2 horizon (steps)")
        ax.set_ylabel("Median task error (final)")
        ax.set_title("(B) Median task error vs horizon")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = results_dir / "phase2_horizon_sweep.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
