"""Phase 2 ablation: separate the effect of posture controller vs longer horizon.

Four conditions, all sharing identical Phase 1 (null_space_probe):

    null_space_probe           plain ctrl,   50 steps  (baseline)
    null_space_probe_plain_200 plain ctrl,  200 steps  (horizon ablation)
    null_space_probe_posture_50 posture ctrl, 50 steps  (controller ablation)
    null_space_probe_posture   posture ctrl, 200 steps  (full condition)

Outputs:
    results/phase2_ablation.json
    results/phase2_ablation.png
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
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (
    run_one,
    N_SEEDS,
    CHANGE_STEP,
    PARAM_RMSE_FAIL,
    TASK_ERR_FAIL,
    _summarize,
)

ABLATION_CONDITIONS = [
    "null_space_probe",            # plain ctrl,   50 steps
    "null_space_probe_plain_200",  # plain ctrl,  200 steps
    "null_space_probe_posture_50", # posture ctrl, 50 steps
    "null_space_probe_posture",    # posture ctrl, 200 steps
]

LABELS = {
    "null_space_probe":            "plain / 50 steps   (baseline)",
    "null_space_probe_plain_200":  "plain / 200 steps  (+horizon only)",
    "null_space_probe_posture_50": "posture / 50 steps  (+controller only)",
    "null_space_probe_posture":    "posture / 200 steps (+both)",
}


def main():
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    raw = {}
    for cond in ABLATION_CONDITIONS:
        print(f"Running '{cond}' ", end="", flush=True)
        runs = []
        for seed in range(N_SEEDS):
            r = run_one(seed, cond)
            runs.append(r)
            print(".", end="", flush=True)
        raw[cond] = runs
        print()

    # Summarize
    summary = {}
    print(
        f"\n{'Condition':42s}  {'RMSE@150':8s}  {'TaskErr':8s}  "
        f"{'EEhold':8s}  {'failRMSE':8s}  {'failTask':8s}"
    )
    for cond in ABLATION_CONDITIONS:
        runs = raw[cond]
        rmse_arr    = np.array([r["rmse_at_change"] for r in runs])
        task_arr    = np.array([r["task_err_final"] for r in runs])
        eehold_arr  = np.array([r["ee_hold_mean"] for r in runs])
        fail_rmse   = float(np.mean(rmse_arr > PARAM_RMSE_FAIL))
        fail_task   = float(np.mean(task_arr > TASK_ERR_FAIL))

        summary[cond] = {
            "label": LABELS[cond],
            "n_seeds": N_SEEDS,
            "rmse_at_change": _summarize(rmse_arr),
            "task_err_final": _summarize(task_arr),
            "ee_hold_err_phase1": _summarize(eehold_arr),
            "rmse_failure_rate": fail_rmse,
            "task_failure_rate": fail_task,
            "per_seed": [
                {
                    "seed": i,
                    "rmse_at_change": float(r["rmse_at_change"]),
                    "task_err_final": float(r["task_err_final"]),
                    "ee_hold_mean": float(r["ee_hold_mean"]),
                    "task_failed": bool(r["task_err_final"] > TASK_ERR_FAIL),
                    "rmse_failed": bool(r["rmse_at_change"] > PARAM_RMSE_FAIL),
                }
                for i, r in enumerate(runs)
            ],
        }
        print(
            f"  {LABELS[cond]:42s}"
            f"  {np.median(rmse_arr):.4f}  "
            f"  {np.median(task_arr):.4f}  "
            f"  {np.median(eehold_arr):.4f}  "
            f"  {fail_rmse:.2f}      {fail_task:.2f}"
        )

    out_path = results_dir / "phase2_ablation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            "Phase 2 ablation: posture controller vs longer horizon\n"
            "(Phase 1 identical across all four conditions: null_space_probe)",
            fontsize=10,
        )

        COLORS = {
            "null_space_probe":            "C0",
            "null_space_probe_plain_200":  "C1",
            "null_space_probe_posture_50": "C2",
            "null_space_probe_posture":    "C3",
        }
        SHORT_LABELS = {
            "null_space_probe":            "plain/50",
            "null_space_probe_plain_200":  "plain/200",
            "null_space_probe_posture_50": "posture/50",
            "null_space_probe_posture":    "posture/200",
        }

        metrics = [
            ("rmse_at_change", "RMSE@150 (θ estimation)", False),
            ("task_err_final", "Task error (Phase 2 final)", False),
        ]

        # Bar chart: failure rates
        ax = axes[0]
        x = np.arange(len(ABLATION_CONDITIONS))
        fail_tasks = [summary[c]["task_failure_rate"] for c in ABLATION_CONDITIONS]
        bars = ax.bar(x, fail_tasks,
                      color=[COLORS[c] for c in ABLATION_CONDITIONS])
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[c] for c in ABLATION_CONDITIONS], fontsize=9)
        ax.set_ylabel("Task failure rate")
        ax.set_title("Task failure rate\n(lower is better)")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, fail_tasks):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=9)

        # Box plot: RMSE
        ax = axes[1]
        data = [np.array([r["rmse_at_change"] for r in raw[c]]) for c in ABLATION_CONDITIONS]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cond in zip(bp["boxes"], ABLATION_CONDITIONS):
            patch.set_facecolor(COLORS[cond])
        ax.set_xticklabels([SHORT_LABELS[c] for c in ABLATION_CONDITIONS], fontsize=9)
        ax.set_ylabel("RMSE@150")
        ax.set_title("Parameter RMSE at Phase 1 end\n(should be equal across conditions)")

        # Box plot: Task error
        ax = axes[2]
        data = [np.array([r["task_err_final"] for r in raw[c]]) for c in ABLATION_CONDITIONS]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cond in zip(bp["boxes"], ABLATION_CONDITIONS):
            patch.set_facecolor(COLORS[cond])
        ax.set_xticklabels([SHORT_LABELS[c] for c in ABLATION_CONDITIONS], fontsize=9)
        ax.set_ylabel("Task error (final)")
        ax.set_title("Phase 2 task error\n(lower is better)")
        ax.axhline(TASK_ERR_FAIL, color="red", linestyle="--", linewidth=0.8,
                   label=f"fail threshold ({TASK_ERR_FAIL})")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = results_dir / "phase2_ablation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
