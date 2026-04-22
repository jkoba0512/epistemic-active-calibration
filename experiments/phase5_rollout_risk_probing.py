"""Phase 5: short rollout risk probing horizon sweep.

Compares null_space_probe (baseline) against null_space_probe_rollout_risk
across Phase 2 horizons [50,80,100,120,150,200] for both plain and posture
Phase 2 controllers.

Key question: does the K-step rollout risk penalty in Phase 1 reduce the
failTask curve and rescue the hard-tail seeds compared to the baseline?

Background:
    Phase 5 follows from the Phase 5 AUC diagnostics showing:
    - K=5  rollout AUC ≥ 0.94 across all horizons
    - K=10 rollout AUC = 1.000 at H≤100, 0.986 at H≥150
    - The posture-potential proxy (Phase 4) was a weak surrogate; AUC ≈ 0.5

Settings:
    PROBE_ROLLOUT_K = 10   (from redundant_arm_calibration.py)
    PROBE_ROLLOUT_PENALTY = 0.2

Outputs:
    results/phase5_rollout_risk_probing.json
    results/phase5_rollout_risk_probing.png
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
    PROBE_ROLLOUT_K,
    PROBE_ROLLOUT_PENALTY,
    _summarize,
)

HORIZONS = [50, 80, 100, 120, 150, 200]
PROBES = ["null_space_probe", "null_space_probe_rollout_risk"]
CONTROLLERS = ["plain", "posture"]

HARD_TAIL_BASELINE = [1, 3, 37]
HARD_TAIL_RISK_AWARE = [1, 3, 22]


def run_sweep(probe_condition, ctrl, horizon):
    runs = []
    for seed in range(N_SEEDS):
        r = run_one(
            seed,
            probe_condition,
            phase2_horizon_override=horizon,
            phase2_controller_override=ctrl,
        )
        task_arr = np.array(r["task_err"])
        phase2 = task_arr[CHANGE_STEP:]
        below = np.where(phase2 < TASK_ERR_FAIL)[0]
        runs.append({
            "seed": seed,
            "task_failed": bool(r["task_err_final"] > TASK_ERR_FAIL),
            "rmse_at_change": float(r["rmse_at_change"]),
            "task_err_final": float(r["task_err_final"]),
            "ee_hold_mean": float(r["ee_hold_mean"]),
            "first_success_step": int(below[0] + CHANGE_STEP) if len(below) > 0 else None,
        })
        print(".", end="", flush=True)
    return runs


def main():
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "settings": {
            "probes": PROBES,
            "horizons": HORIZONS,
            "controllers": CONTROLLERS,
            "n_seeds": N_SEEDS,
            "change_step": CHANGE_STEP,
            "task_err_fail": TASK_ERR_FAIL,
            "probe_rollout_k": int(PROBE_ROLLOUT_K),
            "probe_rollout_penalty": float(PROBE_ROLLOUT_PENALTY),
        },
        "results": {p: {c: {} for c in CONTROLLERS} for p in PROBES},
    }

    for probe in PROBES:
        for ctrl in CONTROLLERS:
            for horizon in HORIZONS:
                print(f"  {probe:34s} / {ctrl:7s} / {horizon:3d} ", end="", flush=True)
                runs = run_sweep(probe, ctrl, horizon)

                task_failed = np.array([r["task_failed"] for r in runs])
                task_err = np.array([r["task_err_final"] for r in runs])
                rmse = np.array([r["rmse_at_change"] for r in runs])
                eehold = np.array([r["ee_hold_mean"] for r in runs])
                fail_rate = float(np.mean(task_failed))
                failed_seeds = [r["seed"] for r in runs if r["task_failed"]]

                print(
                    f"  failTask={fail_rate:.2f}  "
                    f"RMSE={np.median(rmse):.4f}  "
                    f"EEhold={np.median(eehold):.4f}"
                )

                output["results"][probe][ctrl][str(horizon)] = {
                    "task_failure_rate": fail_rate,
                    "n_failed": int(np.sum(task_failed)),
                    "failed_seeds": failed_seeds,
                    "task_err_final": _summarize(task_err),
                    "rmse_at_change": _summarize(rmse),
                    "ee_hold_err_phase1": _summarize(eehold),
                    "per_seed": runs,
                }

    out_path = results_dir / "phase5_rollout_risk_probing.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    for ctrl in CONTROLLERS:
        print(f"\n--- failTask: baseline vs rollout-risk probe ({ctrl} controller) ---")
        print(f"{'horizon':>8}  {'baseline':>10}  {'rollout-risk':>12}  {'delta':>8}")
        for h in HORIZONS:
            b = output["results"]["null_space_probe"][ctrl][str(h)]["task_failure_rate"]
            r = output["results"]["null_space_probe_rollout_risk"][ctrl][str(h)]["task_failure_rate"]
            print(f"{h:8d}  {b:10.2f}  {r:12.2f}  {r - b:+8.2f}")

    print("\n--- Hard-tail seeds at max horizon (200 steps) ---")
    for ctrl in CONTROLLERS:
        print(f"  {ctrl} controller:")
        for probe in PROBES:
            fs = sorted(output["results"][probe][ctrl]["200"]["failed_seeds"])
            print(f"    {probe:34s}: {fs}")

    print("\n--- RMSE comparison (should be similar) ---")
    for probe in PROBES:
        rmse_med = output["results"][probe]["plain"]["50"]["rmse_at_change"]["median"]
        eehold_med = output["results"][probe]["plain"]["50"]["ee_hold_err_phase1"]["median"]
        print(f"  {probe:34s}: RMSE={rmse_med:.4f}  EEhold={eehold_med:.4f}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Phase 5: rollout-risk probing (K={PROBE_ROLLOUT_K}, penalty={PROBE_ROLLOUT_PENALTY})\n"
            "Does K-step rollout penalty in Phase 1 reduce the horizon needed for Phase 2 convergence?",
            fontsize=10,
        )

        horizons_arr = np.array(HORIZONS)
        styles = {
            "null_space_probe":             {"color": "C0", "linestyle": "-",  "label": "baseline probe"},
            "null_space_probe_rollout_risk": {"color": "C2", "linestyle": "--", "label": f"rollout-risk probe (K={PROBE_ROLLOUT_K})"},
        }

        for ax_idx, ctrl in enumerate(CONTROLLERS):
            ax = axes[ax_idx]
            for probe in PROBES:
                rates = [
                    output["results"][probe][ctrl][str(h)]["task_failure_rate"]
                    for h in HORIZONS
                ]
                s = styles[probe]
                ax.plot(horizons_arr, rates, "o" + s["linestyle"],
                        color=s["color"], label=s["label"], linewidth=2)
            ax.set_xlabel("Phase 2 horizon (steps)")
            ax.set_ylabel("Task failure rate")
            ax.set_title(f"Phase 2 controller: {ctrl}")
            ax.set_ylim(-0.02, 0.6)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = results_dir / "phase5_rollout_risk_probing.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
