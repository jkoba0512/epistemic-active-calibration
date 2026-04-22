"""Phase 4: risk-aware probing horizon sweep.

Compares the plain probe (null_space_probe) against the risk-aware probe
(null_space_probe_risk) across Phase 2 horizons [50,80,100,120,150,180,200]
for both plain and posture Phase 2 controllers.

Key question: does the terminal-risk penalty in Phase 1 reduce the failTask
curve compared to the baseline, and does it reduce or eliminate the hard-tail?

Outputs:
    results/phase4_risk_aware_probing.json
    results/phase4_risk_aware_probing.png
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
    PROBE_TERMINAL_PENALTY,
    PROBE_TERMINAL_EPS,
    _summarize,
)

HORIZONS = [50, 80, 100, 120, 150, 180, 200]
PROBES = ["null_space_probe", "null_space_probe_risk"]
CONTROLLERS = ["plain", "posture"]


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
            "probe_terminal_penalty": float(PROBE_TERMINAL_PENALTY),
            "probe_terminal_eps": float(PROBE_TERMINAL_EPS),
        },
        "results": {p: {c: {} for c in CONTROLLERS} for p in PROBES},
    }

    for probe in PROBES:
        for ctrl in CONTROLLERS:
            for horizon in HORIZONS:
                print(f"  {probe:26s} / {ctrl:7s} / {horizon:3d} ", end="", flush=True)
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

    out_path = results_dir / "phase4_risk_aware_probing.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n--- failTask: plain probe vs risk-aware probe (plain controller) ---")
    print(f"{'horizon':>8}  {'baseline':>10}  {'risk-aware':>10}  {'delta':>8}")
    for h in HORIZONS:
        b = output["results"]["null_space_probe"]["plain"][str(h)]["task_failure_rate"]
        r = output["results"]["null_space_probe_risk"]["plain"][str(h)]["task_failure_rate"]
        print(f"{h:8d}  {b:10.2f}  {r:10.2f}  {r - b:+8.2f}")

    print("\n--- failTask: plain probe vs risk-aware probe (posture controller) ---")
    print(f"{'horizon':>8}  {'baseline':>10}  {'risk-aware':>10}  {'delta':>8}")
    for h in HORIZONS:
        b = output["results"]["null_space_probe"]["posture"][str(h)]["task_failure_rate"]
        r = output["results"]["null_space_probe_risk"]["posture"][str(h)]["task_failure_rate"]
        print(f"{h:8d}  {b:10.2f}  {r:10.2f}  {r - b:+8.2f}")

    # Hard-tail analysis: seeds that fail at max horizon
    print("\n--- Hard-tail seeds at max horizon (200 steps, plain ctrl) ---")
    for probe in PROBES:
        fs = set(output["results"][probe]["plain"]["200"]["failed_seeds"])
        print(f"  {probe:26s}: {sorted(fs)}")

    # RMSE comparison (should be similar)
    print("\n--- RMSE@150 median: should be similar across probes ---")
    for probe in PROBES:
        rmse_med = output["results"][probe]["plain"]["50"]["rmse_at_change"]["median"]
        eehold_med = output["results"][probe]["plain"]["50"]["ee_hold_err_phase1"]["median"]
        print(f"  {probe:26s}: RMSE={rmse_med:.4f}  EEhold={eehold_med:.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Phase 4: risk-aware probing (terminal_penalty={PROBE_TERMINAL_PENALTY})\n"
            "Does Phase 1 terminal-risk penalty reduce the horizon needed for Phase 2 convergence?",
            fontsize=10,
        )

        horizons_arr = np.array(HORIZONS)
        styles = {
            "null_space_probe":      {"color": "C0", "linestyle": "-",  "label": "baseline probe"},
            "null_space_probe_risk": {"color": "C1", "linestyle": "--", "label": "risk-aware probe"},
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
        fig_path = results_dir / "phase4_risk_aware_probing.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
