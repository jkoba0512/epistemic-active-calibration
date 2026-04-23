"""Phase 8: q0 generalization check.

Does terminal controllability failure occur only at the fully-extended
degenerate start (q0=[0,0,0,0]), or also at non-degenerate initial
configurations?

RA-L reviewer concern:
  "Single scenario / single initial condition — is failure an artifact?"

Conditions:
  q0_A = [0, 0, 0, 0]             current degenerate baseline
  q0_B = [0, pi/4, -pi/4, 0]      non-degenerate control
  q0_C = [0.1, -0.1, 0.05, -0.05] mild asymmetric near-degenerate

Phase 1: null_space_probe (fixed)
Phase 2: plain / posture × H=[50, 100, 150, 200]
N=50 seeds

Expected:
  q0_A: plain fails (known), posture rescues
  q0_B: if non-degenerate → plain may already succeed → posture adds less
  q0_C: if near-degenerate → similar failure pattern to q0_A → posture rescues
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.redundant_arm_calibration import (
    run_one, CHANGE_STEP, TASK_ERR_FAIL, PARAM_RMSE_FAIL,
)

Q0_CONDITIONS = {
    "q0_A_degenerate":    [0.0, 0.0, 0.0, 0.0],
    "q0_B_nondegenerate": [0.0, math.pi/4, -math.pi/4, 0.0],
    "q0_C_near_degen":    [0.1, -0.1, 0.05, -0.05],
}
PHASE1_CONDITION = "null_space_probe"
CONTROLLERS = ["plain", "posture"]
HORIZONS = [50, 100, 150, 200]
N_RUNS = 50


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def main():
    print(f"Phase 8: q0 generalization check  (N={N_RUNS})")
    print(f"  Phase 1: {PHASE1_CONDITION}")
    print(f"  Controllers: {CONTROLLERS}")
    print(f"  Horizons: {HORIZONS}")
    print()

    results = {
        q0_label: {ctrl: {H: [] for H in HORIZONS} for ctrl in CONTROLLERS}
        for q0_label in Q0_CONDITIONS
    }

    for q0_label, q0_val in Q0_CONDITIONS.items():
        for ctrl in CONTROLLERS:
            for H in HORIZONS:
                ph2_ctrl = ctrl if ctrl != "plain" else None
                ph2_ctrl_posture = "posture" if ctrl == "posture" else None
                print(f"  {q0_label:25s} / {ctrl:7s} / H={H:3d} ", end="", flush=True)
                for seed in range(N_RUNS):
                    r = run_one(seed, PHASE1_CONDITION,
                                phase2_horizon_override=H,
                                phase2_controller_override=ph2_ctrl_posture,
                                q0_override=q0_val)
                    results[q0_label][ctrl][H].append(r)
                    print(".", end="", flush=True)
                rs = results[q0_label][ctrl][H]
                ft = float(np.mean(np.array([r["task_err_final"] for r in rs]) > TASK_ERR_FAIL))
                rmse = float(np.median([r["rmse_at_change"] for r in rs]))
                ee = float(np.median([r["ee_hold_mean"] for r in rs]))
                print(f"  failTask={ft:.2f}  RMSE={rmse:.4f}  EEhold={ee:.4f}")

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    out = {}
    for q0_label, q0_val in Q0_CONDITIONS.items():
        out[q0_label] = {"q0": q0_val, "controllers": {}}
        for ctrl in CONTROLLERS:
            out[q0_label]["controllers"][ctrl] = {}
            for H in HORIZONS:
                rs = results[q0_label][ctrl][H]
                task_errs = np.array([r["task_err_final"] for r in rs])
                rmses = np.array([r["rmse_at_change"] for r in rs])
                ee_holds = np.array([r["ee_hold_mean"] for r in rs])
                n_fail = int(np.sum(task_errs > TASK_ERR_FAIL))
                hard_seeds = [i for i, r in enumerate(rs)
                              if r["task_err_final"] > TASK_ERR_FAIL]
                ci_lo, ci_hi = wilson_ci(n_fail, N_RUNS)
                q_norms = [float(np.linalg.norm(r["q_change"])) for r in rs]
                out[q0_label]["controllers"][ctrl][H] = {
                    "fail_task": float(n_fail / N_RUNS),
                    "fail_task_wilson_lo": ci_lo,
                    "fail_task_wilson_hi": ci_hi,
                    "fail_rmse": float(np.mean(rmses > PARAM_RMSE_FAIL)),
                    "rmse_median": float(np.median(rmses)),
                    "ee_hold_median": float(np.median(ee_holds)),
                    "task_err_median": float(np.median(task_errs)),
                    "q_norm_change_median": float(np.median(q_norms)),
                    "hard_seeds": hard_seeds,
                    "n_seeds": N_RUNS,
                }

    # -----------------------------------------------------------------------
    # Print failTask table
    # -----------------------------------------------------------------------
    print("\n--- failTask (Wilson 95% CI) ---")
    for q0_label in Q0_CONDITIONS:
        print(f"\n  {q0_label}:")
        for ctrl in CONTROLLERS:
            row = f"    {ctrl:8s}"
            for H in HORIZONS:
                d = out[q0_label]["controllers"][ctrl][H]
                ft = d["fail_task"]
                lo, hi = d["fail_task_wilson_lo"], d["fail_task_wilson_hi"]
                row += f"  H={H}: {ft:.2f} [{lo:.2f},{hi:.2f}]"
            print(row)

    print("\n--- q_norm_change median ---")
    for q0_label in Q0_CONDITIONS:
        # q_norm_change is the same across H (Phase 1 fixed)
        qn = out[q0_label]["controllers"]["plain"][HORIZONS[0]]["q_norm_change_median"]
        print(f"  {q0_label:25s}: median ||q_change|| = {qn:.4f}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_json = project_root / "results" / "phase8_q0_generalization.json"
    with open(out_json, "w") as f:
        json.dump(
            {ql: {
                "q0": qd["q0"],
                "controllers": {
                    ctrl: {str(H): v for H, v in hd.items()}
                    for ctrl, hd in qd["controllers"].items()
                }
            } for ql, qd in out.items()},
            f, indent=2,
        )
    print(f"\nSaved → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = {
        "q0_A_degenerate":    "tab:red",
        "q0_B_nondegenerate": "tab:blue",
        "q0_C_near_degen":    "tab:orange",
    }
    STYLES = {"plain": "--", "posture": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Phase 8: q0 generalization  |  Phase 1: {PHASE1_CONDITION}  |  N={N_RUNS}",
        fontsize=10,
    )

    for ax_i, ctrl in enumerate(CONTROLLERS):
        ax = axes[ax_i]
        for q0_label in Q0_CONDITIONS:
            ft = [out[q0_label]["controllers"][ctrl][H]["fail_task"] for H in HORIZONS]
            lo = [out[q0_label]["controllers"][ctrl][H]["fail_task_wilson_lo"] for H in HORIZONS]
            hi = [out[q0_label]["controllers"][ctrl][H]["fail_task_wilson_hi"] for H in HORIZONS]
            ax.plot(HORIZONS, ft, "o-", color=COLORS[q0_label], label=q0_label)
            ax.fill_between(HORIZONS, lo, hi, color=COLORS[q0_label], alpha=0.15)
        ax.set_xlabel("Phase 2 horizon H")
        ax.set_ylabel("failTask rate")
        ax.set_title(f"({chr(65+ax_i)}) {ctrl} controller")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.02, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_fig = project_root / "results" / "phase8_q0_generalization.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
