"""Phase 9: q_ref sensitivity for posture regularizer.

Does N(-k*q) work because q_ref=0 happens to be special, or does any
null-space posture regularizer help?

RA-L reviewer concern:
  "Why N(-kq)? The choice of q_ref=0 lacks justification."

Compares u = J^+ v_task + N(-k(q - q_ref)) for several q_ref values.

q_ref variants:
  q_ref_zero   = [0, 0, 0, 0]           current baseline
  q_ref_bent   = [0.2, -0.2, 0.1, -0.1] arbitrary non-zero
  q_ref_alt    = [0.4, -0.3, 0.2, -0.1] larger displacement
  q_ref_median = median q_change from null_space_probe N=50 seeds

Phase 1: null_space_probe (fixed, q0=[0,0,0,0])
Phase 2: posture controller with varying q_ref
H=[50, 100, 150, 200], N=50
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
    run_one, CHANGE_STEP, TASK_ERR_FAIL, PARAM_RMSE_FAIL, N_SEEDS,
)

PHASE1_CONDITION = "null_space_probe"
HORIZONS = [50, 100, 150, 200]
N_RUNS = 50


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def get_q_ref_median():
    """Compute median q_change across N=50 seeds for null_space_probe."""
    print("  Computing q_ref_median from N=50 Phase 1 runs...", flush=True)
    q_changes = []
    for seed in range(N_RUNS):
        r = run_one(seed, PHASE1_CONDITION, phase2_horizon_override=1)
        q_changes.append(r["q_change"])
        print(".", end="", flush=True)
    print()
    return list(np.median(np.array(q_changes), axis=0))


def main():
    print(f"Phase 9: q_ref sensitivity  (N={N_RUNS})")
    print(f"  Phase 1: {PHASE1_CONDITION}")
    print(f"  Horizons: {HORIZONS}")
    print()

    # Compute q_ref_median first
    q_ref_median = get_q_ref_median()
    print(f"  q_ref_median = {[f'{v:.4f}' for v in q_ref_median]}")

    Q_REF_CONDITIONS = {
        "q_ref_zero":   [0.0, 0.0, 0.0, 0.0],
        "q_ref_bent":   [0.2, -0.2, 0.1, -0.1],
        "q_ref_alt":    [0.4, -0.3, 0.2, -0.1],
        "q_ref_median": q_ref_median,
    }

    results = {
        qr_label: {H: [] for H in HORIZONS}
        for qr_label in Q_REF_CONDITIONS
    }

    for qr_label, q_ref_val in Q_REF_CONDITIONS.items():
        for H in HORIZONS:
            print(f"  {qr_label:15s} / H={H:3d} ", end="", flush=True)
            for seed in range(N_RUNS):
                r = run_one(seed, PHASE1_CONDITION,
                            phase2_horizon_override=H,
                            phase2_controller_override="posture",
                            q_ref_override=q_ref_val)
                results[qr_label][H].append(r)
                print(".", end="", flush=True)
            rs = results[qr_label][H]
            ft = float(np.mean(np.array([r["task_err_final"] for r in rs]) > TASK_ERR_FAIL))
            print(f"  failTask={ft:.2f}")

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    out = {"q_ref_values": {k: v for k, v in Q_REF_CONDITIONS.items()}, "results": {}}
    for qr_label in Q_REF_CONDITIONS:
        out["results"][qr_label] = {}
        for H in HORIZONS:
            rs = results[qr_label][H]
            task_errs = np.array([r["task_err_final"] for r in rs])
            rmses = np.array([r["rmse_at_change"] for r in rs])
            n_fail = int(np.sum(task_errs > TASK_ERR_FAIL))
            ci_lo, ci_hi = wilson_ci(n_fail, N_RUNS)
            hard_seeds = [i for i, r in enumerate(rs)
                          if r["task_err_final"] > TASK_ERR_FAIL]
            out["results"][qr_label][H] = {
                "fail_task": float(n_fail / N_RUNS),
                "fail_task_wilson_lo": ci_lo,
                "fail_task_wilson_hi": ci_hi,
                "fail_rmse": float(np.mean(rmses > PARAM_RMSE_FAIL)),
                "rmse_median": float(np.median(rmses)),
                "task_err_median": float(np.median(task_errs)),
                "hard_seeds": hard_seeds,
                "n_seeds": N_RUNS,
            }

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    print("\n--- failTask by q_ref and horizon (Wilson 95% CI) ---")
    header = f"  {'q_ref':15s}" + "".join(f"  H={H:3d}            " for H in HORIZONS)
    print(header)
    for qr_label in Q_REF_CONDITIONS:
        row = f"  {qr_label:15s}"
        for H in HORIZONS:
            d = out["results"][qr_label][H]
            ft = d["fail_task"]
            lo, hi = d["fail_task_wilson_lo"], d["fail_task_wilson_hi"]
            row += f"  {ft:.2f} [{lo:.2f},{hi:.2f}]"
        print(row)

    print("\n--- hard seeds at H=200 ---")
    for qr_label in Q_REF_CONDITIONS:
        hs = out["results"][qr_label][200]["hard_seeds"]
        print(f"  {qr_label:15s}: {hs}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_json = project_root / "results" / "phase9_qref_sensitivity.json"
    with open(out_json, "w") as f:
        json.dump(
            {"q_ref_values": out["q_ref_values"],
             "results": {qr: {str(H): v for H, v in hd.items()}
                         for qr, hd in out["results"].items()}},
            f, indent=2,
        )
    print(f"\nSaved → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = {
        "q_ref_zero":   "tab:blue",
        "q_ref_bent":   "tab:orange",
        "q_ref_alt":    "tab:green",
        "q_ref_median": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Phase 9: posture q_ref sensitivity  |  Phase 1: {PHASE1_CONDITION}  |  N={N_RUNS}\n"
        "u = J⁺v_task + N(-k(q - q_ref))",
        fontsize=10,
    )

    for qr_label in Q_REF_CONDITIONS:
        ft = [out["results"][qr_label][H]["fail_task"] for H in HORIZONS]
        lo = [out["results"][qr_label][H]["fail_task_wilson_lo"] for H in HORIZONS]
        hi = [out["results"][qr_label][H]["fail_task_wilson_hi"] for H in HORIZONS]
        ax.plot(HORIZONS, ft, "o-", color=COLORS[qr_label], label=qr_label)
        ax.fill_between(HORIZONS, lo, hi, color=COLORS[qr_label], alpha=0.15)

    ax.set_xlabel("Phase 2 horizon H")
    ax.set_ylabel("failTask rate")
    ax.set_title("failTask vs horizon for different q_ref (posture controller)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.02, color="gray", linestyle=":", linewidth=0.8, label="target 0.02")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_fig = project_root / "results" / "phase9_qref_sensitivity.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
