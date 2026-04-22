"""Phase 6: adaptive and manipulability-based posture controllers.

Plan C: Sweep adaptive_posture and manip_posture Phase 2 controllers
against baseline posture across horizons [50, 80, 100, 120, 150, 200].

Research question: can Phase 2 controller improvement reduce failTask
at shorter horizons (target: posture/100 failTask < 0.02)?

Baseline (from Phase 2 horizon sweep):
  null_space_probe + posture controller:
    H=100: failTask=0.16, H=120: 0.02, H=150: 0.00, H=200: 0.00

New controllers:
  adaptive_posture: k_eff = k / (||q|| + 0.1)  — larger gain near q=0
  manip_posture:    N @ grad_q log det(J J^T)   — manipulability gradient
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.redundant_arm_calibration import (
    run_one, CHANGE_STEP, N_SEEDS, TASK_ERR_FAIL, PARAM_RMSE_FAIL, THETA_TRUE, fk,
)

PHASE1_CONDITION = "null_space_probe"
CONTROLLERS = ["posture", "adaptive_posture", "manip_posture"]
HORIZONS = [50, 80, 100, 120, 150, 200]
N_RUNS = 50  # full N


def main():
    print(f"Phase 6: adaptive posture sweep  (N={N_RUNS})")
    print(f"  Phase 1 condition: {PHASE1_CONDITION}")
    print(f"  Controllers: {CONTROLLERS}")
    print(f"  Horizons: {HORIZONS}")
    print()

    results = {ctrl: {H: [] for H in HORIZONS} for ctrl in CONTROLLERS}

    for ctrl in CONTROLLERS:
        for H in HORIZONS:
            print(f"  {ctrl:20s} / H={H:3d} ", end="", flush=True)
            for seed in range(N_RUNS):
                r = run_one(seed, PHASE1_CONDITION,
                            phase2_horizon_override=H,
                            phase2_controller_override=ctrl)
                results[ctrl][H].append(r)
                print(".", end="", flush=True)
            rs = results[ctrl][H]
            task_errs = [r["task_err_final"] for r in rs]
            rmses = [r["rmse_at_change"] for r in rs]
            ee_holds = [r["ee_hold_mean"] for r in rs]
            fail_task = float(np.mean(np.array(task_errs) > TASK_ERR_FAIL))
            rmse_med = float(np.median(rmses))
            ee_hold_med = float(np.median(ee_holds))
            print(f"  failTask={fail_task:.2f}  RMSE={rmse_med:.4f}  EEhold={ee_hold_med:.4f}")

    # -----------------------------------------------------------------------
    # Hard-tail analysis
    # -----------------------------------------------------------------------
    print("\n--- Hard-tail seeds at H=200 ---")
    for ctrl in CONTROLLERS:
        hard_seeds = [
            seed for seed, r in enumerate(results[ctrl][200])
            if r["task_err_final"] > TASK_ERR_FAIL
        ]
        print(f"  {ctrl:20s}: {hard_seeds}")

    # -----------------------------------------------------------------------
    # failTask table
    # -----------------------------------------------------------------------
    print("\n--- failTask by horizon ---")
    header = f"{'ctrl':20s} " + " ".join(f"H={H:3d}" for H in HORIZONS)
    print(header)
    for ctrl in CONTROLLERS:
        row = f"  {ctrl:18s} "
        for H in HORIZONS:
            rs = results[ctrl][H]
            ft = float(np.mean(np.array([r["task_err_final"] for r in rs]) > TASK_ERR_FAIL))
            row += f"  {ft:.2f}  "
        print(row)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out = {}
    for ctrl in CONTROLLERS:
        out[ctrl] = {}
        for H in HORIZONS:
            rs = results[ctrl][H]
            task_errs = np.array([r["task_err_final"] for r in rs])
            rmses = np.array([r["rmse_at_change"] for r in rs])
            ee_holds = np.array([r["ee_hold_mean"] for r in rs])
            hard_seeds = [i for i, r in enumerate(rs)
                         if r["task_err_final"] > TASK_ERR_FAIL]
            out[ctrl][H] = {
                "fail_task": float(np.mean(task_errs > TASK_ERR_FAIL)),
                "fail_rmse": float(np.mean(rmses > PARAM_RMSE_FAIL)),
                "rmse_median": float(np.median(rmses)),
                "ee_hold_median": float(np.median(ee_holds)),
                "task_err_median": float(np.median(task_errs)),
                "hard_seeds": hard_seeds,
            }

    out_json = project_root / "results" / "phase6_adaptive_posture.json"
    with open(out_json, "w") as f:
        # JSON keys must be strings
        json.dump(
            {ctrl: {str(H): v for H, v in hd.items()} for ctrl, hd in out.items()},
            f, indent=2,
        )
    print(f"\nSaved → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = {
        "posture": "C0",
        "adaptive_posture": "C1",
        "manip_posture": "C2",
    }
    LABELS = {
        "posture": "posture (baseline)",
        "adaptive_posture": "adaptive posture (k_eff = k / (||q||+ε))",
        "manip_posture": "manip posture (∇ log det J J^T)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Phase 6: Phase 2 adaptive posture controllers vs baseline\n"
        f"Phase 1: {PHASE1_CONDITION}  |  N={N_RUNS}",
        fontsize=10,
    )

    # (A) failTask vs horizon
    ax = axes[0]
    for ctrl in CONTROLLERS:
        ft = [out[ctrl][H]["fail_task"] for H in HORIZONS]
        ax.plot(HORIZONS, ft, "o-", color=COLORS[ctrl], label=LABELS[ctrl])
    ax.set_xlabel("Phase 2 horizon H")
    ax.set_ylabel("failTask rate")
    ax.set_title("(A) Task failure rate vs horizon")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.02, color="gray", linestyle=":", linewidth=0.8, label="target 0.02")

    # (B) RMSE@150 (should be identical across controllers - Phase 1 frozen)
    ax = axes[1]
    for ctrl in CONTROLLERS:
        rmse = [out[ctrl][H]["rmse_median"] for H in HORIZONS]
        ax.plot(HORIZONS, rmse, "s--", color=COLORS[ctrl], label=LABELS[ctrl])
    ax.set_xlabel("Phase 2 horizon H")
    ax.set_ylabel("Median RMSE@Phase1end")
    ax.set_title("(B) RMSE at Phase 1 end (sanity: should be flat)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_fig = project_root / "results" / "phase6_adaptive_posture.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")

    # -----------------------------------------------------------------------
    # Print final comparison table
    # -----------------------------------------------------------------------
    print("\n--- Final comparison (H=100, H=200) ---")
    for ctrl in CONTROLLERS:
        ft100 = out[ctrl][100]["fail_task"]
        ft200 = out[ctrl][200]["fail_task"]
        hard200 = out[ctrl][200]["hard_seeds"]
        print(f"  {ctrl:20s}  failTask@100={ft100:.2f}  failTask@200={ft200:.2f}  "
              f"hard@200={hard200}")


if __name__ == "__main__":
    main()
