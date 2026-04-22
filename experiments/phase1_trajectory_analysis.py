"""Phase 1 trajectory analysis: hard-tail seeds vs success seeds.

Plan D: Compare q(t), sigma_min(t), probe_weight(t), RMSE(t) between
hard-tail seeds [1, 3] and matched success seeds [0, 2, 27, 47] under
null_space_probe / plain controller.

Goal: determine whether Phase 1 explores a qualitatively different
manifold for hard-tail seeds, identifying the structural cause of failure.
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
    run_one, CHANGE_STEP, N_DOF, THETA_TRUE, fk,
)

# Seeds to compare
HARD_TAIL_SEEDS = [1, 3]
SUCCESS_SEEDS = [0, 2, 27, 47]
ALL_SEEDS = HARD_TAIL_SEEDS + SUCCESS_SEEDS

CONDITION = "null_space_probe"


def run_with_trajectory(seed):
    return run_one(seed, CONDITION,
                   phase2_horizon_override=1,
                   return_trajectory=True)


def main():
    print("Phase 1 trajectory analysis")
    print(f"  Hard-tail seeds: {HARD_TAIL_SEEDS}")
    print(f"  Success seeds:   {SUCCESS_SEEDS}")
    print(f"  Condition:       {CONDITION}")
    print()

    results = {}
    for seed in ALL_SEEDS:
        print(f"  Running seed {seed}...", flush=True)
        results[seed] = run_with_trajectory(seed)

    # -----------------------------------------------------------------------
    # Compute derived metrics
    # -----------------------------------------------------------------------
    import jax.numpy as jnp

    def compute_sigma_min_traj(q_traj, theta):
        """sigma_min of J_ee at each step."""
        import jax
        sigma_mins = []
        for q in q_traj:
            J = jax.jacfwd(lambda qi: fk(qi, theta))(jnp.array(q))
            svals = jnp.linalg.svd(J, compute_uv=False)
            sigma_mins.append(float(jnp.min(svals)))
        return np.array(sigma_mins)

    def compute_q_norm_traj(q_traj):
        return np.array([np.linalg.norm(q) for q in q_traj])

    # -----------------------------------------------------------------------
    # Print summary statistics
    # -----------------------------------------------------------------------
    print("\n--- Summary at Phase 1 end (step 149) ---")
    print(f"{'Seed':6s}  {'Group':10s}  {'RMSE@149':9s}  {'sigma_min@end':14s}  "
          f"{'||q||@end':10s}  {'probe_wt_mean':14s}  {'EEhold_mean':12s}")
    for seed in ALL_SEEDS:
        r = results[seed]
        group = "hard-tail" if seed in HARD_TAIL_SEEDS else "success"
        rmse_end = r["rmse"][CHANGE_STEP - 1]
        q_traj = r["q_trajectory_phase1"]
        sigma_mins = compute_sigma_min_traj(q_traj, THETA_TRUE)
        sigma_end = sigma_mins[-1]
        q_norm_end = np.linalg.norm(q_traj[-1])
        pw_mean = r["probe_weight_mean_phase1"]
        ee_hold = r["ee_hold_mean"]
        print(f"  {seed:4d}  {group:10s}  {rmse_end:.5f}    {sigma_end:.6f}          "
              f"{q_norm_end:.5f}    {pw_mean:.5f}          {ee_hold:.5f}")

    # -----------------------------------------------------------------------
    # Save diagnostics JSON
    # -----------------------------------------------------------------------
    diag = {}
    for seed in ALL_SEEDS:
        r = results[seed]
        q_traj = r["q_trajectory_phase1"]
        sigma_mins = compute_sigma_min_traj(q_traj, THETA_TRUE)
        q_norms = compute_q_norm_traj(q_traj)
        diag[seed] = {
            "group": "hard_tail" if seed in HARD_TAIL_SEEDS else "success",
            "rmse_at_change": float(r["rmse_at_change"]),
            "ee_hold_mean": float(r["ee_hold_mean"]),
            "probe_weight_mean_phase1": float(r["probe_weight_mean_phase1"]),
            "p_theta_rank_at_change": int(r["p_theta_rank_at_change"]),
            "q_change": r["q_change"].tolist(),
            "q_norm_at_change": float(np.linalg.norm(r["q_change"])),
            "sigma_min_at_change": float(sigma_mins[-1]),
            "sigma_min_traj": sigma_mins.tolist(),
            "q_norm_traj": q_norms.tolist(),
            "rmse_traj": r["rmse"][:CHANGE_STEP].tolist(),
            "probe_mode_traj": r["probe_mode"][:CHANGE_STEP].tolist(),
            "probe_gain_traj": r["probe_gain"][:CHANGE_STEP].tolist(),
        }

    out_json = project_root / "results" / "phase1_trajectory_analysis.json"
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in diag.items()}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Phase 1 trajectory analysis: hard-tail seeds [1,3] vs success seeds [0,2,27,47]\n"
        f"Condition: {CONDITION}  |  Phase 1: steps 0–{CHANGE_STEP - 1}",
        fontsize=10,
    )

    steps = np.arange(CHANGE_STEP)

    COLOR = {
        1: "tab:red", 3: "tab:orange",
        0: "tab:blue", 2: "tab:cyan", 27: "tab:green", 47: "tab:purple",
    }
    STYLE = {s: "--" if s in HARD_TAIL_SEEDS else "-" for s in ALL_SEEDS}

    def label(seed):
        g = "fail" if seed in HARD_TAIL_SEEDS else "ok"
        return f"seed {seed} ({g})"

    # (A) RMSE over Phase 1
    ax = axes[0, 0]
    for seed in ALL_SEEDS:
        r = results[seed]
        ax.plot(steps, r["rmse"][:CHANGE_STEP],
                color=COLOR[seed], linestyle=STYLE[seed], label=label(seed))
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE(θ)")
    ax.set_title("(A) Parameter RMSE")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # (B) sigma_min of J_ee over Phase 1
    ax = axes[0, 1]
    for seed in ALL_SEEDS:
        r = results[seed]
        q_traj = r["q_trajectory_phase1"]
        sigma_mins = compute_sigma_min_traj(q_traj, THETA_TRUE)
        ax.plot(steps, sigma_mins,
                color=COLOR[seed], linestyle=STYLE[seed], label=label(seed))
    ax.set_xlabel("Step")
    ax.set_ylabel("σ_min(J_ee)")
    ax.set_title("(B) Min singular value of J_ee (manipulability)")
    ax.legend(fontsize=7)

    # (C) ||q|| over Phase 1
    ax = axes[0, 2]
    for seed in ALL_SEEDS:
        r = results[seed]
        q_traj = r["q_trajectory_phase1"]
        q_norms = compute_q_norm_traj(q_traj)
        ax.plot(steps, q_norms,
                color=COLOR[seed], linestyle=STYLE[seed], label=label(seed))
    ax.set_xlabel("Step")
    ax.set_ylabel("||q||")
    ax.set_title("(C) Joint angle norm (departure from degenerate)")
    ax.legend(fontsize=7)

    # (D) probe_weight over Phase 1
    ax = axes[1, 0]
    for seed in ALL_SEEDS:
        r = results[seed]
        ax.plot(steps, r["probe_mode"][:CHANGE_STEP],
                color=COLOR[seed], linestyle=STYLE[seed], label=label(seed))
    ax.set_xlabel("Step")
    ax.set_ylabel("probe weight (1 − α_N)")
    ax.set_title("(D) Probe weight (1=finite-step, 0=gradient)")
    ax.legend(fontsize=7)

    # (E) probe_gain over Phase 1
    ax = axes[1, 1]
    for seed in ALL_SEEDS:
        r = results[seed]
        ax.plot(steps, r["probe_gain"][:CHANGE_STEP],
                color=COLOR[seed], linestyle=STYLE[seed], label=label(seed))
    ax.set_xlabel("Step")
    ax.set_ylabel("IG gain (best probe)")
    ax.set_title("(E) Best probe IG gain per step")
    ax.legend(fontsize=7)

    # (F) q_change scatter: joint 0 vs joint 1
    ax = axes[1, 2]
    for seed in ALL_SEEDS:
        r = results[seed]
        qc = r["q_change"]
        marker = "X" if seed in HARD_TAIL_SEEDS else "o"
        ax.scatter(qc[0], qc[1], color=COLOR[seed], marker=marker,
                   s=120, label=label(seed), zorder=5)
    ax.set_xlabel("q[0] at Phase 2 start")
    ax.set_ylabel("q[1] at Phase 2 start")
    ax.set_title("(F) q_change configuration (Phase 1 end)")
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out_fig = project_root / "results" / "phase1_trajectory_analysis.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
