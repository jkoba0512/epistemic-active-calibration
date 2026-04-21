"""Transition/recovery ablation sweep for null_space_recovery.

Systematically sweeps:
    RECOVERY_START  ∈ {100, 110, 120, 130}
    RECOVERY_LENGTH ∈ {10, 20, 30}   (capped so start+length ≤ CHANGE_STEP)
    blend_type      ∈ {linear, smoothstep}

Phase 1: null_space exploration until RECOVERY_START, then EE blend toward task goal.
Phase 2: standard VFE task control (50 steps).

Goal: confirm whether any recovery variant achieves failTask < 10%
      (benchmark: null_space_posture achieved 8% via Phase 2 extension + posture).

Output:
    results/redundant_arm_transition_sweep.json
    results/redundant_arm_transition_sweep.png
"""

import sys
import json
from pathlib import Path
from itertools import product

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (
    N_DOF, THETA_TRUE, THETA_INIT, Q0, DT, N_STEPS, CHANGE_STEP,
    U_MAX, K_TASK, SIGMA_OBS, PI_Y, PI_X, PARAMS_PRIOR_PI, KAPPA_P,
    N_ESTEP_ITER, ALPHA_NS, ESTEP_FREQ, PARAM_RMSE_FAIL, TASK_ERR_FAIL,
    fk, Y_GOAL_HOLD, Y_GOAL_TASK, _matrix_diag, _summarize,
    rollout_step, _build_estep,
    compute_null_space_action, compute_vfe_only_action,
)

N_SEEDS_SWEEP = 20

# ---------------------------------------------------------------------------
# Sweep grid (only valid combos where start + length <= CHANGE_STEP)
# ---------------------------------------------------------------------------
RECOVERY_STARTS  = [100, 110, 120, 130]
RECOVERY_LENGTHS = [10, 20, 30]
BLEND_TYPES      = ["linear", "smoothstep"]


def _blend_alpha(t, start, length, blend_type):
    """Blend factor α ∈ [0,1] from step t in [start, start+length)."""
    raw = (t - start) / length
    raw = float(np.clip(raw, 0.0, 1.0))
    if blend_type == "smoothstep":
        return raw * raw * (3.0 - 2.0 * raw)
    return raw  # linear


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(seed, rec_start, rec_length, blend_type):
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(N_DOF)
    estep = _build_estep(theta_est)

    q = Q0.copy()
    u = jnp.zeros(N_DOF)
    q_hist, v_hist, y_hist = [], [], []
    theta_frozen = None
    rec_end = rec_start + rec_length

    rmse_hist, task_err_hist, ee_hold_hist = [], [], []

    for t in range(N_STEPS):
        rmse_hist.append(float(jnp.sqrt(jnp.mean((theta_est - THETA_TRUE) ** 2))))
        ee_true = fk(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_TASK) ** 2))))
        ee_hold_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_HOLD) ** 2))))

        if t < CHANGE_STEP:
            if t < rec_start:
                u = compute_null_space_action(q, theta_est, P_theta, Y_GOAL_HOLD)
            elif t < rec_end:
                alpha = _blend_alpha(t, rec_start, rec_length, blend_type)
                y_blend = (1.0 - alpha) * Y_GOAL_HOLD + alpha * Y_GOAL_TASK
                u = compute_vfe_only_action(q, theta_est, y_blend)
            else:
                # Recovery done; hold at task goal until Phase 2
                u = compute_vfe_only_action(q, theta_est, Y_GOAL_TASK)
        else:
            if theta_frozen is None:
                theta_frozen = theta_est
            u = compute_vfe_only_action(q, theta_frozen, Y_GOAL_TASK)

        q = rollout_step(q, u)
        y_obs = fk(q, THETA_TRUE) + rng.normal(0, SIGMA_OBS, size=(2,))
        y_obs = jnp.array(y_obs)
        q_hist.append(q)
        v_hist.append(jnp.zeros(1))
        y_hist.append(y_obs)

        if t < CHANGE_STEP and t > 5 and t % ESTEP_FREQ == 0:
            theta_est = estep.run(q_hist, v_hist, y_hist, theta_est, n_iter=N_ESTEP_ITER)
            theta_est = jnp.clip(theta_est, 0.05, 2.0)
            P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
            estep = _build_estep(theta_est)

    return {
        "rmse_at_change": float(rmse_hist[CHANGE_STEP - 1]),
        "task_err_final": float(task_err_hist[-1]),
        "ee_hold_mean_ph1": float(np.mean(ee_hold_hist[:CHANGE_STEP])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build valid sweep list
    sweep_configs = []
    for start, length, blend in product(RECOVERY_STARTS, RECOVERY_LENGTHS, BLEND_TYPES):
        if start + length <= CHANGE_STEP:
            sweep_configs.append((start, length, blend))

    print(f"Total sweep conditions: {len(sweep_configs)}  (N={N_SEEDS_SWEEP} each)", flush=True)
    print(f"Baseline null_space failTask=28%, null_space_posture failTask=8%\n")

    results = {}
    for start, length, blend in sweep_configs:
        key = f"start{start}_len{length}_{blend}"
        seed_results = []
        print(f"  {key}", end="", flush=True)
        for seed in range(N_SEEDS_SWEEP):
            r = run_one(seed, start, length, blend)
            seed_results.append(r)
            print(".", end="", flush=True)
        task_finals = np.array([r["task_err_final"] for r in seed_results])
        rmse_finals = np.array([r["rmse_at_change"] for r in seed_results])
        fail_task = float(np.mean(task_finals > TASK_ERR_FAIL))
        fail_rmse = float(np.mean(rmse_finals > PARAM_RMSE_FAIL))
        results[key] = {
            "recovery_start": start,
            "recovery_length": length,
            "blend_type": blend,
            "task_failure_rate": fail_task,
            "rmse_failure_rate": fail_rmse,
            "task_err_final": _summarize(task_finals),
            "rmse_at_change": _summarize(rmse_finals),
        }
        print(f"  failTask={fail_task:.2f}  failRMSE={fail_rmse:.2f}", flush=True)

    # Save JSON
    out_json = project_root / "results" / "redundant_arm_transition_sweep.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")

    # --- Heatmap: failTask per (start, length) for each blend type ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Transition sweep: task failure rate\n"
        f"null_space_recovery variants  (N={N_SEEDS_SWEEP} seeds)\n"
        "Baseline: null_space=28%,  null_space_posture=8%",
        fontsize=9,
    )

    for ax, blend in zip(axes, BLEND_TYPES):
        valid_starts  = sorted(set(c[0] for c in sweep_configs if c[2] == blend))
        valid_lengths = sorted(set(c[1] for c in sweep_configs if c[2] == blend))
        grid = np.full((len(valid_starts), len(valid_lengths)), np.nan)
        for i, start in enumerate(valid_starts):
            for j, length in enumerate(valid_lengths):
                key = f"start{start}_len{length}_{blend}"
                if key in results:
                    grid[i, j] = results[key]["task_failure_rate"]

        im = ax.imshow(grid, vmin=0, vmax=0.5, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(valid_lengths)))
        ax.set_xticklabels([f"len={l}" for l in valid_lengths])
        ax.set_yticks(range(len(valid_starts)))
        ax.set_yticklabels([f"start={s}" for s in valid_starts])
        ax.set_title(f"blend={blend}")
        for i in range(len(valid_starts)):
            for j in range(len(valid_lengths)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if grid[i,j] > 0.25 else "black")
        plt.colorbar(im, ax=ax, label="failTask")

    plt.tight_layout()
    out_fig = project_root / "results" / "redundant_arm_transition_sweep.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")

    # Print best results
    print("\n--- Top 5 by failTask ---")
    sorted_keys = sorted(results, key=lambda k: results[k]["task_failure_rate"])
    for k in sorted_keys[:5]:
        r = results[k]
        print(f"  {k}: failTask={r['task_failure_rate']:.2f}  "
              f"median={r['task_err_final']['median']:.4f}")


if __name__ == "__main__":
    main()
