"""Redundant arm failure diagnostics: per-seed breakdown of null_space failures.

Runs the null_space condition (and optionally vfe_only) with full per-step
diagnostics to classify why 28% of seeds fail the Phase 2 task.

Key additions over redundant_arm_calibration.py:
    - projection_leak = ||J_ee(q) @ u_null|| per step
    - action_energy   = ||u||^2 per step
    - null_motion_energy = ||u_null||^2 per step
    - q_at_phase1_end, task_err_at_phase2_start
    - theta_error_vector at Phase 1 end

Output:
    results/redundant_arm_diagnostics.json
    results/redundant_arm_diagnostics_summary.json
    results/redundant_arm_diagnostics.png
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep

# Import shared constants from main experiment
from experiments.redundant_arm_calibration import (
    N_DOF, THETA_TRUE, THETA_INIT, Q0, DT, N_STEPS, CHANGE_STEP, N_SEEDS,
    U_MAX, K_TASK, SIGMA_OBS, PI_Y, PI_X, PARAMS_PRIOR_PI, KAPPA_P,
    N_ESTEP_ITER, ALPHA_NS, ESTEP_FREQ, PARAM_RMSE_FAIL, TASK_ERR_FAIL,
    fk, Y_GOAL_HOLD, Y_GOAL_TASK, _matrix_diag, _summarize,
    rollout_step, _build_estep, _ig_at_q,
)

CONDITIONS_DIAG = ["null_space"]   # focus on the condition with partial success


# ---------------------------------------------------------------------------
# Action computation with diagnostics
# ---------------------------------------------------------------------------

def compute_null_space_action_diag(q, theta_est, P_theta, y_goal):
    """Returns u, u_task, u_null separately for projection leak analysis."""
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    J_pinv = jnp.linalg.pinv(J_ee)

    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)
    u_task = J_pinv @ v_task

    N_mat = jnp.eye(N_DOF) - J_pinv @ J_ee
    ig_grad = jax.grad(lambda qi: _ig_at_q(qi, theta_est, P_theta))(q)
    u_epis = ALPHA_NS * N_mat @ ig_grad

    u = jnp.clip(u_task + u_epis, -U_MAX, U_MAX)
    # u_epis is the raw null-space component (before clipping).
    # Do NOT use (u - u_task): u_task is unclipped and can be huge near singularities,
    # making that difference meaningless as a null-motion energy metric.
    return u, u_task, u_epis


def compute_vfe_only_action_diag(q, theta_est, y_goal):
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    J_pinv = jnp.linalg.pinv(J_ee)
    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)
    u = jnp.clip(J_pinv @ v_task, -U_MAX, U_MAX)
    return u, u, jnp.zeros(N_DOF)


# ---------------------------------------------------------------------------
# Single run with full diagnostics
# ---------------------------------------------------------------------------

def run_one_diag(seed, condition):
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(N_DOF)
    estep = _build_estep(theta_est)

    q = Q0.copy()
    u = jnp.zeros(N_DOF)

    rmse_hist = []
    task_err_hist = []
    ee_hold_err_hist = []
    p_theta_rank_hist = []
    p_theta_min_eig_hist = []
    projection_leak_hist = []   # ||J_ee @ u_null|| per step
    action_energy_hist = []     # ||u||^2 per step
    null_energy_hist = []       # ||u_null||^2 per step

    theta_frozen = None
    q_at_phase1_end = None

    q_hist, v_hist, y_hist = [], [], []

    for t in range(N_STEPS):
        # Metrics
        rmse_hist.append(float(jnp.sqrt(jnp.mean((theta_est - THETA_TRUE) ** 2))))
        ee_true = fk(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_TASK) ** 2))))
        ee_hold_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_HOLD) ** 2))))

        data_fim = P_theta - PARAMS_PRIOR_PI * jnp.eye(N_DOF)
        data_diag = _matrix_diag(data_fim, eps=1e-3)
        p_theta_rank_hist.append(data_diag["rank"])
        p_theta_min_eig_hist.append(data_diag["min_eig"])

        # Action
        if t < CHANGE_STEP:
            if condition == "null_space":
                u, u_task, u_null = compute_null_space_action_diag(
                    q, theta_est, P_theta, Y_GOAL_HOLD)
            else:
                u, u_task, u_null = compute_vfe_only_action_diag(
                    q, theta_est, Y_GOAL_HOLD)

            # Projection leak: how much does u_null move the EE?
            J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
            leak = float(jnp.sqrt(jnp.sum((J_ee @ u_null) ** 2)))
            projection_leak_hist.append(leak)
            action_energy_hist.append(float(jnp.sum(u ** 2)))
            null_energy_hist.append(float(jnp.sum(u_null ** 2)))

        else:
            if theta_frozen is None:
                theta_frozen = theta_est
                q_at_phase1_end = np.array(q)
            u, _, _ = compute_vfe_only_action_diag(q, theta_frozen, Y_GOAL_TASK)
            projection_leak_hist.append(0.0)
            action_energy_hist.append(float(jnp.sum(u ** 2)))
            null_energy_hist.append(0.0)

        # Step
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

    theta_end_ph1 = theta_frozen if theta_frozen is not None else theta_est
    theta_error = np.array(theta_end_ph1) - np.array(THETA_TRUE)

    p_final = _matrix_diag(P_theta)

    rmse_arr = np.array(rmse_hist)
    task_arr = np.array(task_err_hist)
    ee_arr = np.array(ee_hold_err_hist[:CHANGE_STEP])
    leak_arr = np.array(projection_leak_hist[:CHANGE_STEP])
    energy_arr = np.array(action_energy_hist[:CHANGE_STEP])
    null_e_arr = np.array(null_energy_hist[:CHANGE_STEP])

    return {
        "seed": seed,
        "condition": condition,
        # Scalars
        "rmse_at_change": float(rmse_arr[CHANGE_STEP - 1]),
        "task_err_final": float(task_arr[-1]),
        "task_err_at_phase2_start": float(task_arr[CHANGE_STEP]),
        "ee_hold_err_phase1_mean": float(np.mean(ee_arr)),
        "ee_hold_err_phase1_max": float(np.max(ee_arr)),
        "projection_leak_mean": float(np.mean(leak_arr)),
        "projection_leak_max": float(np.max(leak_arr)),
        "action_energy_phase1_mean": float(np.mean(energy_arr)),
        "null_motion_energy_mean": float(np.mean(null_e_arr)),
        "p_theta_rank_at_change": int(p_theta_rank_hist[CHANGE_STEP - 1]),
        "p_theta_min_eig_at_change": float(p_theta_min_eig_hist[CHANGE_STEP - 1]),
        "p_theta_min_eig_final": p_final["min_eig"],
        "p_theta_cond": p_final["cond"],
        "theta_error_vector": [float(x) for x in theta_error],
        "theta_final_ph1": [float(x) for x in theta_end_ph1],
        "q_at_phase1_end": q_at_phase1_end.tolist() if q_at_phase1_end is not None else None,
        # Time series (for trajectory plots)
        "rmse_hist": rmse_arr.tolist(),
        "task_err_hist": task_arr.tolist(),
        "ee_hold_err_hist": ee_arr.tolist(),
        "projection_leak_hist": leak_arr.tolist(),
    }


def classify_failure(r):
    rmse_ch = r["rmse_at_change"]
    task_f = r["task_err_final"]
    prank = r["p_theta_rank_at_change"]
    ee_hold = r["ee_hold_err_phase1_mean"]

    if rmse_ch > PARAM_RMSE_FAIL:
        return "estimation_failure"
    elif task_f > TASK_ERR_FAIL:
        if prank < N_DOF:
            return "incomplete_identifiability"
        elif ee_hold > 0.05:
            return "ee_drift"
        else:
            return "task_control"
    else:
        return "success"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = []

    for cond in CONDITIONS_DIAG:
        print(f"Running diagnostics for '{cond}'", flush=True)
        for seed in range(N_SEEDS):
            print(f"  seed={seed}", flush=True)
            r = run_one_diag(seed, cond)
            r["failure_mode"] = classify_failure(r)
            all_results.append(r)

    # -----------------------------------------------------------------------
    # Save full diagnostics JSON (without long time series to keep size small)
    # -----------------------------------------------------------------------
    out_json = project_root / "results" / "redundant_arm_diagnostics.json"
    # Save scalar fields only (strip time series for the main JSON)
    scalar_keys = [k for k in all_results[0] if k not in
                   ("rmse_hist", "task_err_hist", "ee_hold_err_hist", "projection_leak_hist")]
    json_data = [{k: r[k] for k in scalar_keys} for r in all_results]
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved diagnostics → {out_json}")

    # -----------------------------------------------------------------------
    # Summary statistics by failure mode
    # -----------------------------------------------------------------------
    modes = ["success", "task_control", "estimation_failure",
             "incomplete_identifiability", "ee_drift"]
    mode_results = {m: [r for r in all_results
                        if r["failure_mode"] == m and r["condition"] == "null_space"]
                    for m in modes}
    mode_counts = {m: len(mode_results[m]) for m in modes if mode_results[m]}

    print("\n=== null_space failure mode breakdown ===")
    for m, cnt in mode_counts.items():
        print(f"  {m}: {cnt}/{N_SEEDS} ({cnt/N_SEEDS:.0%})")

    def mode_stat(mode, key):
        vals = [r[key] for r in mode_results.get(mode, [])]
        if not vals:
            return None
        return {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    summary = {
        "n_seeds": N_SEEDS,
        "failure_mode_counts": mode_counts,
        "by_mode": {}
    }
    diag_keys = [
        "rmse_at_change", "task_err_final", "task_err_at_phase2_start",
        "ee_hold_err_phase1_mean", "ee_hold_err_phase1_max",
        "projection_leak_mean", "projection_leak_max",
        "p_theta_rank_at_change", "p_theta_min_eig_at_change",
        "action_energy_phase1_mean", "null_motion_energy_mean",
    ]
    for m in mode_counts:
        summary["by_mode"][m] = {k: mode_stat(m, k) for k in diag_keys}

    out_summary = project_root / "results" / "redundant_arm_diagnostics_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved diagnostics summary → {out_summary}")

    # Print key stats
    print("\n=== Diagnostic stats by mode ===")
    print(f"{'Mode':25s}  {'RMSE@150':8s}  {'TaskErr':8s}  "
          f"{'EEhold_mean':11s}  {'Leak_mean':9s}  {'Rank':5s}  {'MinEig':7s}")
    for m in mode_counts:
        s = summary["by_mode"][m]
        def v(key):
            x = s.get(key)
            return f"{x['median']:.4f}" if x else "  ---  "
        print(f"  {m:23s}  {v('rmse_at_change')}  {v('task_err_final')}  "
              f"{v('ee_hold_err_phase1_mean')}  {v('projection_leak_mean')}  "
              f"{v('p_theta_rank_at_change')}  {v('p_theta_min_eig_at_change')}")

    # -----------------------------------------------------------------------
    # Diagnostic figure
    # -----------------------------------------------------------------------
    MODE_COLORS = {
        "success": "C2",
        "task_control": "C1",
        "estimation_failure": "C3",
        "incomplete_identifiability": "C7",
        "ee_drift": "C5",
    }
    MODE_LABELS = {
        "success": "Success",
        "task_control": "Task control fail",
        "estimation_failure": "Estimation fail",
        "incomplete_identifiability": "Incomplete ident.",
        "ee_drift": "EE drift",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Redundant arm (4-DOF): null_space failure diagnostics\n"
        f"N={N_SEEDS} seeds, conditions: {CONDITIONS_DIAG}",
        fontsize=10,
    )

    present_modes = [m for m in modes if mode_results[m]]

    # (A) RMSE@150 vs TaskErr@200 scatter
    ax = axes[0, 0]
    for m in present_modes:
        xs = [r["rmse_at_change"] for r in mode_results[m]]
        ys = [r["task_err_final"] for r in mode_results[m]]
        ax.scatter(xs, ys, c=MODE_COLORS[m], label=f"{MODE_LABELS[m]} (n={len(xs)})",
                   s=40, alpha=0.8, edgecolors="none")
    ax.axvline(PARAM_RMSE_FAIL, color="gray", linestyle="--", linewidth=0.8, label="RMSE threshold")
    ax.axhline(TASK_ERR_FAIL, color="gray", linestyle=":", linewidth=0.8, label="Task threshold")
    ax.set_xlabel("RMSE at phase change")
    ax.set_ylabel("Task error at final step")
    ax.set_title("(A) RMSE vs Task error")
    ax.legend(fontsize=7)

    # (B) EE hold error vs TaskErr
    ax = axes[0, 1]
    for m in present_modes:
        xs = [r["ee_hold_err_phase1_mean"] for r in mode_results[m]]
        ys = [r["task_err_final"] for r in mode_results[m]]
        ax.scatter(xs, ys, c=MODE_COLORS[m], label=MODE_LABELS[m], s=40, alpha=0.8,
                   edgecolors="none")
    ax.axhline(TASK_ERR_FAIL, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("EE hold error (Phase 1 mean)")
    ax.set_ylabel("Task error at final step")
    ax.set_title("(B) EE drift vs Task error")
    ax.legend(fontsize=7)

    # (C) Projection leak vs TaskErr
    ax = axes[0, 2]
    for m in present_modes:
        xs = [r["projection_leak_mean"] for r in mode_results[m]]
        ys = [r["task_err_final"] for r in mode_results[m]]
        ax.scatter(xs, ys, c=MODE_COLORS[m], label=MODE_LABELS[m], s=40, alpha=0.8,
                   edgecolors="none")
    ax.axhline(TASK_ERR_FAIL, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Projection leak: ||J_ee u_null|| (Phase 1 mean)")
    ax.set_ylabel("Task error at final step")
    ax.set_title("(C) Projection leak vs Task error")
    ax.legend(fontsize=7)

    # (D) FIM min eigenvalue vs RMSE
    ax = axes[1, 0]
    for m in present_modes:
        xs = [r["p_theta_min_eig_at_change"] for r in mode_results[m]]
        ys = [r["rmse_at_change"] for r in mode_results[m]]
        ax.scatter(xs, ys, c=MODE_COLORS[m], label=MODE_LABELS[m], s=40, alpha=0.8,
                   edgecolors="none")
    ax.axhline(PARAM_RMSE_FAIL, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("P_theta min eigenvalue at phase change")
    ax.set_ylabel("RMSE at phase change")
    ax.set_title("(D) FIM min eig vs RMSE")
    ax.legend(fontsize=7)

    # (E) Median RMSE trajectory: success vs task_control failure
    ax = axes[1, 1]
    steps = np.arange(CHANGE_STEP)
    for m in ["success", "task_control"]:
        if not mode_results.get(m):
            continue
        mat = np.array([r["rmse_hist"][:CHANGE_STEP] for r in mode_results[m]])
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(steps, med, color=MODE_COLORS[m], label=f"{MODE_LABELS[m]} (n={len(mode_results[m])})")
        ax.fill_between(steps, q25, q75, color=MODE_COLORS[m], alpha=0.2)
    ax.set_xlabel("Step (Phase 1)")
    ax.set_ylabel("RMSE(θ)")
    ax.set_title("(E) RMSE trajectory: success vs task_control fail")
    ax.set_yscale("log")
    ax.legend(fontsize=7)

    # (F) Median EE hold error: success vs task_control failure
    ax = axes[1, 2]
    for m in ["success", "task_control"]:
        if not mode_results.get(m):
            continue
        mat = np.array([r["ee_hold_err_hist"] for r in mode_results[m]])
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(np.arange(CHANGE_STEP), med, color=MODE_COLORS[m],
                label=f"{MODE_LABELS[m]} (n={len(mode_results[m])})")
        ax.fill_between(np.arange(CHANGE_STEP), q25, q75, color=MODE_COLORS[m], alpha=0.2)
    ax.set_xlabel("Step (Phase 1)")
    ax.set_ylabel("EE hold error (m)")
    ax.set_title("(F) EE hold error: success vs task_control fail")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_fig = project_root / "results" / "redundant_arm_diagnostics.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
