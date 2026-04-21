"""Noise sweep for 2-DOF 1D-observation experiment.

Sweeps sigma_obs across 7 levels with 3 conditions and N_SEEDS=20.
Theta_init is fixed at the standard degenerate setting (scale=1.0).

Conditions:
  vfe_only               : task-only control
  dual_weak              : dual control lambda=0.5
  fim_greedy_cost_matched: D-optimal greedy at fixed energy budget

Sigma sweep:
  {0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10}

Bootstrap 95% CI is computed for failure rate across seeds.

Output:
  results/noise_sweep_2d.json
  results/noise_sweep_2d.png
  results/bootstrap_ci_2d.json

Usage:
    uv run python experiments/noise_sweep_2d.py
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

# ---------------------------------------------------------------------------
# Fixed system parameters (identical to dual_control_1d_obs.py)
# ---------------------------------------------------------------------------
THETA_TRUE = jnp.array([0.5, 0.5])
THETA_INIT = jnp.array([0.9, 0.1])   # standard degenerate init (scale=1.0)
Q0 = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET_2 = jnp.array([jnp.pi / 6, jnp.pi / 3])

DT = 0.05
N_STEPS = 100
CHANGE_STEP = 50
N_SEEDS = 20
U_MAX = 0.8
LR_ACTION = 0.05
N_ACTION_ITER = 30

PI_X = 1.0
PARAMS_PRIOR_PI = 1.0
KAPPA_P = 0.5
N_ESTEP_ITER = 3

LAMBDA_WEAK = 0.5
FIM_GREEDY_CM_TARGET = 0.667

PARAM_RMSE_FAILURE_THRESHOLD = 0.10
TASK_ERR_FAILURE_THRESHOLD = 0.05

# Noise sweep axis
SIGMA_OBS_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
CONDITIONS = ["vfe_only", "dual_weak", "fim_greedy_cost_matched"]

# Precompute candidate directions for fim_greedy_cost_matched
_N_ANGLES = 60
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _N_ANGLES, endpoint=False)
_TARGET_NORM = float(np.sqrt(FIM_GREEDY_CM_TARGET))
_U_CANDIDATES = jnp.array(
    np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1) * _TARGET_NORM
).clip(-U_MAX, U_MAX)

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def fk_2d(q, theta):
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    y = l1 * jnp.sin(q[0]) + l2 * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])

def fk_1d(q, theta):
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    return jnp.array([x])

Y_GOAL = fk_1d(Q_TARGET, THETA_TRUE)
Y_GOAL_2D = fk_2d(Q_TARGET_2, THETA_TRUE)

# ---------------------------------------------------------------------------
# FIM / IG helpers
# ---------------------------------------------------------------------------
def rollout_step(q, u):
    return jnp.clip(q + u * DT, -jnp.pi, jnp.pi)

def rollout(q0, u, n_steps=5):
    def step(q, _):
        q_next = rollout_step(q, u)
        return q_next, q_next
    _, qs = jax.lax.scan(step, q0, None, length=n_steps)
    return qs

def compute_fim(q, u, theta, pi_y):
    def y_future_fn(th):
        qs = rollout(q, u, n_steps=5)
        return jnp.concatenate([fk_1d(qi, th) for qi in qs])
    J = jax.jacfwd(y_future_fn)(theta)
    R_inv = jnp.eye(5) * pi_y
    return J.T @ R_inv @ J

def compute_info_gain(P_theta, fim):
    sign1, ld1 = jnp.linalg.slogdet(P_theta + fim)
    sign0, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)

# ---------------------------------------------------------------------------
# Per-sigma JIT-compiled functions
# ---------------------------------------------------------------------------
PI_Y_TASK = 1.0 / 0.05**2

def make_optimize_action(pi_y):
    @jax.jit
    def optimize_action(q, theta_est, P_theta, y_goal, lambda_eff, u_init):
        def objective(u):
            q_pred = rollout_step(q, u)
            y_pred = fk_1d(q_pred, theta_est)
            vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)
            fim = compute_fim(q, u, theta_est, pi_y)
            ig = compute_info_gain(P_theta, fim)
            return vfe - lambda_eff * ig
        def descent_step(u, _):
            g = jax.grad(objective)(u)
            return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None
        u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
        return u_opt
    return optimize_action

def make_fim_greedy_cm(pi_y):
    @jax.jit
    def _fim_greedy_cm(q, theta, u_candidates):
        def score_one(u_cand):
            fim = compute_fim(q, u_cand, theta, pi_y)
            sign, logdet = jnp.linalg.slogdet(fim + 1e-6 * jnp.eye(2))
            return jnp.where(sign > 0, logdet, -1e9)
        scores = jax.vmap(score_one)(u_candidates)
        return u_candidates[jnp.argmax(scores)]
    return _fim_greedy_cm

@jax.jit
def optimize_task_action_2d(q, theta_est, y_goal_2d, u_init):
    def objective(u):
        q_pred = rollout_step(q, u)
        y_pred = fk_2d(q_pred, theta_est)
        return 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal_2d) ** 2)
    def descent_step(u, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None
    u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
    return u_opt

def _build_estep(theta_init, pi_y):
    model = DEMModel(
        f=lambda x, v, p: jnp.zeros(2),
        g=lambda x, v, p: fk_1d(x, p),
        n_x=2, n_v=1, n_y=1, n_order=1,
        pi_y=pi_y, pi_x=PI_X,
        params=theta_init,
        params_prior_pi=PARAMS_PRIOR_PI,
    )
    return EStep(model, kappa_p=KAPPA_P, use_gauss_newton=True)

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(seed, condition, sigma_obs, optimize_action_fn, fim_greedy_cm_fn):
    pi_y = 1.0 / sigma_obs**2
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(2)
    estep = _build_estep(theta_est, pi_y)

    q = Q0.copy()
    u = jnp.zeros(2)
    q_hist, v_hist, y_hist = [], [], []
    action_energy_phase1 = []

    for t in range(N_STEPS):
        if t < CHANGE_STEP:
            if condition == "vfe_only":
                u = optimize_action_fn(q, theta_est, P_theta, Y_GOAL, 0.0, u)
            elif condition == "dual_weak":
                u = optimize_action_fn(q, theta_est, P_theta, Y_GOAL, LAMBDA_WEAK, u)
            elif condition == "fim_greedy_cost_matched":
                u = fim_greedy_cm_fn(q, theta_est, _U_CANDIDATES)

            action_energy_phase1.append(float(jnp.sum(u**2)))
            q = rollout_step(q, u)
            y_obs = fk_1d(q, THETA_TRUE) + rng.normal(0, sigma_obs, size=(1,))
            q_hist.append(q)
            v_hist.append(jnp.zeros(1))
            y_hist.append(jnp.array(y_obs))

            if t > 5 and t % 5 == 0:
                theta_est = estep.run(q_hist, v_hist, y_hist, theta_est, n_iter=N_ESTEP_ITER)
                theta_est = jnp.clip(theta_est, 0.05, 2.0)
                P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
                estep = _build_estep(theta_est, pi_y)
        else:
            u = optimize_task_action_2d(q, theta_est, Y_GOAL_2D, u)
            q = rollout_step(q, u)

    err = theta_est - THETA_TRUE
    rmse_at_change = float(jnp.sqrt(jnp.mean((theta_est - THETA_TRUE) ** 2)))
    ee_true = fk_2d(q, THETA_TRUE)
    task_err_final = float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_2D) ** 2)))
    energy_median = float(np.median(action_energy_phase1)) if action_energy_phase1 else 0.0

    return {
        "rmse_at_change": rmse_at_change,
        "task_err_final": task_err_final,
        "action_energy_phase1": energy_median,
    }

# ---------------------------------------------------------------------------
# Bootstrap CI for failure rate
# ---------------------------------------------------------------------------
def bootstrap_ci_failure_rate(failures, n_bootstrap=2000, ci=95):
    """Return (mean, ci_low, ci_high) for binary failure rate via bootstrap."""
    failures = np.asarray(failures, dtype=float)
    n = len(failures)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(failures, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "mean": float(np.mean(failures)),
        "ci_low": float(np.percentile(boot_means, lo)),
        "ci_high": float(np.percentile(boot_means, hi)),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # results_raw[sigma][cond] = list of per-seed dicts
    results_raw = {}
    for sigma in SIGMA_OBS_VALUES:
        print(f"\n--- sigma_obs={sigma} ---", flush=True)
        pi_y = 1.0 / sigma**2
        optimize_action_fn = make_optimize_action(pi_y)
        fim_greedy_cm_fn = make_fim_greedy_cm(pi_y)

        results_raw[sigma] = {}
        for cond in CONDITIONS:
            results_raw[sigma][cond] = []
            for seed in range(N_SEEDS):
                r = run_one(seed, cond, sigma, optimize_action_fn, fim_greedy_cm_fn)
                results_raw[sigma][cond].append(r)
            rmse_vals = np.array([r["rmse_at_change"] for r in results_raw[sigma][cond]])
            task_vals = np.array([r["task_err_final"] for r in results_raw[sigma][cond]])
            fail_r = float(np.mean(rmse_vals > PARAM_RMSE_FAILURE_THRESHOLD))
            fail_t = float(np.mean(task_vals > TASK_ERR_FAILURE_THRESHOLD))
            print(f"  {cond:30s}: failRMSE={fail_r:.0%}  failTask={fail_t:.0%}", flush=True)

    # -----------------------------------------------------------------------
    # Compute summary statistics
    # -----------------------------------------------------------------------
    summary = {}
    for sigma in SIGMA_OBS_VALUES:
        summary[str(sigma)] = {}
        for cond in CONDITIONS:
            rmse_vals = np.array([r["rmse_at_change"] for r in results_raw[sigma][cond]])
            task_vals = np.array([r["task_err_final"] for r in results_raw[sigma][cond]])
            energy_vals = np.array([r["action_energy_phase1"] for r in results_raw[sigma][cond]])
            fail_rmse = rmse_vals > PARAM_RMSE_FAILURE_THRESHOLD
            fail_task = task_vals > TASK_ERR_FAILURE_THRESHOLD
            summary[str(sigma)][cond] = {
                "rmse_at_change_median": float(np.median(rmse_vals)),
                "rmse_at_change_q25": float(np.percentile(rmse_vals, 25)),
                "rmse_at_change_q75": float(np.percentile(rmse_vals, 75)),
                "task_err_final_median": float(np.median(task_vals)),
                "task_err_final_q25": float(np.percentile(task_vals, 25)),
                "task_err_final_q75": float(np.percentile(task_vals, 75)),
                "action_energy_median": float(np.median(energy_vals)),
                "rmse_failure_rate": float(np.mean(fail_rmse)),
                "task_failure_rate": float(np.mean(fail_task)),
                "n_seeds": N_SEEDS,
            }

    out_json = project_root / "results" / "noise_sweep_2d.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_json}")

    # -----------------------------------------------------------------------
    # Bootstrap CI for task failure rate
    # -----------------------------------------------------------------------
    bootstrap_ci = {}
    for sigma in SIGMA_OBS_VALUES:
        bootstrap_ci[str(sigma)] = {}
        for cond in CONDITIONS:
            task_vals = np.array([r["task_err_final"] for r in results_raw[sigma][cond]])
            fail_task = (task_vals > TASK_ERR_FAILURE_THRESHOLD).astype(float)
            bootstrap_ci[str(sigma)][cond] = bootstrap_ci_failure_rate(fail_task)

    out_ci = project_root / "results" / "bootstrap_ci_2d.json"
    with open(out_ci, "w") as f:
        json.dump(bootstrap_ci, f, indent=2)
    print(f"Saved → {out_ci}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    COLORS = {
        "vfe_only":                "C3",
        "dual_weak":               "C1",
        "fim_greedy_cost_matched": "C0",
    }
    LABELS = {
        "vfe_only":                "VFE-only",
        "dual_weak":               "Dual (λ=0.5)",
        "fim_greedy_cost_matched": "FIM-greedy\n(cost-matched)",
    }
    MARKERS = {
        "vfe_only":                "o",
        "dual_weak":               "s",
        "fim_greedy_cost_matched": "^",
    }

    sigma_vals = SIGMA_OBS_VALUES
    x = np.array(sigma_vals)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(
        f"Fig. 4 – Noise sweep: 2-DOF 1D-obs (N={N_SEEDS} seeds per point)",
        fontsize=9, y=1.01,
    )

    ax_rmse, ax_task, ax_fail = axes

    for cond in CONDITIONS:
        rmse_med = [summary[str(s)][cond]["rmse_at_change_median"] for s in sigma_vals]
        rmse_q25 = [summary[str(s)][cond]["rmse_at_change_q25"] for s in sigma_vals]
        rmse_q75 = [summary[str(s)][cond]["rmse_at_change_q75"] for s in sigma_vals]
        task_med = [summary[str(s)][cond]["task_err_final_median"] for s in sigma_vals]
        fail_rate = [summary[str(s)][cond]["task_failure_rate"] for s in sigma_vals]
        fail_ci_lo = [bootstrap_ci[str(s)][cond]["ci_low"] for s in sigma_vals]
        fail_ci_hi = [bootstrap_ci[str(s)][cond]["ci_high"] for s in sigma_vals]

        color = COLORS[cond]
        marker = MARKERS[cond]
        label = LABELS[cond]

        ax_rmse.plot(x, rmse_med, color=color, marker=marker, label=label, linewidth=1.5, markersize=5)
        ax_rmse.fill_between(x, rmse_q25, rmse_q75, color=color, alpha=0.15)

        ax_task.plot(x, task_med, color=color, marker=marker, label=label, linewidth=1.5, markersize=5)

        ax_fail.plot(x, fail_rate, color=color, marker=marker, label=label, linewidth=1.5, markersize=5)
        ax_fail.fill_between(x, fail_ci_lo, fail_ci_hi, color=color, alpha=0.15)

    for ax in axes:
        ax.set_xlabel("σ_obs (observation noise std)")
        ax.set_xticks(sigma_vals)
        ax.set_xticklabels([str(s) for s in sigma_vals], rotation=30, ha="right", fontsize=6.5)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)

    ax_rmse.set_ylabel("RMSE at phase change (median ± IQR)")
    ax_rmse.set_title("(a) Parameter RMSE")

    ax_task.set_ylabel("Task error at step 100 (median)")
    ax_task.set_title("(b) Task error")

    ax_fail.set_ylabel("Task failure rate (bootstrap 95% CI)")
    ax_fail.set_title("(c) Failure rate")
    ax_fail.set_ylim(0, 1.05)

    fig.tight_layout()
    out_png = project_root / "results" / "noise_sweep_2d.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_png}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n=== Noise sweep summary (task failure rate) ===")
    header = f"  {'sigma':8s}" + "".join(f"  {c:30s}" for c in CONDITIONS)
    print(header)
    for sigma in sigma_vals:
        row = f"  {sigma:<8.3f}"
        for cond in CONDITIONS:
            fr = summary[str(sigma)][cond]["task_failure_rate"]
            ci = bootstrap_ci[str(sigma)][cond]
            row += f"  {fr:.0%} [{ci['ci_low']:.0%}–{ci['ci_high']:.0%}]          "
        print(row)


if __name__ == "__main__":
    main()
