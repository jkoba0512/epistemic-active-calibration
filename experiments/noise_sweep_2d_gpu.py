"""Noise sweep for 2-DOF 1D-observation experiment (GPU / vmap version).

Pure-JAX reimplementation of experiments/noise_sweep_2d.py that vmap's over
seeds to parallelize on GPU. Numerical setup and constants match the CPU
reference; only the RNG backend differs (jax.random vs numpy), so per-seed
values will differ but ensemble statistics should match.

Simplifications valid for this experiment's model:
  - n_order=1, so D*x_tilde = 0 and the eps_x term vanishes (f=0).
  - tilde_Pi_y reduces to pi_y * I(1).
  - E-step therefore becomes standard Gauss-Newton least-squares on y.

Output:
  results/noise_sweep_2d_gpu.json
  results/noise_sweep_2d_gpu.png
  results/bootstrap_ci_2d_gpu.json

Usage:
    uv run python experiments/noise_sweep_2d_gpu.py
"""

import sys
import json
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

print(f"JAX devices: {jax.devices()}", flush=True)

# ---------------------------------------------------------------------------
# Fixed system parameters (identical to noise_sweep_2d.py)
# ---------------------------------------------------------------------------
THETA_TRUE = jnp.array([0.5, 0.5])
THETA_INIT = jnp.array([0.9, 0.1])
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

PARAMS_PRIOR_PI = 1.0
N_ESTEP_ITER = 3

LAMBDA_WEAK = 0.5
FIM_GREEDY_CM_TARGET = 0.667

PARAM_RMSE_FAILURE_THRESHOLD = 0.10
TASK_ERR_FAILURE_THRESHOLD = 0.05

SIGMA_OBS_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
CONDITIONS = ["vfe_only", "dual_weak", "fim_greedy_cost_matched"]

PI_Y_TASK = 1.0 / 0.05**2

MODE_VFE = 0
MODE_DUAL_WEAK = 1
MODE_FIM_GREEDY = 2
COND_TO_MODE = {
    "vfe_only": MODE_VFE,
    "dual_weak": MODE_DUAL_WEAK,
    "fim_greedy_cost_matched": MODE_FIM_GREEDY,
}

_N_ANGLES = 60
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _N_ANGLES, endpoint=False)
_TARGET_NORM = float(np.sqrt(FIM_GREEDY_CM_TARGET))
U_CANDIDATES = jnp.array(
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
# Rollout / FIM / IG
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
    _, ld1 = jnp.linalg.slogdet(P_theta + fim)
    _, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)


# ---------------------------------------------------------------------------
# Action optimizers
# ---------------------------------------------------------------------------
def optimize_action_dual(q, theta_est, P_theta, y_goal, lambda_eff, u_init, pi_y):
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


def fim_greedy_cm(q, theta, pi_y):
    def score_one(u_cand):
        fim = compute_fim(q, u_cand, theta, pi_y)
        sign, logdet = jnp.linalg.slogdet(fim + 1e-6 * jnp.eye(2))
        return jnp.where(sign > 0, logdet, -1e9)
    scores = jax.vmap(score_one)(U_CANDIDATES)
    return U_CANDIDATES[jnp.argmax(scores)]


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


# ---------------------------------------------------------------------------
# E-step (inlined for n_order=1, f=0): pure Gauss-Newton least-squares on y
# ---------------------------------------------------------------------------
def gn_contrib(q, y, theta, pi_y):
    J_g = jax.jacobian(lambda p: fk_1d(q, p))(theta)   # (1, 2)
    J_y = -J_g                                          # J of eps_y = y - g
    eps_y = y - fk_1d(q, theta)                         # (1,)
    dFdp = J_y.T @ (pi_y * eps_y)                       # (2,)
    dFdpp = J_y.T @ (pi_y * J_y)                        # (2, 2)
    return dFdp, dFdpp


def estep_run(theta_start, theta_prior_mean, q_hist, y_hist, mask, pi_y, n_iter):
    def iter_step(theta, _):
        dFdp_all, dFdpp_all = jax.vmap(gn_contrib, in_axes=(0, 0, None, None))(
            q_hist, y_hist, theta, pi_y
        )
        dFdp = jnp.sum(dFdp_all * mask[:, None], axis=0)
        dFdpp = jnp.sum(dFdpp_all * mask[:, None, None], axis=0)
        prior_grad = PARAMS_PRIOR_PI * (theta - theta_prior_mean)
        prior_curv = PARAMS_PRIOR_PI * jnp.eye(2)
        total_dFdp = dFdp + prior_grad
        total_dFdpp = dFdpp + prior_curv
        dp = jnp.linalg.solve(total_dFdpp, total_dFdp)
        return theta - dp, None

    theta_new, _ = jax.lax.scan(iter_step, theta_start, None, length=n_iter)
    return theta_new


def compute_precision(q_hist, y_hist, mask, theta, pi_y):
    _, dFdpp_all = jax.vmap(gn_contrib, in_axes=(0, 0, None, None))(
        q_hist, y_hist, theta, pi_y
    )
    dFdpp = jnp.sum(dFdpp_all * mask[:, None, None], axis=0)
    return dFdpp + PARAMS_PRIOR_PI * jnp.eye(2)


# ---------------------------------------------------------------------------
# Single run (fully JAX-compatible)
# ---------------------------------------------------------------------------
def run_one(key, sigma_obs, action_mode):
    pi_y = 1.0 / sigma_obs**2

    noise_all = jax.random.normal(key, (N_STEPS, 1)) * sigma_obs

    q_hist0 = jnp.zeros((N_STEPS, 2))
    y_hist0 = jnp.zeros((N_STEPS, 1))

    def phase1_body(carry, t):
        q, u_prev, theta_est, theta_prior, P_theta, q_hist, y_hist = carry

        def vfe_branch(_):
            return optimize_action_dual(q, theta_est, P_theta, Y_GOAL, 0.0, u_prev, pi_y)

        def dual_weak_branch(_):
            return optimize_action_dual(q, theta_est, P_theta, Y_GOAL, LAMBDA_WEAK, u_prev, pi_y)

        def fim_branch(_):
            return fim_greedy_cm(q, theta_est, pi_y)

        u = jax.lax.switch(action_mode, [vfe_branch, dual_weak_branch, fim_branch], None)

        energy_t = jnp.sum(u**2)
        q_new = rollout_step(q, u)
        y_obs = fk_1d(q_new, THETA_TRUE) + noise_all[t]
        q_hist = q_hist.at[t].set(q_new)
        y_hist = y_hist.at[t].set(y_obs)

        mask = (jnp.arange(N_STEPS) <= t).astype(jnp.float64)
        do_estep = (t > 5) & (t % 5 == 0)

        def run_estep(_):
            theta_new = estep_run(
                theta_est, theta_prior, q_hist, y_hist, mask, pi_y, N_ESTEP_ITER
            )
            theta_new = jnp.clip(theta_new, 0.05, 2.0)
            P_new = compute_precision(q_hist, y_hist, mask, theta_new, pi_y)
            return theta_new, theta_new, P_new

        def skip_estep(_):
            return theta_est, theta_prior, P_theta

        theta_new, prior_new, P_new = jax.lax.cond(do_estep, run_estep, skip_estep, None)

        return (q_new, u, theta_new, prior_new, P_new, q_hist, y_hist), energy_t

    carry0 = (
        Q0,
        jnp.zeros(2),
        THETA_INIT,
        THETA_INIT,
        PARAMS_PRIOR_PI * jnp.eye(2),
        q_hist0,
        y_hist0,
    )
    carry_p1, energies_p1 = jax.lax.scan(
        phase1_body, carry0, jnp.arange(CHANGE_STEP)
    )
    q_p1, u_p1, theta_p1, _, _, _, _ = carry_p1

    rmse_at_change = jnp.sqrt(jnp.mean((theta_p1 - THETA_TRUE) ** 2))
    energy_median = jnp.median(energies_p1)

    def phase2_body(carry, _):
        q, u_prev = carry
        u = optimize_task_action_2d(q, theta_p1, Y_GOAL_2D, u_prev)
        q_new = rollout_step(q, u)
        return (q_new, u), None

    (q_end, _), _ = jax.lax.scan(
        phase2_body, (q_p1, u_p1), None, length=N_STEPS - CHANGE_STEP
    )
    ee_true = fk_2d(q_end, THETA_TRUE)
    task_err_final = jnp.sqrt(jnp.sum((ee_true - Y_GOAL_2D) ** 2))

    return rmse_at_change, task_err_final, energy_median


# Vmap over seeds + JIT
run_many = jax.jit(jax.vmap(run_one, in_axes=(0, None, None)))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci_failure_rate(failures, n_bootstrap=2000, ci=95):
    failures = np.asarray(failures, dtype=float)
    n = len(failures)
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [np.mean(rng.choice(failures, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
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

    t_start = time.time()

    base_key = jax.random.PRNGKey(0)
    keys = jax.random.split(base_key, N_SEEDS)

    results_raw = {}
    for sigma in SIGMA_OBS_VALUES:
        print(f"\n--- sigma_obs={sigma} ---", flush=True)
        results_raw[sigma] = {}
        for cond in CONDITIONS:
            mode = jnp.int32(COND_TO_MODE[cond])
            t0 = time.time()
            rmse_arr, task_arr, energy_arr = run_many(keys, jnp.float64(sigma), mode)
            rmse_arr = np.asarray(jax.block_until_ready(rmse_arr))
            task_arr = np.asarray(task_arr)
            energy_arr = np.asarray(energy_arr)
            dt = time.time() - t0

            fail_r = float(np.mean(rmse_arr > PARAM_RMSE_FAILURE_THRESHOLD))
            fail_t = float(np.mean(task_arr > TASK_ERR_FAILURE_THRESHOLD))
            print(
                f"  {cond:30s}: failRMSE={fail_r:.0%}  failTask={fail_t:.0%}  ({dt:.2f}s)",
                flush=True,
            )

            results_raw[sigma][cond] = {
                "rmse": rmse_arr,
                "task": task_arr,
                "energy": energy_arr,
            }

    print(f"\nTotal compute time: {time.time() - t_start:.2f} s", flush=True)

    # Summary stats
    summary = {}
    for sigma in SIGMA_OBS_VALUES:
        summary[str(sigma)] = {}
        for cond in CONDITIONS:
            d = results_raw[sigma][cond]
            rmse_vals = d["rmse"]
            task_vals = d["task"]
            energy_vals = d["energy"]
            summary[str(sigma)][cond] = {
                "rmse_at_change_median": float(np.median(rmse_vals)),
                "rmse_at_change_q25": float(np.percentile(rmse_vals, 25)),
                "rmse_at_change_q75": float(np.percentile(rmse_vals, 75)),
                "task_err_final_median": float(np.median(task_vals)),
                "task_err_final_q25": float(np.percentile(task_vals, 25)),
                "task_err_final_q75": float(np.percentile(task_vals, 75)),
                "action_energy_median": float(np.median(energy_vals)),
                "rmse_failure_rate": float(np.mean(rmse_vals > PARAM_RMSE_FAILURE_THRESHOLD)),
                "task_failure_rate": float(np.mean(task_vals > TASK_ERR_FAILURE_THRESHOLD)),
                "n_seeds": N_SEEDS,
            }

    out_json = project_root / "results" / "noise_sweep_2d_gpu.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_json}")

    # Bootstrap CI
    bootstrap_ci = {}
    for sigma in SIGMA_OBS_VALUES:
        bootstrap_ci[str(sigma)] = {}
        for cond in CONDITIONS:
            task_vals = results_raw[sigma][cond]["task"]
            fail_task = (task_vals > TASK_ERR_FAILURE_THRESHOLD).astype(float)
            bootstrap_ci[str(sigma)][cond] = bootstrap_ci_failure_rate(fail_task)

    out_ci = project_root / "results" / "bootstrap_ci_2d_gpu.json"
    with open(out_ci, "w") as f:
        json.dump(bootstrap_ci, f, indent=2)
    print(f"Saved → {out_ci}")

    # Plot
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
        f"Fig. 4 – Noise sweep (GPU): 2-DOF 1D-obs (N={N_SEEDS} seeds per point)",
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
    out_png = project_root / "results" / "noise_sweep_2d_gpu.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_png}")

    # Summary table
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
