"""Dual-control: task (VFE goal-reaching) + parameter calibration (IG).

Core claim: when body parameters are unknown, an agent that minimizes
    J(u) = VFE(u; θ_est) - λ_eff · IG(u; P_θ)
simultaneously reaches the goal and gathers information for calibration,
outperforming pure task control (λ=0) especially in early steps.

With wrong θ_est, VFE-only drives the arm to the WRONG target position.
As E-step calibrates θ_est → θ_true, the arm must re-converge.
Dual-control with adaptive λ accelerates calibration and corrects faster.

Conditions:
  vfe_only      λ = 0.0          pure task control (calibration via E-step on VFE observations)
  dual_weak     λ = 0.5          mild epistemic bonus
  dual_strong   λ = 3.0          strong epistemic (exploration-heavy)
  dual_adaptive λ = f(P_θ)       high when uncertain, low when calibrated

Action objective minimized by gradient descent:
  J(u) = 0.5 · π_y · ||fk(q + u·dt, θ_est) − y_goal||²   ← VFE (goal)
         − λ_eff · IG(u; P_θ)                              ← epistemic

System:
  q = [q1, q2]      joint angles
  u = [dq1, dq2]    velocity command
  θ = [l1, l2]      unknown link lengths
  y = [x_ee, y_ee]  end-effector position (2D observation)
  y_goal = fk(q_target, θ_true)  target EE position

Usage:
    .venv/bin/python experiments/dual_control_2dof.py

Output:
    results/dual_control_2dof.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep


# ── Parameters ────────────────────────────────────────────────────────────────

N_SEEDS  = 10
N_STEPS  = 100

THETA_TRUE      = jnp.array([0.5, 0.5])
THETA_INIT      = jnp.array([0.8, 0.2])   # intentionally wrong (±0.3m)
THETA_PRIOR_PI  = 0.1                      # very weak prior
Q_START         = jnp.array([0.0, 0.1])   # initial joint angles
Q_TARGET        = jnp.array([0.9, 0.7])   # goal joint angles
Q_CLIP          = 1.5
U_MAX           = 0.8
DT              = 0.05
SIGMA_OBS       = 0.02                     # EE observation noise (m)
DAMPING         = 1e-8

PI_Y_TASK       = 15.0                     # goal-tracking precision weight
LR_ACTION       = 0.15                     # action gradient descent step
N_ACTION_ITER   = 30                       # gradient descent steps per action
N_ESTEP_ITER    = 4
N_FUTURE_STEPS  = 6                        # FIM rollout horizon

# Adaptive lambda parameters
LAMBDA_0        = 4.0                      # initial exploration weight
LAMBDA_SCALE    = 8.0                      # uncertainty scale for decay

CONDITIONS = ["vfe_only", "dual_weak", "dual_strong", "dual_adaptive"]
COLORS = {
    "vfe_only":      "tab:blue",
    "dual_weak":     "tab:orange",
    "dual_strong":   "tab:red",
    "dual_adaptive": "tab:green",
}
LAMBDA_FIXED = {
    "vfe_only":    0.0,
    "dual_weak":   0.5,
    "dual_strong": 3.0,
    "dual_adaptive": None,  # computed per step
}


# ── Kinematic model ───────────────────────────────────────────────────────────

def fk(q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Forward kinematics: [x_ee, y_ee]."""
    x = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    y = theta[0] * jnp.sin(q[0]) + theta[1] * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])


def rollout_fn(q: jnp.ndarray, u: jnp.ndarray, n_steps: int) -> jnp.ndarray:
    def step(q_curr, _): return q_curr + u * DT, None
    q_f, _ = jax.lax.scan(step, q, None, length=n_steps)
    return q_f


def compute_fim(q: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """FIM = J.T @ R_obs_inv @ J, J = d(fk(q_future))/d(theta), shape (2,2)."""
    R_obs_inv = jnp.eye(2) / SIGMA_OBS**2
    J = jax.jacfwd(
        lambda t: fk(rollout_fn(q, u, N_FUTURE_STEPS), t)
    )(theta)    # shape (2, 2)
    return J.T @ R_obs_inv @ J


def compute_info_gain(P_theta: jnp.ndarray, FIM_future: jnp.ndarray) -> jnp.ndarray:
    I = jnp.eye(2)
    _, ld0 = jnp.linalg.slogdet(P_theta + DAMPING * I)
    _, ld1 = jnp.linalg.slogdet(P_theta + FIM_future + DAMPING * I)
    return 0.5 * (ld1 - ld0)


def adaptive_lambda(P_theta: jnp.ndarray) -> jnp.ndarray:
    """λ_eff = λ_0 / (1 + λ_min(P_θ) / scale).

    High when P_θ has small eigenvalues (uncertain).
    Low when P_θ is well-conditioned (calibrated).
    """
    eigs = jnp.linalg.eigvalsh(P_theta)
    min_eig = jnp.maximum(eigs[0], DAMPING)
    return LAMBDA_0 / (1.0 + min_eig / LAMBDA_SCALE)


# ── E-step ────────────────────────────────────────────────────────────────────

def _build_estep() -> EStep:
    def f_zero(x, v, p): return jnp.zeros(2)
    def g_fk(x, v, p):   return fk(x, p)

    model = DEMModel(
        f=f_zero, g=g_fk,
        n_x=2, n_v=2, n_y=2, n_order=1,
        pi_y=1.0 / SIGMA_OBS**2,
        pi_x=1.0,
        params=THETA_INIT,
        params_prior_mean=THETA_INIT,
        params_prior_pi=THETA_PRIOR_PI,
    )
    return EStep(model, use_gauss_newton=True)

ESTEP = _build_estep()


# ── Action optimizer ──────────────────────────────────────────────────────────

@jax.jit
def optimize_action(
    q: jnp.ndarray,
    theta_est: jnp.ndarray,
    P_theta: jnp.ndarray,
    y_goal: jnp.ndarray,
    lambda_eff: jnp.ndarray,
    u_init: jnp.ndarray,
) -> jnp.ndarray:
    """Minimize J(u) = VFE - λ·IG via gradient descent.

    VFE  = 0.5 * π_y * ||fk(q + u·dt, θ_est) - y_goal||²
    IG   = 0.5 * (logdet P_post - logdet P_prior)
    """
    def objective(u: jnp.ndarray) -> jnp.ndarray:
        # VFE: drive predicted next observation toward goal
        q_pred = q + u * DT
        y_pred = fk(q_pred, theta_est)
        vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)

        # Epistemic: expected information gain from longer rollout
        fim = compute_fim(q, u, theta_est)
        ig = compute_info_gain(P_theta, fim)

        return vfe - lambda_eff * ig

    def descent_step(u: jnp.ndarray, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
    return u_opt


# ── Single-seed simulation ────────────────────────────────────────────────────

def run_one_seed(condition: str, seed: int, y_goal: jnp.ndarray) -> dict:
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    q = Q_START.copy()
    P_theta = THETA_PRIOR_PI * jnp.eye(2)
    u_current = jnp.zeros(2)
    q_acc, v_acc, y_acc = [], [], []

    lam_fixed = LAMBDA_FIXED[condition]

    # Initial metrics
    task_err_hist  = [float(jnp.linalg.norm(fk(q, THETA_TRUE) - y_goal))]
    rmse_hist      = [float(jnp.linalg.norm(theta_est - THETA_TRUE))]
    lambda_hist    = [float(lam_fixed) if lam_fixed is not None else float(adaptive_lambda(P_theta))]
    q2_hist        = [float(q[1])]

    for t in range(N_STEPS):
        # ── Compute effective lambda ─────────────────────────────────────────
        if lam_fixed is not None:
            lam = jnp.array(lam_fixed)
        else:
            lam = adaptive_lambda(P_theta)

        # ── Optimize action ──────────────────────────────────────────────────
        u = optimize_action(q, theta_est, P_theta, y_goal, lam, u_current)
        u_current = u

        # ── Step arm (true dynamics) ─────────────────────────────────────────
        q_next = jnp.clip(q + u * DT, -Q_CLIP, Q_CLIP)

        # Observe true EE position with noise
        y_obs = fk(q_next, THETA_TRUE) + jnp.array(rng.standard_normal(2) * SIGMA_OBS)

        q_acc.append(q_next)
        v_acc.append(jnp.zeros(2))
        y_acc.append(y_obs)

        # ── E-step: update θ_est ─────────────────────────────────────────────
        theta_est = ESTEP.run(q_acc, v_acc, y_acc, theta_est, n_iter=N_ESTEP_ITER)
        theta_est = jnp.clip(theta_est, 0.05, 1.8)
        P_theta   = ESTEP.compute_precision(q_acc, v_acc, y_acc, theta_est)

        # ── Log ──────────────────────────────────────────────────────────────
        task_err = float(jnp.linalg.norm(fk(q_next, THETA_TRUE) - y_goal))
        rmse     = float(jnp.linalg.norm(theta_est - THETA_TRUE))
        task_err_hist.append(task_err)
        rmse_hist.append(rmse)
        lambda_hist.append(float(lam))
        q2_hist.append(float(q_next[1]))
        q = q_next

    return {
        "task_err": np.array(task_err_hist),
        "rmse":     np.array(rmse_hist),
        "lambda":   np.array(lambda_hist),
        "q2":       np.array(q2_hist),
    }


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(data2d: np.ndarray, n_boot: int = 1000, ci: float = 0.95):
    n = data2d.shape[0]
    rng = np.random.default_rng(0)
    means = np.mean(data2d, axis=0)
    boots = np.stack([
        np.mean(data2d[rng.integers(0, n, n)], axis=0) for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (1 - ci) / 2 * 100, axis=0)
    hi = np.percentile(boots, (1 + ci) / 2 * 100, axis=0)
    return means, lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # True target EE position
    y_goal = fk(Q_TARGET, THETA_TRUE)

    print("=" * 65)
    print("Dual-Control: VFE Goal-Reaching + Epistemic Calibration")
    print(f"  θ_true={list(THETA_TRUE)}, θ_init={list(THETA_INIT)}")
    print(f"  q_target={list(Q_TARGET)}")
    print(f"  y_goal=[{float(y_goal[0]):.3f}, {float(y_goal[1]):.3f}]  "
          f"(true EE target)")
    print(f"  N_SEEDS={N_SEEDS}, N_STEPS={N_STEPS}, σ_obs={SIGMA_OBS}m")
    print()
    print("  VFE-only (λ=0): arm converges to WRONG target while θ_est is wrong,")
    print("  then slowly corrects as E-step calibrates.")
    print("  dual_adaptive: explores first, calibrates fast, then converges correctly.")
    print("=" * 65)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    all_data = {}
    for cond in CONDITIONS:
        lam_str = (f"λ={LAMBDA_FIXED[cond]}" if LAMBDA_FIXED[cond] is not None
                   else "λ=adaptive")
        print(f"Running '{cond}' ({lam_str}, {N_SEEDS} seeds)...", end="", flush=True)
        seed_results = [run_one_seed(cond, seed * 100 + 13, y_goal)
                        for seed in range(N_SEEDS) if not print(".", end="", flush=True)]
        all_data[cond] = {
            key: np.stack([r[key] for r in seed_results])
            for key in seed_results[0]
        }
        m_task = np.mean(all_data[cond]["task_err"][:, -1])
        m_rmse = np.mean(all_data[cond]["rmse"][:, -1])
        print(f" done.  task_err@end={m_task:.4f}m  RMSE@end={m_rmse:.4f}m")

    print()
    print("=" * 65)
    print(f"{'Condition':<16} {'Task err@end':>13} {'RMSE@end':>10} "
          f"{'AUC task':>10} {'AUC RMSE':>10}")
    print("-" * 65)
    for cond in CONDITIONS:
        te = all_data[cond]["task_err"]
        rm = all_data[cond]["rmse"]
        print(f"  {cond:<14} {np.mean(te[:,-1]):>13.4f} {np.mean(rm[:,-1]):>10.4f} "
              f"{np.mean(te[:,1:].sum(1)):>10.2f} {np.mean(rm[:,1:].sum(1)):>10.2f}")
    print("=" * 65)

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(N_STEPS + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        def plot_ci(ax, data2d, color, label, semilogy=False):
            mean, lo, hi = bootstrap_ci(data2d)
            fn = ax.semilogy if semilogy else ax.plot
            fn(steps, mean, color=color, lw=2.2, label=label)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)

        # Panel 1: Task error (main result)
        ax = axes[0, 0]
        for cond in CONDITIONS:
            plot_ci(ax, all_data[cond]["task_err"], COLORS[cond],
                    f"{cond} (λ={LAMBDA_FIXED[cond] or 'adapt'})")
        # Annotate: EE error of task_only if θ were correct from the start
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.3, label="perfect")
        ax.set_xlabel("Step")
        ax.set_ylabel("||fk(q, θ_true) − y_goal||  (m)")
        ax.set_title("Task Error: Distance of True EE from Goal\n"
                     "← key metric; lower = arm closer to intended target")
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

        # Panel 2: Parameter RMSE
        ax = axes[0, 1]
        for cond in CONDITIONS:
            plot_ci(ax, all_data[cond]["rmse"], COLORS[cond], cond)
        ax.set_xlabel("Step"); ax.set_ylabel("||θ_est − θ_true||  (m)")
        ax.set_title("Parameter RMSE\n(all conditions run E-step)")
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

        # Panel 3: Lambda trajectory (adaptive vs fixed)
        ax = axes[1, 0]
        for cond in CONDITIONS:
            mean, lo, hi = bootstrap_ci(all_data[cond]["lambda"])
            ax.plot(steps, mean, color=COLORS[cond], lw=2.2,
                    label=f"{cond}")
            ax.fill_between(steps, lo, hi, color=COLORS[cond], alpha=0.15)
        ax.set_xlabel("Step"); ax.set_ylabel("λ_eff")
        ax.set_title("Effective λ over Time\n"
                     "adaptive: starts high (explore) → falls (exploit task)")
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

        # Panel 4: AUC task error (bar plot)
        ax = axes[1, 1]
        cond_labels = [f"{c}\n(λ={LAMBDA_FIXED[c] or 'adapt'})" for c in CONDITIONS]
        auc_means = [np.mean(all_data[c]["task_err"][:, 1:].sum(1)) for c in CONDITIONS]
        auc_stds  = [np.std(all_data[c]["task_err"][:, 1:].sum(1))  for c in CONDITIONS]
        bars = ax.bar(range(len(CONDITIONS)), auc_means,
                      yerr=auc_stds,
                      color=[COLORS[c] for c in CONDITIONS],
                      alpha=0.75, capsize=6)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_ylabel("AUC Task Error  (Σ_t ||EE−goal||, lower = better)")
        ax.set_title("Total Accumulated Task Error")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Dual-Control: Task (VFE) + Calibration (Epistemic IG)\n"
            f"θ_true={list(THETA_TRUE)}, θ_init={list(THETA_INIT)}, "
            f"σ_obs={SIGMA_OBS}m, {N_SEEDS}seeds×{N_STEPS}steps",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig_path = results_dir / "dual_control_2dof.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    print("Done.")


if __name__ == "__main__":
    main()
