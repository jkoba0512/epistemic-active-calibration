"""Phase 1: Epistemic self-calibration of 2-DOF planar arm.

Compares three action policies for online parameter estimation (no MuJoCo required):
  1. Random babbling    — random velocity commands
  2. Sinusoidal         — sinusoidal joint velocity excitation
  3. Epistemic A-step   — gradient ascent on expected information gain (IG)

System:
  q = [q1, q2]        joint angles (rad)
  u = [dq1, dq2]      velocity command (rad/s)
  theta = [l1, l2]    unknown link lengths (m)  ← to be estimated
  y = [x_ee, y_ee]    end-effector position observation (m)

Estimation loop (online, closed-loop):
  At each step t:
    1. Apply action u_t → q_{t+1} = q_t + u_t * dt
    2. Observe y_{t+1} = fk(q_{t+1}, theta_true) + noise
    3. Run E-step (Gauss-Newton) on all accumulated (q, y) pairs → update theta_est
    4. Compute P_theta (posterior precision) from accumulated data
    5. (Epistemic only) optimize u_{t+1} via gradient ascent on IG(u; P_theta)

Metrics:
  - Parameter RMSE: ||theta_est - theta_true||
  - Posterior uncertainty: logdet(Sigma_theta) = -logdet(P_theta)

Usage:
    .venv/bin/python experiments/epistemic_calibration_2dof.py

Output:
    results/epistemic_calibration_2dof.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep


# ── Simulation parameters ─────────────────────────────────────────────────────

THETA_TRUE = jnp.array([0.5, 0.5])   # true link lengths [l1, l2] (m)
THETA_INIT = jnp.array([0.7, 0.3])   # initial (wrong) estimate
THETA_PRIOR_MEAN = THETA_INIT         # prior mean = initial guess
THETA_PRIOR_PI = 0.5                  # weak prior precision (regularizer)

Q0 = jnp.array([0.0, 0.5])           # initial joint angles (rad)
Q_CLIP = 1.5                          # joint angle limit (rad)
U_MAX = 1.0                           # velocity command limit (rad/s)

DT = 0.05                             # simulation time step (s)
N_STEPS = 120                         # total steps per condition
SIGMA_OBS = 0.01                      # EE position noise std (m)
R_OBS_INV = jnp.eye(2) / (SIGMA_OBS ** 2)  # observation precision

N_ESTEP_ITER = 3                      # E-step Gauss-Newton iterations per step
N_EPISTEMIC_ITER = 25                 # gradient ascent steps for epistemic action
LR_EPISTEMIC = 0.3                    # learning rate for epistemic action optimization
N_FUTURE_STEPS = 8                    # rollout horizon for FIM computation
DAMPING = 1e-6                        # P_theta regularization

SIN_AMP = 0.8                         # sinusoidal velocity amplitude (rad/s)
SIN_OMEGA1 = 0.6                      # angular frequency for joint 1
SIN_OMEGA2 = 0.4                      # angular frequency for joint 2 (different)
SIN_PHASE2 = 1.1                      # phase offset for joint 2 (rad)

SEED = 42


# ── Kinematic model ────────────────────────────────────────────────────────────

def fk(q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Forward kinematics: end-effector position."""
    x = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    y = theta[0] * jnp.sin(q[0]) + theta[1] * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])


def rollout_single(
    q: jnp.ndarray, u: jnp.ndarray, dt: float, n_steps: int
) -> jnp.ndarray:
    """Pure-JAX kinematic rollout. q_{t+1} = q_t + u * dt (Euler)."""
    def step(q_curr, _):
        return q_curr + u * dt, None

    q_future, _ = jax.lax.scan(step, q, None, length=n_steps)
    return q_future


def y_future_fn(
    q: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray, dt: float, n_steps: int
) -> jnp.ndarray:
    """EE position after rolling out action u. Differentiable w.r.t. theta and u."""
    return fk(rollout_single(q, u, dt, n_steps), theta)


def compute_fim(
    q: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray
) -> jnp.ndarray:
    """Fisher Information Matrix: FIM = J.T @ R_obs_inv @ J."""
    J = jax.jacfwd(lambda t: y_future_fn(q, u, t, DT, N_FUTURE_STEPS))(theta)
    return J.T @ R_OBS_INV @ J


def compute_info_gain(
    P_theta: jnp.ndarray, FIM_future: jnp.ndarray
) -> jnp.ndarray:
    """IG = 0.5 * (logdet(P_post) - logdet(P_prior)), precision form."""
    I = jnp.eye(P_theta.shape[0])
    _, ld_prior = jnp.linalg.slogdet(P_theta + DAMPING * I)
    _, ld_post = jnp.linalg.slogdet(P_theta + FIM_future + DAMPING * I)
    return 0.5 * (ld_post - ld_prior)


# ── DEM model and E-step ──────────────────────────────────────────────────────

def build_estep(theta_init: jnp.ndarray) -> EStep:
    """Build DEM model and EStep for kinematic parameter estimation.

    n_order=1: generalized coordinates reduce to plain state.
    f = 0 (no dynamics), g = fk (forward kinematics observation).
    P_theta is driven entirely by d(fk)/d(theta) Jacobians.
    """
    def f_zero(x_t, v_t, p):
        return jnp.zeros(2)

    def g_fk(x_t, v_t, p):
        return fk(x_t, p)

    model = DEMModel(
        f=f_zero,
        g=g_fk,
        n_x=2,
        n_v=2,
        n_y=2,
        n_order=1,
        pi_y=1.0 / (SIGMA_OBS ** 2),
        pi_x=1.0,
        params=theta_init,
        params_prior_mean=THETA_PRIOR_MEAN,
        params_prior_pi=THETA_PRIOR_PI,
    )
    return EStep(model, use_gauss_newton=True)


# ── Epistemic action optimization ─────────────────────────────────────────────

@jax.jit
def optimize_epistemic_action(
    q: jnp.ndarray,
    theta_est: jnp.ndarray,
    P_theta: jnp.ndarray,
    u_init: jnp.ndarray,
) -> jnp.ndarray:
    """Find action maximizing expected information gain via gradient ascent.

    Maximizes IG(u) = 0.5 * (logdet(P_theta + FIM(u)) - logdet(P_theta))
    subject to ||u||_inf <= U_MAX.
    """
    def ig_of_u(u):
        fim = compute_fim(q, u, theta_est)
        return compute_info_gain(P_theta, fim)

    def ascent_step(u, _):
        grad_u = jax.grad(ig_of_u)(u)
        u_new = jnp.clip(u + LR_EPISTEMIC * grad_u, -U_MAX, U_MAX)
        return u_new, None

    u_opt, _ = jax.lax.scan(ascent_step, u_init, None, length=N_EPISTEMIC_ITER)
    return u_opt


# ── Simulation loop ───────────────────────────────────────────────────────────

def run_condition(condition: str, seed: int) -> dict:
    """Run one calibration condition and return history.

    Args:
        condition: "random", "sinusoidal", or "epistemic"
        seed: random seed for reproducibility
    Returns:
        dict with keys: rmse, logdet_sigma, theta_hist, q_hist
    """
    rng = np.random.default_rng(seed)

    # Initialize
    theta_est = THETA_INIT.copy()
    q = Q0.copy()
    estep = build_estep(theta_est)

    # Accumulated observation history (for E-step)
    q_hist_acc = []    # joint angles at each observation
    v_hist_acc = []    # DEM causes (zeros)
    y_hist_acc = []    # EE position observations

    # Metrics history
    rmse_hist = [float(jnp.linalg.norm(theta_est - THETA_TRUE))]
    logdet_sigma_hist = [0.0]  # will be filled after first E-step
    theta_est_hist = [np.array(theta_est)]
    q_hist_out = [np.array(q)]

    # Initial P_theta (prior only)
    P_theta = THETA_PRIOR_PI * jnp.eye(2)

    # Current action (will be updated each step for epistemic)
    u_current = jnp.zeros(2)

    for t in range(N_STEPS):
        # ── Choose action ──────────────────────────────────────────────────
        if condition == "random":
            u = jnp.array(rng.standard_normal(2) * 0.6).clip(-U_MAX, U_MAX)

        elif condition == "sinusoidal":
            t_sec = t * DT
            u = jnp.array([
                SIN_AMP * np.sin(SIN_OMEGA1 * t_sec),
                SIN_AMP * np.sin(SIN_OMEGA2 * t_sec + SIN_PHASE2),
            ]).clip(-U_MAX, U_MAX)

        elif condition == "epistemic":
            # Optimize u to maximize IG given current P_theta
            u_init = u_current  # warm start from previous action
            u = optimize_epistemic_action(q, theta_est, P_theta, u_init)
            u_current = u

        else:
            raise ValueError(f"Unknown condition: {condition}")

        # ── Simulate arm ───────────────────────────────────────────────────
        q_next = jnp.clip(q + u * DT, -Q_CLIP, Q_CLIP)

        # Observe EE position with noise
        y_obs = fk(q_next, THETA_TRUE) + jnp.array(
            rng.standard_normal(2) * SIGMA_OBS
        )

        # ── Accumulate observation ─────────────────────────────────────────
        q_hist_acc.append(q_next)
        v_hist_acc.append(jnp.zeros(2))
        y_hist_acc.append(y_obs)

        # ── E-step: update theta_est ───────────────────────────────────────
        theta_est = estep.run(
            mu_x_sequence=q_hist_acc,
            mu_v_sequence=v_hist_acc,
            y_sequence=y_hist_acc,
            params=theta_est,
            n_iter=N_ESTEP_ITER,
        )
        # Clip to physically plausible range
        theta_est = jnp.clip(theta_est, 0.1, 1.5)

        # ── Compute P_theta ────────────────────────────────────────────────
        P_theta = estep.compute_precision(
            mu_x_sequence=q_hist_acc,
            mu_v_sequence=v_hist_acc,
            y_sequence=y_hist_acc,
            params=theta_est,
        )

        # ── Log metrics ────────────────────────────────────────────────────
        rmse = float(jnp.linalg.norm(theta_est - THETA_TRUE))
        # Posterior uncertainty: logdet(Sigma) = -logdet(P_theta)
        _, ld_P = jnp.linalg.slogdet(P_theta + DAMPING * jnp.eye(2))
        logdet_sigma = float(-ld_P)

        rmse_hist.append(rmse)
        logdet_sigma_hist.append(logdet_sigma)
        theta_est_hist.append(np.array(theta_est))
        q_hist_out.append(np.array(q_next))

        q = q_next

    return {
        "rmse": np.array(rmse_hist),
        "logdet_sigma": np.array(logdet_sigma_hist),
        "theta_hist": np.array(theta_est_hist),
        "q_hist": np.array(q_hist_out),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Phase 1: Epistemic Self-Calibration — 2-DOF Planar Arm")
    print(f"  theta_true = {THETA_TRUE},  theta_init = {THETA_INIT}")
    print(f"  N_STEPS={N_STEPS}, dt={DT}, sigma_obs={SIGMA_OBS}")
    print("=" * 65)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    conditions = ["random", "sinusoidal", "epistemic"]
    colors = {"random": "tab:blue", "sinusoidal": "tab:orange", "epistemic": "tab:green"}
    results = {}

    for cond in conditions:
        print(f"Running condition: {cond} ...")
        results[cond] = run_condition(cond, seed=SEED)
        final_rmse = results[cond]["rmse"][-1]
        final_theta = results[cond]["theta_hist"][-1]
        print(f"  Final RMSE:       {final_rmse:.4f} m")
        print(f"  Final theta_est:  [{final_theta[0]:.4f}, {final_theta[1]:.4f}]  "
              f"(true: [{float(THETA_TRUE[0]):.4f}, {float(THETA_TRUE[1]):.4f}])")
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 65)
    print("Summary: Final parameter RMSE and uncertainty")
    print(f"  {'Condition':<14} {'RMSE (m)':>10} {'logdet(Σ)':>12} "
          f"{'l1_est':>8} {'l2_est':>8}")
    print("-" * 65)
    for cond in conditions:
        r = results[cond]
        th = r["theta_hist"][-1]
        print(f"  {cond:<14} {r['rmse'][-1]:>10.4f} "
              f"{r['logdet_sigma'][-1]:>12.4f} "
              f"{th[0]:>8.4f} {th[1]:>8.4f}")
    print("=" * 65)
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(N_STEPS + 1)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Panel 1: Parameter RMSE vs steps
        ax = axes[0, 0]
        for cond in conditions:
            ax.plot(steps, results[cond]["rmse"],
                    color=colors[cond], lw=2.0, label=cond)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("Step")
        ax.set_ylabel("||θ_est - θ_true|| (m)")
        ax.set_title("Parameter RMSE")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 2: Posterior uncertainty vs steps
        ax = axes[0, 1]
        for cond in conditions:
            ax.plot(steps, results[cond]["logdet_sigma"],
                    color=colors[cond], lw=2.0, label=cond)
        ax.set_xlabel("Step")
        ax.set_ylabel("logdet(Σ_θ)  [lower = more certain]")
        ax.set_title("Posterior Uncertainty (logdet Σ_θ = −logdet P_θ)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 3: l1 estimate vs steps
        ax = axes[1, 0]
        ax.axhline(float(THETA_TRUE[0]), color="k", lw=1.2, ls="--",
                   alpha=0.7, label=f"l1_true = {float(THETA_TRUE[0])}")
        ax.axhline(float(THETA_INIT[0]), color="gray", lw=0.8, ls=":",
                   alpha=0.5, label=f"l1_init = {float(THETA_INIT[0])}")
        for cond in conditions:
            ax.plot(steps, results[cond]["theta_hist"][:, 0],
                    color=colors[cond], lw=1.8, label=cond)
        ax.set_xlabel("Step")
        ax.set_ylabel("l1 estimate (m)")
        ax.set_title("Link Length l1 Estimation")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 4: l2 estimate vs steps
        ax = axes[1, 1]
        ax.axhline(float(THETA_TRUE[1]), color="k", lw=1.2, ls="--",
                   alpha=0.7, label=f"l2_true = {float(THETA_TRUE[1])}")
        ax.axhline(float(THETA_INIT[1]), color="gray", lw=0.8, ls=":",
                   alpha=0.5, label=f"l2_init = {float(THETA_INIT[1])}")
        for cond in conditions:
            ax.plot(steps, results[cond]["theta_hist"][:, 1],
                    color=colors[cond], lw=1.8, label=cond)
        ax.set_xlabel("Step")
        ax.set_ylabel("l2 estimate (m)")
        ax.set_title("Link Length l2 Estimation")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Phase 1: Epistemic Self-Calibration — 2-DOF Planar Arm\n"
            f"θ_true=[l1={float(THETA_TRUE[0])}, l2={float(THETA_TRUE[1])}],  "
            f"θ_init=[{float(THETA_INIT[0])}, {float(THETA_INIT[1])}],  "
            f"σ_obs={SIGMA_OBS}m,  N={N_STEPS} steps",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        fig_path = results_dir / "epistemic_calibration_2dof.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    print("Done.")


if __name__ == "__main__":
    main()
