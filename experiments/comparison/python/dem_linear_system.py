"""DEM comparison experiment script using Python + JAX

Runs DEM with exactly the same settings as the MATLAB version
(experiments/comparison/matlab/dem_linear_system.m) and saves results to CSV.

Target system: damped linear system
    dx/dt = a * x + v    (true a = -1.0, v = 0)
    y     = x + epsilon  (observation noise)

Algorithm (3-phase design matching MATLAB/SPM spm_nlsi_GN):

    Phase 1 — D-step (a=0 fixed):
        Estimate state trajectory with initial a=0.

    Phase 2 — Trajectory Gauss-Newton (matching spm_nlsi_GN):
        Fix the ODE and integrate forward from x0.
        Optimize a to fit y_obs by minimizing sum of squared residuals.
        Jacobian dr/da is computed via JAX autodiff on Euler integration.
        This matches spm_nlsi_GN which integrates via spm_int_J and fits y.

    Phase 3 — D-step (estimated a, fixed):
        Re-run D-step with a_est to get final state estimates.

Key alignment with MATLAB:
    - Phase 2 uses trajectory fitting (same as spm_nlsi_GN), not collocation
    - Phase 1 and 3 use D-step (same as spm_DEM with pC=0)
    - Same hyperparameters: pi_y=2.0, pi_x=8.0, n_order=4, s=1.0, pC=1000

Output: results/comparison_python.csv
    Columns: t, x_true, x_estimated, a_estimated, vfe
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import math
import numpy as np
import jax
import jax.numpy as jnp
import csv

from src.dem.model import DEMModel
from src.dem.inference import DStep


# ============================================================
# Common parameter settings (matching MATLAB/SPM version)
# ============================================================

A_TRUE     = -1.0
X0_TRUE    = 1.0
V_INPUT    = 0.0

DT         = 0.1           # Observation time step
T_END      = 4.0
N_STEPS    = round(T_END / DT)   # 40

NOISE_STD  = math.exp(-2)         # ≈ 0.135
SEED       = 42

# Model hyperparameters (matching MATLAB: pi_y=2, pi_x=8, n=4, s=1)
PI_Y       = 2.0
PI_X       = 8.0
N_ORDER    = 4
S_SMOOTH   = 1.0

A_INIT     = 0.0
# Prior covariance for a: pC = 1000 (matching MATLAB pC_a=1000)
PC_A       = 1000.0
PRIOR_PI_A = 1.0 / PC_A   # = 0.001

# D-step settings
N_ITER_D_STEP = 32
D_STEP_DT  = DT / N_ITER_D_STEP   # ≈ 0.003125
KAPPA_MU   = 1.0

# Trajectory GN settings (Phase 2, matching spm_nlsi_GN)
N_GN_ITER  = 8   # Gauss-Newton iterations (matching SPM default nE=8)
# Observation precision for trajectory fit: 1/noise_std^2 (matching MATLAB YG.Q)
PI_Y_TRAJ  = 1.0 / (NOISE_STD ** 2)


def generate_trajectory():
    rng = np.random.default_rng(SEED)
    t_vec = np.arange(N_STEPS) * DT

    x_true = np.zeros(N_STEPS)
    x = X0_TRUE
    for k in range(N_STEPS):
        x_true[k] = x
        x = x + DT * (A_TRUE * x + V_INPUT)

    x_analytic = X0_TRUE * np.exp(A_TRUE * t_vec)
    noise = rng.normal(0.0, NOISE_STD, N_STEPS)
    y_obs = x_true + noise

    return t_vec, x_true, y_obs, x_analytic


def build_model(a_val: float) -> DEMModel:
    """Build a DEMModel with current parameter value a_val."""
    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        a = params[0]
        # Full generalized f: d/dt [x, x', x'', ...] = a * [x, x', x'', ...]
        # For linear system dx/dt = a*x: x^{(n+1)} = a * x^{(n)}
        return a * x_tilde

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        return x_tilde

    return DEMModel(
        f=f,
        g=g,
        n_x=1,
        n_v=1,
        n_y=1,
        n_order=N_ORDER,
        pi_y=PI_Y,
        pi_x=PI_X,
        s_y=S_SMOOTH,
        s_x=S_SMOOTH,
        params=jnp.array([a_val]),
        params_prior_mean=jnp.array([A_INIT]),
        params_prior_pi=PRIOR_PI_A,
    )


def build_y_tilde_seq(y_obs: np.ndarray) -> list:
    """Build the generalized observation sequence."""
    N = len(y_obs)
    y_tilde_seq = []
    for t in range(N):
        y_tilde = np.zeros(N_ORDER)
        y_tilde[0] = y_obs[t]
        if N_ORDER > 1:
            if 0 < t < N - 1:
                y_tilde[1] = (y_obs[t + 1] - y_obs[t - 1]) / (2 * DT)
            elif t == 0:
                y_tilde[1] = (y_obs[1] - y_obs[0]) / DT
            else:
                y_tilde[1] = (y_obs[-1] - y_obs[-2]) / DT
        y_tilde_seq.append(jnp.array(y_tilde))
    return y_tilde_seq


def run_d_step(model: DEMModel, y_tilde_seq: list) -> tuple:
    """D-step: estimate states at all observation time points (pure D-step)."""
    d_step = DStep(model, kappa_mu=KAPPA_MU, dt=D_STEP_DT,
                   n_iter=N_ITER_D_STEP, use_d_operator=False)

    mu_x = jnp.zeros(N_ORDER)
    mu_x = mu_x.at[0].set(float(y_tilde_seq[0][0]))
    mu_v = jnp.zeros(N_ORDER)

    mu_x_list, vfe_list = [], []
    for y_tilde in y_tilde_seq:
        mu_x, mu_v, vfe = d_step.run(mu_x, mu_v, y_tilde)
        mu_x_list.append(mu_x)
        vfe_list.append(float(vfe))

    return mu_x_list, vfe_list


def integrate_euler(a_scalar, x0, dt, N):
    """Euler integration of dx/dt = a*x from x0.

    Matches spm_int_J (Jacobian-based ODE integration).
    For linear system, exact at each step via matrix exponential:
        x[k+1] = x[k] * exp(a * dt)

    Uses exp-Euler for accuracy (exact for linear ODEs).
    """
    def step(x_k, _):
        x_next = x_k * jnp.exp(a_scalar * dt)
        return x_next, x_k

    _, x_traj = jax.lax.scan(step, jnp.array(x0), None, length=N)
    return x_traj  # shape (N,)


def estimate_params_gn(y_obs: np.ndarray) -> tuple:
    """Phase 2: Trajectory Gauss-Newton (matching spm_nlsi_GN).

    Integrates dx/dt = a*x forward from x0 and minimizes the weighted
    sum of squared residuals between the integrated trajectory and y_obs.

    This matches MATLAB's spm_nlsi_GN which:
    - Uses spm_int_J for ODE integration
    - Computes Jacobian of trajectory w.r.t. parameters via numerical differentiation
    - Applies Gauss-Newton update: da = -(J.T J + prior_pi*I)^{-1} (J.T r + prior_pi*(a - a0))

    Returns:
        a_est: estimated parameter value
        a_history: list of a values at each GN iteration
    """
    y_jax = jnp.array(y_obs)
    a = jnp.array([A_INIT])
    prior_mean = jnp.array([A_INIT])

    def residuals(a_arr):
        """Residuals: y_obs - x_integrated(a)"""
        a_val = a_arr[0]
        x_traj = integrate_euler(a_val, X0_TRUE, DT, N_STEPS)
        return y_jax - x_traj

    # JIT-compile the Jacobian computation
    jac_fn = jax.jit(jax.jacobian(residuals))

    a_history = [float(a[0])]
    for i in range(N_GN_ITER):
        r = residuals(a)                   # shape (N,)
        J = jac_fn(a)                      # shape (N, 1)

        # Gauss-Newton: accumulate gradient and curvature
        # F = 0.5 * Pi_y * ||r||^2 + 0.5 * prior_pi * (a-a0)^2
        # dF/da = Pi_y * J.T @ r + prior_pi * (a - a0)  where J = dr/da
        # dFdpp = J.T @ Pi_y @ J + prior_pi * I
        dFdp = PI_Y_TRAJ * J.T @ r + PRIOR_PI_A * (a - prior_mean)
        dFdpp = PI_Y_TRAJ * J.T @ J + PRIOR_PI_A * jnp.eye(1)

        # Gauss-Newton step: a_new = a - dFdpp^{-1} dFdp
        dp = jnp.linalg.solve(dFdpp, dFdp)
        a = a - dp

        a_history.append(float(a[0]))
        print(f"    GN iter {i+1:2d}/{N_GN_ITER}: a = {float(a[0]):.5f}")

    return float(a[0]), np.array(a_history)


def save_results_csv(output_path, t_vec, x_true, x_estimated, a_history, vfe_history):
    a_per_step = np.interp(
        np.arange(N_STEPS),
        np.linspace(0, N_STEPS - 1, len(a_history)),
        a_history,
    )
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_true', 'x_estimated', 'a_estimated', 'vfe'])
        for k in range(len(t_vec)):
            writer.writerow([
                f'{t_vec[k]:.6f}',
                f'{x_true[k]:.6f}',
                f'{x_estimated[k]:.6f}',
                f'{a_per_step[k]:.6f}',
                f'{vfe_history[k]:.6f}' if k < len(vfe_history) else 'nan',
            ])
    print(f"  Saved: {output_path}")


def save_metadata(output_path, x_true, x_estimated, a_final):
    rmse_state = float(np.sqrt(np.mean((x_estimated - x_true) ** 2)))
    with open(output_path, 'w') as f:
        f.write(f'a_true={A_TRUE:.4f}\n')
        f.write(f'a_estimated={a_final:.4f}\n')
        f.write(f'rmse_state={rmse_state:.6f}\n')
        f.write(f'n_order={N_ORDER}\n')
        f.write(f'pi_y={PI_Y:.4f}\n')
        f.write(f'pi_x={PI_X:.4f}\n')
        f.write(f's={S_SMOOTH:.4f}\n')
        f.write(f'dt={DT:.4f}\n')
        f.write(f'N={N_STEPS}\n')
        f.write(f'noise_std={NOISE_STD:.6f}\n')
        f.write(f'pi_y_traj={PI_Y_TRAJ:.4f}\n')
        f.write(f'prior_pi_a={PRIOR_PI_A:.6e}\n')
        f.write(f'n_gn_iter={N_GN_ITER}\n')
        f.write(f'estep_type=trajectory_gauss_newton\n')


def main():
    print("=" * 60)
    print("DEM comparison: Python + JAX")
    print("  (3-phase: D-step | trajectory GN | D-step)")
    print("  (Phase 2 aligned to MATLAB spm_nlsi_GN)")
    print("=" * 60)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Step 1: Generate data")
    t_vec, x_true, y_obs, x_analytic = generate_trajectory()
    print(f"  State range: [{x_true.min():.4f}, {x_true.max():.4f}]")
    print(f"  Noise std:   {np.std(y_obs - x_true):.4f}")
    print()

    y_tilde_seq = build_y_tilde_seq(y_obs)

    print("Step 2: Phase 1 — D-step (a=0.0, fixed)")
    model_init = build_model(A_INIT)
    mu_x_init, vfe_init = run_d_step(model_init, y_tilde_seq)
    x_init = np.array([float(mx[0]) for mx in mu_x_init])
    rmse_init = np.sqrt(np.mean((x_init - x_true) ** 2))
    print(f"  State RMSE (a=0.0): {rmse_init:.4f}")
    print()

    print("Step 3: Phase 2 — Trajectory Gauss-Newton (matching spm_nlsi_GN)")
    print(f"  a_init={A_INIT} -> a_true={A_TRUE} (target)")
    a_final, a_history = estimate_params_gn(y_obs)
    a_error = abs(a_final - A_TRUE)
    print(f"  a_estimated = {a_final:.4f} (true={A_TRUE}, error={a_error:.4f})")
    print()

    print(f"Step 4: Phase 3 — D-step (a={a_final:.4f}, fixed)")
    model_final = build_model(a_final)
    mu_x_final, vfe_final_arr = run_d_step(model_final, y_tilde_seq)
    x_estimated = np.array([float(mx[0]) for mx in mu_x_final])
    vfe_history = np.array(vfe_final_arr)
    rmse_final = np.sqrt(np.mean((x_estimated - x_true) ** 2))
    rmse_analytic = np.sqrt(np.mean((x_estimated - x_analytic) ** 2))
    print(f"  State RMSE (a={a_final:.4f}): {rmse_final:.4f}")
    print(f"  State RMSE vs analytic: {rmse_analytic:.4f}")
    print()

    print("Step 5: Accuracy evaluation")
    print(f"  State RMSE (vs true trajectory):   {rmse_final:.4f}")
    print(f"  State RMSE (vs analytic solution): {rmse_analytic:.4f}")
    print(f"  Parameter error |a_est - a_true|:  {a_error:.4f}")
    print()

    print("Step 6: Save CSV")
    output_csv = str(results_dir / "comparison_python.csv")
    save_results_csv(output_csv, t_vec, x_true, x_estimated, a_history, vfe_history)
    meta_path = str(results_dir / "comparison_python_meta.txt")
    save_metadata(meta_path, x_true, x_estimated, a_final)
    print(f"  Metadata: {meta_path}")
    print()

    print("Step 7: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        axes[0].plot(t_vec, x_true, 'b-', lw=2, label='True x(t)')
        axes[0].plot(t_vec, x_analytic, 'b--', lw=1, label='Analytic exp(-t)')
        axes[0].scatter(t_vec, y_obs, c='k', s=15, alpha=0.5, label='Obs y(t)')
        axes[0].plot(t_vec, x_init, 'g--', lw=1.5, alpha=0.7,
                     label=f'Phase1 (a=0.0)')
        axes[0].plot(t_vec, x_estimated, 'r-', lw=2,
                     label=f'Phase3 (a={a_final:.3f})')
        axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('State')
        axes[0].set_title('State estimation')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        import matplotlib.ticker as ticker
        ax2 = axes[1]
        bars = ax2.bar([0, 1, 2], [A_INIT, a_final, A_TRUE],
                       color=[0.5, 0.7, 1.0])
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['a_init', f'a_est (GN)', 'a_true'])
        ax2.set_ylabel('Value')
        ax2.set_title(f'Parameter estimation: a_est={a_final:.4f}, a_true={A_TRUE}')
        ax2.grid(True, alpha=0.3)

        axes[2].plot(t_vec, vfe_history, 'm-', lw=1.5)
        axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('VFE')
        axes[2].set_title('VFE (Phase 3)'); axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            'DEM Comparison: Python + JAX (3-phase, trajectory GN)',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        fig_path = str(results_dir / "comparison_python_plot.png")
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found.")
    print()

    print("=" * 60)
    print("Python DEM (3-phase) complete")
    print("=" * 60)
    print(f"  Phase 1 RMSE (a=0):       {rmse_init:.4f}")
    print(f"  Phase 2 a_est (traj GN):  {a_final:.4f}")
    print(f"  Phase 3 RMSE (a={a_final:.4f}): {rmse_final:.4f}")
    print(f"  Output: {output_csv}")
    print("=" * 60)
    print()
    print("Next: python experiments/comparison/compare_results.py")


if __name__ == "__main__":
    main()
