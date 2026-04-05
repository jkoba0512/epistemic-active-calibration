"""DEM E-step demo: simultaneous state tracking and parameter estimation.

Demonstrates D-step (state inference) + E-step (parameter inference) on a
linear damping system:

    dx/dt = a * x + v,    y = x + noise

True parameter: a = -1.0
Initial estimate: a_0 = 0.0

Results are saved to: results/dem_demo_param_estimation.png
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so that `src` package is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dem.model import DEMModel
from src.dem.inference import DStep, compute_vfe
from src.dem.estep import EStep, _compute_vfe_wrt_params


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def make_demo_model(a_init: float, n_order: int = 4) -> DEMModel:
    """Create a 1D linear damping model with learnable scalar a.

    Model:
        f̃[0] = a * x̃[0] + ṽ[0]   (zeroth-order dynamics; θ=a enters here)
        f̃[i] = 0                   (higher orders)
        g̃ = x̃                      (identity observation; y = x)

    Args:
        a_init: Initial parameter estimate for the decay coefficient a.
        n_order: Generalized coordinate embedding order.

    Returns:
        DEMModel configured for this system.
    """
    n_x = 1

    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        a = params[0]
        n_v = v_tilde.shape[0] // n_order
        x0 = x_tilde[:n_x]
        v0 = v_tilde[:n_v]
        result = [a * x0 + v0]
        for _ in range(1, n_order):
            result.append(jnp.zeros(n_x))
        return jnp.concatenate(result)

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        return x_tilde  # y = x

    params0 = jnp.array([a_init])
    return DEMModel(
        f=f,
        g=g,
        n_x=1,
        n_v=1,
        n_y=1,
        n_order=n_order,
        pi_y=2.0,
        pi_x=8.0,
        s_y=1.0,
        s_x=1.0,
        params=params0,
        params_prior_mean=params0,
        params_prior_pi=0.001,
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_trajectory(
    a_true: float,
    x0: float,
    dt: float,
    T: int,
    noise_std: float,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy trajectory from dx/dt = a*x, y = x + noise.

    Args:
        a_true: True decay coefficient.
        x0: Initial state value.
        dt: Integration time step.
        T: Number of time steps.
        noise_std: Standard deviation of observation noise.
        key: JAX random key.

    Returns:
        Tuple of (true_states, noisy_observations), each shape (T,).
    """
    xs = np.zeros(T)
    x = x0
    for t in range(T):
        xs[t] = x
        x = x + dt * a_true * x  # Euler integration

    noise = np.array(jax.random.normal(key, shape=(T,))) * noise_std
    ys = xs + noise
    return xs, ys


def build_generalized_obs(
    ys: np.ndarray,
    dt: float,
    n_order: int,
) -> list:
    """Build generalized observation vectors from a noisy scalar trajectory.

    Uses finite differences to approximate higher-order time derivatives.

    Args:
        ys: Noisy scalar observations, shape (T,).
        dt: Time step used for finite differences.
        n_order: Number of generalized coordinate orders.

    Returns:
        List of T arrays each of shape (n_order,).
    """
    T = len(ys)
    y_tilde_list = []
    for t in range(T):
        y_tilde = np.zeros(n_order)
        y_tilde[0] = ys[t]
        if n_order > 1:
            if 0 < t < T - 1:
                y_tilde[1] = (ys[t + 1] - ys[t - 1]) / (2 * dt)
            elif t == 0:
                y_tilde[1] = (ys[1] - ys[0]) / dt
            else:
                y_tilde[1] = (ys[-1] - ys[-2]) / dt
        y_tilde_list.append(jnp.array(y_tilde))
    return y_tilde_list


# ---------------------------------------------------------------------------
# Inference: D-step state tracking + E-step parameter estimation
# ---------------------------------------------------------------------------

def run_dstep_tracking(
    model: DEMModel,
    y_tilde_list: list,
    dt_obs: float,
    n_dstep_iter: int = 32,
    kappa_mu: float = 1.0,
    n_order: int = 4,
) -> tuple[list, list, list]:
    """Run D-step state tracking over all observations.

    Args:
        model: DEMModel (parameters fixed during this phase).
        y_tilde_list: List of T generalized observation vectors.
        dt_obs: Observation time step.
        n_dstep_iter: Number of Euler steps per D-step.
        kappa_mu: Learning rate for state inference.
        n_order: Generalized coordinate order.

    Returns:
        Tuple of:
            mu_x_list: List of T estimated state vectors.
            mu_v_list: List of T estimated cause vectors.
            vfe_list: List of T VFE values.
    """
    T = len(y_tilde_list)
    d_step = DStep(model, kappa_mu=kappa_mu, dt=dt_obs / n_dstep_iter,
                   n_iter=n_dstep_iter, use_d_operator=False)

    mu_x = jnp.zeros(model.dim_x_tilde)
    mu_x = mu_x.at[0].set(float(y_tilde_list[0][0]))
    mu_v = jnp.zeros(model.dim_v_tilde)

    mu_x_list, mu_v_list, vfe_list = [], [], []
    for y_tilde in y_tilde_list:
        mu_x, mu_v, vfe = d_step.run(mu_x, mu_v, y_tilde)
        mu_x_list.append(mu_x)
        mu_v_list.append(mu_v)
        vfe_list.append(float(vfe))

    return mu_x_list, mu_v_list, vfe_list


def run_estep_estimation(
    model: DEMModel,
    mu_x_list: list,
    mu_v_list: list,
    y_tilde_list: list,
    n_iter: int = 500,
    kappa_p: float = 0.0005,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Run E-step parameter estimation over all accumulated state estimates.

    Args:
        model: DEMModel with initial parameters.
        mu_x_list: List of T estimated state vectors (from D-step).
        mu_v_list: List of T estimated cause vectors (from D-step).
        y_tilde_list: List of T generalized observation vectors.
        n_iter: Total number of E-step gradient descent iterations.
        kappa_p: Learning rate for parameter update.

    Returns:
        Tuple of:
            params_final: Final parameter estimate, shape (n_params,).
            params_history: Array of shape (n_iter, n_params) — θ at each iteration.
    """
    e_step = EStep(model, kappa_p=kappa_p)
    params = jnp.asarray(model.params)
    params_history = [np.array(params)]

    for _ in range(n_iter):
        params = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=1)
        params_history.append(np.array(params))

    return params, np.array(params_history)


def run_joint_iterative(
    model: DEMModel,
    y_tilde_list: list,
    dt_obs: float,
    n_em_iter: int = 5,
    n_dstep_iter: int = 32,
    n_estep_iter: int = 100,
    kappa_mu: float = 1.0,
    kappa_p: float = 0.0005,
    n_order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run iterative EM: alternating D-step and E-step phases.

    Each EM iteration:
        1. D-step phase: run state tracking with current parameter estimate.
        2. E-step phase: update parameters using tracked states.

    Args:
        model: DEMModel with initial parameters.
        y_tilde_list: List of T generalized observation vectors.
        dt_obs: Observation time step.
        n_em_iter: Number of EM (outer) iterations.
        n_dstep_iter: Number of Euler steps per D-step.
        n_estep_iter: Number of E-step gradient steps per EM iteration.
        kappa_mu: Learning rate for state inference.
        kappa_p: Learning rate for parameter inference.
        n_order: Generalized coordinate order.

    Returns:
        Tuple of:
            mu_x0_history: np.ndarray (T,) — final state estimates.
            params_em_history: np.ndarray (n_em_iter+1, n_params) — θ per EM iter.
            params_fine_history: np.ndarray (n_estep_iter+1, n_params) — θ final E-step.
            vfe_history: np.ndarray (T,) — final D-step VFE values.
    """
    T = len(y_tilde_list)
    params = jnp.asarray(model.params)
    params_em_history = [np.array(params)]

    for em_iter in range(n_em_iter):
        # Create a new model with current parameter estimate
        updated_model = DEMModel(
            f=model.f,
            g=model.g,
            n_x=model.n_x,
            n_v=model.n_v,
            n_y=model.n_y,
            n_order=model.n_order,
            pi_y=model.pi_y,
            pi_x=model.pi_x,
            s_y=model.s_y,
            s_x=model.s_x,
            params=params,
            params_prior_mean=model.params_prior_mean,
            params_prior_pi=model.params_prior_pi,
        )

        # D-step: state tracking with current params
        mu_x_list, mu_v_list, vfe_list = run_dstep_tracking(
            updated_model, y_tilde_list, dt_obs, n_dstep_iter, kappa_mu, n_order
        )

        # E-step: parameter update using tracked states
        e_step = EStep(updated_model, kappa_p=kappa_p)
        for _ in range(n_estep_iter):
            params = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=1)

        params_em_history.append(np.array(params))

    # Final D-step for plotting
    final_model = DEMModel(
        f=model.f,
        g=model.g,
        n_x=model.n_x,
        n_v=model.n_v,
        n_y=model.n_y,
        n_order=model.n_order,
        pi_y=model.pi_y,
        pi_x=model.pi_x,
        s_y=model.s_y,
        s_x=model.s_x,
        params=params,
        params_prior_mean=model.params_prior_mean,
        params_prior_pi=model.params_prior_pi,
    )
    mu_x_list, mu_v_list, vfe_list = run_dstep_tracking(
        final_model, y_tilde_list, dt_obs, n_dstep_iter, kappa_mu, n_order
    )

    # Final E-step with fine-grained history for convergence plot
    e_step = EStep(final_model, kappa_p=kappa_p)
    params_fine_history = [np.array(params)]
    for _ in range(200):
        params = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=1)
        params_fine_history.append(np.array(params))

    mu_x0_history = np.array([float(mx[0]) for mx in mu_x_list])
    return (
        mu_x0_history,
        np.array(params_em_history),
        np.array(params_fine_history),
        np.array(vfe_list),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the demo and save results to results/dem_demo_param_estimation.png.

    Two-phase strategy:
        Phase 1 (D-step): State tracking using initial parameter estimate.
            Provides state sequence {mu_x_t} for the E-step.
        Phase 2 (E-step): Parameter inference using the tracked states.
            Accumulates gradient over all T observations and iterates.

    Note: In a full DEM implementation, D-step and E-step would alternate
    (EM-style). This demo shows each phase clearly for illustration.
    """
    # Settings
    a_true = -1.0
    a_init = 0.0
    x0 = 1.0
    dt = 0.05
    T = 150
    noise_std = 0.05
    n_order = 4
    key = jax.random.PRNGKey(42)

    print("DEM E-step Demo: parameter estimation")
    print(f"  True a = {a_true}, Initial a_0 = {a_init}")
    print(f"  T = {T} observations, dt = {dt}, noise_std = {noise_std}")
    print()

    # Generate data
    xs, ys = generate_trajectory(a_true, x0, dt, T, noise_std, key)
    y_tilde_list = build_generalized_obs(ys, dt, n_order)

    # Build model with wrong initial parameter
    model = make_demo_model(a_init=a_init, n_order=n_order)

    # -----------------------------------------------------------------
    # Phase 1: D-step — state tracking (with initial wrong parameter)
    # -----------------------------------------------------------------
    print("Phase 1: D-step state tracking (initial parameter a=0.0)...")
    mu_x_list, mu_v_list_dstep, vfe_hist_init = run_dstep_tracking(
        model, y_tilde_list, dt_obs=dt, n_dstep_iter=32, kappa_mu=1.0, n_order=n_order
    )

    # Build FD-based state estimates for E-step.
    # These encode the true dx/dt from data via finite differences,
    # and are used with zero causes so that ε_x[0] = x' - a*x
    # directly reflects the true dynamics.
    mu_x_fd_list = []
    for t in range(T):
        xval = float(ys[t])
        if 0 < t < T - 1:
            xdot = float((ys[t + 1] - ys[t - 1]) / (2 * dt))
        elif t == 0:
            xdot = float((ys[1] - ys[0]) / dt)
        else:
            xdot = float((ys[-1] - ys[-2]) / dt)
        mu_x_fd = jnp.zeros(n_order).at[0].set(xval).at[1].set(xdot)
        mu_x_fd_list.append(mu_x_fd)

    # Use zero causes for E-step: v=0 means f(x, 0, a) = a*x,
    # so ε_x[0] = x' - a*x = 0 gives a = x'/x ≈ a_true.
    mu_v_zero_list = [jnp.zeros(model.dim_v_tilde) for _ in range(T)]

    # -----------------------------------------------------------------
    # Phase 2: E-step — parameter estimation
    # -----------------------------------------------------------------
    print("Phase 2: E-step parameter estimation...")
    n_estep = 800
    e_step = EStep(model, kappa_p=0.0005)
    params = jnp.asarray(model.params)
    params_history = [float(params[0])]

    for i in range(n_estep):
        params = e_step.run(
            mu_x_fd_list, mu_v_zero_list, y_tilde_list, params, n_iter=1
        )
        params_history.append(float(params[0]))
        if i % 200 == 0:
            print(f"  E-step iter {i:4d}: a = {float(params[0]):.4f}")

    a_final = float(params[0])
    print(f"  E-step iter {n_estep:4d}: a = {a_final:.4f} (true = {a_true})")
    print()

    # -----------------------------------------------------------------
    # Phase 3: D-step with final parameter — improved state tracking
    # -----------------------------------------------------------------
    print("Phase 3: D-step with estimated parameter...")
    final_model = DEMModel(
        f=model.f, g=model.g,
        n_x=model.n_x, n_v=model.n_v, n_y=model.n_y,
        n_order=model.n_order, pi_y=model.pi_y, pi_x=model.pi_x,
        s_y=model.s_y, s_x=model.s_x,
        params=params,
        params_prior_mean=model.params_prior_mean,
        params_prior_pi=model.params_prior_pi,
    )
    mu_x_final_list, _, vfe_hist_final = run_dstep_tracking(
        final_model, y_tilde_list, dt_obs=dt,
        n_dstep_iter=32, kappa_mu=1.0, n_order=n_order
    )

    mu_x0_init = np.array([float(mx[0]) for mx in mu_x_list])
    mu_x0_final = np.array([float(mx[0]) for mx in mu_x_final_list])

    # --- Plot ---
    t_arr = np.arange(T) * dt
    iter_arr = np.arange(n_estep + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(
        "DEM: State Tracking (D-step) + Parameter Estimation (E-step)\n"
        r"$\dot{x} = a \cdot x + v$,  $y = x + \epsilon$"
        f"\n(True $a={a_true}$, Initial $a_0={a_init}$, Estimated $\\hat{{a}}={a_final:.3f}$)",
        fontsize=12,
    )

    # Panel 1: State tracking comparison
    ax = axes[0]
    ax.plot(t_arr, xs, "k-", lw=2, label=r"True state $x(t)$", alpha=0.7)
    ax.plot(t_arr, ys, ".", color="gray", ms=3, label="Noisy obs $y$", alpha=0.5)
    ax.plot(t_arr, mu_x0_init, "b--", lw=1.5, alpha=0.7,
            label=r"D-step ($a_0=0.0$)")
    ax.plot(t_arr, mu_x0_final, "r-", lw=2,
            label=rf"D-step ($\hat{{a}}={a_final:.3f}$)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State $x$")
    ax.set_title("State Tracking: Before vs After Parameter Estimation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Parameter convergence curve
    ax = axes[1]
    ax.plot(iter_arr, params_history, "r-", lw=2, label=r"E-step: $\hat{a}$")
    ax.axhline(a_true, color="k", ls="--", lw=1.5, label=f"True $a = {a_true}$")
    ax.axhline(a_init, color="gray", ls=":", lw=1.5, label=f"Initial $a_0 = {a_init}$")
    ax.fill_between(
        iter_arr[[0, -1]],
        a_true - 0.3, a_true + 0.3,
        alpha=0.1, color="green", label="±0.3 tolerance",
    )
    ax.set_xlabel("E-step iteration")
    ax.set_ylabel("Parameter $a$")
    ax.set_title("Parameter Convergence (E-step)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: VFE comparison
    ax = axes[2]
    ax.semilogy(t_arr, np.maximum(vfe_hist_init, 1e-10), "b--", lw=1.5,
                alpha=0.7, label=rf"VFE ($a_0={a_init}$)")
    ax.semilogy(t_arr, np.maximum(vfe_hist_final, 1e-10), "r-", lw=1.5,
                label=rf"VFE ($\hat{{a}}={a_final:.3f}$)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("VFE (log scale)")
    ax.set_title("Variational Free Energy: Improvement After Parameter Estimation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = project_root / "results" / "dem_demo_param_estimation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    # Final summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  a_true   = {a_true:.4f}")
    print(f"  a_init   = {a_init:.4f}")
    print(f"  a_final  = {a_final:.4f}")
    print(f"  Error    = {abs(a_final - a_true):.4f}")
    converged = abs(a_final - a_true) < 0.3
    print(f"  Converged (tol=0.3): {'YES' if converged else 'NO'}")
    vfe_ratio = float(np.mean(vfe_hist_final)) / max(float(np.mean(vfe_hist_init)), 1e-10)
    print(f"  VFE reduction: {(1 - vfe_ratio) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
