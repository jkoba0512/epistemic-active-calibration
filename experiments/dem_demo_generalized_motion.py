"""DEM generalized motion demo: gradient descent vs local linearization comparison on a harmonic oscillator.

Harmonic oscillator model:
    ẍ + ω²x = v
    y = x + noise

Compares two integration modes:
1. Gradient descent (use_d_operator=False): stable but without D operator
2. Full DEM (use_d_operator=True, use_local_linearization=True):
   stable generalized motion via Ozaki (1992) local linearization

Comparison metrics:
- Convergence speed (VFE trajectory)
- State estimation accuracy
- Final VFE value

Results are saved to results/dem_demo_generalized_motion.png.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dem.model import DEMModel
from src.dem.core import make_D_matrix, make_tilde_precision
from src.dem.inference import DStep, compute_vfe


# ---------------------------------------------------------------------------
# Harmonic oscillator model definition
# ---------------------------------------------------------------------------

def make_harmonic_oscillator_model(
    omega: float = 1.0,
    n_order: int = 4,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
    s_y: float = 1.0,
    s_x: float = 1.0,
) -> DEMModel:
    """Create a DEM model for the harmonic oscillator ẍ + ω²x = v.

    State variables: q = [x, ẋ] (position and velocity)
    Equations of motion:
        ẋ₁ = x₂
        ẋ₂ = -ω²x₁ + v₁

    Args:
        omega: Angular frequency.
        n_order: Generalized coordinate embedding order.
        pi_y: Observation precision.
        pi_x: State noise precision.
        s_y: Observation noise smoothing parameter.
        s_x: State noise smoothing parameter.

    Returns:
        DEMModel for the harmonic oscillator.
    """
    n_x = 2  # State dimension (position and velocity)
    n_v = 1  # Cause dimension (external input)
    n_y = 1  # Observation dimension (position only)

    omega_sq = omega ** 2

    def f_osc(
        x_tilde: jnp.ndarray,
        v_tilde: jnp.ndarray,
        params: None,
    ) -> jnp.ndarray:
        """Generalized state transition function for the harmonic oscillator.

        Applies A at each order: f([x,ẋ]) = [ẋ, -ω²x + v]

        Args:
            x_tilde: Generalized state, shape (n_x * n_order,).
            v_tilde: Generalized cause, shape (n_v * n_order,).
            params: Unused.

        Returns:
            f(x_tilde, v_tilde), shape (n_x * n_order,).
        """
        A = jnp.array([[0.0, 1.0], [-omega_sq, 0.0]])
        B = jnp.array([[0.0], [1.0]])

        result = []
        for i in range(n_order):
            x_i = x_tilde[i * n_x : (i + 1) * n_x]
            v_i_start = i * n_v
            v_i = v_tilde[v_i_start : v_i_start + n_v]
            Ax_i = A @ x_i + B @ v_i
            result.append(Ax_i)
        return jnp.concatenate(result)

    def g_osc(
        x_tilde: jnp.ndarray,
        v_tilde: jnp.ndarray,
        params: None,
    ) -> jnp.ndarray:
        """Generalized observation function for the harmonic oscillator y = x (position only).

        Args:
            x_tilde: Generalized state, shape (n_x * n_order,).
            v_tilde: Generalized cause, unused.
            params: Unused.

        Returns:
            g(x_tilde), shape (n_y * n_order,).
        """
        C = jnp.array([[1.0, 0.0]])  # Position only
        result = []
        for i in range(n_order):
            x_i = x_tilde[i * n_x : (i + 1) * n_x]
            result.append(C @ x_i)
        return jnp.concatenate(result)

    return DEMModel(
        f=f_osc,
        g=g_osc,
        n_x=n_x,
        n_v=n_v,
        n_y=n_y,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        s_y=s_y,
        s_x=s_x,
        params=None,
    )


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_oscillator_trajectory(
    omega: float = 1.0,
    x0: float = 1.0,
    v_input: float = 0.0,
    dt: float = 0.1,
    T: int = 100,
    noise_std: float = 0.2,
    key: jax.Array | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a harmonic oscillator trajectory.

    Args:
        omega: Angular frequency.
        x0: Initial position.
        v_input: Constant external input.
        dt: Time step.
        T: Number of time steps.
        noise_std: Standard deviation of observation noise.
        key: JAX random key.

    Returns:
        (true position trajectory shape (T,), true velocity trajectory shape (T,),
         noisy observations shape (T,)).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    x_pos = np.zeros(T)
    x_vel = np.zeros(T)

    x = x0
    xdot = 0.0
    for t in range(T):
        x_pos[t] = x
        x_vel[t] = xdot
        # Euler integration
        xddot = -omega ** 2 * x + v_input
        xdot = xdot + dt * xddot
        x = x + dt * xdot

    noise = np.array(jax.random.normal(key, (T,))) * noise_std
    y_obs = x_pos + noise
    return x_pos, x_vel, y_obs


# ---------------------------------------------------------------------------
# VFE trajectory collection
# ---------------------------------------------------------------------------

def collect_vfe_trajectory(
    d_step: DStep,
    mu_x0: jnp.ndarray,
    mu_v0: jnp.ndarray,
    y_tilde: jnp.ndarray,
    n_outer_steps: int = 50,
) -> tuple[list[float], jnp.ndarray, jnp.ndarray]:
    """Run D-step n_outer_steps times and collect VFE at each iteration.

    Args:
        d_step: DStep instance.
        mu_x0: Initial state mean.
        mu_v0: Initial cause mean.
        y_tilde: Generalized observation vector.
        n_outer_steps: Number of outer loop iterations.

    Returns:
        (list of VFE values, final mu_x_tilde, final mu_v_tilde).
    """
    mu_x = mu_x0
    mu_v = mu_v0
    vfe_history = []

    for _ in range(n_outer_steps):
        mu_x, mu_v = d_step.run_single_step(mu_x, mu_v, y_tilde)
        vfe = float(compute_vfe(mu_x, mu_v, y_tilde, d_step.model))
        vfe_history.append(vfe)

    return vfe_history, mu_x, mu_v


# ---------------------------------------------------------------------------
# Main comparison experiment
# ---------------------------------------------------------------------------

def run_comparison(
    omega: float = 1.0,
    n_order: int = 4,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
    dt_gd: float = 0.001,
    dt_ll: float = 0.01,
    n_outer_steps: int = 100,
) -> dict:
    """Compare gradient descent and local linearization.

    Args:
        omega: Angular frequency of the oscillator.
        n_order: Generalized coordinate embedding order.
        pi_y: Observation precision.
        pi_x: State noise precision.
        dt_gd: Gradient descent time step.
        dt_ll: Local linearization time step (larger and stable).
        n_outer_steps: Number of outer loop iterations.

    Returns:
        Dictionary of comparison results.
    """
    print(f"Harmonic oscillator model: omega={omega}, n_order={n_order}")
    print(f"Precisions: pi_y={pi_y}, pi_x={pi_x}")
    print(f"GD dt={dt_gd}, LL dt={dt_ll}, n_steps={n_outer_steps}")
    print()

    model = make_harmonic_oscillator_model(
        omega=omega, n_order=n_order, pi_y=pi_y, pi_x=pi_x
    )

    # Prepare observation (stationary state at position = 1.0)
    y_tilde = jnp.zeros(model.dim_y_tilde)
    y_tilde = y_tilde.at[0].set(1.0)  # Zeroth-order observation y₀ = 1.0

    # Initial estimates (zero)
    mu_x0 = jnp.zeros(model.dim_x_tilde)
    mu_v0 = jnp.zeros(model.dim_v_tilde)

    # Initial VFE
    vfe_init = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))
    print(f"Initial VFE: {vfe_init:.4f}")

    # ---- Gradient descent (use_d_operator=False) ----
    d_step_gd = DStep(
        model,
        kappa_mu=1.0,
        dt=dt_gd,
        n_iter=1,
        use_d_operator=False,
        use_local_linearization=False,
    )
    vfe_gd, mu_x_gd, mu_v_gd = collect_vfe_trajectory(
        d_step_gd, mu_x0, mu_v0, y_tilde, n_outer_steps
    )

    # ---- Full DEM (use_d_operator=True, use_local_linearization=True) ----
    d_step_ll = DStep(
        model,
        kappa_mu=1.0,
        dt=dt_ll,
        n_iter=1,
        use_d_operator=True,
        use_local_linearization=True,
    )
    vfe_ll, mu_x_ll, mu_v_ll = collect_vfe_trajectory(
        d_step_ll, mu_x0, mu_v0, y_tilde, n_outer_steps
    )

    # Display results
    print(f"GD  Final VFE: {vfe_gd[-1]:.4f}, Position estimate: {float(mu_x_gd[0]):.4f}")
    print(f"LL  Final VFE: {vfe_ll[-1]:.4f}, Position estimate: {float(mu_x_ll[0]):.4f}")
    print()

    return {
        "model": model,
        "y_tilde": y_tilde,
        "vfe_init": vfe_init,
        "vfe_gd": vfe_gd,
        "vfe_ll": vfe_ll,
        "mu_x_gd": mu_x_gd,
        "mu_x_ll": mu_x_ll,
        "mu_v_gd": mu_v_gd,
        "mu_v_ll": mu_v_ll,
        "dt_gd": dt_gd,
        "dt_ll": dt_ll,
        "n_outer_steps": n_outer_steps,
    }


def run_tracking_comparison(
    omega: float = 1.0,
    noise_std: float = 0.2,
    n_order: int = 4,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
    T: int = 30,
    dt_obs: float = 0.1,
    dt_gd: float = 0.001,
    dt_ll: float = 0.01,
    n_inner_steps: int = 50,
) -> dict:
    """Time-series tracking comparison experiment.

    Args:
        omega: Angular frequency of the oscillator.
        noise_std: Observation noise standard deviation.
        n_order: Embedding order.
        pi_y: Observation precision.
        pi_x: State noise precision.
        T: Length of the time series.
        dt_obs: Observation time step.
        dt_gd: GD integration step.
        dt_ll: LL integration step.
        n_inner_steps: Number of internal update steps per observation.

    Returns:
        Dictionary of tracking comparison results.
    """
    print(f"Time-series tracking experiment: T={T}, noise_std={noise_std}")

    # Generate true trajectory
    x_pos, x_vel, y_obs = generate_oscillator_trajectory(
        omega=omega, x0=1.0, dt=dt_obs, T=T, noise_std=noise_std
    )

    model = make_harmonic_oscillator_model(
        omega=omega, n_order=n_order, pi_y=pi_y, pi_x=pi_x
    )

    # DStep configuration
    d_step_gd = DStep(
        model, kappa_mu=1.0, dt=dt_gd, n_iter=n_inner_steps,
        use_d_operator=False, use_local_linearization=False,
    )
    d_step_ll = DStep(
        model, kappa_mu=1.0, dt=dt_ll, n_iter=n_inner_steps,
        use_d_operator=True, use_local_linearization=True,
    )

    mu_x_gd = jnp.zeros(model.dim_x_tilde)
    mu_v_gd = jnp.zeros(model.dim_v_tilde)
    mu_x_ll = jnp.zeros(model.dim_x_tilde)
    mu_v_ll = jnp.zeros(model.dim_v_tilde)

    est_gd = []  # Position estimate (GD)
    est_ll = []  # Position estimate (LL)
    vfe_gd_track = []
    vfe_ll_track = []

    for t in range(T):
        # Generalized observation vector (only zeroth order set)
        y_tilde = jnp.zeros(model.dim_y_tilde)
        y_tilde = y_tilde.at[0].set(float(y_obs[t]))

        # GD update
        mu_x_gd, mu_v_gd, vfe_gd = d_step_gd.run(mu_x_gd, mu_v_gd, y_tilde)
        est_gd.append(float(mu_x_gd[0]))
        vfe_gd_track.append(vfe_gd)

        # LL update
        mu_x_ll, mu_v_ll, vfe_ll = d_step_ll.run(mu_x_ll, mu_v_ll, y_tilde)
        est_ll.append(float(mu_x_ll[0]))
        vfe_ll_track.append(vfe_ll)

    est_gd = np.array(est_gd)
    est_ll = np.array(est_ll)

    rmse_gd = float(np.sqrt(np.mean((est_gd - x_pos) ** 2)))
    rmse_ll = float(np.sqrt(np.mean((est_ll - x_pos) ** 2)))

    print(f"GD  RMSE: {rmse_gd:.4f}")
    print(f"LL  RMSE: {rmse_ll:.4f}")

    return {
        "x_pos": x_pos,
        "y_obs": y_obs,
        "est_gd": est_gd,
        "est_ll": est_ll,
        "vfe_gd_track": vfe_gd_track,
        "vfe_ll_track": vfe_ll_track,
        "rmse_gd": rmse_gd,
        "rmse_ll": rmse_ll,
        "T": T,
        "dt_obs": dt_obs,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_comparison_figure(
    conv_result: dict,
    track_result: dict,
    save_path: str,
) -> None:
    """Generate and save the comparison results plot.

    Args:
        conv_result: Return value of run_comparison().
        track_result: Return value of run_tracking_comparison().
        save_path: File path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "DEM D-step: Gradient Descent vs Local Linearization (Ozaki 1992)",
        fontsize=13, fontweight="bold"
    )

    steps = np.arange(1, conv_result["n_outer_steps"] + 1)

    # ---- Panel 1: VFE convergence curve ----
    ax = axes[0, 0]
    vfe_gd = np.array(conv_result["vfe_gd"])
    vfe_ll = np.array(conv_result["vfe_ll"])

    # Exclude NaN/Inf
    valid_gd = np.isfinite(vfe_gd)
    valid_ll = np.isfinite(vfe_ll)

    ax.plot(steps[valid_gd], vfe_gd[valid_gd], "b-", lw=1.5,
            label=f"Gradient Descent (dt={conv_result['dt_gd']})")
    ax.plot(steps[valid_ll], vfe_ll[valid_ll], "r-", lw=1.5,
            label=f"Local Linearization (dt={conv_result['dt_ll']})")
    ax.axhline(y=conv_result["vfe_init"], color="k", linestyle="--", alpha=0.5,
               label=f"Initial VFE = {conv_result['vfe_init']:.2f}")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("VFE", fontsize=11)
    ax.set_title("VFE Convergence (Static Observation)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: VFE log scale ----
    ax = axes[0, 1]
    vfe_gd_pos = np.where(vfe_gd > 0, vfe_gd, np.nan)
    vfe_ll_pos = np.where(vfe_ll > 0, vfe_ll, np.nan)

    ax.semilogy(steps, vfe_gd_pos, "b-", lw=1.5,
                label=f"Gradient Descent (dt={conv_result['dt_gd']})")
    ax.semilogy(steps, vfe_ll_pos, "r-", lw=1.5,
                label=f"Local Linearization (dt={conv_result['dt_ll']})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("VFE (log scale)", fontsize=11)
    ax.set_title("VFE Convergence (Log Scale)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ---- Panel 3: Time-series tracking ----
    ax = axes[1, 0]
    t_arr = np.arange(track_result["T"]) * track_result["dt_obs"]

    ax.plot(t_arr, track_result["x_pos"], "k-", lw=2, label="True position")
    ax.scatter(t_arr, track_result["y_obs"], s=10, c="gray", alpha=0.5,
               label="Observations")
    ax.plot(t_arr, track_result["est_gd"], "b--", lw=1.5,
            label=f"GD (RMSE={track_result['rmse_gd']:.3f})")
    ax.plot(t_arr, track_result["est_ll"], "r-.", lw=1.5,
            label=f"LL (RMSE={track_result['rmse_ll']:.3f})")
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title("Oscillator Tracking", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Tracking VFE comparison ----
    ax = axes[1, 1]
    t_arr = np.arange(track_result["T"]) * track_result["dt_obs"]

    vfe_gd_t = np.array(track_result["vfe_gd_track"])
    vfe_ll_t = np.array(track_result["vfe_ll_track"])

    ax.plot(t_arr, vfe_gd_t, "b-", lw=1.5,
            label=f"Gradient Descent")
    ax.plot(t_arr, vfe_ll_t, "r-", lw=1.5,
            label=f"Local Linearization")
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("VFE", fontsize=11)
    ax.set_title("VFE During Tracking", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotation text
    gd_final = vfe_gd[-1] if np.isfinite(vfe_gd[-1]) else float("nan")
    ll_final = vfe_ll[-1] if np.isfinite(vfe_ll[-1]) else float("nan")

    info_text = (
        f"Final VFE: GD={gd_final:.3f}, LL={ll_final:.3f}\n"
        f"LL effective step = {conv_result['dt_ll'] / conv_result['dt_gd']:.0f}x GD\n"
        f"Integration: Ozaki (1992) local linearization"
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, color="gray",
             verticalalignment="bottom")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the main comparison experiment."""
    print("=" * 60)
    print("DEM D-step Numerical Stabilization Demo")
    print("Harmonic Oscillator: Gradient Descent vs Local Linearization")
    print("=" * 60)
    print()

    # Experiment parameters
    omega = 1.0
    n_order = 4
    pi_y = 4.0
    pi_x = 1.0

    # Experiment 1: convergence comparison on static observation
    print("[Experiment 1] Convergence comparison on static observation")
    print("-" * 40)
    conv_result = run_comparison(
        omega=omega,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        dt_gd=0.001,
        dt_ll=0.01,   # 10x larger dt than GD yet still stable
        n_outer_steps=100,
    )

    # Experiment 2: time-series tracking comparison
    print("[Experiment 2] Time-series tracking comparison")
    print("-" * 40)
    track_result = run_tracking_comparison(
        omega=omega,
        noise_std=0.2,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        T=30,
        dt_obs=0.1,
        dt_gd=0.001,
        dt_ll=0.01,
        n_inner_steps=50,
    )

    # Generate and save plot
    save_path = str(project_root / "results" / "dem_demo_generalized_motion.png")
    make_comparison_figure(conv_result, track_result, save_path)

    print()
    print("=" * 60)
    print("Demo complete")
    print(f"Results: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
