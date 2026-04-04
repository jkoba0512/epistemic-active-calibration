"""DEM demonstration on a linear dynamical system.

This script demonstrates the DEM (Dynamic Expectation Maximization) algorithm
for state estimation in a linear first-order system:

    dx/dt = A*x + v = -x + v
    y = C*x = x + noise

Steps:
1. Generate true state trajectory x(t) from the linear system
2. Create noisy observations y(t)
3. Run DEMAgent to infer the hidden state from observations
4. Plot true state vs estimated state and VFE convergence
5. Save figure to results/dem_demo_linear.png
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.dem.model import LinearDEMModel
from src.dem.agent import DEMAgent


def generate_linear_system_trajectory(
    A: np.ndarray,
    x0: float,
    v: float,
    dt: float,
    T: int,
    noise_std: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate trajectory from dx/dt = A*x + v, y = x + noise.

    Args:
        A: System matrix (scalar for 1D system).
        x0: Initial state.
        v: Constant cause/input.
        dt: Time step.
        T: Number of time steps.
        noise_std: Observation noise standard deviation.
        seed: Random seed.

    Returns:
        Tuple of (times, true states, noisy observations).
    """
    rng = np.random.default_rng(seed)
    times = np.arange(T) * dt
    states = np.zeros(T)
    x = x0
    for t in range(T):
        states[t] = x
        # Euler integration: dx/dt = A*x + v
        dx = A[0, 0] * x + v
        x = x + dt * dx

    noise = rng.normal(0.0, noise_std, T)
    observations = states + noise
    return times, states, observations


def run_dem_inference(
    states: np.ndarray,
    observations: np.ndarray,
    dt: float,
    n_order: int = 4,
    pi_y: float = 16.0,
    pi_x: float = 1.0,
    kappa_mu: float = 1.0,
    n_iter_per_step: int = 200,
) -> tuple[np.ndarray, list[float]]:
    """Run DEM agent to infer states from observations.

    Args:
        states: True state trajectory, shape (T,).
        observations: Noisy observations, shape (T,).
        dt: Observation time step.
        n_order: Generalized coordinates embedding order.
        pi_y: Observation precision.
        pi_x: State noise precision.
        kappa_mu: D-step learning rate.
        n_iter_per_step: Number of gradient steps per observation.

    Returns:
        Tuple of (estimated states, VFE history).
    """
    T = len(states)

    # Build linear model: dx/dt = -x + v, y = x
    A = jnp.array([[-1.0]])
    C = jnp.array([[1.0]])
    model = LinearDEMModel(
        A, C,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        s_y=1.0,
        s_x=1.0,
    )

    # Create agent (gradient descent mode for stability)
    agent = DEMAgent(
        model,
        kappa_mu=kappa_mu,
        dt=0.001,  # Internal integration step
        n_iter_per_step=n_iter_per_step,
        use_d_operator=False,  # Stable gradient descent
    )

    # Build generalized observation sequence
    # Only zeroth-order observation is set; higher orders remain 0
    y_tilde_sequence = []
    for t in range(T):
        y_t = jnp.zeros(model.dim_y_tilde)
        y_t = y_t.at[0].set(float(observations[t]))
        y_tilde_sequence.append(y_t)

    # Initialize state estimate at 0 (far from true value 1.0)
    mu_x0 = jnp.zeros(model.dim_x_tilde)
    mu_v0 = jnp.zeros(model.dim_v_tilde)

    print(f"  Running DEMAgent: T={T}, n_order={n_order}, pi_y={pi_y}")
    print(f"  n_iter_per_step={n_iter_per_step}, total steps={T * n_iter_per_step}")

    final_state, mu_x_history = agent.run(y_tilde_sequence, mu_x0, mu_v0)

    # Extract zeroth-order state estimates
    estimated_states = np.array([float(mu[0]) for mu in mu_x_history[1:]])

    return estimated_states, final_state.vfe_history


def plot_results(
    times: np.ndarray,
    states: np.ndarray,
    observations: np.ndarray,
    estimated_states: np.ndarray,
    vfe_history: list[float],
    output_path: str,
) -> None:
    """Create and save the results figure.

    Args:
        times: Time axis, shape (T,).
        states: True states, shape (T,).
        observations: Noisy observations, shape (T,).
        estimated_states: DEM state estimates, shape (T,).
        vfe_history: VFE at each observation step.
        output_path: Path to save the figure.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: State estimation (full trajectory) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, states, "b-", linewidth=2.0, label="True state $x(t)$", zorder=3)
    ax1.scatter(
        times, observations,
        color="lightgray", s=12, alpha=0.7, label="Noisy observations $y(t)$",
        zorder=2
    )
    ax1.plot(
        times, estimated_states,
        "r--", linewidth=1.8, label="DEM estimate $\\hat{x}(t)$", zorder=4
    )
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("State", fontsize=12)
    ax1.set_title(
        "DEM State Estimation: Linear System  $dx/dt = -x$, $y = x + \\epsilon$",
        fontsize=13
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Estimation error over time ---
    ax2 = fig.add_subplot(gs[1, 0])
    error = np.abs(estimated_states - states)
    ax2.plot(times, error, "m-", linewidth=1.5, label="|True - Estimate|")
    ax2.axhline(y=np.mean(error), color="m", linestyle=":", alpha=0.7,
                label=f"Mean error = {np.mean(error):.4f}")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Absolute error", fontsize=11)
    ax2.set_title("State Estimation Error", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # --- Panel 3: VFE convergence ---
    ax3 = fig.add_subplot(gs[1, 1])
    steps = np.arange(1, len(vfe_history) + 1)
    ax3.plot(steps, vfe_history, "g-", linewidth=1.5, label="VFE")
    ax3.set_xlabel("Observation step", fontsize=11)
    ax3.set_ylabel("Variational Free Energy", fontsize=11)
    ax3.set_title("VFE Convergence over Observations", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    if vfe_history and vfe_history[0] > 0:
        ax3.set_yscale("log")

    # Add summary text
    rmse = np.sqrt(np.mean((estimated_states - states) ** 2))
    fig.text(
        0.5, 0.01,
        f"RMSE = {rmse:.4f}  |  Final VFE = {vfe_history[-1]:.4f}  |  "
        f"Observations = {len(states)}",
        ha="center", fontsize=10, color="gray",
        bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3")
    )

    plt.suptitle(
        "DEM (Dynamic Expectation Maximization) Demo — Linear System",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main() -> None:
    """Main entry point for the linear DEM demonstration."""
    print("=" * 60)
    print("DEM Demo: Linear Dynamical System")
    print("=" * 60)
    print()

    # System parameters
    dt = 0.1        # Observation time step (s)
    T = 60          # Number of time steps
    x0 = 1.0        # Initial state
    v = 0.0         # No external input (dx/dt = -x decays to 0)
    noise_std = 0.15  # Observation noise
    A = np.array([[-1.0]])

    print(f"System: dx/dt = -x,  y = x + N(0, {noise_std}^2)")
    print(f"Parameters: x0={x0}, T={T}, dt={dt}")
    print()

    # 1. Generate true trajectory and noisy observations
    print("Step 1: Generating true trajectory...")
    times, states, observations = generate_linear_system_trajectory(
        A=A, x0=x0, v=v, dt=dt, T=T, noise_std=noise_std, seed=42
    )
    print(f"  True state range: [{states.min():.4f}, {states.max():.4f}]")
    print(f"  Observation noise: std={np.std(observations - states):.4f}")
    print()

    # 2. Run DEM inference
    print("Step 2: Running DEM inference...")
    estimated_states, vfe_history = run_dem_inference(
        states=states,
        observations=observations,
        dt=dt,
        n_order=4,
        pi_y=16.0,
        pi_x=1.0,
        kappa_mu=1.0,
        n_iter_per_step=300,
    )
    print()

    # 3. Compute metrics
    rmse = np.sqrt(np.mean((estimated_states - states) ** 2))
    mean_abs_error = np.mean(np.abs(estimated_states - states))
    print(f"Step 3: Performance metrics:")
    print(f"  RMSE:            {rmse:.4f}")
    print(f"  Mean abs error:  {mean_abs_error:.4f}")
    print(f"  Initial VFE:     {vfe_history[0]:.4f}")
    print(f"  Final VFE:       {vfe_history[-1]:.4f}")
    print(f"  VFE reduction:   {(1 - vfe_history[-1] / vfe_history[0]) * 100:.1f}%")
    print()

    # 4. Plot and save
    print("Step 4: Plotting results...")
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / "dem_demo_linear.png")

    plot_results(
        times=times,
        states=states,
        observations=observations,
        estimated_states=estimated_states,
        vfe_history=vfe_history,
        output_path=output_path,
    )

    print()
    print("=" * 60)
    print("Demo completed successfully!")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
