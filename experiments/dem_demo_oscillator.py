"""DEM demonstration on a harmonic oscillator system.

This script demonstrates the DEM (Dynamic Expectation Maximization) algorithm
for state tracking in a 2D harmonic oscillator:

    dx/dt = [[0, 1], [-omega^2, -gamma]] * x
    y = [1, 0] * x  (observe position only)

The agent infers both position and velocity from noisy position observations.

Steps:
1. Generate true oscillator trajectory (position + velocity)
2. Create noisy position observations y(t) = x_pos(t) + noise
3. Run DEMAgent to infer position from observations
4. Plot true trajectory vs DEM estimates
5. Save figure to results/dem_demo_oscillator.png
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.dem.model import LinearDEMModel
from src.dem.agent import DEMAgent


def generate_oscillator_trajectory(
    omega: float,
    gamma: float,
    x0: np.ndarray,
    dt: float,
    T: int,
    noise_std: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate trajectory from a damped harmonic oscillator.

    System:
        dx1/dt = x2                (x1 = position)
        dx2/dt = -omega^2*x1 - gamma*x2  (x2 = velocity)
        y = x1 + noise

    Args:
        omega: Natural frequency (rad/s).
        gamma: Damping coefficient.
        x0: Initial state [position, velocity], shape (2,).
        dt: Time step (s).
        T: Number of time steps.
        noise_std: Observation noise standard deviation.
        seed: Random seed.

    Returns:
        Tuple of (times, true_states shape (T,2), noisy observations shape (T,)).
    """
    rng = np.random.default_rng(seed)
    times = np.arange(T) * dt

    # System matrix
    A = np.array([[0.0, 1.0], [-(omega**2), -gamma]])

    states = np.zeros((T, 2))
    x = x0.copy()
    for t in range(T):
        states[t] = x
        # Euler integration
        dx = A @ x
        x = x + dt * dx

    noise = rng.normal(0.0, noise_std, T)
    observations = states[:, 0] + noise  # observe position only
    return times, states, observations


def run_dem_inference_oscillator(
    states: np.ndarray,
    observations: np.ndarray,
    omega: float,
    n_order: int = 4,
    pi_y: float = 16.0,
    pi_x: float = 0.5,
    kappa_mu: float = 1.0,
    n_iter_per_step: int = 300,
) -> tuple[np.ndarray, list[float]]:
    """Run DEM agent to infer oscillator position from noisy observations.

    Args:
        states: True state trajectory, shape (T, 2).
        observations: Noisy position observations, shape (T,).
        omega: Natural frequency (used to define model).
        n_order: Generalized coordinates embedding order.
        pi_y: Observation precision.
        pi_x: State noise precision.
        kappa_mu: D-step learning rate.
        n_iter_per_step: Number of gradient steps per observation.

    Returns:
        Tuple of (estimated positions, VFE history).
    """
    T = len(states)

    # Build linear model for the oscillator
    A = jnp.array([[0.0, 1.0], [-(omega**2), -1.0]])  # slight damping
    C = jnp.array([[1.0, 0.0]])  # observe position only

    model = LinearDEMModel(
        A, C,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        s_y=1.0,
        s_x=1.0,
    )

    print(f"  Oscillator model: n_x={model.n_x}, n_y={model.n_y}, n_order={n_order}")
    print(f"  State dim: {model.dim_x_tilde}, Obs dim: {model.dim_y_tilde}")

    agent = DEMAgent(
        model,
        kappa_mu=kappa_mu,
        dt=0.001,
        n_iter_per_step=n_iter_per_step,
        use_d_operator=False,
    )

    # Build generalized observation sequence (only position, zeroth order)
    y_tilde_sequence = []
    for t in range(T):
        y_t = jnp.zeros(model.dim_y_tilde)
        y_t = y_t.at[0].set(float(observations[t]))
        y_tilde_sequence.append(y_t)

    # Initialize at 0
    mu_x0 = jnp.zeros(model.dim_x_tilde)
    mu_v0 = jnp.zeros(model.dim_v_tilde)

    final_state, mu_x_history = agent.run(y_tilde_sequence, mu_x0, mu_v0)

    # Extract zeroth-order position estimate (first component)
    estimated_positions = np.array([float(mu[0]) for mu in mu_x_history[1:]])

    return estimated_positions, final_state.vfe_history


def plot_oscillator_results(
    times: np.ndarray,
    states: np.ndarray,
    observations: np.ndarray,
    estimated_positions: np.ndarray,
    vfe_history: list[float],
    output_path: str,
) -> None:
    """Create and save the oscillator tracking figure.

    Args:
        times: Time axis, shape (T,).
        states: True states [position, velocity], shape (T, 2).
        observations: Noisy position observations, shape (T,).
        estimated_positions: DEM position estimates, shape (T,).
        vfe_history: VFE values over observation steps.
        output_path: Path to save the figure.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    true_pos = states[:, 0]
    true_vel = states[:, 1]

    # --- Panel 1: Position tracking ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, true_pos, "b-", linewidth=2.0, label="True position $x_1(t)$",
             zorder=3)
    ax1.scatter(
        times, observations,
        color="lightblue", s=12, alpha=0.6,
        label="Noisy observation $y(t) = x_1 + \\epsilon$", zorder=2
    )
    ax1.plot(
        times, estimated_positions,
        "r--", linewidth=1.8, label="DEM estimate $\\hat{x}_1(t)$", zorder=4
    )
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Position", fontsize=12)
    ax1.set_title(
        "DEM Tracking: Damped Harmonic Oscillator  (position observation only)",
        fontsize=13
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Phase portrait ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(true_pos, true_vel, "b-", linewidth=1.5, label="True trajectory",
             alpha=0.8)
    ax2.scatter(true_pos[0], true_vel[0], color="blue", s=60, zorder=5,
                label="Start")
    ax2.scatter(true_pos[-1], true_vel[-1], color="darkblue", s=60, marker="*",
                zorder=5, label="End")
    ax2.set_xlabel("Position $x_1$", fontsize=11)
    ax2.set_ylabel("Velocity $x_2$", fontsize=11)
    ax2.set_title("Phase Portrait (True trajectory)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: VFE convergence ---
    ax3 = fig.add_subplot(gs[1, 1])
    steps = np.arange(1, len(vfe_history) + 1)
    ax3.plot(steps, vfe_history, "g-", linewidth=1.5)
    ax3.set_xlabel("Observation step", fontsize=11)
    ax3.set_ylabel("Variational Free Energy", fontsize=11)
    ax3.set_title("VFE over Observation Steps", fontsize=12)
    ax3.grid(True, alpha=0.3)
    if vfe_history and vfe_history[0] > 0 and vfe_history[-1] > 0:
        ax3.set_yscale("log")

    # Summary
    rmse = np.sqrt(np.mean((estimated_positions - true_pos) ** 2))
    fig.text(
        0.5, 0.01,
        f"RMSE = {rmse:.4f}  |  Final VFE = {vfe_history[-1]:.4f}  |  T = {len(times)} steps",
        ha="center", fontsize=10, color="gray",
        bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3")
    )

    plt.suptitle(
        "DEM Demo — Damped Harmonic Oscillator Position Tracking",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main() -> None:
    """Main entry point for the oscillator DEM demonstration."""
    print("=" * 60)
    print("DEM Demo: Damped Harmonic Oscillator")
    print("=" * 60)
    print()

    # System parameters
    omega = 1.5         # Natural frequency (rad/s)
    gamma = 0.5         # Damping coefficient
    x0 = np.array([1.0, 0.0])  # [position, velocity]
    dt = 0.05           # Time step (s)
    T = 80              # Number of steps
    noise_std = 0.1     # Observation noise

    print(f"System: omega={omega} rad/s, gamma={gamma}")
    print(f"Initial state: position={x0[0]}, velocity={x0[1]}")
    print(f"Observation: position only, noise_std={noise_std}")
    print(f"Parameters: T={T}, dt={dt}")
    print()

    # 1. Generate trajectory
    print("Step 1: Generating oscillator trajectory...")
    times, states, observations = generate_oscillator_trajectory(
        omega=omega, gamma=gamma, x0=x0, dt=dt, T=T,
        noise_std=noise_std, seed=7
    )
    print(f"  Position range: [{states[:, 0].min():.4f}, {states[:, 0].max():.4f}]")
    print()

    # 2. Run DEM inference
    print("Step 2: Running DEM inference (position only)...")
    estimated_positions, vfe_history = run_dem_inference_oscillator(
        states=states,
        observations=observations,
        omega=omega,
        n_order=4,
        pi_y=16.0,
        pi_x=0.5,
        kappa_mu=1.0,
        n_iter_per_step=300,
    )
    print()

    # 3. Metrics
    true_pos = states[:, 0]
    rmse = np.sqrt(np.mean((estimated_positions - true_pos) ** 2))
    print(f"Step 3: Performance metrics:")
    print(f"  Position RMSE: {rmse:.4f}")
    print(f"  Initial VFE:   {vfe_history[0]:.4f}")
    print(f"  Final VFE:     {vfe_history[-1]:.4f}")
    print()

    # 4. Plot
    print("Step 4: Plotting results...")
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / "dem_demo_oscillator.png")

    plot_oscillator_results(
        times=times,
        states=states,
        observations=observations,
        estimated_positions=estimated_positions,
        vfe_history=vfe_history,
        output_path=output_path,
    )

    print()
    print("=" * 60)
    print("Oscillator demo completed successfully!")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
