"""DEM + MuJoCo: Pendulum proprioceptive state estimation

Demonstrates how a DEMAgent estimates both joint angle AND angular
velocity from noisy angle-only proprioceptive (encoder) observations.

Task:
    A hanging pendulum swings under gravity and sinusoidal torque.
    The only sensor is a noisy joint angle encoder.
    There is NO direct velocity measurement.

DEM output:
    - Filtered joint angle (smoother than raw encoder)
    - Angular velocity estimate (inferred from temporal structure)

Key insight:
    DEM's generalized coordinates [q, q', q'', q'''] embed the full
    temporal trajectory at each observation time. The D-step infers
    all generalized coordinates simultaneously by balancing:
        - Observation fit:  eps_y = y_tilde - x_tilde → small
        - Smoothness prior: eps_x = D @ x_tilde - f(x_tilde) → small

    This allows velocity estimation from position-only sensors.

Usage:
    .venv/bin/python experiments/dem_mujoco_pendulum.py

Output:
    results/dem_mujoco_pendulum.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
import numpy as np
import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.agent import DEMAgent


# ============================================================
# MuJoCo pendulum XML
# ============================================================

PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 -0.5" mass="1.0" diaginertia="0.083 0.083 0.001"/>
      <geom type="capsule" fromto="0 0 0  0 0 -1" size="0.04"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# ============================================================
# Simulation parameters
# ============================================================

SIM_DT      = 0.002        # MuJoCo integration timestep (s)
OBS_DT      = 0.05         # Observation interval (s) = 25 sim steps
T_END       = 5.0          # Total simulation time (s)
N_SIM_STEPS = int(T_END / SIM_DT)   # 2500
N_OBS       = int(T_END / OBS_DT)   # 100
OBS_EVERY   = int(OBS_DT / SIM_DT)  # 25

NOISE_STD   = 0.05         # Encoder noise (rad)
SEED        = 42

# Sinusoidal torque: tau = A * sin(2*pi*f*t)
TORQUE_AMP  = 1.5          # Torque amplitude (N·m)
TORQUE_FREQ = 0.3          # Forcing frequency (Hz) — below natural frequency (~0.62 Hz)

# Initial condition
Q0          = 0.3          # Initial angle (rad)
DQ0         = 0.0          # Initial angular velocity (rad/s)

# ============================================================
# DEM parameters
# ============================================================

N_ORDER       = 4
PI_Y          = 8.0        # Observation precision (high: trust encoder)
PI_X          = 2.0        # Dynamics precision (moderate: allow deviation)
S_SMOOTH      = 1.0        # Smoothness (noise correlation width)

N_ITER_D_STEP = 128        # D-step Euler iterations per observation
D_STEP_DT     = 0.01       # D-step virtual time step (chosen for convergence, not OBS_DT/N)
KAPPA_MU      = 1.0        # D-step learning rate


# ============================================================
# Step 1: MuJoCo simulation
# ============================================================

def run_mujoco_simulation():
    """Run pendulum simulation and collect ground truth + noisy observations.

    Returns:
        t_obs:   Observation times, shape (N_OBS,)
        q_true:  True joint angle, shape (N_OBS,)
        dq_true: True angular velocity, shape (N_OBS,)
        y_obs:   Noisy angle observations, shape (N_OBS,)
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data  = mujoco.MjData(model)

    data.qpos[0] = Q0
    data.qvel[0] = DQ0

    rng = np.random.default_rng(SEED)

    t_obs   = np.zeros(N_OBS)
    q_true  = np.zeros(N_OBS)
    dq_true = np.zeros(N_OBS)
    y_obs   = np.zeros(N_OBS)

    obs_idx = 0
    for step in range(N_SIM_STEPS):
        t = step * SIM_DT

        # Record observation at regular intervals
        if step % OBS_EVERY == 0 and obs_idx < N_OBS:
            t_obs[obs_idx]   = t
            q_true[obs_idx]  = float(data.qpos[0])
            dq_true[obs_idx] = float(data.qvel[0])
            y_obs[obs_idx]   = float(data.qpos[0]) + rng.normal(0.0, NOISE_STD)
            obs_idx += 1

        # Apply sinusoidal torque
        data.ctrl[0] = TORQUE_AMP * math.sin(2.0 * math.pi * TORQUE_FREQ * t)
        mujoco.mj_step(model, data)

    return t_obs, q_true, dq_true, y_obs


# ============================================================
# Step 2: Build DEM model
# ============================================================

def build_dem_model() -> DEMModel:
    """Build DEM model for pendulum proprioception.

    Uses a smoothness prior (f = 0) with full generalized-coordinate
    observation (g = identity). The D-step infers all orders of the
    generalized state [q, q', q'', q'''] from noisy angle observations.

    Smoothness prior: eps_x = D @ x_tilde - 0 = D @ x_tilde → 0
        Encourages temporal smoothness (higher derivatives small).

    Observation: eps_y = y_tilde - x_tilde → 0
        Pushes estimated generalized state to match observations.

    With y_tilde = [q_obs, q_dot_fd, 0, 0] (finite-diff velocity as
    second observation), the DEM jointly smooths both angle and velocity.
    """
    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        # Smoothness prior: no explicit dynamics model
        # eps_x = D @ x_tilde pushes higher-order derivatives toward 0
        return jnp.zeros(N_ORDER)

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        # Full generalized-coordinate observation: y_tilde ≈ x_tilde
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
    )


# ============================================================
# Step 3: Build generalized observation sequence
# ============================================================

def build_y_tilde_sequence(y_obs: np.ndarray) -> list:
    """Build generalized observation sequence.

    y_tilde[0] = q_obs          (noisy angle from encoder)
    y_tilde[1] = central-diff velocity estimate (noisy)
    y_tilde[2:] = 0             (no higher-order observations)

    The DEM will smooth and denoise both components.

    Args:
        y_obs: Noisy angle observations, shape (N_OBS,)

    Returns:
        List of N_OBS generalized observation vectors, each shape (N_ORDER,).
    """
    N = len(y_obs)
    y_tilde_list = []

    for i in range(N):
        y_tilde = np.zeros(N_ORDER)
        y_tilde[0] = y_obs[i]

        # Central finite-difference velocity estimate
        if 0 < i < N - 1:
            y_tilde[1] = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * OBS_DT)
        elif i == 0:
            y_tilde[1] = (y_obs[1] - y_obs[0]) / OBS_DT
        else:
            y_tilde[1] = (y_obs[-1] - y_obs[-2]) / OBS_DT

        y_tilde_list.append(jnp.array(y_tilde))

    return y_tilde_list


# ============================================================
# Step 4: Run DEM estimation
# ============================================================

def run_dem_estimation(y_tilde_list: list, y_obs: np.ndarray):
    """Run DEMAgent for proprioceptive state estimation.

    Args:
        y_tilde_list: List of generalized observation vectors.
        y_obs:        Raw noisy angle observations.

    Returns:
        q_est:   DEM angle estimates, shape (N_OBS,)
        dq_est:  DEM velocity estimates, shape (N_OBS,)
        vfe_arr: VFE at each timestep, shape (N_OBS,)
    """
    model = build_dem_model()
    agent = DEMAgent(
        model,
        kappa_mu=KAPPA_MU,
        dt=D_STEP_DT,
        n_iter_per_step=N_ITER_D_STEP,
        use_d_operator=False,
    )

    # Initialize state at first observation
    mu_x0 = jnp.zeros(N_ORDER).at[0].set(float(y_obs[0]))
    mu_v0 = jnp.zeros(N_ORDER)

    state, mu_x_history = agent.run(y_tilde_list, mu_x0=mu_x0, mu_v0=mu_v0)

    # Extract 0th-order (angle) and 1st-order (velocity) estimates
    # mu_x_history[0] is init, [1:] are post-step estimates
    q_est  = np.array([float(mu_x[0]) for mu_x in mu_x_history[1:]])
    dq_est = np.array([float(mu_x[1]) for mu_x in mu_x_history[1:]])
    vfe_arr = np.array(state.vfe_history)

    return q_est, dq_est, vfe_arr


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DEM + MuJoCo: Pendulum Proprioception Demo")
    print("  Velocity estimation from angle-only observations")
    print("=" * 60)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Simulation ---
    print("Step 1: MuJoCo pendulum simulation")
    t_obs, q_true, dq_true, y_obs = run_mujoco_simulation()
    rmse_obs = float(np.sqrt(np.mean((y_obs - q_true) ** 2)))
    print(f"  {N_OBS} observations at dt={OBS_DT}s over {T_END}s")
    print(f"  Angle range:    [{q_true.min():.3f}, {q_true.max():.3f}] rad")
    print(f"  Velocity range: [{dq_true.min():.3f}, {dq_true.max():.3f}] rad/s")
    print(f"  Encoder RMSE:   {rmse_obs:.4f} rad (noise std={NOISE_STD})")
    print()

    # --- Finite-difference baseline ---
    print("Step 2: Finite-difference velocity baseline")
    dq_fd = np.zeros(N_OBS)
    for i in range(N_OBS):
        if 0 < i < N_OBS - 1:
            dq_fd[i] = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * OBS_DT)
        elif i == 0:
            dq_fd[i] = (y_obs[1] - y_obs[0]) / OBS_DT
        else:
            dq_fd[i] = (y_obs[-1] - y_obs[-2]) / OBS_DT
    rmse_fd = float(np.sqrt(np.mean((dq_fd - dq_true) ** 2)))
    print(f"  Finite-diff velocity RMSE: {rmse_fd:.4f} rad/s")
    print()

    # --- DEM estimation ---
    print("Step 3: DEM state estimation")
    y_tilde_list = build_y_tilde_sequence(y_obs)
    q_est, dq_est, vfe_arr = run_dem_estimation(y_tilde_list, y_obs)

    rmse_q  = float(np.sqrt(np.mean((q_est  - q_true)  ** 2)))
    rmse_dq = float(np.sqrt(np.mean((dq_est - dq_true) ** 2)))
    print(f"  DEM angle RMSE:    {rmse_q:.4f} rad  (encoder: {rmse_obs:.4f})")
    print(f"  DEM velocity RMSE: {rmse_dq:.4f} rad/s (FD: {rmse_fd:.4f})")
    print()

    # --- Results summary ---
    angle_improve    = (rmse_obs - rmse_q)  / rmse_obs  * 100
    velocity_improve = (rmse_fd  - rmse_dq) / rmse_fd   * 100
    print("Step 4: Results")
    print(f"  Angle:    DEM {angle_improve:+.1f}% vs encoder noise")
    print(f"  Velocity: DEM {velocity_improve:+.1f}% vs finite differences")
    print()

    # --- Plot ---
    print("Step 5: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 10))

        # Panel 1: Angle estimation
        axes[0].plot(t_obs, q_true, 'b-', lw=2, label='True q(t) [MuJoCo]')
        axes[0].scatter(t_obs, y_obs, c='gray', s=12, alpha=0.6,
                        label=f'Encoder (RMSE={rmse_obs:.3f} rad)')
        axes[0].plot(t_obs, q_est, 'r-', lw=1.5,
                     label=f'DEM estimate (RMSE={rmse_q:.3f} rad)')
        axes[0].set_ylabel('Angle (rad)')
        axes[0].set_title('Joint angle: DEM filtering of noisy encoder')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Velocity estimation (DEM vs FD baseline)
        axes[1].plot(t_obs, dq_true, 'b-', lw=2, label='True dq/dt [MuJoCo]')
        axes[1].plot(t_obs, dq_fd, 'g--', lw=1.2, alpha=0.7,
                     label=f'Finite diff (RMSE={rmse_fd:.3f} rad/s)')
        axes[1].plot(t_obs, dq_est, 'r-', lw=1.5,
                     label=f'DEM estimate (RMSE={rmse_dq:.3f} rad/s)')
        axes[1].set_ylabel('Angular velocity (rad/s)')
        axes[1].set_title('Velocity estimation from angle-only observations')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Panel 3: VFE
        axes[2].plot(t_obs, vfe_arr, 'm-', lw=1.5)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('VFE')
        axes[2].set_title('Variational Free Energy (lower = better fit)')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            'DEM + MuJoCo: Pendulum Proprioception\n'
            f'(pi_y={PI_Y}, pi_x={PI_X}, n_order={N_ORDER})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_mujoco_pendulum.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping plot")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Encoder noise:         {rmse_obs:.4f} rad")
    print(f"  DEM angle RMSE:        {rmse_q:.4f} rad   [{angle_improve:+.1f}%]")
    print(f"  FD velocity RMSE:      {rmse_fd:.4f} rad/s")
    print(f"  DEM velocity RMSE:     {rmse_dq:.4f} rad/s [{velocity_improve:+.1f}%]")
    print("=" * 60)


if __name__ == "__main__":
    main()
