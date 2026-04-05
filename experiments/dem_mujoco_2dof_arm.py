"""DEM + MuJoCo: 2-DOF arm proprioceptive state estimation

Demonstrates DEMAgent estimating joint angles AND angular velocities
for a 2-DOF planar arm from noisy angle-only encoder observations.

Setup:
    A horizontal 2-DOF arm (zero gravity) is driven by independent
    sinusoidal torques at different frequencies.
    Each joint has a noisy angle encoder. No velocity sensors.

DEM output:
    - Filtered angles for both joints (x_tilde[0], x_tilde[1])
    - Angular velocity estimates (x_tilde[2], x_tilde[3])

Key design:
    n_x=2 requires kron(R, Pi_base) precision layout (order-first).
    This was the motivation for fixing make_tilde_precision in core.py.

Usage:
    .venv/bin/python experiments/dem_mujoco_2dof_arm.py

Output:
    results/dem_mujoco_2dof_arm.png
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
# MuJoCo 2-DOF arm XML  (horizontal, zero gravity)
# ============================================================

ARM_XML = """
<mujoco model="2dof_arm">
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <body name="link1">
      <joint name="joint1" type="hinge" axis="0 0 1" damping="5"/>
      <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.083 0.083 0.083"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="5"/>
        <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.083 0.083 0.083"/>
        <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="torque2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# ============================================================
# Simulation parameters
# ============================================================

SIM_DT      = 0.002
OBS_DT      = 0.05
T_END       = 8.0
N_SIM       = int(T_END / SIM_DT)
N_OBS       = int(T_END / OBS_DT)
OBS_EVERY   = int(OBS_DT / SIM_DT)

NOISE_STD   = 0.05   # encoder noise (rad)
SEED        = 42
N_JOINTS    = 2

# Independent sinusoidal torques at different frequencies
TORQUE_AMP1  = 1.5
TORQUE_FREQ1 = 0.3   # Hz — joint 1
TORQUE_AMP2  = 1.0
TORQUE_FREQ2 = 0.5   # Hz — joint 2

# Initial joint angles
Q0 = [0.0, 0.0]    # rad (start at rest; torques drive oscillation)
DQ0 = [0.0, 0.0]

# ============================================================
# DEM parameters
# ============================================================

N_ORDER       = 4
PI_Y          = 8.0
PI_X          = 2.0
S_SMOOTH      = 1.0
N_ITER_D_STEP = 128
D_STEP_DT     = 0.01    # fixed for convergence (not OBS_DT/N_ITER)
KAPPA_MU      = 1.0


# ============================================================
# Step 1: MuJoCo simulation
# ============================================================

def run_simulation():
    """Run 2-DOF arm simulation and collect ground truth + noisy observations.

    Returns:
        t_obs:   Observation times, shape (N_OBS,)
        q_true:  True joint angles, shape (N_OBS, 2)
        dq_true: True angular velocities, shape (N_OBS, 2)
        y_obs:   Noisy angle observations, shape (N_OBS, 2)
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data  = mujoco.MjData(model)

    data.qpos[:2] = Q0
    data.qvel[:2] = DQ0

    rng = np.random.default_rng(SEED)

    t_obs   = np.zeros(N_OBS)
    q_true  = np.zeros((N_OBS, N_JOINTS))
    dq_true = np.zeros((N_OBS, N_JOINTS))
    y_obs   = np.zeros((N_OBS, N_JOINTS))

    obs_idx = 0
    for step in range(N_SIM):
        t = step * SIM_DT

        if step % OBS_EVERY == 0 and obs_idx < N_OBS:
            t_obs[obs_idx]   = t
            q_true[obs_idx]  = data.qpos[:2].copy()
            dq_true[obs_idx] = data.qvel[:2].copy()
            y_obs[obs_idx]   = data.qpos[:2] + rng.normal(0.0, NOISE_STD, 2)
            obs_idx += 1

        data.ctrl[0] = TORQUE_AMP1 * math.sin(2.0 * math.pi * TORQUE_FREQ1 * t)
        data.ctrl[1] = TORQUE_AMP2 * math.sin(2.0 * math.pi * TORQUE_FREQ2 * t)
        mujoco.mj_step(model, data)

    return t_obs, q_true, dq_true, y_obs


# ============================================================
# Step 2: Build DEM model  (n_x=2, n_y=2, n_order=4)
# ============================================================

def build_dem_model() -> DEMModel:
    """Build DEM model for 2-DOF arm proprioception.

    x_tilde layout (order-first, interleaved, shape (8,)):
        [θ1, θ2,  θ1', θ2',  θ1'', θ2'',  θ1''', θ2''']

    Smoothness prior: f = 0  →  eps_x = D @ x_tilde pushes derivatives toward 0.
    Full generalized observation: g = identity  →  eps_y = y_tilde - x_tilde.
    """
    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        return jnp.zeros(N_ORDER * N_JOINTS)

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        return x_tilde

    return DEMModel(
        f=f,
        g=g,
        n_x=N_JOINTS,
        n_v=N_JOINTS,
        n_y=N_JOINTS,
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
    """Build generalized observation sequence for 2-DOF arm.

    y_tilde layout (order-first, interleaved, shape (8,)):
        [θ1_obs, θ2_obs,  θ1'_fd, θ2'_fd,  0, 0,  0, 0]

    Args:
        y_obs: Noisy angle observations, shape (N_OBS, 2)

    Returns:
        List of N_OBS generalized observation vectors, each shape (8,).
    """
    N = len(y_obs)
    y_tilde_list = []

    for i in range(N):
        y_tilde = np.zeros(N_ORDER * N_JOINTS)

        # Zeroth order: noisy encoder readings
        y_tilde[0] = y_obs[i, 0]   # θ1
        y_tilde[1] = y_obs[i, 1]   # θ2

        # First order: finite-difference velocity estimates
        if 0 < i < N - 1:
            dq = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * OBS_DT)
        elif i == 0:
            dq = (y_obs[1] - y_obs[0]) / OBS_DT
        else:
            dq = (y_obs[-1] - y_obs[-2]) / OBS_DT

        y_tilde[2] = dq[0]   # θ1'
        y_tilde[3] = dq[1]   # θ2'
        # y_tilde[4:] = 0 (higher orders not observed)

        y_tilde_list.append(jnp.array(y_tilde))

    return y_tilde_list


# ============================================================
# Step 4: Run DEM estimation
# ============================================================

def run_dem_estimation(y_tilde_list: list, y_obs: np.ndarray):
    """Run DEMAgent for 2-DOF arm state estimation.

    Returns:
        q_est:   DEM angle estimates, shape (N_OBS, 2)
        dq_est:  DEM velocity estimates, shape (N_OBS, 2)
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

    # Initialize from first observation (warm start)
    mu_x0 = jnp.zeros(N_ORDER * N_JOINTS)
    mu_x0 = mu_x0.at[0].set(float(y_obs[0, 0]))
    mu_x0 = mu_x0.at[1].set(float(y_obs[0, 1]))
    mu_v0 = jnp.zeros(N_ORDER * N_JOINTS)

    state, mu_x_history = agent.run(y_tilde_list, mu_x0=mu_x0, mu_v0=mu_v0)

    # Extract estimates from history (skip init)
    # x_tilde layout: [θ1, θ2, θ1', θ2', θ1'', θ2'', θ1''', θ2''']
    q_est  = np.array([[float(mu_x[0]), float(mu_x[1])] for mu_x in mu_x_history[1:]])
    dq_est = np.array([[float(mu_x[2]), float(mu_x[3])] for mu_x in mu_x_history[1:]])
    vfe_arr = np.array(state.vfe_history)

    return q_est, dq_est, vfe_arr


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DEM + MuJoCo: 2-DOF Arm Proprioception Demo")
    print("  Multi-joint velocity estimation from encoders only")
    print("=" * 60)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Simulation ---
    print("Step 1: MuJoCo 2-DOF arm simulation")
    t_obs, q_true, dq_true, y_obs = run_simulation()
    rmse_obs = np.sqrt(np.mean((y_obs - q_true) ** 2, axis=0))
    print(f"  {N_OBS} observations at dt={OBS_DT}s over {T_END}s")
    print(f"  Joint 1 range: [{q_true[:,0].min():.3f}, {q_true[:,0].max():.3f}] rad")
    print(f"  Joint 2 range: [{q_true[:,1].min():.3f}, {q_true[:,1].max():.3f}] rad")
    print(f"  Encoder RMSE:  J1={rmse_obs[0]:.4f}  J2={rmse_obs[1]:.4f} rad")
    print()

    # --- Finite-difference baseline ---
    print("Step 2: Finite-difference velocity baseline")
    dq_fd = np.zeros_like(dq_true)
    for i in range(N_OBS):
        if 0 < i < N_OBS - 1:
            dq_fd[i] = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * OBS_DT)
        elif i == 0:
            dq_fd[i] = (y_obs[1] - y_obs[0]) / OBS_DT
        else:
            dq_fd[i] = (y_obs[-1] - y_obs[-2]) / OBS_DT
    rmse_fd = np.sqrt(np.mean((dq_fd - dq_true) ** 2, axis=0))
    print(f"  FD velocity RMSE: J1={rmse_fd[0]:.4f}  J2={rmse_fd[1]:.4f} rad/s")
    print()

    # --- DEM estimation ---
    print("Step 3: DEM state estimation")
    y_tilde_list = build_y_tilde_sequence(y_obs)
    q_est, dq_est, vfe_arr = run_dem_estimation(y_tilde_list, y_obs)

    rmse_q  = np.sqrt(np.mean((q_est  - q_true)  ** 2, axis=0))
    rmse_dq = np.sqrt(np.mean((dq_est - dq_true) ** 2, axis=0))
    print(f"  DEM angle RMSE: J1={rmse_q[0]:.4f}  J2={rmse_q[1]:.4f} rad")
    print(f"    (encoder:     J1={rmse_obs[0]:.4f}  J2={rmse_obs[1]:.4f})")
    print(f"  DEM vel RMSE:   J1={rmse_dq[0]:.4f}  J2={rmse_dq[1]:.4f} rad/s")
    print(f"    (FD:          J1={rmse_fd[0]:.4f}  J2={rmse_fd[1]:.4f})")
    print()

    # --- Results summary ---
    print("Step 4: Results")
    for j, (enc, dem, fd, dem_v) in enumerate(
        zip(rmse_obs, rmse_q, rmse_fd, rmse_dq), start=1
    ):
        a_improve = (enc - dem) / enc * 100
        v_improve = (fd - dem_v) / fd * 100
        print(f"  Joint {j}: angle {a_improve:+.1f}% vs encoder | "
              f"velocity {v_improve:+.1f}% vs FD")
    print()

    # --- Plot ---
    print("Step 5: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(13, 11))

        for j in range(2):
            jlabel = f"Joint {j+1}"

            # Angle estimation
            ax = axes[0, j]
            ax.plot(t_obs, q_true[:, j], 'b-', lw=2, label=f'True θ{j+1}(t)')
            ax.scatter(t_obs, y_obs[:, j], c='gray', s=10, alpha=0.5,
                       label=f'Encoder (RMSE={rmse_obs[j]:.3f} rad)')
            ax.plot(t_obs, q_est[:, j], 'r-', lw=1.5,
                    label=f'DEM (RMSE={rmse_q[j]:.3f} rad)')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(f'{jlabel} — Angle filtering')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Velocity estimation
            ax = axes[1, j]
            ax.plot(t_obs, dq_true[:, j], 'b-', lw=2, label=f'True θ{j+1}\'(t)')
            ax.plot(t_obs, dq_fd[:, j], 'g--', lw=1.2, alpha=0.7,
                    label=f'FD (RMSE={rmse_fd[j]:.3f} rad/s)')
            ax.plot(t_obs, dq_est[:, j], 'r-', lw=1.5,
                    label=f'DEM (RMSE={rmse_dq[j]:.3f} rad/s)')
            ax.set_ylabel('Velocity (rad/s)')
            ax.set_title(f'{jlabel} — Velocity estimation')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # VFE
        ax = axes[2, 0]
        ax.plot(t_obs, vfe_arr, 'm-', lw=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VFE')
        ax.set_title('Variational Free Energy')
        ax.grid(True, alpha=0.3)

        # RMSE summary bar chart
        ax = axes[2, 1]
        labels = ['J1 angle', 'J2 angle', 'J1 vel', 'J2 vel']
        baseline = [rmse_obs[0], rmse_obs[1], rmse_fd[0], rmse_fd[1]]
        dem_vals = [rmse_q[0], rmse_q[1], rmse_dq[0], rmse_dq[1]]
        x = np.arange(4)
        ax.bar(x - 0.2, baseline, 0.4, label='Encoder/FD baseline', color='steelblue', alpha=0.8)
        ax.bar(x + 0.2, dem_vals, 0.4, label='DEM estimate', color='tomato', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            'DEM + MuJoCo: 2-DOF Arm Proprioception\n'
            f'(n_x=2, n_order={N_ORDER}, pi_y={PI_Y}, pi_x={PI_X})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_mujoco_2dof_arm.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping plot")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for j in range(2):
        a_improve = (rmse_obs[j] - rmse_q[j]) / rmse_obs[j] * 100
        v_improve = (rmse_fd[j] - rmse_dq[j]) / rmse_fd[j] * 100
        print(f"  Joint {j+1}: angle {a_improve:+.1f}%  |  velocity {v_improve:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
