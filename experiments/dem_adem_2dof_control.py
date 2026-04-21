"""DEM + MuJoCo: 2-DOF Arm Goal-Reaching via ADEM

Demonstrates ADEMAgent controlling a 2-DOF horizontal arm to reach a
target joint configuration using Active Inference.

D-step perceives current joint angles/velocities from noisy encoders.
A-step generates joint torques to minimize VFE toward target.

The two joints are controlled INDEPENDENTLY via the decoupled g_action:
    a[0] → joint 1 torque only
    a[1] → joint 2 torque only

Usage:
    .venv/bin/python experiments/dem_adem_2dof_control.py

Output:
    results/dem_adem_2dof_control.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.inference import DStep
from src.dem.action import ActionUpdate


# ============================================================
# MuJoCo 2-DOF arm XML  (horizontal, zero gravity, damping=0.3)
# ============================================================

ARM_XML = """
<mujoco model="2dof_arm">
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <body name="link1">
      <joint name="joint1" type="hinge" axis="0 0 1" damping="0.3"/>
      <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.083 0.083 0.083"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="0.3"/>
        <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.083 0.083 0.083"/>
        <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-15 15"/>
    <motor name="torque2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""

# ============================================================
# Simulation parameters
# ============================================================

SIM_DT      = 0.002
OBS_DT      = 0.05
T_END       = 8.0
N_OBS       = int(T_END / OBS_DT)
OBS_EVERY   = int(OBS_DT / SIM_DT)

NOISE_STD   = 0.05
SEED        = 42

N_JOINTS    = 2

# Start and goal joint configurations
Q0        = [0.4, 0.3]      # initial joint angles (rad)
Q_TARGET  = [-0.3, 0.5]    # target joint angles (rad)

# ============================================================
# DEM parameters
# ============================================================

N_ORDER       = 4
PI_Y          = 8.0
PI_X          = 2.0
S_SMOOTH      = 1.0
N_ITER_D_STEP = 128
D_STEP_DT     = 0.01    # fixed for convergence, NOT OBS_DT/N_ITER
KAPPA_MU      = 1.0

N_ITER_A_STEP = 16
KAPPA_A       = 2.0
A_DT          = 0.01
COUPLING_POS  = 0.08    # smaller → stronger gain
COUPLING_VEL  = 0.12
TORQUE_LIMIT  = 15.0


# ============================================================
# Build DEM model  (n_x=2, n_y=2, n_order=4)
# ============================================================

def build_dem_model() -> DEMModel:
    """Build DEM model for 2-DOF arm ADEM control.

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


def make_g_action():
    """Observation function that includes action (proprioceptive prediction).

    Actions shift expected position and velocity for each joint independently:
        a[0] → joint 1 torque only (shifts θ1, θ1')
        a[1] → joint 2 torque only (shifts θ2, θ2')

    At A-step equilibrium:
        a[j]* = Kp*(θ_target[j] - θ_est[j]) - Kd*ω_est[j]
    where Kp = coupling_pos / (cp² + cv²),  Kd = coupling_vel / (cp² + cv²)
    """
    def g_action(x_tilde, v_tilde, a, params):
        # a[0] = torque joint1, a[1] = torque joint2
        # Actions shift expected position and velocity for each joint independently
        return (x_tilde
                .at[0].add(a[0] * COUPLING_POS)   # θ1 ← a[0]
                .at[1].add(a[1] * COUPLING_POS)   # θ2 ← a[1]
                .at[2].add(a[0] * COUPLING_VEL)   # θ1' ← a[0]
                .at[3].add(a[1] * COUPLING_VEL))  # θ2' ← a[1]
    return g_action


# ============================================================
# Build y_tilde from observations
# ============================================================

def build_y_tilde(q_obs_curr, q_obs_prev, q_obs_next, dt) -> jnp.ndarray:
    """Build generalized observation vector for a single timestep.

    y_tilde layout (order-first, interleaved, shape (8,)):
        [θ1_obs, θ2_obs,  θ1'_fd, θ2'_fd,  0, 0,  0, 0]

    Args:
        q_obs_curr: Current noisy angle observation, shape (2,)
        q_obs_prev: Previous noisy angle observation (or None at start), shape (2,)
        q_obs_next: Next noisy angle observation (or None at end), shape (2,)
        dt: Time step for finite difference

    Returns:
        y_tilde: Generalized observation vector, shape (8,)
    """
    y_tilde = np.zeros(N_ORDER * N_JOINTS)

    # Zeroth order: noisy encoder readings
    y_tilde[0] = q_obs_curr[0]   # θ1
    y_tilde[1] = q_obs_curr[1]   # θ2

    # First order: finite-difference velocity estimates
    if q_obs_prev is not None and q_obs_next is not None:
        dq = (q_obs_next - q_obs_prev) / (2.0 * dt)
    elif q_obs_prev is None and q_obs_next is not None:
        dq = (q_obs_next - q_obs_curr) / dt
    else:
        dq = (q_obs_curr - q_obs_prev) / dt

    y_tilde[2] = dq[0]   # θ1'
    y_tilde[3] = dq[1]   # θ2'
    # y_tilde[4:] = 0 (higher orders not observed)

    return jnp.array(y_tilde)


# ============================================================
# Run ADEM control loop with MuJoCo
# ============================================================

def run_adem_control():
    """Run ADEM 2-DOF arm goal-reaching with MuJoCo simulation.

    Returns:
        t_hist:     Observation times, shape (N_OBS+1,)
        q_true:     True joint angles at observation times, shape (N_OBS+1, 2)
        tau_hist:   Applied torques, shape (N_OBS, 2)
        vfe_hist:   VFE values, shape (N_OBS,)
    """
    import mujoco

    model_mj = mujoco.MjModel.from_xml_string(ARM_XML)
    data_mj  = mujoco.MjData(model_mj)

    # Initialize simulation state
    data_mj.qpos[:2] = Q0
    data_mj.qvel[:2] = [0.0, 0.0]

    rng = np.random.default_rng(SEED)

    # Build DEM components
    dem_model   = build_dem_model()
    g_action_fn = make_g_action()

    d_step = DStep(
        dem_model,
        kappa_mu=KAPPA_MU,
        dt=D_STEP_DT,
        n_iter=N_ITER_D_STEP,
        use_d_operator=False,
    )
    action_update = ActionUpdate(
        dem_model,
        g_action=g_action_fn,
        kappa_a=KAPPA_A,
        dt=A_DT,
    )

    # Goal prior: desire to be at Q_TARGET with zero velocity
    y_tilde_goal = jnp.zeros(N_ORDER * N_JOINTS)
    y_tilde_goal = y_tilde_goal.at[0].set(Q_TARGET[0])
    y_tilde_goal = y_tilde_goal.at[1].set(Q_TARGET[1])

    # Initialize DEM state from first observation (warm start)
    q0_obs = np.array(data_mj.qpos[:2]) + rng.normal(0.0, NOISE_STD, 2)
    mu_x = jnp.zeros(N_ORDER * N_JOINTS).at[0].set(float(q0_obs[0])).at[1].set(float(q0_obs[1]))
    mu_v = jnp.zeros(N_ORDER * N_JOINTS)
    a    = jnp.zeros(2)

    # Storage
    t_hist   = [0.0]
    q_true   = [data_mj.qpos[:2].copy()]
    tau_hist = []
    vfe_hist = []

    # We need a rolling buffer of observations for finite differences
    q_obs_buf = [q0_obs]   # will grow one step ahead if possible

    # Pre-collect first two observations to enable FD at step 0
    # We rely on lookahead by one step from the observation buffer
    q_obs_prev = None

    for obs_idx in range(N_OBS):
        # Collect current noisy encoder reading
        q_obs_curr = np.array(data_mj.qpos[:2]) + rng.normal(0.0, NOISE_STD, 2)

        # We need lookahead for central FD — temporarily step sim to get next obs
        # Save state to restore
        qpos_save = data_mj.qpos[:2].copy()
        qvel_save = data_mj.qvel[:2].copy()
        ctrl_save = data_mj.ctrl[:2].copy()

        # Peek one OBS_EVERY steps ahead (using current torques if already set)
        for _ in range(OBS_EVERY):
            mujoco.mj_step(model_mj, data_mj)
        q_obs_next = np.array(data_mj.qpos[:2]) + rng.normal(0.0, NOISE_STD, 2)

        # Restore state
        data_mj.qpos[:2] = qpos_save
        data_mj.qvel[:2] = qvel_save
        data_mj.ctrl[:2] = ctrl_save
        mujoco.mj_forward(model_mj, data_mj)

        # Build y_tilde from actual observations
        y_tilde_obs = build_y_tilde(q_obs_curr, q_obs_prev, q_obs_next, OBS_DT)

        # D-step: update belief from actual observation
        mu_x, mu_v, vfe = d_step.run(mu_x, mu_v, y_tilde_obs)
        vfe_hist.append(vfe)

        # A-step: update action to minimize VFE toward goal
        for _ in range(N_ITER_A_STEP):
            a = action_update.step(a, mu_x, mu_v, y_tilde_goal)

        # Check for NaN before applying to MuJoCo
        if jnp.any(jnp.isnan(a)):
            print(f"  WARNING: NaN in action at obs_idx={obs_idx}, resetting to zero")
            a = jnp.zeros(2)

        # Clamp and apply torques
        tau = jnp.clip(a, -TORQUE_LIMIT, TORQUE_LIMIT)
        data_mj.ctrl[0] = float(tau[0])
        data_mj.ctrl[1] = float(tau[1])
        tau_hist.append(np.array([float(tau[0]), float(tau[1])]))

        # Step simulation (OBS_EVERY steps)
        for _ in range(OBS_EVERY):
            mujoco.mj_step(model_mj, data_mj)

        t_hist.append((obs_idx + 1) * OBS_DT)
        q_true.append(data_mj.qpos[:2].copy())
        q_obs_prev = q_obs_curr

    return (
        np.array(t_hist),
        np.array(q_true),
        np.array(tau_hist),
        np.array(vfe_hist),
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DEM + MuJoCo: 2-DOF Arm Goal-Reaching via ADEM")
    print(f"  Start:  θ1={Q0[0]} rad, θ2={Q0[1]} rad")
    print(f"  Target: θ1={Q_TARGET[0]} rad, θ2={Q_TARGET[1]} rad")
    print(f"  n_x={N_JOINTS}, n_order={N_ORDER}, pi_y={PI_Y}, pi_x={PI_X}")
    print(f"  coupling_pos={COUPLING_POS}, coupling_vel={COUPLING_VEL}")
    print("=" * 60)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Running ADEM control loop...")
    t_hist, q_true, tau_hist, vfe_hist = run_adem_control()

    if np.isnan(q_true).any():
        print("  ERROR: NaN in joint angle trajectory!")
        return

    # --- Summary statistics ---
    q_final = q_true[-1]
    q_err   = q_final - np.array(Q_TARGET)

    print()
    print("Results:")
    print(f"  Initial angles:  θ1={Q0[0]:.3f} rad, θ2={Q0[1]:.3f} rad")
    print(f"  Target angles:   θ1={Q_TARGET[0]:.3f} rad, θ2={Q_TARGET[1]:.3f} rad")
    print(f"  Final angles:    θ1={q_final[0]:.3f} rad, θ2={q_final[1]:.3f} rad")
    print(f"  Final error:     θ1={q_err[0]:+.4f} rad, θ2={q_err[1]:+.4f} rad")
    print()

    # Time to settle within 0.05 rad for each joint
    settle_thr = 0.05
    t_obs = t_hist[1:]   # observation times (after first step)
    for j in range(N_JOINTS):
        settle_idx = next(
            (i for i in range(len(t_obs))
             if abs(q_true[i + 1, j] - Q_TARGET[j]) < settle_thr),
            None,
        )
        if settle_idx is not None:
            print(f"  Joint {j+1} settled within {settle_thr} rad at t={t_obs[settle_idx]:.2f} s")
        else:
            print(f"  Joint {j+1} did NOT settle within {settle_thr} rad in {T_END}s")

    print()
    peak_tau = np.abs(tau_hist).max(axis=0)
    print(f"  Peak torques: τ1={peak_tau[0]:.2f} N·m, τ2={peak_tau[1]:.2f} N·m")
    print()

    # Equivalent PD gains
    Kp = COUPLING_POS / (COUPLING_POS**2 + COUPLING_VEL**2)
    Kd = COUPLING_VEL / (COUPLING_POS**2 + COUPLING_VEL**2)
    print(f"  Equivalent PD gains: Kp≈{Kp:.2f}, Kd≈{Kd:.2f}")
    print()

    # --- Plot ---
    print("Plotting...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(11, 14))

        t_obs_arr = t_hist[1:]   # times of observations (shape N_OBS)
        # q_true[0] is initial, q_true[1:] are after each obs step
        q_traj = q_true[1:]      # shape (N_OBS, 2) — post-step angles

        # Panel 1: θ1(t)
        ax = axes[0]
        ax.axhline(Q_TARGET[0], color='k', ls='--', lw=1.2, alpha=0.6,
                   label=f'Goal θ1_target = {Q_TARGET[0]} rad')
        ax.axhline(Q0[0], color='gray', ls=':', lw=1.0, alpha=0.5,
                   label=f'Start θ1_0 = {Q0[0]} rad')
        ax.plot(t_obs_arr, q_traj[:, 0], 'r-', lw=2.0, label='ADEM θ1(t)')
        ax.set_ylabel('Angle θ1 (rad)')
        ax.set_title('Joint 1: θ1(t) — Goal Reaching via ADEM')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: θ2(t)
        ax = axes[1]
        ax.axhline(Q_TARGET[1], color='k', ls='--', lw=1.2, alpha=0.6,
                   label=f'Goal θ2_target = {Q_TARGET[1]} rad')
        ax.axhline(Q0[1], color='gray', ls=':', lw=1.0, alpha=0.5,
                   label=f'Start θ2_0 = {Q0[1]} rad')
        ax.plot(t_obs_arr, q_traj[:, 1], 'b-', lw=2.0, label='ADEM θ2(t)')
        ax.set_ylabel('Angle θ2 (rad)')
        ax.set_title('Joint 2: θ2(t) — Goal Reaching via ADEM')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 3: torques τ1(t) and τ2(t)
        ax = axes[2]
        ax.plot(t_obs_arr, tau_hist[:, 0], 'm-', lw=1.5, label='τ1(t)')
        ax.plot(t_obs_arr, tau_hist[:, 1], 'c-', lw=1.5, label='τ2(t)')
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.axhline(TORQUE_LIMIT, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.axhline(-TORQUE_LIMIT, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylabel('Torque (N·m)')
        ax.set_title('Applied Torques: τ1(t) and τ2(t)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 4: VFE(t)
        ax = axes[3]
        ax.plot(t_obs_arr, vfe_hist, 'g-', lw=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VFE')
        ax.set_title('Variational Free Energy (lower = belief closer to goal)')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            'DEM + MuJoCo: 2-DOF Arm Goal-Reaching via ADEM\n'
            f'(coupling_pos={COUPLING_POS}, coupling_vel={COUPLING_VEL}, '
            f'pi_y={PI_Y}, pi_x={PI_X})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_adem_2dof_control.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping plot")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Initial: θ1={Q0[0]:.3f}, θ2={Q0[1]:.3f} rad")
    print(f"  Target:  θ1={Q_TARGET[0]:.3f}, θ2={Q_TARGET[1]:.3f} rad")
    print(f"  Final:   θ1={q_final[0]:.3f}, θ2={q_final[1]:.3f} rad")
    print(f"  Error:   θ1={q_err[0]:+.4f}, θ2={q_err[1]:+.4f} rad")
    print(f"  Peak torques: τ1={peak_tau[0]:.2f} N·m, τ2={peak_tau[1]:.2f} N·m")
    print("=" * 60)


if __name__ == "__main__":
    main()
