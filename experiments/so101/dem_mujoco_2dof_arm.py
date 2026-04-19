"""DEM + MuJoCo: 2-DOF arm proprioceptive state estimation (gravity-aware)

Demonstrates DEMAgent estimating joint angles AND angular velocities
for a 2-DOF planar arm in a VERTICAL plane (gravity enabled).

Setup:
    A vertical-plane 2-DOF arm starts hanging straight down (stable
    equilibrium). Sinusoidal torques drive oscillations around the
    hanging position.
    Each joint has a noisy angle encoder. No velocity sensors.

DEM models compared:
    1. Smoothness-only (f=0): original approach, no physics knowledge.
    2. Gravity-aware (f=dynamics): f encodes decoupled gravity dynamics,
       v_tilde is inferred as the unknown applied torque.

Key design:
    n_x=2, n_order=4, order-first layout.
    f(x_tilde, v_tilde) = [theta', gravity_dynamics] for gravity model.
    v_tilde is estimated by DEM as the unknown torque cause.

Usage:
    uv run --group so101 python experiments/so101/dem_mujoco_2dof_arm.py

Output:
    results/dem_mujoco_2dof_arm.png
"""

import sys
import math
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.agent import DEMAgent
from src.dem.inference import compute_vfe


# ============================================================
# MuJoCo 2-DOF arm XML  (vertical plane, gravity enabled)
# ============================================================

ARM_XML = """
<mujoco model="2dof_arm_gravity">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="link1">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="2.0"/>
      <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.0008 0.0208 0.0208"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="2.0"/>
        <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.0008 0.0208 0.0208"/>
        <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="torque2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# ============================================================
# Simulation parameters
# ============================================================

SIM_DT      = 0.002
OBS_DT      = 0.05
T_END       = 10.0
N_SIM       = int(T_END / SIM_DT)
N_OBS       = int(T_END / OBS_DT)
OBS_EVERY   = int(OBS_DT / SIM_DT)

NOISE_STD   = 0.05   # encoder noise (rad)
SEED        = 42
N_JOINTS    = 2

# Sinusoidal torques (near natural frequency of pendulum)
TORQUE_AMP1  = 2.5
TORQUE_FREQ1 = 0.6   # Hz — joint 1
TORQUE_AMP2  = 1.5
TORQUE_FREQ2 = 0.8   # Hz — joint 2

# Initial joint angles: hanging straight down (stable equilibrium)
# axis="0 1 0": Ry(pi/2) takes +x toward -z, so qpos=pi/2 → link points down
Q0  = [math.pi / 2, 0.0]
DQ0 = [0.0, 0.0]

# ============================================================
# Physical arm parameters (must match XML)
# ============================================================

MASS      = 1.0     # kg per link
LINK_LEN  = 0.5     # m
LC        = 0.25    # m, distance from joint to link CoM
G_ACCEL   = 9.81    # m/s²
DAMPING   = 2.0     # viscous joint damping (matches XML damping="2.0")

# Effective inertia about each joint (parallel axis theorem, decoupled approx)
# I1_eff = (I1_cm + m1*lc1²) + (I2_cm + m2*(L1+lc2)²)
I1_EFF = (MASS * LINK_LEN**2 / 12 + MASS * LC**2) + \
         (MASS * LINK_LEN**2 / 12 + MASS * (LINK_LEN + LC)**2)  # ≈ 0.667
# I2_eff = I2_cm + m2*lc2²
I2_EFF = MASS * LINK_LEN**2 / 12 + MASS * LC**2               # ≈ 0.083

# ============================================================
# DEM parameters
# ============================================================

N_ORDER       = 4
PI_Y          = 8.0
S_SMOOTH      = 1.0

# EKF parameters
EKF_Q_POS  = 1e-4    # process noise: position (rad²)
EKF_Q_VEL  = 0.1     # process noise: velocity (rad²/s²)
EKF_R_POS  = NOISE_STD ** 2   # observation noise (rad²)

# Smoothness model (f=0): light Pi_x to avoid over-damping fast gravity motion
PI_X_SMOOTH   = 1.0
N_ITER_SMOOTH = 128
DT_SMOOTH     = 0.01
KAPPA_SMOOTH  = 1.0

# Gravity model: fixed known torques, diagonal precision matrices (R=I).
# Diagonal Pi_y and Pi_x avoid cross-order coupling from the R matrix
# (e.g., R[0,2]=-1 would otherwise couple θ ↔ θ'' via observations).
# With diagonal precision: angle is constrained ONLY by observation,
# velocity by FD observation + Coriolis coupling, acceleration by physics.
# PI_X_ACC uses 4x pi_y so acceleration tracks physics at ~80%.
PI_X_GRAVITY  = 4.0    # acceleration precision
N_ITER_GRAV   = 400
DT_GRAV       = 0.001
KAPPA_GRAV    = 0.5


# ============================================================
# Step 1: MuJoCo simulation
# ============================================================

def run_simulation():
    """Run 2-DOF arm simulation in vertical plane.

    Returns:
        t_obs:    Observation times, shape (N_OBS,)
        q_true:   True joint angles, shape (N_OBS, 2)
        dq_true:  True angular velocities, shape (N_OBS, 2)
        y_obs:    Noisy angle observations, shape (N_OBS, 2)
        tau_obs:  Applied torques at each observation, shape (N_OBS, 2)
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
    tau_obs = np.zeros((N_OBS, N_JOINTS))

    obs_idx = 0
    for step in range(N_SIM):
        t = step * SIM_DT
        tau1 = TORQUE_AMP1 * math.sin(2.0 * math.pi * TORQUE_FREQ1 * t)
        tau2 = TORQUE_AMP2 * math.sin(2.0 * math.pi * TORQUE_FREQ2 * t)

        if step % OBS_EVERY == 0 and obs_idx < N_OBS:
            t_obs[obs_idx]   = t
            q_true[obs_idx]  = data.qpos[:2].copy()
            dq_true[obs_idx] = data.qvel[:2].copy()
            y_obs[obs_idx]   = data.qpos[:2] + rng.normal(0.0, NOISE_STD, 2)
            tau_obs[obs_idx] = [tau1, tau2]
            obs_idx += 1

        data.ctrl[0] = tau1
        data.ctrl[1] = tau2
        mujoco.mj_step(model, data)

    return t_obs, q_true, dq_true, y_obs, tau_obs


# ============================================================
# Step 2: Build DEM models
# ============================================================

def _arm_dynamics(theta1, theta2, dtheta1, dtheta2, tau1, tau2):
    """Full coupled 2-DOF arm dynamics: q'' = M(q)^{-1} * (tau + G(q) - C(q,dq)*dq).

    Mass matrix M (theta2-dependent due to inertia coupling):
        M11 = 2*(I_cm+m*lc^2) + m*L^2 + 2*m*L*lc*cos(theta2)
        M12 = I_cm + m*lc^2 + m*L*lc*cos(theta2)
        M22 = I_cm + m*lc^2

    Gravity generalized forces:
        G1 = (m*g*lc + m*g*L)*cos(theta1) + m*g*lc*cos(theta1+theta2)
        G2 = m*g*lc*cos(theta1+theta2)

    Coriolis/centrifugal:
        h = m*L*lc*sin(theta2)
        C*dq = [-h*dtheta2*(2*dtheta1+dtheta2), h*dtheta1^2]

    Returns:
        (ddtheta1, ddtheta2)
    """
    I_cm = MASS * LINK_LEN**2 / 12.0   # inertia about CoM

    c2 = jnp.cos(theta2)
    s2 = jnp.sin(theta2)

    # Mass matrix
    M11 = (2.0 * (I_cm + MASS * LC**2) + MASS * LINK_LEN**2
           + 2.0 * MASS * LINK_LEN * LC * c2)
    M12 = I_cm + MASS * LC**2 + MASS * LINK_LEN * LC * c2
    M22 = I_cm + MASS * LC**2

    # Gravity
    G1 = ((MASS * G_ACCEL * LC + MASS * G_ACCEL * LINK_LEN) * jnp.cos(theta1)
          + MASS * G_ACCEL * LC * jnp.cos(theta1 + theta2))
    G2 = MASS * G_ACCEL * LC * jnp.cos(theta1 + theta2)

    # Coriolis
    h = MASS * LINK_LEN * LC * s2
    Cq1 = -h * dtheta2 * (2.0 * dtheta1 + dtheta2)
    Cq2 = h * dtheta1**2

    # Effective generalized forces (including viscous damping)
    f1 = tau1 + G1 - Cq1 - DAMPING * dtheta1
    f2 = tau2 + G2 - Cq2 - DAMPING * dtheta2

    # Inverse mass matrix (2x2)
    det = M11 * M22 - M12**2
    ddtheta1 = (M22 * f1 - M12 * f2) / det
    ddtheta2 = (-M12 * f1 + M11 * f2) / det

    return ddtheta1, ddtheta2


def build_dem_model_smoothness() -> DEMModel:
    """Original model: f=0 (smoothness prior, no physics)."""
    def f(x_tilde, v_tilde, params):
        return jnp.zeros(N_ORDER * N_JOINTS)

    def g(x_tilde, v_tilde, params):
        return x_tilde

    return DEMModel(
        f=f, g=g,
        n_x=N_JOINTS, n_v=N_JOINTS, n_y=N_JOINTS,
        n_order=N_ORDER, pi_y=PI_Y, pi_x=PI_X_SMOOTH,
        s_y=S_SMOOTH, s_x=S_SMOOTH,
    )


def build_dem_model_gravity() -> DEMModel:
    """Gravity-aware model: f encodes decoupled arm dynamics.

    x_tilde layout (order-first): [theta1, theta2, dtheta1, dtheta2, ...]
    v_tilde layout (order-first): [tau1, tau2, ...]   (inferred torque causes)

    f[0:2] = [dtheta1, dtheta2]         (kinematic consistency)
    f[2:4] = [(tau+G)/I_eff, ...]       (gravity + torque dynamics)
    f[4:8] = 0                           (no model for higher orders)
    """
    def f(x_tilde, v_tilde, params):
        theta1  = x_tilde[0]
        theta2  = x_tilde[1]
        dtheta1 = x_tilde[2]
        dtheta2 = x_tilde[3]
        tau1    = v_tilde[0]   # known torque for joint1 (fixed externally)
        tau2    = v_tilde[1]   # known torque for joint2 (fixed externally)

        ddtheta1, ddtheta2 = _arm_dynamics(
            theta1, theta2, dtheta1, dtheta2, tau1, tau2
        )

        result = jnp.zeros(N_ORDER * N_JOINTS)
        result = result.at[0].set(dtheta1)
        result = result.at[1].set(dtheta2)
        result = result.at[2].set(ddtheta1)
        result = result.at[3].set(ddtheta2)
        return result

    def g(x_tilde, v_tilde, params):
        return x_tilde

    return DEMModel(
        f=f, g=g,
        n_x=N_JOINTS, n_v=N_JOINTS, n_y=N_JOINTS,
        n_order=N_ORDER, pi_y=PI_Y, pi_x=PI_X_GRAVITY,
        s_y=S_SMOOTH, s_x=S_SMOOTH,
    )


# ============================================================
# Step 3: Build generalized observation sequence
# ============================================================

def build_y_tilde_sequence(y_obs: np.ndarray) -> list:
    """Build generalized observation sequence.

    y_tilde layout (order-first):
        [theta1_obs, theta2_obs, dtheta1_fd, dtheta2_fd, 0, 0, 0, 0]
    """
    N = len(y_obs)
    y_tilde_list = []

    for i in range(N):
        y_tilde = np.zeros(N_ORDER * N_JOINTS)

        y_tilde[0] = y_obs[i, 0]
        y_tilde[1] = y_obs[i, 1]

        if 0 < i < N - 1:
            dq = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * OBS_DT)
        elif i == 0:
            dq = (y_obs[1] - y_obs[0]) / OBS_DT
        else:
            dq = (y_obs[-1] - y_obs[-2]) / OBS_DT

        y_tilde[2] = dq[0]
        y_tilde[3] = dq[1]

        y_tilde_list.append(jnp.array(y_tilde))

    return y_tilde_list


# ============================================================
# Step 4: Run DEM estimation (both models)
# ============================================================

def _init_mu_x(y_obs: np.ndarray, dq_fd: np.ndarray,
               tau0: np.ndarray = None) -> jnp.ndarray:
    """Warm-start x_tilde from first observation, FD velocity, and initial acceleration."""
    mu_x0 = jnp.zeros(N_ORDER * N_JOINTS)
    mu_x0 = mu_x0.at[0].set(float(y_obs[0, 0]))
    mu_x0 = mu_x0.at[1].set(float(y_obs[0, 1]))
    mu_x0 = mu_x0.at[2].set(float(dq_fd[0, 0]))
    mu_x0 = mu_x0.at[3].set(float(dq_fd[0, 1]))
    if tau0 is not None:
        # Initialize acceleration from dynamics to reduce initial eps_x residual
        import numpy as np
        theta1, theta2 = float(y_obs[0, 0]), float(y_obs[0, 1])
        dth1, dth2 = float(dq_fd[0, 0]), float(dq_fd[0, 1])
        dd1, dd2 = _arm_dynamics(
            jnp.array(theta1), jnp.array(theta2),
            jnp.array(dth1), jnp.array(dth2),
            jnp.array(float(tau0[0])), jnp.array(float(tau0[1]))
        )
        mu_x0 = mu_x0.at[4].set(float(dd1))
        mu_x0 = mu_x0.at[5].set(float(dd2))
    return mu_x0


def run_dem_smoothness(model: DEMModel, y_tilde_list: list, y_obs: np.ndarray,
                       dq_fd: np.ndarray):
    """Run DEMAgent with smoothness model (f=0).

    Returns:
        q_est, dq_est, vfe_arr
    """
    agent = DEMAgent(
        model,
        kappa_mu=KAPPA_SMOOTH,
        dt=DT_SMOOTH,
        n_iter_per_step=N_ITER_SMOOTH,
        use_d_operator=False,
    )
    mu_x0 = _init_mu_x(y_obs, dq_fd)
    mu_v0 = jnp.zeros(N_ORDER * N_JOINTS)
    state, mu_x_history = agent.run(y_tilde_list, mu_x0=mu_x0, mu_v0=mu_v0)
    q_est  = np.array([[float(h[0]), float(h[1])] for h in mu_x_history[1:]])
    dq_est = np.array([[float(h[2]), float(h[3])] for h in mu_x_history[1:]])
    return q_est, dq_est, np.array(state.vfe_history)


def run_dem_gravity(model: DEMModel, y_tilde_list: list, tau_obs: np.ndarray,
                    y_obs: np.ndarray, dq_fd: np.ndarray):
    """Physics-augmented DEM state estimation with diagonal precision matrices.

    Uses diagonal Pi_y = pi_y * I and Pi_x = pi_x * I instead of the
    standard R-coupled tilde precision matrices. This avoids cross-order
    coupling (e.g., R[0,2]=-1 would couple θ and θ'' through observations).

    VFE per timestep:
        F = 0.5 * pi_y * ||x_tilde - y_tilde||²         (observation)
          + 0.5 * pi_x * ||D@x_tilde - f(θ_obs,ω,τ)||² (physics dynamics)

    where f uses OBSERVED angles θ_obs (not inferred x_tilde[0:2]) so that
    ∂F/∂θ = 0 from the dynamics term → angle is purely from observations.

    State: x_tilde = [θ1, θ2, ω1, ω2, α1, α2, j1, j2]  (n_order=4, n_x=2)

    Returns:
        q_est, dq_est, vfe_arr
    """
    import jax
    from src.dem.core import make_D_matrix

    D_mat = make_D_matrix(N_ORDER, N_JOINTS)   # 8×8 shift operator

    def _vfe_diag(mu_x, tau_tilde, y_tilde, theta_obs, omega_obs):
        """VFE with diagonal precision matrices and observation-angle dynamics."""
        omega1, omega2 = mu_x[2], mu_x[3]
        tau1,   tau2   = tau_tilde[0], tau_tilde[1]

        # Physics prediction using observed angles AND observed (FD) velocities.
        # This ensures d(alpha_p)/d(omega) = 0, so omega inference is driven
        # purely by the observation term (PI_Y * (omega - omega_fd)).
        alpha1_p, alpha2_p = _arm_dynamics(
            theta_obs[0], theta_obs[1], omega_obs[0], omega_obs[1], tau1, tau2
        )

        # Generalized dynamics prediction f_tilde
        f_tilde = jnp.zeros(N_ORDER * N_JOINTS)
        f_tilde = f_tilde.at[0].set(omega1)    # kinematic: θ' = ω
        f_tilde = f_tilde.at[1].set(omega2)
        f_tilde = f_tilde.at[2].set(alpha1_p)  # dynamics: ω' = acceleration
        f_tilde = f_tilde.at[3].set(alpha2_p)
        # higher orders: f=0 (smoothness prior)

        eps_y = (y_tilde - mu_x)[:4]   # angle + FD-velocity only; ignore zeros at order>1
        eps_x = D_mat @ mu_x - f_tilde

        # Diagonal precision matrices (no cross-order R coupling)
        vfe_y = 0.5 * PI_Y         * jnp.sum(eps_y ** 2)
        vfe_x = 0.5 * PI_X_GRAVITY * jnp.sum(eps_x ** 2)
        return vfe_y + vfe_x

    grad_fn = jax.jit(jax.grad(_vfe_diag, argnums=0))
    vfe_fn  = jax.jit(_vfe_diag)

    mu_x = _init_mu_x(y_obs, dq_fd, tau0=tau_obs[0])

    q_est, dq_est, vfe_arr = [], [], []

    for y_tilde, tau in zip(y_tilde_list, tau_obs):
        tau_tilde = jnp.zeros(N_ORDER * N_JOINTS)
        tau_tilde = tau_tilde.at[0].set(float(tau[0]))
        tau_tilde = tau_tilde.at[1].set(float(tau[1]))
        theta_obs = jnp.array([float(y_tilde[0]), float(y_tilde[1])])
        omega_obs = jnp.array([float(y_tilde[2]), float(y_tilde[3])])

        for _ in range(N_ITER_GRAV):
            grad_x = grad_fn(mu_x, tau_tilde, y_tilde, theta_obs, omega_obs)
            grad_x = jnp.clip(grad_x, -50.0, 50.0)
            mu_x   = mu_x - KAPPA_GRAV * DT_GRAV * grad_x

        vfe = float(vfe_fn(mu_x, tau_tilde, y_tilde, theta_obs, omega_obs))
        q_est.append([float(mu_x[0]), float(mu_x[1])])
        dq_est.append([float(mu_x[2]), float(mu_x[3])])
        vfe_arr.append(vfe)

    return np.array(q_est), np.array(dq_est), np.array(vfe_arr)


# ============================================================
# Step 4b: EKF estimation
# ============================================================

def run_ekf(y_obs: np.ndarray, tau_obs: np.ndarray, dq_fd: np.ndarray):
    """Extended Kalman Filter for 2-DOF arm state estimation.

    State:       x = [theta1, theta2, omega1, omega2]
    Observation: y = [theta1_obs, theta2_obs]   (angle encoder only)
    Input:       u = [tau1, tau2]               (known torques)

    Dynamics (continuous):
        dx/dt = [omega1, omega2, alpha1(x,u), alpha2(x,u)]
    Discretized with RK4 at OBS_DT.

    Jacobian F = d(f_discrete)/dx computed via JAX autodiff.

    Returns:
        q_est:  shape (N_OBS, 2)
        dq_est: shape (N_OBS, 2)
    """
    import jax

    N = len(y_obs)
    n_x = 4   # [theta1, theta2, omega1, omega2]
    n_y = 2   # [theta1_obs, theta2_obs]
    dt  = OBS_DT

    # --- Noise matrices ---
    Q = np.diag([EKF_Q_POS, EKF_Q_POS, EKF_Q_VEL, EKF_Q_VEL])
    R = np.diag([EKF_R_POS, EKF_R_POS])
    H = np.zeros((n_y, n_x))
    H[0, 0] = 1.0   # theta1 observed
    H[1, 1] = 1.0   # theta2 observed

    # --- Dynamics function (JAX) ---
    def _f_continuous(x_jax, u_jax):
        th1, th2, om1, om2 = x_jax
        ta1, ta2 = u_jax
        al1, al2 = _arm_dynamics(th1, th2, om1, om2, ta1, ta2)
        return jnp.stack([om1, om2, al1, al2])

    def _f_rk4_sub(x_jax, u_jax, dt_local):
        """RK4 step with explicit dt (for sub-stepping)."""
        k1 = _f_continuous(x_jax,                    u_jax)
        k2 = _f_continuous(x_jax + dt_local/2 * k1,  u_jax)
        k3 = _f_continuous(x_jax + dt_local/2 * k2,  u_jax)
        k4 = _f_continuous(x_jax + dt_local    * k3,  u_jax)
        return x_jax + (dt_local / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Sub-step integration: divide OBS_DT into n_sub steps so that each
    # Jacobian F_sub ≈ I + dt_sub * df/dx stays close to identity,
    # preventing P from blowing up between observations.
    n_sub  = int(round(OBS_DT / SIM_DT))   # = 25
    dt_sub = OBS_DT / n_sub
    Q_sub  = Q / n_sub    # distribute process noise across sub-steps

    _f_sub   = jax.jit(lambda xj, uj: _f_rk4_sub(xj, uj, dt_sub))
    _jac_sub = jax.jit(jax.jacobian(lambda xj, uj: _f_rk4_sub(xj, uj, dt_sub),
                                     argnums=0))

    # --- Initial state and covariance ---
    # Initialise velocity at zero (not FD-based): FD velocity at t=0 has
    # O(NOISE_STD/OBS_DT) ≈ 1 rad/s noise and causes early divergence.
    # The EKF will learn velocity from angle observations within a few steps.
    x = np.array([y_obs[0, 0], y_obs[0, 1], 0.0, 0.0])
    P = np.diag([EKF_R_POS, EKF_R_POS, 10.0, 10.0])

    q_est  = [[x[0], x[1]]]
    dq_est = [[x[2], x[3]]]

    I4 = np.eye(n_x)

    for i in range(1, N):
        y = y_obs[i]
        u = tau_obs[i - 1].astype(float)
        u_j = jnp.array(u)

        # --- Predict: n_sub sub-steps from t[i-1] to t[i] ---
        P_pred = P.copy()
        x_pred = x.copy()
        for _ in range(n_sub):
            x_j    = jnp.array(x_pred)
            F_sub  = np.array(_jac_sub(x_j, u_j))
            x_pred = np.array(_f_sub(x_j, u_j))
            P_pred = F_sub @ P_pred @ F_sub.T + Q_sub

        # --- Update (Joseph form) ---
        innov = y - H @ x_pred
        S     = H @ P_pred @ H.T + R
        K     = P_pred @ H.T @ np.linalg.inv(S)
        x     = x_pred + K @ innov
        I_KH  = I4 - K @ H
        P     = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        q_est.append([x[0], x[1]])
        dq_est.append([x[2], x[3]])

    return np.array(q_est), np.array(dq_est)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("DEM + MuJoCo: 2-DOF Arm (Gravity-Aware)")
    print("  Vertical plane, physics-based vs smoothness-only DEM")
    print("=" * 65)
    print()
    print(f"Arm parameters: m={MASS}kg, L={LINK_LEN}m, g={G_ACCEL}m/s²")
    print(f"I1_eff={I1_EFF:.4f}, I2_eff={I2_EFF:.4f} kg·m²")
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Simulation ---
    print("Step 1: MuJoCo simulation (vertical plane, gravity)")
    t_obs, q_true, dq_true, y_obs, tau_obs = run_simulation()
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

    # --- Build observations ---
    y_tilde_list = build_y_tilde_sequence(y_obs)

    # --- Model 1: smoothness only (f=0) ---
    print("Step 3a: DEM (smoothness-only, f=0)")
    model_smooth = build_dem_model_smoothness()
    q_smooth, dq_smooth, vfe_smooth = run_dem_smoothness(
        model_smooth, y_tilde_list, y_obs, dq_fd
    )
    rmse_q_smooth  = np.sqrt(np.mean((q_smooth  - q_true)  ** 2, axis=0))
    rmse_dq_smooth = np.sqrt(np.mean((dq_smooth - dq_true) ** 2, axis=0))
    print(f"  Angle RMSE: J1={rmse_q_smooth[0]:.4f}  J2={rmse_q_smooth[1]:.4f} rad")
    print(f"  Vel RMSE:   J1={rmse_dq_smooth[0]:.4f}  J2={rmse_dq_smooth[1]:.4f} rad/s")
    print()

    # --- Model 2: gravity-aware with known torques ---
    print("Step 3b: DEM (gravity-aware, fixed known torques)")
    model_gravity = build_dem_model_gravity()
    q_grav, dq_grav, vfe_grav = run_dem_gravity(
        model_gravity, y_tilde_list, tau_obs, y_obs, dq_fd
    )
    rmse_q_grav  = np.sqrt(np.mean((q_grav  - q_true)  ** 2, axis=0))
    rmse_dq_grav = np.sqrt(np.mean((dq_grav - dq_true) ** 2, axis=0))
    print(f"  Angle RMSE: J1={rmse_q_grav[0]:.4f}  J2={rmse_q_grav[1]:.4f} rad")
    print(f"  Vel RMSE:   J1={rmse_dq_grav[0]:.4f}  J2={rmse_dq_grav[1]:.4f} rad/s")
    print()

    # --- Model 3: EKF ---
    print("Step 3c: EKF (Extended Kalman Filter)")
    q_ekf, dq_ekf = run_ekf(y_obs, tau_obs, dq_fd)
    rmse_q_ekf  = np.sqrt(np.mean((q_ekf  - q_true)  ** 2, axis=0))
    rmse_dq_ekf = np.sqrt(np.mean((dq_ekf - dq_true) ** 2, axis=0))
    print(f"  Angle RMSE: J1={rmse_q_ekf[0]:.4f}  J2={rmse_q_ekf[1]:.4f} rad")
    print(f"  Vel RMSE:   J1={rmse_dq_ekf[0]:.4f}  J2={rmse_dq_ekf[1]:.4f} rad/s")
    print()

    # --- Summary ---
    print("=" * 75)
    print(f"{'Method':<18} {'J1 angle':>10} {'J2 angle':>10} {'J1 vel':>10} {'J2 vel':>10}")
    print("-" * 75)
    print(f"{'Encoder/FD':<18} {rmse_obs[0]:>10.4f} {rmse_obs[1]:>10.4f} {rmse_fd[0]:>10.4f} {rmse_fd[1]:>10.4f}")
    print(f"{'DEM smooth':<18} {rmse_q_smooth[0]:>10.4f} {rmse_q_smooth[1]:>10.4f} {rmse_dq_smooth[0]:>10.4f} {rmse_dq_smooth[1]:>10.4f}")
    print(f"{'DEM gravity':<18} {rmse_q_grav[0]:>10.4f} {rmse_q_grav[1]:>10.4f} {rmse_dq_grav[0]:>10.4f} {rmse_dq_grav[1]:>10.4f}")
    print(f"{'EKF':<18} {rmse_q_ekf[0]:>10.4f} {rmse_q_ekf[1]:>10.4f} {rmse_dq_ekf[0]:>10.4f} {rmse_dq_ekf[1]:>10.4f}")
    print("=" * 75)
    print()
    print("Improvement vs Encoder/FD baseline:")
    print(f"{'Method':<18} {'J1 angle':>10} {'J2 angle':>10} {'J1 vel':>10} {'J2 vel':>10}")
    print("-" * 75)
    for name, rq, rdq in [("DEM smooth",   rmse_q_smooth, rmse_dq_smooth),
                           ("DEM gravity",  rmse_q_grav,   rmse_dq_grav),
                           ("EKF",          rmse_q_ekf,    rmse_dq_ekf)]:
        ia = [(rmse_obs[j] - rq[j])  / rmse_obs[j] * 100 for j in range(2)]
        iv = [(rmse_fd[j]  - rdq[j]) / rmse_fd[j]  * 100 for j in range(2)]
        print(f"  {name:<16} {ia[0]:>+9.1f}% {ia[1]:>+9.1f}% {iv[0]:>+9.1f}% {iv[1]:>+9.1f}%")
    print("=" * 75)
    print()

    # --- Plot ---
    print("Step 4: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        for j in range(2):
            jlabel = f"Joint {j+1}"

            # Angle estimation
            ax = axes[0, j]
            ax.plot(t_obs, q_true[:, j], 'b-', lw=2, label=f'True θ{j+1}')
            ax.scatter(t_obs, y_obs[:, j], c='gray', s=8, alpha=0.4,
                       label=f'Encoder (RMSE={rmse_obs[j]:.3f})')
            ax.plot(t_obs, q_smooth[:, j], 'g--', lw=1.3, alpha=0.8,
                    label=f'DEM smooth (RMSE={rmse_q_smooth[j]:.3f})')
            ax.plot(t_obs, q_grav[:, j], 'r-', lw=1.5,
                    label=f'DEM gravity (RMSE={rmse_q_grav[j]:.3f})')
            ax.plot(t_obs, q_ekf[:, j], 'm:', lw=1.8,
                    label=f'EKF (RMSE={rmse_q_ekf[j]:.3f})')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(f'{jlabel} — Angle filtering')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Velocity estimation
            ax = axes[1, j]
            ax.plot(t_obs, dq_true[:, j], 'b-', lw=2, label=f'True θ{j+1}\'')
            ax.plot(t_obs, dq_fd[:, j], 'k--', lw=1.0, alpha=0.5,
                    label=f'FD (RMSE={rmse_fd[j]:.3f})')
            ax.plot(t_obs, dq_smooth[:, j], 'g--', lw=1.3, alpha=0.8,
                    label=f'DEM smooth (RMSE={rmse_dq_smooth[j]:.3f})')
            ax.plot(t_obs, dq_grav[:, j], 'r-', lw=1.5,
                    label=f'DEM gravity (RMSE={rmse_dq_grav[j]:.3f})')
            ax.plot(t_obs, dq_ekf[:, j], 'm:', lw=1.8,
                    label=f'EKF (RMSE={rmse_dq_ekf[j]:.3f})')
            ax.set_ylabel('Velocity (rad/s)')
            ax.set_title(f'{jlabel} — Velocity estimation')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # VFE comparison
        ax = axes[2, 0]
        ax.plot(t_obs, vfe_smooth, 'g--', lw=1.3, label='Smoothness (f=0)')
        ax.plot(t_obs, vfe_grav,   'r-',  lw=1.5, label='Gravity-aware')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VFE')
        ax.set_title('Variational Free Energy')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # RMSE bar chart (4-way comparison)
        ax = axes[2, 1]
        labels   = ['J1 angle', 'J2 angle', 'J1 vel', 'J2 vel']
        baseline = [rmse_obs[0],      rmse_obs[1],      rmse_fd[0],       rmse_fd[1]]
        smooth_v = [rmse_q_smooth[0], rmse_q_smooth[1], rmse_dq_smooth[0], rmse_dq_smooth[1]]
        grav_v   = [rmse_q_grav[0],   rmse_q_grav[1],   rmse_dq_grav[0],   rmse_dq_grav[1]]
        ekf_v    = [rmse_q_ekf[0],    rmse_q_ekf[1],    rmse_dq_ekf[0],    rmse_dq_ekf[1]]
        x = np.arange(4)
        w = 0.19
        ax.bar(x - 1.5*w, baseline, w, label='Encoder/FD',  color='steelblue', alpha=0.85)
        ax.bar(x - 0.5*w, smooth_v, w, label='DEM smooth',  color='seagreen',  alpha=0.85)
        ax.bar(x + 0.5*w, grav_v,   w, label='DEM gravity', color='tomato',    alpha=0.85)
        ax.bar(x + 1.5*w, ekf_v,    w, label='EKF',         color='mediumpurple', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison (4-way)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            'DEM + MuJoCo: 2-DOF Vertical Arm — 4-Way Comparison\n'
            f'(pi_y={PI_Y}, pi_x_smooth={PI_X_SMOOTH}, pi_x_grav={PI_X_GRAVITY},'
            f' EKF Q_pos={EKF_Q_POS}, Q_vel={EKF_Q_VEL})',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_mujoco_2dof_arm.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping plot")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
