"""ADEM: Goal-reaching via Active Inference (A-step)

Demonstrates how ADEMAgent minimizes Variational Free Energy by
generating control actions to bring a system to a desired state.

System:
    Horizontal-axis joint: θ'' = -(c/I)*θ' + τ/I
    No gravity component on rotation (pure damping + inertia).
    c=2.0 Nms/rad, I=0.333 kg·m²

    This gives a first-order reachable system: any target angle
    is achievable with finite torque, and steady-state torque = 0.

Agent (ADEM):
    D-step: perceives current state [θ, ω] from noisy encoder
            (smoothness prior f=0, Wiener filter on observations)
    A-step: generates torque τ to minimize VFE toward θ_target

    g_action encodes proprioceptive prediction:
        "if I apply action a, I expect my angle/velocity to change by ..."
    This is a PD-like closed-loop controller emergent from VFE minimization.

Comparison:
    Free motion: no control — θ damps to rest at initial position
    ADEM:        active torque drives θ → θ_target from opposite side

Usage:
    .venv/bin/python experiments/dem_adem_goal_reaching.py

Output:
    results/dem_adem_goal_reaching.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.inference import DStep
from src.dem.action import ActionUpdate


# ============================================================
# Physical parameters (horizontal joint, no gravity)
# ============================================================

INERTIA     = 1.0 / 3.0  # I = mL²/3 ≈ 0.333 kg·m²
DAMPING     = 0.3         # viscous damping (Nms/rad) — light, for faster response

THETA_0     = -0.4        # initial angle (rad)
OMEGA_0     = 0.0
THETA_TARGET = 0.3        # goal angle (rad)  — opposite side from start

T_END       = 6.0
OBS_DT      = 0.05        # observation interval (s)
N_STEPS     = int(T_END / OBS_DT)

NOISE_STD   = 0.05        # encoder noise (rad)
SEED        = 42

# ============================================================
# DEM parameters
# ============================================================

N_ORDER       = 4
PI_Y          = 8.0
PI_X          = 2.0
S_SMOOTH      = 1.0
N_ITER_D_STEP = 128
D_STEP_DT     = 0.01     # fixed for convergence (not OBS_DT/N_ITER)
KAPPA_MU      = 1.0

# A-step parameters
N_ITER_A_STEP = 16
KAPPA_A       = 2.0
A_DT          = 0.01

# Proprioceptive coupling: action a[0] (N·m) shifts expected angle/velocity.
# Equilibrium: a* = Kp*(θ_target - θ_est) - Kd*ω_est  (PD-like)
# where Kp = coupling_pos / (coupling_pos² + coupling_vel²)
COUPLING_POS  = 0.08   # smaller coupling → stronger effective gain Kp≈3.85
COUPLING_VEL  = 0.12   # Kd≈5.77 (accounts for velocity attenuation in D-step)

TORQUE_LIMIT  = 15.0  # N·m


# ============================================================
# Build DEM model
# ============================================================

def build_model() -> DEMModel:
    """DEM model with smoothness prior (f=0, g=identity).

    D-step acts as a Wiener filter, estimating θ and ω from noisy encoder.
    Using f=0 avoids dynamics conflicts in the D-step Hessian.
    The physical dynamics are captured in the closed-loop via g_action.
    """
    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        return jnp.zeros(N_ORDER)

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params) -> jnp.ndarray:
        return x_tilde

    return DEMModel(
        f=f, g=g, n_x=1, n_v=1, n_y=1,
        n_order=N_ORDER,
        pi_y=PI_Y, pi_x=PI_X,
        s_y=S_SMOOTH, s_x=S_SMOOTH,
    )


def make_g_action():
    """Observation function that includes action (proprioceptive prediction).

    g_action(x_tilde, v_tilde, a) = x_tilde + [a*COUPLING_POS, a*COUPLING_VEL, 0, 0]

    This encodes: "if I apply torque a (N·m), I expect my angle to shift
    by a*COUPLING_POS and my velocity to shift by a*COUPLING_VEL."

    At A-step equilibrium:
        a* = Kp*(θ_target - θ_est) - Kd*ω_est
    where Kp = coupling_pos / (cp² + cv²),  Kd = coupling_vel / (cp² + cv²)
    """
    def g_action(x_tilde, v_tilde, a, params):
        return (x_tilde
                .at[0].add(a[0] * COUPLING_POS)
                .at[1].add(a[0] * COUPLING_VEL))
    return g_action


# ============================================================
# Physics simulation
# ============================================================

def sim_step(theta: float, omega: float, tau: float) -> tuple:
    """One Euler step: horizontal joint (no gravity).

        dω/dt = -(c/I)*ω + τ/I
        dθ/dt = ω
    """
    alpha = -(DAMPING / INERTIA) * omega + tau / INERTIA
    omega_new = omega + OBS_DT * alpha
    theta_new = theta + OBS_DT * omega_new
    return theta_new, omega_new


# ============================================================
# Run simulations
# ============================================================

def run_free() -> tuple:
    """Free motion: no control torque. θ damps to rest near THETA_0."""
    theta, omega = THETA_0, OMEGA_0
    theta_hist = [theta]
    omega_hist = [omega]
    t_hist = [0.0]

    for k in range(N_STEPS):
        theta, omega = sim_step(theta, omega, tau=0.0)
        theta_hist.append(theta)
        omega_hist.append(omega)
        t_hist.append((k + 1) * OBS_DT)

    return np.array(t_hist), np.array(theta_hist), np.array(omega_hist)


def run_adem(rng) -> tuple:
    """ADEM control: D-step perception + A-step action toward THETA_TARGET."""
    model = build_model()
    g_action_fn = make_g_action()

    d_step = DStep(
        model, kappa_mu=KAPPA_MU, dt=D_STEP_DT,
        n_iter=N_ITER_D_STEP, use_d_operator=False,
    )
    action_update = ActionUpdate(
        model, g_action=g_action_fn, kappa_a=KAPPA_A, dt=A_DT,
    )

    # Goal prior: desire to be at θ_target with zero velocity
    y_tilde_goal = jnp.zeros(N_ORDER).at[0].set(THETA_TARGET)

    theta, omega = THETA_0, OMEGA_0
    mu_x = jnp.zeros(N_ORDER).at[0].set(THETA_0)
    mu_v = jnp.zeros(N_ORDER)
    a    = jnp.zeros(1)

    theta_hist = [theta]
    omega_hist = [omega]
    a_hist     = [0.0]
    vfe_hist   = []
    t_hist     = [0.0]

    theta_prev_obs = THETA_0

    for k in range(N_STEPS):
        # Noisy angle observation
        theta_obs = theta + rng.normal(0.0, NOISE_STD)

        # Finite-difference velocity estimate
        omega_fd = (theta_obs - theta_prev_obs) / OBS_DT

        # Build y_tilde from actual observations (for perception)
        y_tilde_obs = jnp.zeros(N_ORDER).at[0].set(theta_obs).at[1].set(omega_fd)

        # D-step: update belief from actual observation
        mu_x, mu_v, vfe = d_step.run(mu_x, mu_v, y_tilde_obs)
        vfe_hist.append(vfe)

        # A-step: update action to minimize VFE toward goal
        for _ in range(N_ITER_A_STEP):
            a = action_update.step(a, mu_x, mu_v, y_tilde_goal)

        # Clamp torque and apply to physics
        tau = float(jnp.clip(a[0], -TORQUE_LIMIT, TORQUE_LIMIT))
        theta_prev_obs = theta_obs
        theta, omega = sim_step(theta, omega, tau=tau)

        theta_hist.append(theta)
        omega_hist.append(omega)
        a_hist.append(tau)
        t_hist.append((k + 1) * OBS_DT)

    return (
        np.array(t_hist),
        np.array(theta_hist),
        np.array(omega_hist),
        np.array(a_hist),
        np.array(vfe_hist),
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("ADEM: Goal-Reaching via Active Inference")
    print(f"  System: horizontal joint, I={INERTIA:.3f} kg·m², c={DAMPING} Nms/rad")
    print(f"  Start: θ0 = {THETA_0} rad → Goal: θ_target = {THETA_TARGET} rad")
    print("=" * 60)
    print()

    rng = np.random.default_rng(SEED)
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Free motion ---
    print("Step 1: Free motion (no control)")
    t_free, theta_free, omega_free = run_free()
    final_free = theta_free[-1]
    print(f"  Final θ = {final_free:.4f} rad  (target: {THETA_TARGET})")
    print()

    # --- ADEM control ---
    print("Step 2: ADEM control")
    t_adem, theta_adem, omega_adem, a_adem, vfe_adem = run_adem(rng)

    if np.isnan(theta_adem).any():
        print("  ERROR: NaN in ADEM trajectory!")
        return

    final_adem = theta_adem[-1]
    final_err  = abs(final_adem - THETA_TARGET)
    settle_thr = 0.03   # within 0.03 rad of target
    settle_idx = next(
        (i for i in range(len(theta_adem))
         if abs(theta_adem[i] - THETA_TARGET) < settle_thr),
        len(theta_adem) - 1,
    )
    settle_time = t_adem[settle_idx]

    print(f"  Final θ = {final_adem:.4f} rad  (error = {final_err:.4f} rad)")
    print(f"  Settled within {settle_thr} rad at t = {settle_time:.2f} s")
    print(f"  Peak torque: {np.abs(a_adem).max():.2f} N·m")
    print(f"  Steady-state torque: {a_adem[-10:].mean():.3f} N·m "
          f"(≈0 expected for damped-only system)")
    print()

    # --- PD control equivalent (for reference) ---
    Kp = COUPLING_POS / (COUPLING_POS**2 + COUPLING_VEL**2)
    Kd = COUPLING_VEL / (COUPLING_POS**2 + COUPLING_VEL**2)
    print(f"  Equivalent PD gains: Kp≈{Kp:.2f}, Kd≈{Kd:.2f}")
    print()

    # --- Plot ---
    print("Step 3: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 10))

        # Panel 1: angle trajectory
        ax = axes[0]
        ax.axhline(THETA_TARGET, color='k', ls='--', lw=1.2, alpha=0.6,
                   label=f'Goal θ_target = {THETA_TARGET} rad')
        ax.axhline(THETA_0, color='gray', ls=':', lw=1.0, alpha=0.5,
                   label=f'Start θ0 = {THETA_0} rad')
        ax.plot(t_free, theta_free, 'b-', lw=1.8, alpha=0.7,
                label=f'Free motion (final={final_free:.3f} rad)')
        ax.plot(t_adem, theta_adem, 'r-', lw=2.0,
                label=f'ADEM control (settled at t={settle_time:.1f}s, '
                      f'error={final_err:.3f} rad)')
        ax.set_ylabel('Angle θ (rad)')
        ax.set_title('Joint Angle: Goal-Reaching via ADEM Active Inference')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: torque
        ax = axes[1]
        ax.plot(t_adem[1:], a_adem[1:], 'm-', lw=1.5)
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_ylabel('Torque τ (N·m)')
        ax.set_title('ADEM Action (A-step output: PD-like torque)')
        ax.grid(True, alpha=0.3)

        # Panel 3: VFE
        ax = axes[2]
        t_vfe = t_adem[1:]
        ax.plot(t_vfe, vfe_adem, 'g-', lw=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VFE')
        ax.set_title('Variational Free Energy (lower = belief closer to goal)')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            'ADEM Goal-Reaching: Active Inference for Joint Control\n'
            f'(coupling_pos={COUPLING_POS}, coupling_vel={COUPLING_VEL}, '
            f'pi_y={PI_Y}, pi_x={PI_X})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_adem_goal_reaching.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Free motion:  final θ = {final_free:.4f} rad  "
          f"(error from target = {abs(final_free-THETA_TARGET):.4f} rad)")
    print(f"  ADEM control: final θ = {final_adem:.4f} rad  "
          f"(error = {final_err:.4f} rad)")
    print(f"  Settled at t = {settle_time:.2f} s  |  "
          f"Peak torque = {np.abs(a_adem).max():.2f} N·m")
    print("=" * 60)


if __name__ == "__main__":
    main()
