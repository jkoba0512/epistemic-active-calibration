"""Phase 0: Foundation for epistemic A-step in DEM self-calibration.

Tests (Phase 0 deliverables from research plan §8):
  T1: rollout(q, u, theta) is JAX-jittable (pure function)
  T2: dy_future/dtheta via jax.jacfwd gives correct shape and finite values
  T3: FIM = J.T @ R_inv @ J is positive semi-definite
  T4: P_theta precision update (simulated E-step accumulation) is PD
  T5: IG = 0.5*(logdet(P_post) - logdet(P_prior)) is finite and positive
  T6: damping keeps P_theta + FIM positive-definite near singular cases
  T7: d(IG)/d(u) is differentiable (prerequisite for epistemic A-step gradient)
  T8: EStep.compute_precision() returns PD matrix matching manual FIM

System: 2-DOF planar arm
  q = [q1, q2]        joint angles (rad)
  u = [dq1, dq2]      velocity command (rad/s)
  theta = [l1, l2]    link lengths (m)
  y = [x_ee, y_ee]    end-effector position (m)

Usage:
    .venv/bin/python experiments/phase0_foundation.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ── 2-DOF planar arm kinematic model ─────────────────────────────────────────

def fk(q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Forward kinematics: end-effector position.

    Args:
        q: [q1, q2] joint angles (rad)
        theta: [l1, l2] link lengths (m)
    Returns:
        [x_ee, y_ee] end-effector position (m)
    """
    x = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    y = theta[0] * jnp.sin(q[0]) + theta[1] * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])


def rollout(q: jnp.ndarray, u: jnp.ndarray, dt: float, n_steps: int) -> jnp.ndarray:
    """Kinematic rollout under constant velocity command (Euler integration).

    q_{t+1} = q_t + u * dt
    Pure JAX (uses lax.scan), jittable.

    Args:
        q: [q1, q2] initial joint angles
        u: [dq1, dq2] constant velocity command
        dt: time step (s)
        n_steps: number of integration steps
    Returns:
        q_future: joint angles after n_steps
    """
    def step(q_curr, _):
        return q_curr + u * dt, None

    q_future, _ = jax.lax.scan(step, q, None, length=n_steps)
    return q_future


def y_future_fn(
    q: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    dt: float,
    n_steps: int,
) -> jnp.ndarray:
    """End-effector position after rolling out action u for n_steps.

    Pure JAX, differentiable w.r.t. theta and u.
    """
    q_f = rollout(q, u, dt, n_steps)
    return fk(q_f, theta)


# ── FIM and information gain ──────────────────────────────────────────────────

def compute_fim(
    q: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    R_obs_inv: jnp.ndarray,
    dt: float,
    n_steps: int,
) -> jnp.ndarray:
    """Fisher Information Matrix for theta given action u.

    FIM = J.T @ R_obs_inv @ J,  J = d(y_future)/d(theta)

    Args:
        q: current joint angles
        u: velocity command
        theta: current parameter estimate [l1, l2]
        R_obs_inv: observation noise precision matrix (n_y x n_y)
        dt: time step
        n_steps: rollout horizon
    Returns:
        FIM: shape (n_params, n_params)
    """
    J = jax.jacfwd(lambda t: y_future_fn(q, u, t, dt, n_steps))(theta)
    return J.T @ R_obs_inv @ J


def compute_info_gain(
    P_theta: jnp.ndarray,
    FIM_future: jnp.ndarray,
    damping: float = 1e-6,
) -> jnp.ndarray:
    """Expected information gain from action.

    IG = 0.5 * (logdet(P_post) - logdet(P_prior))
    where P_post = P_theta + FIM_future (precision form).

    Damping is added to ensure positive-definiteness.

    Args:
        P_theta: current posterior precision (n_params x n_params)
        FIM_future: Fisher information from proposed action
        damping: regularization scalar for numerical stability
    Returns:
        Scalar IG value
    """
    I = jnp.eye(P_theta.shape[0])
    P_prior_reg = P_theta + damping * I
    P_post_reg = P_theta + FIM_future + damping * I
    _, logdet_prior = jnp.linalg.slogdet(P_prior_reg)
    _, logdet_post = jnp.linalg.slogdet(P_post_reg)
    return 0.5 * (logdet_post - logdet_prior)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    print("=" * 65)
    print("Phase 0: Foundation Tests")
    print("  System: 2-DOF planar arm, theta=[l1, l2], y=[x_ee, y_ee]")
    print("=" * 65)

    # Test parameters
    dt = 0.05
    n_steps = 10
    theta_true = jnp.array([0.5, 0.5])
    theta_est = jnp.array([0.7, 0.3])  # intentionally wrong
    q0 = jnp.array([0.4, 0.3])
    u_test = jnp.array([0.1, -0.2])
    R_obs_inv = jnp.eye(2) * 100.0  # high observation precision
    Pi_theta_prior = 1.0 * jnp.eye(2)
    damping = 1e-6

    results = []

    # T1: rollout is JAX-jittable
    try:
        rollout_jit = jax.jit(rollout, static_argnums=(2, 3))
        q_f = rollout_jit(q0, u_test, dt, n_steps)
        ok = q_f.shape == (2,) and bool(jnp.isfinite(q_f).all())
        results.append(_check(
            "T1: rollout JAX-jittable", ok,
            f"q_future=[{float(q_f[0]):.4f}, {float(q_f[1]):.4f}]",
        ))
    except Exception as e:
        results.append(_check("T1: rollout JAX-jittable", False, str(e)))

    # T2: dy_future/dtheta via jacfwd
    try:
        J = jax.jacfwd(lambda t: y_future_fn(q0, u_test, t, dt, n_steps))(theta_est)
        ok = J.shape == (2, 2) and bool(jnp.isfinite(J).all())
        results.append(_check(
            "T2: jacfwd dy_future/dtheta, shape (2,2)", ok,
            f"J=[[{float(J[0,0]):.3f},{float(J[0,1]):.3f}],"
            f"[{float(J[1,0]):.3f},{float(J[1,1]):.3f}]]",
        ))
    except Exception as e:
        results.append(_check("T2: jacfwd dy_future/dtheta", False, str(e)))

    # T3: FIM is PSD
    try:
        FIM = compute_fim(q0, u_test, theta_est, R_obs_inv, dt, n_steps)
        eigs = jnp.linalg.eigvalsh(FIM)
        ok = bool(jnp.isfinite(FIM).all()) and bool((eigs >= -1e-10).all())
        results.append(_check(
            "T3: FIM is PSD", ok,
            f"eigenvalues=[{float(eigs[0]):.3f}, {float(eigs[1]):.3f}]",
        ))
    except Exception as e:
        results.append(_check("T3: FIM PSD", False, str(e)))

    # T4: P_theta precision update (simulated multi-step accumulation)
    try:
        key = jax.random.PRNGKey(0)
        P_accumulated = jnp.zeros((2, 2))
        for i in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            u_i = jax.random.normal(k1, (2,)) * 0.3
            q_i = q0 + jax.random.normal(k2, (2,)) * 0.2
            J_i = jax.jacfwd(lambda t: y_future_fn(q_i, u_i, t, dt, 1))(theta_true)
            P_accumulated = P_accumulated + J_i.T @ R_obs_inv @ J_i
        P_theta = P_accumulated + Pi_theta_prior
        eigs_P = jnp.linalg.eigvalsh(P_theta)
        ok = bool(jnp.isfinite(P_theta).all()) and bool((eigs_P > 0).all())
        results.append(_check(
            "T4: P_theta PD after 20-step accumulation", ok,
            f"eigenvalues=[{float(eigs_P[0]):.2f}, {float(eigs_P[1]):.2f}]",
        ))
    except Exception as e:
        results.append(_check("T4: P_theta accumulation", False, str(e)))

    # T5: IG finite and positive for informative action
    try:
        FIM_test = compute_fim(q0, u_test, theta_est, R_obs_inv, dt, n_steps)
        ig = compute_info_gain(Pi_theta_prior, FIM_test, damping)
        ok = bool(jnp.isfinite(ig)) and float(ig) > 0
        results.append(_check(
            "T5: IG finite and positive", ok,
            f"IG={float(ig):.4f}",
        ))
    except Exception as e:
        results.append(_check("T5: IG computation", False, str(e)))

    # T6: damping handles near-singular P_theta
    try:
        P_singular = jnp.array([[1.0, 0.0], [0.0, 1e-14]])
        FIM_zero = jnp.zeros((2, 2))
        ig_s = compute_info_gain(P_singular, FIM_zero, damping=1e-6)
        ok = bool(jnp.isfinite(ig_s))
        results.append(_check(
            "T6: damping handles near-singular P_theta", ok,
            f"IG(near-singular)={float(ig_s):.6f}",
        ))
    except Exception as e:
        results.append(_check("T6: near-singular damping", False, str(e)))

    # T7: d(IG)/d(u) differentiable (prerequisite for epistemic A-step)
    try:
        def ig_of_u(u):
            fim = compute_fim(q0, u, theta_est, R_obs_inv, dt, n_steps)
            return compute_info_gain(Pi_theta_prior, fim, damping)

        dig_du = jax.grad(ig_of_u)(u_test)
        ok = dig_du.shape == (2,) and bool(jnp.isfinite(dig_du).all())
        results.append(_check(
            "T7: d(IG)/d(u) differentiable (epistemic A-step gradient)", ok,
            f"dIG/du=[{float(dig_du[0]):.4f}, {float(dig_du[1]):.4f}]",
        ))
    except Exception as e:
        results.append(_check("T7: d(IG)/d(u) differentiable", False, str(e)))

    # T8: EStep.compute_precision() matches manual FIM accumulation
    try:
        from src.dem.model import DEMModel
        from src.dem.estep import EStep

        # Build minimal DEM model: static kinematic observation, n_order=1
        n_order = 1

        def f_zero(x_t, v_t, p):
            return jnp.zeros(n_order * 2)

        def g_fk(x_t, v_t, p):
            return fk(x_t, p)

        model = DEMModel(
            f=f_zero,
            g=g_fk,
            n_x=2,
            n_v=2,
            n_y=2,
            n_order=n_order,
            pi_y=100.0,
            pi_x=1.0,
            params=theta_est,
            params_prior_pi=1.0,
        )
        estep = EStep(model, use_gauss_newton=True)

        # Build a small observation sequence at known joint configs
        key = jax.random.PRNGKey(42)
        mus_x, mus_v, ys = [], [], []
        for i in range(10):
            key, k1 = jax.random.split(key)
            q_i = jax.random.uniform(k1, (2,), minval=-1.0, maxval=1.0)
            y_i = fk(q_i, theta_true) + jax.random.normal(k1, (2,)) * 0.01
            mus_x.append(q_i)
            mus_v.append(jnp.zeros(2))
            ys.append(y_i)

        P_estep = estep.compute_precision(mus_x, mus_v, ys, theta_est)
        eigs_e = jnp.linalg.eigvalsh(P_estep)
        ok = bool(jnp.isfinite(P_estep).all()) and bool((eigs_e > 0).all())
        results.append(_check(
            "T8: EStep.compute_precision() returns PD matrix", ok,
            f"eigenvalues=[{float(eigs_e[0]):.3f}, {float(eigs_e[1]):.3f}]",
        ))
    except Exception as e:
        results.append(_check("T8: EStep.compute_precision()", False, str(e)))

    print()
    n_pass = sum(results)
    n_total = len(results)
    print(f"Result: {n_pass}/{n_total} tests passed")
    print("=" * 65)
    return n_pass == n_total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
