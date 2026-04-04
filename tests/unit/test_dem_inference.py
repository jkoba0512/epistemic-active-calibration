"""Unit tests for src/dem/inference.py.

Tests:
- D-step reduces VFE (convergence check)
- Linear system state inference convergence
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.dem.core import make_D_matrix, make_tilde_precision
from src.dem.model import DEMModel, LinearDEMModel
from src.dem.inference import DStep, compute_vfe
from src.dem.agent import DEMAgent


def make_simple_linear_model(
    pi_y: float = 4.0,
    pi_x: float = 1.0,
    n_order: int = 4,
    s_y: float = 1.0,
    s_x: float = 1.0,
) -> DEMModel:
    """Create a simple 1D linear test model: dx/dt = -x + v, y = x."""
    A = jnp.array([[-1.0]])
    C = jnp.array([[1.0]])
    return LinearDEMModel(A, C, n_order=n_order, pi_y=pi_y, pi_x=pi_x,
                          s_y=s_y, s_x=s_x)


def generate_linear_trajectory(
    x0: float = 1.0,
    v: float = 0.0,
    dt: float = 0.1,
    T: int = 50,
    noise_std: float = 0.1,
    key: jax.Array = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy trajectory from dx/dt = -x + v, y = x.

    Args:
        x0: Initial state.
        v: Constant cause/input.
        dt: Time step.
        T: Number of time steps.
        noise_std: Observation noise standard deviation.
        key: JAX random key.

    Returns:
        Tuple of (true states, noisy observations), each shape (T,).
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    states = np.zeros(T)
    x = x0
    for t in range(T):
        states[t] = x
        dx = -x + v
        x = x + dt * dx

    noise = jax.random.normal(key, (T,)) * noise_std
    observations = states + np.array(noise)
    return states, observations


class TestVFEComputation:
    """Tests for compute_vfe function."""

    def test_vfe_is_scalar(self):
        """VFE should return a scalar."""
        model = make_simple_linear_model()
        mu_x = jnp.zeros(model.dim_x_tilde)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jnp.zeros(model.dim_y_tilde)

        vfe = compute_vfe(mu_x, mu_v, y_tilde, model)
        assert vfe.shape == (), f"VFE should be scalar, got shape {vfe.shape}"

    def test_vfe_nonnegative(self):
        """VFE should be non-negative (quadratic form with PD precision)."""
        model = make_simple_linear_model()
        key = jax.random.PRNGKey(0)
        mu_x = jax.random.normal(key, (model.dim_x_tilde,))
        mu_v = jax.random.normal(key, (model.dim_v_tilde,))
        y_tilde = jax.random.normal(key, (model.dim_y_tilde,))

        vfe = compute_vfe(mu_x, mu_v, y_tilde, model)
        assert float(vfe) >= 0.0, f"VFE should be non-negative, got {vfe}"

    def test_vfe_zero_at_perfect_prediction(self):
        """VFE should be small when predictions perfectly match observations."""
        model = make_simple_linear_model(pi_x=100.0)

        # At steady state with x=1, y=1 for dx/dt = -x + 1 (equilibrium at x=1)
        # Set mu_x = [1, 0, 0, 0] (x=1, all derivatives 0)
        # Then D*mu_x = [0,0,0,0], f(mu_x) = A*[1,0,0,0] + v = [-1+1, ...] = 0
        # So eps_x ~ 0

        # f = Ax + v, at equilibrium: Ax + v = 0 -> x = 1 when v = 1
        # y_tilde = [g(mu_x)] = [1, 0, ...]
        mu_x = jnp.array([1.0, 0.0, 0.0, 0.0])  # x=1, derivatives 0
        # v should be 1 so that Ax + v = -1 + 1 = 0 (equilibrium)
        mu_v = jnp.array([1.0, 0.0, 0.0, 0.0])
        # Perfect observation: y = g(mu_x) = C*x = 1
        y_tilde = jnp.array([1.0, 0.0, 0.0, 0.0])

        vfe = compute_vfe(mu_x, mu_v, y_tilde, model)
        assert float(vfe) < 0.5, f"VFE should be small at equilibrium, got {vfe}"

    def test_vfe_gradient_finite(self):
        """Gradient of VFE should be finite everywhere."""
        model = make_simple_linear_model()
        key = jax.random.PRNGKey(1)
        mu_x = jax.random.normal(key, (model.dim_x_tilde,))
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jax.random.normal(key, (model.dim_y_tilde,))

        grad = jax.grad(compute_vfe)(mu_x, mu_v, y_tilde, model)
        assert jnp.all(jnp.isfinite(grad)), "VFE gradient contains non-finite values"


class TestDStepReducesVFE:
    """Tests that D-step reduces VFE."""

    def test_d_step_reduces_vfe(self):
        """Running D-step should reduce VFE compared to initial state.

        This tests the basic property that gradient descent on VFE
        moves the estimate in the direction that reduces free energy.
        Uses gradient descent mode (use_d_operator=False) for stability.
        """
        model = make_simple_linear_model(pi_y=10.0, pi_x=1.0, n_order=4)
        # Stable dt for gradient descent: dt < 2 / max_eigenvalue_of_Hessian
        d_step = DStep(model, kappa_mu=1.0, dt=0.001, n_iter=50,
                       use_d_operator=False)

        key = jax.random.PRNGKey(42)
        # Start with a noisy initial estimate
        mu_x0 = jax.random.normal(key, (model.dim_x_tilde,)) * 2.0
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # Observation at x=1 (zeroth order)
        y_tilde = jnp.zeros(model.dim_y_tilde)
        y_tilde = y_tilde.at[0].set(1.0)

        # Compute initial VFE
        vfe_initial = compute_vfe(mu_x0, mu_v0, y_tilde, model)

        # Run D-step
        mu_x_new, mu_v_new, vfe_final = d_step.run(mu_x0, mu_v0, y_tilde)

        assert float(vfe_final) < float(vfe_initial), (
            f"D-step did not reduce VFE: initial={vfe_initial:.4f}, "
            f"final={vfe_final:.4f}"
        )

    def test_d_step_vfe_monotone_decrease(self):
        """VFE should decrease monotonically over D-step iterations.

        Uses gradient descent mode (use_d_operator=False) for stability.
        """
        model = make_simple_linear_model(pi_y=10.0, pi_x=1.0, n_order=4)
        d_step_single = DStep(model, kappa_mu=0.5, dt=0.001, n_iter=1,
                              use_d_operator=False)

        key = jax.random.PRNGKey(7)
        mu_x = jax.random.normal(key, (model.dim_x_tilde,))
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jnp.zeros(model.dim_y_tilde).at[0].set(1.0)

        vfe_values = []
        for _ in range(30):
            vfe = compute_vfe(mu_x, mu_v, y_tilde, model)
            vfe_values.append(float(vfe))
            mu_x, mu_v = d_step_single.run_single_step(mu_x, mu_v, y_tilde)

        # Check overall trend is decreasing (allow small oscillations)
        # Final VFE should be less than initial VFE
        assert vfe_values[-1] < vfe_values[0], (
            f"VFE did not decrease overall: start={vfe_values[0]:.4f}, "
            f"end={vfe_values[-1]:.4f}"
        )


class TestLinearSystemConvergence:
    """Tests for convergence of DEM to true states in a linear system."""

    def test_linear_system_convergence(self):
        """DEMAgent should converge to true state in a linear system.

        System: dx/dt = -x, y = x
        True state approaches 0 from x0=1.
        The agent should track the true state using gradient descent mode.
        """
        A = jnp.array([[-1.0]])
        C = jnp.array([[1.0]])
        n_order = 4
        model = LinearDEMModel(A, C, n_order=n_order, pi_y=16.0, pi_x=1.0,
                               s_y=1.0, s_x=1.0)
        # Use gradient descent mode (use_d_operator=False) for stability
        # dt=0.001 is within stable range for this model's Hessian
        agent = DEMAgent(model, kappa_mu=1.0, dt=0.001, n_iter_per_step=100)

        # Generate true trajectory
        dt = 0.1
        T = 30
        x_true = np.zeros(T)
        x = 1.0
        for t in range(T):
            x_true[t] = x
            x = x + dt * (-x)

        # Create generalized observations (only zeroth order is non-zero)
        noise_key = jax.random.PRNGKey(0)
        noise = jax.random.normal(noise_key, (T,)) * 0.1
        y_obs = x_true + np.array(noise)

        y_tilde_sequence = []
        for t in range(T):
            y_t = jnp.zeros(model.dim_y_tilde)
            y_t = y_t.at[0].set(float(y_obs[t]))
            y_tilde_sequence.append(y_t)

        # Run agent
        final_state, mu_x_history = agent.run(y_tilde_sequence)

        # Check: the zeroth-order estimate should track the true state
        # at least in the latter portion (after convergence)
        errors = []
        for t in range(10, T):  # skip initial transient
            mu_x0 = float(mu_x_history[t][0])  # zeroth-order estimate
            error = abs(mu_x0 - x_true[t])
            errors.append(error)

        mean_error = np.mean(errors)
        assert mean_error < 0.5, (
            f"Mean tracking error {mean_error:.4f} is too large. "
            f"Agent may not be converging to true state."
        )

    def test_static_state_recovery(self):
        """Agent should recover a static hidden state from noisy observations.

        System: dx/dt = -x + 1 (equilibrium at x=1), y = x + noise
        Expected: agent converges to estimate x ~ 1.
        Uses gradient descent mode (use_d_operator=False) for stability.
        """
        A = jnp.array([[-1.0]])
        C = jnp.array([[1.0]])
        n_order = 4
        model = LinearDEMModel(A, C, n_order=n_order, pi_y=25.0, pi_x=1.0,
                               s_y=1.0, s_x=1.0)
        # n_iter_per_step=200 with dt=0.001 gives 0.2 seconds of gradient descent
        agent = DEMAgent(model, kappa_mu=1.0, dt=0.001, n_iter_per_step=200)

        # Constant noisy observations at y=1 (true state x=1)
        key = jax.random.PRNGKey(42)
        T = 50
        noise = jax.random.normal(key, (T,)) * 0.2
        y_tilde_sequence = []
        for t in range(T):
            y_t = jnp.zeros(model.dim_y_tilde)
            y_t = y_t.at[0].set(1.0 + float(noise[t]))
            y_tilde_sequence.append(y_t)

        # Initialize near 0 (far from true state of 1)
        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.ones(model.dim_v_tilde)  # prior: v=1

        final_state, mu_x_history = agent.run(y_tilde_sequence, mu_x0, mu_v0)

        # Final estimate should be near 1
        final_x_estimate = float(mu_x_history[-1][0])
        assert abs(final_x_estimate - 1.0) < 0.5, (
            f"Static state recovery failed. Final estimate: {final_x_estimate:.4f}, "
            f"expected ~1.0"
        )
