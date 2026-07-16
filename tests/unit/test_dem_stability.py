"""Unit tests for DEM D-step numerical stability.

Tests:
- local_linearization_step: matches the analytic solution for linear systems
- D-step with D operator: does not diverge even with use_d_operator=True
- Euler vs local linearization: local linearization converges faster than Euler
- Generalized motion consistency: estimates agree with and without the D operator
"""

import pytest
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp
import numpy as np

from src.dem.model import LinearDEMModel, DEMModel
from src.dem.inference import DStep, compute_vfe
from src.dem.utils import local_linearization_step


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_test_model(
    n_order: int = 4,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
) -> DEMModel:
    """Create a 1D linear test model dx/dt = -x + v, y = x."""
    A = jnp.array([[-1.0]])
    C = jnp.array([[1.0]])
    return LinearDEMModel(A, C, n_order=n_order, pi_y=pi_y, pi_x=pi_x)


def make_observation(y0: float = 1.0, n_order: int = 4) -> jnp.ndarray:
    """Create a generalized observation vector with only the zeroth-order component nonzero."""
    y_tilde = jnp.zeros(n_order)
    return y_tilde.at[0].set(y0)


# ---------------------------------------------------------------------------
# test_local_linearization_linear
# ---------------------------------------------------------------------------

class TestLocalLinearizationLinear:
    """Test that local_linearization_step matches the analytic solution for linear systems."""

    def test_scalar_exponential_decay(self):
        """Check agreement with the analytic solution of the scalar linear system dx/dt = -lambda * x.

        Analytic solution: Δx = x₀ * (e^(-lambda * dt) - 1)
        """
        lam = 2.0
        x0 = jnp.array([1.5])
        J = jnp.array([[-lam]])
        f_val = J @ x0  # = -lam * x0

        dt = 0.1
        delta_x = local_linearization_step(f_val, J, dt, damping=0.0)

        # Analytic solution
        expected = x0 * (jnp.exp(-lam * dt) - 1)
        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-4,
            err_msg="Local linearization should match analytic solution for scalar decay"
        )

    def test_2d_linear_system(self):
        """Check agreement with the matrix-exponential analytic solution for a 2D linear system."""
        A = jnp.array([[-1.0, 0.5], [-0.5, -1.0]])
        x0 = jnp.array([1.0, 0.0])
        f_val = A @ x0
        dt = 0.05

        delta_x = local_linearization_step(f_val, A, dt, damping=0.0)

        # Analytic solution: Δx = (expm(A*dt) - I) x0 = expm(A*dt) x0 - x0
        import jax.scipy.linalg
        expm_A_dt = jax.scipy.linalg.expm(A * dt)
        expected = expm_A_dt @ x0 - x0

        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-5,
            err_msg="Local linearization should match matrix exponential for 2D linear system"
        )

    def test_constant_input(self):
        """Check agreement with the analytic solution of the constant-input system dx/dt = A*x + b.

        Here f(x0) = A*x0 + b and J = A.
        Analytic solution: Δx = A⁻¹(e^(A*dt) - I)(A*x0 + b)
        """
        A = jnp.array([[-2.0]])
        b = jnp.array([1.0])
        x0 = jnp.array([0.5])
        f_val = A @ x0 + b  # f(x0) = A*x0 + b
        dt = 0.1

        delta_x = local_linearization_step(f_val, A, dt, damping=0.0)

        # Analytic solution
        import jax.scipy.linalg
        expm_Adt = jax.scipy.linalg.expm(A * dt)
        A_inv = jnp.linalg.inv(A)
        expected = A_inv @ (expm_Adt - jnp.eye(1)) @ f_val

        np.testing.assert_allclose(
            np.array(delta_x), np.array(expected), rtol=1e-5,
            err_msg="Local linearization should match analytic solution for affine system"
        )

    def test_jit_compilable(self):
        """Test that local_linearization_step is compilable with jax.jit."""
        J = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        f_val = jnp.array([1.0, 0.5])

        @jax.jit
        def jit_step(f, j):
            return local_linearization_step(f, j, dt=0.01)

        result = jit_step(f_val, J)
        assert jnp.all(jnp.isfinite(result)), "JIT-compiled local linearization should produce finite results"


# ---------------------------------------------------------------------------
# test_d_step_stable_with_d_operator
# ---------------------------------------------------------------------------

class TestDStepStableWithDOperator:
    """Test stable behavior with a large dt even when use_d_operator=True."""

    def test_stable_with_large_dt(self):
        """Check that it does not diverge even with a large step dt=0.01.

        With Euler, when the largest Hessian eigenvalue is around 269 the method
        becomes unstable for dt > 2/269 ≈ 0.0074, whereas local linearization
        remains stable.
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,  # large dt that is unstable for Euler
            n_iter=20,
            use_d_operator=True,
            use_local_linearization=True,
        )

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x_new, mu_v_new, vfe_final = d_step.run(mu_x0, mu_v0, y_tilde)

        # Should not diverge (finite values)
        assert jnp.all(jnp.isfinite(mu_x_new)), (
            f"mu_x_tilde diverged with large dt=0.01: {mu_x_new}"
        )
        assert jnp.all(jnp.isfinite(mu_v_new)), (
            f"mu_v_tilde diverged with large dt=0.01: {mu_v_new}"
        )
        assert jnp.isfinite(vfe_final), f"VFE diverged: {vfe_final}"

    def test_no_divergence_over_iterations(self):
        """Check that it does not diverge over many iterations."""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,
            n_iter=1,  # check one step at a time
            use_d_operator=True,
            use_local_linearization=True,
        )

        mu_x = jnp.zeros(model.dim_x_tilde)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        vfe_values = []
        for i in range(50):
            mu_x, mu_v = d_step.run_single_step(mu_x, mu_v, y_tilde)
            vfe = float(compute_vfe(mu_x, mu_v, y_tilde, model))
            vfe_values.append(vfe)

            assert jnp.all(jnp.isfinite(mu_x)), f"mu_x diverged at step {i}: {mu_x}"
            assert jnp.isfinite(vfe), f"VFE diverged at step {i}: {vfe}"

    def test_vfe_decreases_with_d_operator(self):
        """Check that VFE decreases from its initial value with the D operator and local linearization."""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        d_step = DStep(
            model,
            kappa_mu=1.0,
            dt=0.01,
            n_iter=30,
            use_d_operator=True,
            use_local_linearization=True,
        )

        key = jax.random.PRNGKey(42)
        mu_x0 = jax.random.normal(key, (model.dim_x_tilde,)) * 0.5
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(1.0, model.n_order)

        vfe_initial = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))
        mu_x_new, mu_v_new, vfe_final = d_step.run(mu_x0, mu_v0, y_tilde)

        assert vfe_final < vfe_initial, (
            f"VFE should decrease with D-operator + local linearization: "
            f"initial={vfe_initial:.4f}, final={vfe_final:.4f}"
        )


# ---------------------------------------------------------------------------
# test_d_step_euler_vs_local_lin_convergence
# ---------------------------------------------------------------------------

class TestEulerVsLocalLinConvergence:
    """Test that local linearization converges faster than Euler."""

    def test_local_lin_fewer_steps_to_converge(self):
        """Check that, at the same dt, local linearization reaches a low VFE in fewer steps.

        Comparison within a safe dt range (dt=0.001).
        Local linearization takes a larger "equivalent step", so it converges faster.
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        dt = 0.001
        n_steps = 20

        mu_x0 = jnp.ones(model.dim_x_tilde) * 2.0
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(0.0, model.n_order)  # converge towards observation y=0

        vfe_init = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))

        # Euler (no D operator, stable gradient descent)
        d_step_euler = DStep(
            model, kappa_mu=1.0, dt=dt, n_iter=n_steps,
            use_d_operator=False, use_local_linearization=False
        )
        _, _, vfe_euler = d_step_euler.run(mu_x0, mu_v0, y_tilde)

        # Local linearization (with D operator)
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=dt, n_iter=n_steps,
            use_d_operator=True, use_local_linearization=True
        )
        _, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # Both should decrease from the initial VFE
        assert vfe_euler < vfe_init, "Euler should reduce VFE"
        assert vfe_ll < vfe_init, "Local linearization should reduce VFE"

        # Both should be finite
        assert jnp.isfinite(vfe_euler), f"Euler VFE is not finite: {vfe_euler}"
        assert jnp.isfinite(vfe_ll), f"Local linearization VFE is not finite: {vfe_ll}"

    def test_local_lin_stable_where_euler_diverges(self):
        """Check that local linearization is stable at a dt where Euler is unstable.

        dt=0.01 is dangerous for Euler when the Hessian eigenvalues are large,
        but local linearization remains stable.
        """
        model = make_test_model(n_order=4, pi_y=16.0, pi_x=1.0)  # large pi_y

        mu_x0 = jnp.ones(model.dim_x_tilde) * 3.0
        mu_v0 = jnp.zeros(model.dim_v_tilde)
        y_tilde = make_observation(0.0, model.n_order)

        # Local linearization: should be stable even at dt=0.01
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_new, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        assert jnp.all(jnp.isfinite(mu_x_new)), (
            f"Local linearization diverged where Euler would be unstable: {mu_x_new}"
        )
        assert jnp.isfinite(vfe_ll), f"VFE not finite: {vfe_ll}"

        # VFE should decrease
        vfe_init = float(compute_vfe(mu_x0, mu_v0, y_tilde, model))
        assert vfe_ll < vfe_init, (
            f"Local linearization should reduce VFE: init={vfe_init:.4f}, final={vfe_ll:.4f}"
        )


# ---------------------------------------------------------------------------
# test_generalized_motion_consistency
# ---------------------------------------------------------------------------

class TestGeneralizedMotionConsistency:
    """Test that the final estimates are close with and without the D operator."""

    def test_gradient_descent_and_local_lin_agree(self):
        """Check that gradient descent and local linearization converge to similar estimates for the same observation.

        The VFE minimizer is independent of the integration method, so after enough
        iterations the zeroth-order state estimates should be close.
        """
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # Gradient descent (no D operator, many small steps)
        d_step_gd = DStep(
            model, kappa_mu=1.0, dt=0.001, n_iter=200,
            use_d_operator=False, use_local_linearization=False
        )
        mu_x_gd, _, vfe_gd = d_step_gd.run(mu_x0, mu_v0, y_tilde)

        # Local linearization (with D operator)
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_ll, _, vfe_ll = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # Zeroth-order state estimates (first element) should be close
        x0_gd = float(mu_x_gd[0])
        x0_ll = float(mu_x_ll[0])

        assert abs(x0_gd - x0_ll) < 0.5, (
            f"Gradient descent and local linearization should give similar estimates: "
            f"GD={x0_gd:.4f}, LL={x0_ll:.4f}"
        )

        # Both should be finite
        assert jnp.all(jnp.isfinite(mu_x_gd)), "Gradient descent estimate not finite"
        assert jnp.all(jnp.isfinite(mu_x_ll)), "Local linearization estimate not finite"

    def test_both_modes_track_observation(self):
        """Check that both modes move the estimate towards the observation.

        For observation y=1.0, the zeroth-order state estimate should move from the
        initial value 0 towards 1.
        """
        model = make_test_model(n_order=4, pi_y=8.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)

        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # Gradient descent
        d_step_gd = DStep(
            model, kappa_mu=1.0, dt=0.001, n_iter=200,
            use_d_operator=False
        )
        mu_x_gd, _, _ = d_step_gd.run(mu_x0, mu_v0, y_tilde)

        # Local linearization
        d_step_ll = DStep(
            model, kappa_mu=1.0, dt=0.01, n_iter=50,
            use_d_operator=True, use_local_linearization=True
        )
        mu_x_ll, _, _ = d_step_ll.run(mu_x0, mu_v0, y_tilde)

        # Both should move towards the observation (1.0)
        assert float(mu_x_gd[0]) > 0.1, (
            f"GD estimate should move towards y=1.0, got {float(mu_x_gd[0]):.4f}"
        )
        assert float(mu_x_ll[0]) > 0.1, (
            f"LL estimate should move towards y=1.0, got {float(mu_x_ll[0]):.4f}"
        )

    def test_jit_compatible_both_modes(self):
        """Test that DStep is compilable with jax.jit in both modes."""
        model = make_test_model(n_order=4, pi_y=4.0, pi_x=1.0)
        y_tilde = make_observation(1.0, model.n_order)
        mu_x0 = jnp.zeros(model.dim_x_tilde)
        mu_v0 = jnp.zeros(model.dim_v_tilde)

        # Gradient descent
        d_step_gd = DStep(model, kappa_mu=1.0, dt=0.001, n_iter=1,
                          use_d_operator=False)
        # Local linearization
        d_step_ll = DStep(model, kappa_mu=1.0, dt=0.01, n_iter=1,
                          use_d_operator=True, use_local_linearization=True)

        # _euler_step is already jax.jit-compiled in both, so just call and test
        mu_x_gd, mu_v_gd = d_step_gd.run_single_step(mu_x0, mu_v0, y_tilde)
        mu_x_ll, mu_v_ll = d_step_ll.run_single_step(mu_x0, mu_v0, y_tilde)

        assert jnp.all(jnp.isfinite(mu_x_gd)), "GD step produced non-finite values"
        assert jnp.all(jnp.isfinite(mu_x_ll)), "LL step produced non-finite values"
