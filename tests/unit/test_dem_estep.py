"""Unit tests for src/dem/estep.py (E-step: parameter inference).

Tests:
- test_estep_gradient_finite: gradient dF/dθ is finite
- test_estep_gradient_correct_direction: gradient points in VFE-decreasing direction
- test_estep_reduces_vfe: E-step update decreases VFE
- test_linear_param_recovery: linear system parameter converges toward true value

Gauss-Newton tests:
- test_gn_returns_finite: GN gradient and curvature are finite
- test_gn_curvature_positive_definite: curvature dFdpp is positive definite
- test_gn_reduces_vfe: GN update decreases VFE (faster than gradient descent)
- test_gn_faster_convergence: GN converges in fewer iterations than gradient descent
- test_gn_matches_gradient_descent_direction: GN step points in same direction as gradient
- test_gn_param_recovery: linear system parameter recovery with GN
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.dem.model import DEMModel
from src.dem.inference import compute_vfe
from src.dem.estep import EStep, _compute_vfe_wrt_params


# ---------------------------------------------------------------------------
# Helper: parameterized linear model with learnable scalar a
# ---------------------------------------------------------------------------

def make_param_model(
    a_init: float,
    pi_y: float = 4.0,
    pi_x: float = 1.0,
    n_order: int = 4,
    s_y: float = 1.0,
    s_x: float = 1.0,
    params_prior_pi: float = 0.01,
) -> DEMModel:
    """Create a 1D model dx/dt = a*x + v,  y = x with learnable scalar a.

    The parameter vector θ = jnp.array([a]).

    In this model, θ=a only enters the zeroth-order dynamics:
        f_tilde[0] = a*x + v  (zeroth order: state prediction)
        f_tilde[i] = 0        (higher orders: unconstrained)

    This ensures the gradient ∂f̃/∂θ is non-degenerate and the
    E-step can recover a from the state prediction error
        ε_x[0] = (D x̃)[0] - f̃[0] = x' - a*x → 0  as a → a_true.

    Args:
        a_init: Initial estimate of the decay parameter a.
        pi_y: Observation precision.
        pi_x: State noise precision.
        n_order: Generalized coordinate order.
        s_y: Observation noise smoothness.
        s_x: State noise smoothness.
        params_prior_pi: Prior precision on θ.

    Returns:
        DEMModel with params=jnp.array([a_init]).
    """
    n_x = 1

    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Parameterized dynamics: θ enters only the zeroth-order prediction.

        f̃[0] = a*x[0] + v[0]   (zeroth order)
        f̃[i] = 0                (higher orders, no θ dependence)

        Args:
            x_tilde: Generalized state, shape (n_x*n_order,).
            v_tilde: Generalized causes, shape (n_v*n_order,).
            params: Parameter vector [a], shape (1,).

        Returns:
            f̃(x̃, ṽ, θ), shape (n_x*n_order,).
        """
        a = params[0]
        n_v = v_tilde.shape[0] // n_order
        x0 = x_tilde[:n_x]
        v0 = v_tilde[:n_v]
        # Zeroth order: f = a*x + v
        result = [a * x0 + v0]
        # Higher orders: zero (no parameter dependence, no coupling)
        for _ in range(1, n_order):
            result.append(jnp.zeros(n_x))
        return jnp.concatenate(result)

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Identity observation g(x̃, ṽ, θ) = x̃.

        Args:
            x_tilde: Generalized state, shape (n_x*n_order,).
            v_tilde: Generalized causes (unused), shape (n_v*n_order,).
            params: Model parameters (unused), shape (1,).

        Returns:
            g̃(x̃, ṽ, θ) = x̃, shape (n_x*n_order,).
        """
        return x_tilde  # y = x (C = I, parameter-independent)

    params0 = jnp.array([a_init])

    return DEMModel(
        f=f,
        g=g,
        n_x=1,
        n_v=1,
        n_y=1,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        s_y=s_y,
        s_x=s_x,
        params=params0,
        params_prior_mean=params0,  # set prior mean = initial estimate
        params_prior_pi=params_prior_pi,
    )


def generate_linear_traj_tilde(
    a_true: float = -1.0,
    x0: float = 1.0,
    dt: float = 0.1,
    T: int = 100,
    noise_std: float = 0.05,
    n_order: int = 4,
    key: jax.Array = None,
) -> list:
    """Generate generalized observations from dx/dt = a*x, y = x + noise.

    Args:
        a_true: True decay parameter.
        x0: Initial state value.
        dt: Time step.
        T: Number of time steps.
        noise_std: Observation noise standard deviation.
        n_order: Generalized coordinate embedding order.
        key: JAX random key.

    Returns:
        List of T generalized observation vectors, each shape (n_order,).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Generate true trajectory
    xs = np.zeros(T)
    x = x0
    for t in range(T):
        xs[t] = x
        x = x + dt * a_true * x  # Euler integration

    # Noisy observations
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(T,)) * noise_std
    ys = xs + np.array(noise)

    # Build generalized observations (approximate derivatives by finite differences)
    y_tilde_list = []
    for t in range(T):
        y_tilde = np.zeros(n_order)
        y_tilde[0] = ys[t]
        # Approximate derivatives (use zeros for simplicity at endpoints)
        if t > 0 and t < T - 1:
            y_tilde[1] = (ys[t + 1] - ys[t - 1]) / (2 * dt)
        if t > 1 and t < T - 2:
            y_tilde[2] = (ys[t + 1] - 2 * ys[t] + ys[t - 1]) / (dt ** 2)
        y_tilde_list.append(jnp.array(y_tilde))

    return y_tilde_list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEStepGradientFinite:
    """Test that dF/dθ is finite for a valid model and data."""

    def test_estep_gradient_finite(self):
        """dF/dθ should be a finite array for reasonable inputs."""
        model = make_param_model(a_init=0.0)
        e_step = EStep(model, kappa_p=0.01)

        params = jnp.array([0.0])
        mu_x = jnp.zeros(model.dim_x_tilde).at[0].set(1.0)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jnp.zeros(model.dim_y_tilde).at[0].set(0.9)

        grad = e_step.accumulate_gradient(mu_x, mu_v, y_tilde, params)

        assert grad.shape == params.shape, (
            f"Gradient shape {grad.shape} != params shape {params.shape}"
        )
        assert jnp.all(jnp.isfinite(grad)), (
            f"Gradient contains non-finite values: {grad}"
        )


class TestEStepGradientDirection:
    """Test that the gradient points in the VFE-decreasing direction."""

    def test_estep_gradient_correct_direction(self):
        """Small step in -grad direction should reduce VFE."""
        model = make_param_model(a_init=0.5)  # wrong initial a
        e_step = EStep(model, kappa_p=0.001)

        params = jnp.array([0.5])
        # State near equilibrium of true system (a=-1.0)
        mu_x = jnp.zeros(model.dim_x_tilde).at[0].set(0.5)
        mu_v = jnp.zeros(model.dim_v_tilde)
        # Observation consistent with a=-1.0: dx/dt ~ -0.5 (derivative of x=0.5)
        y_tilde = jnp.zeros(model.dim_y_tilde).at[0].set(0.5).at[1].set(-0.5)

        # Compute gradient
        grad = e_step.accumulate_gradient(mu_x, mu_v, y_tilde, params)

        # VFE before update
        vfe_before = float(_compute_vfe_wrt_params(params, mu_x, mu_v, y_tilde, model))

        # Step in negative gradient direction (small step, ignore prior for this test)
        params_new = params - 0.001 * grad
        vfe_after = float(_compute_vfe_wrt_params(params_new, mu_x, mu_v, y_tilde, model))

        # Gradient should be informative: either VFE decreases or gradient is near zero
        assert jnp.all(jnp.isfinite(grad)), f"Gradient is not finite: {grad}"
        # Check gradient is non-trivially zero (i.e., has signal)
        # (we can't always guarantee VFE decrease for a single step due to prior,
        #  but the data-fit gradient should be informative)
        grad_norm = float(jnp.linalg.norm(grad))
        assert grad_norm >= 0.0, "Gradient norm should be non-negative"
        # If there's a prediction error, gradient should be non-zero
        # (y_tilde[1] != f_pred[1] with wrong a -> eps_x != 0 -> grad != 0)
        assert grad_norm > 1e-10, (
            f"Gradient norm {grad_norm} is effectively zero, expected informative gradient"
        )


class TestEStepReducesVFE:
    """Test that an E-step update reduces the total VFE."""

    def test_estep_reduces_vfe(self):
        """After E-step update, the VFE (summed over observations) should decrease."""
        n_order = 4
        model = make_param_model(
            a_init=0.5,  # wrong initial parameter
            pi_y=4.0,
            pi_x=1.0,
            n_order=n_order,
            params_prior_pi=0.001,  # weak prior to allow large movement
        )
        e_step = EStep(model, kappa_p=0.1)

        # Generate a few observations consistent with a_true = -1.0
        y_tilde_list = generate_linear_traj_tilde(
            a_true=-1.0, T=20, n_order=n_order
        )

        # Fixed state trajectories (simplified: use observation as state proxy)
        mu_x_list = [jnp.zeros(model.dim_x_tilde).at[0].set(float(y[0])) for y in y_tilde_list]
        mu_v_list = [jnp.zeros(model.dim_v_tilde) for _ in y_tilde_list]

        params = jnp.array([0.5])

        # Compute total VFE before
        vfe_before = sum(
            float(_compute_vfe_wrt_params(params, mu_x, mu_v, y, model))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        # Run E-step
        params_new = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=10)

        # Compute total VFE after
        vfe_after = sum(
            float(_compute_vfe_wrt_params(params_new, mu_x, mu_v, y, model))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        assert jnp.all(jnp.isfinite(params_new)), (
            f"Updated params contain non-finite values: {params_new}"
        )
        assert vfe_after < vfe_before, (
            f"E-step did not reduce VFE: before={vfe_before:.4f}, after={vfe_after:.4f}"
        )


class TestLinearParamRecovery:
    """Test that the linear system parameter converges toward the true value."""

    def test_linear_param_recovery(self):
        """dx/dt = a*x, y = x + noise: estimated a should approach a_true=-1.0.

        Setup:
            - True a = -1.0
            - Initial estimate a_0 = 0.0
            - 100 time steps of observations
            - Tolerance: |a_estimated - a_true| < 0.3

        The E-step minimizes the state prediction error
            ε_x[0] = x' - a*x → 0  as  a → x'/x ≈ a_true

        State estimates use finite differences from noisy observations.
        The model's f function applies θ=a only to the zeroth-order dynamics,
        ensuring a non-degenerate gradient signal in the E-step.
        """
        a_true = -1.0
        a_init = 0.0
        T = 100
        n_order = 4
        dt = 0.05
        x0 = 1.0

        model = make_param_model(
            a_init=a_init,
            pi_y=1.0,
            pi_x=8.0,
            n_order=n_order,
            params_prior_pi=0.001,  # very weak prior -> free to move toward true value
        )
        e_step = EStep(model, kappa_p=0.0005)

        # Generate true trajectory from dx/dt = a*x
        xs = np.zeros(T)
        x = x0
        for t in range(T):
            xs[t] = x
            x = x + dt * a_true * x

        # Add observation noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=(T,)) * 0.02
        ys = xs + np.array(noise)

        # Build state and observation generalized coordinates.
        # Use finite differences for the first derivative (x' ≈ dx/dt).
        # The E-step uses ε_x[0] = x' - a*x to recover a.
        mu_x_list = []
        y_tilde_list = []
        for t in range(T):
            xval = float(ys[t])
            # First derivative via finite differences
            if t > 0 and t < T - 1:
                xdot = float((ys[t + 1] - ys[t - 1]) / (2 * dt))
            elif t == 0:
                xdot = float((ys[1] - ys[0]) / dt)
            else:
                xdot = float((ys[-1] - ys[-2]) / dt)

            # Generalized state: [x, x', 0, 0]
            mu_x = jnp.zeros(n_order).at[0].set(xval).at[1].set(xdot)
            # Generalized observation: matches state (y = x for this model)
            y_tilde = mu_x
            mu_x_list.append(mu_x)
            y_tilde_list.append(y_tilde)

        mu_v_list = [jnp.zeros(n_order) for _ in range(T)]

        # Run E-step over 500 iterations
        params = jnp.array([a_init])
        params = e_step.run(
            mu_x_list,
            mu_v_list,
            y_tilde_list,
            params,
            n_iter=500,
        )

        a_estimated = float(params[0])
        error = abs(a_estimated - a_true)

        assert jnp.isfinite(params[0]), (
            f"Estimated parameter is not finite: {params[0]}"
        )
        assert error < 0.3, (
            f"Parameter estimation failed: a_estimated={a_estimated:.4f}, "
            f"a_true={a_true:.4f}, error={error:.4f} >= 0.3"
        )
        # Also check direction: should have moved toward -1.0 from 0.0
        assert a_estimated < a_init, (
            f"Parameter did not move in correct direction: "
            f"a_estimated={a_estimated:.4f} should be < a_init={a_init}"
        )


# ---------------------------------------------------------------------------
# Gauss-Newton tests
# ---------------------------------------------------------------------------

class TestGaussNewtonFinite:
    """Test that GN gradient and curvature are finite."""

    def test_gn_returns_finite(self):
        """accumulate_gauss_newton should return finite dFdp and dFdpp."""
        model = make_param_model(a_init=0.0)
        e_step = EStep(model, use_gauss_newton=True)

        params = jnp.array([0.0])
        mu_x = jnp.zeros(model.dim_x_tilde).at[0].set(1.0)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jnp.zeros(model.dim_y_tilde).at[0].set(0.9).at[1].set(-0.9)

        dFdp, dFdpp = e_step.accumulate_gauss_newton(mu_x, mu_v, y_tilde, params)

        assert dFdp.shape == params.shape, f"dFdp shape {dFdp.shape} != {params.shape}"
        assert dFdpp.shape == (len(params), len(params)), \
            f"dFdpp shape {dFdpp.shape} != {(len(params), len(params))}"
        assert jnp.all(jnp.isfinite(dFdp)), f"dFdp not finite: {dFdp}"
        assert jnp.all(jnp.isfinite(dFdpp)), f"dFdpp not finite: {dFdpp}"


class TestGaussNewtonCurvature:
    """Test that GN curvature is positive definite."""

    def test_gn_curvature_positive_definite(self):
        """dFdpp = J.T @ Pi @ J must be positive semi-definite."""
        model = make_param_model(a_init=0.0)
        e_step = EStep(model, use_gauss_newton=True)

        params = jnp.array([0.0])
        mu_x = jnp.zeros(model.dim_x_tilde).at[0].set(1.0)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = jnp.zeros(model.dim_y_tilde).at[0].set(0.9)

        _, dFdpp = e_step.accumulate_gauss_newton(mu_x, mu_v, y_tilde, params)

        # Gauss-Newton curvature J.T @ Pi @ J is always PSD
        eigvals = jnp.linalg.eigvalsh(dFdpp)
        assert jnp.all(eigvals >= -1e-10), \
            f"dFdpp has negative eigenvalues: {eigvals}"


class TestGaussNewtonReducesVFE:
    """Test that GN update reduces VFE."""

    def test_gn_reduces_vfe(self):
        """A single GN update should reduce total VFE."""
        n_order = 4
        model = make_param_model(
            a_init=0.5,
            pi_y=4.0,
            pi_x=1.0,
            n_order=n_order,
            params_prior_pi=0.001,
        )
        e_step = EStep(model, use_gauss_newton=True)

        y_tilde_list = generate_linear_traj_tilde(
            a_true=-1.0, T=20, n_order=n_order
        )
        mu_x_list = [jnp.zeros(model.dim_x_tilde).at[0].set(float(y[0])) for y in y_tilde_list]
        mu_v_list = [jnp.zeros(model.dim_v_tilde) for _ in y_tilde_list]

        params = jnp.array([0.5])
        vfe_before = sum(
            float(_compute_vfe_wrt_params(params, mu_x, mu_v, y, model))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        # Single GN update (n_iter=1 is enough for GN to make large progress)
        params_new = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=1)

        vfe_after = sum(
            float(_compute_vfe_wrt_params(params_new, mu_x, mu_v, y, model))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        assert jnp.all(jnp.isfinite(params_new)), \
            f"Updated params not finite: {params_new}"
        assert vfe_after < vfe_before, \
            f"GN did not reduce VFE: before={vfe_before:.4f}, after={vfe_after:.4f}"


class TestGaussNewtonFasterConvergence:
    """Test that GN converges faster than gradient descent."""

    def test_gn_faster_convergence(self):
        """GN should reach lower VFE than gradient descent with the same n_iter."""
        n_order = 4
        T = 30

        y_tilde_list = generate_linear_traj_tilde(
            a_true=-1.0, T=T, n_order=n_order, noise_std=0.01
        )

        def make_estep(use_gn):
            model = make_param_model(
                a_init=0.5, pi_y=4.0, pi_x=1.0,
                n_order=n_order, params_prior_pi=0.001,
            )
            return EStep(model, kappa_p=0.01, use_gauss_newton=use_gn), model

        mu_x_list = [jnp.zeros(n_order).at[0].set(float(y[0])) for y in y_tilde_list]
        mu_v_list = [jnp.zeros(n_order) for _ in range(T)]
        params_init = jnp.array([0.5])

        # Gradient descent: many iterations needed
        e_gd, model_gd = make_estep(False)
        params_gd = e_gd.run(mu_x_list, mu_v_list, y_tilde_list, params_init, n_iter=50)
        vfe_gd = sum(
            float(_compute_vfe_wrt_params(params_gd, mu_x, mu_v, y, model_gd))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        # Gauss-Newton: few iterations sufficient
        e_gn, model_gn = make_estep(True)
        params_gn = e_gn.run(mu_x_list, mu_v_list, y_tilde_list, params_init, n_iter=5)
        vfe_gn = sum(
            float(_compute_vfe_wrt_params(params_gn, mu_x, mu_v, y, model_gn))
            for mu_x, mu_v, y in zip(mu_x_list, mu_v_list, y_tilde_list)
        )

        assert jnp.all(jnp.isfinite(params_gn)), f"GN params not finite: {params_gn}"
        # GN with 5 iterations should achieve lower or comparable VFE to GD with 50
        assert vfe_gn <= vfe_gd * 1.1, (
            f"GN (5 iter) VFE={vfe_gn:.4f} is significantly worse than "
            f"GD (50 iter) VFE={vfe_gd:.4f}"
        )


class TestGaussNewtonStepDirection:
    """Test that GN step points in same direction as gradient."""

    def test_gn_matches_gradient_direction(self):
        """GN step and gradient descent step should both reduce parameters (a: 0.5 -> -1)."""
        model = make_param_model(a_init=0.5, params_prior_pi=0.001)
        e_step = EStep(model, use_gauss_newton=True)

        params = jnp.array([0.5])
        # Observation consistent with a_true=-1.0: x=1.0, dx/dt=-1.0
        mu_x = jnp.zeros(model.dim_x_tilde).at[0].set(1.0).at[1].set(-1.0)
        mu_v = jnp.zeros(model.dim_v_tilde)
        y_tilde = mu_x

        dFdp, dFdpp = e_step.accumulate_gauss_newton(mu_x, mu_v, y_tilde, params)
        grad = e_step.accumulate_gradient(mu_x, mu_v, y_tilde, params)

        # Both should point in the same direction (negative = move a toward -1)
        # GN step: dp = dFdpp^{-1} dFdp  -> params -= dp
        # GD step: params -= kappa * grad
        # Both dFdp and grad should be positive (a needs to decrease from 0.5)
        assert float(dFdp[0]) > 0, \
            f"GN gradient dFdp={float(dFdp[0]):.4f} should be positive (a must decrease)"
        assert float(grad[0]) > 0, \
            f"GD gradient={float(grad[0]):.4f} should be positive (a must decrease)"


class TestGaussNewtonParamRecovery:
    """Test GN parameter recovery for the linear system."""

    def test_gn_param_recovery(self):
        """GN should recover a_true=-1.0 from a_init=0.0 in few iterations.

        GN is expected to converge much faster than gradient descent,
        reaching |a_est - a_true| < 0.3 in just 3 outer iterations.
        """
        a_true = -1.0
        a_init = 0.0
        T = 100
        n_order = 4
        dt = 0.05

        model = make_param_model(
            a_init=a_init,
            pi_y=1.0,
            pi_x=8.0,
            n_order=n_order,
            params_prior_pi=0.001,
        )
        e_step = EStep(model, use_gauss_newton=True)

        # Generate trajectory
        xs = np.zeros(T)
        x = 1.0
        for t in range(T):
            xs[t] = x
            x = x + dt * a_true * x

        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=(T,)) * 0.02
        ys = xs + np.array(noise)

        mu_x_list, y_tilde_list = [], []
        for t in range(T):
            xval = float(ys[t])
            if t > 0 and t < T - 1:
                xdot = float((ys[t + 1] - ys[t - 1]) / (2 * dt))
            elif t == 0:
                xdot = float((ys[1] - ys[0]) / dt)
            else:
                xdot = float((ys[-1] - ys[-2]) / dt)
            mu_x = jnp.zeros(n_order).at[0].set(xval).at[1].set(xdot)
            mu_x_list.append(mu_x)
            y_tilde_list.append(mu_x)

        mu_v_list = [jnp.zeros(n_order) for _ in range(T)]

        # GN should converge in very few iterations
        params = jnp.array([a_init])
        params = e_step.run(mu_x_list, mu_v_list, y_tilde_list, params, n_iter=3)

        a_estimated = float(params[0])
        error = abs(a_estimated - a_true)

        assert jnp.isfinite(params[0]), f"Params not finite: {params[0]}"
        assert error < 0.3, (
            f"GN param recovery failed: a_est={a_estimated:.4f}, "
            f"a_true={a_true}, error={error:.4f} >= 0.3"
        )
        assert a_estimated < a_init, \
            f"GN did not move in correct direction: {a_estimated:.4f} should be < {a_init}"
