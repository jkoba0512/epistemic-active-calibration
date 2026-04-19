"""E-step: Parameter inference for DEM (Dynamic Expectation Maximization).

Implements the E-step of DEM, which updates the model parameters θ by
gradient descent (or Gauss-Newton) on the Variational Free Energy (VFE)
accumulated over multiple time steps.

Gradient descent update rule (following Friston's spm_ADEM.m):

    Accumulated gradient (summed over all time steps t):
        dF/dθ = Σ_t [ (∂g̃/∂θ)ᵀ Π̃_y ε_y + (∂f̃/∂θ)ᵀ Π̃_x ε_x ]
               + Π_θ (θ - μ_θ)   ← prior contribution

    Parameter update:
        θ ← θ - κ_θ · dF/dθ

Gauss-Newton update rule (following Friston's spm_DEM.m E-step):

    Accumulated gradient and curvature over all time steps t:
        dF/dθ  = Σ_t [ Jᵀ_y Π̃_y ε_y + Jᵀ_x Π̃_x ε_x ]  + Π_θ (θ - μ_θ)
        d²F/dθ² = Σ_t [ Jᵀ_y Π̃_y J_y + Jᵀ_x Π̃_x J_x ] + Π_θ I

    where J_y = ∂ε_y/∂θ = -∂g̃/∂θ, J_x = ∂ε_x/∂θ = -∂f̃/∂θ

    Gauss-Newton step:
        dp = (d²F/dθ²)⁻¹ dF/dθ
        θ ← θ - dp

where:
    ε_y = ỹ - g̃(x̃, ṽ, θ)   (observation prediction error)
    ε_x = D x̃ - f̃(x̃, ṽ, θ) (state prediction error)
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp

from .model import DEMModel


def _compute_vfe_wrt_params(
    params: jnp.ndarray,
    mu_x_tilde: jnp.ndarray,
    mu_v_tilde: jnp.ndarray,
    y_tilde: jnp.ndarray,
    model: DEMModel,
) -> jnp.ndarray:
    """Compute VFE as a function of parameters θ (for E-step gradient).

    Computes the data-fit term of VFE treating θ as the variable:
        F(θ) = 0.5 ε_y^T Π̃_y ε_y + 0.5 ε_x^T Π̃_x ε_x

    Args:
        params: Model parameters θ, shape (n_params,).
        mu_x_tilde: Posterior mean over generalized states, shape (n_x*n_order,).
        mu_v_tilde: Posterior mean over generalized causes, shape (n_v*n_order,).
        y_tilde: Generalized observation vector, shape (n_y*n_order,).
        model: DEMModel specifying f, g, and precision matrices.

    Returns:
        Scalar VFE value.
    """
    g_pred = model.g(mu_x_tilde, mu_v_tilde, params)
    f_pred = model.f(mu_x_tilde, mu_v_tilde, params)

    eps_y = y_tilde - g_pred
    eps_x = model.D @ mu_x_tilde - f_pred

    vfe_y = 0.5 * eps_y @ model.tilde_Pi_y @ eps_y
    vfe_x = 0.5 * eps_x @ model.tilde_Pi_x @ eps_x

    return vfe_y + vfe_x


class EStep:
    """E-step: parameter inference via gradient descent or Gauss-Newton on VFE.

    Implements the E-step update for model parameter estimation. The gradient
    (and curvature for Gauss-Newton) of VFE with respect to parameters θ is
    accumulated over all time steps, then used to update θ.

    Gradient descent update:
        dF/dθ = Σ_t [ (∂g̃/∂θ)ᵀ Π̃_y ε_y + (∂f̃/∂θ)ᵀ Π̃_x ε_x ]
               + Π_θ (θ - μ_θ)
        θ ← θ - κ_θ · dF/dθ

    Gauss-Newton update (matching spm_DEM.m E-step):
        dF/dθ  = Σ_t [ Jᵀ_y Π̃_y ε_y + Jᵀ_x Π̃_x ε_x ]  + Π_θ (θ - μ_θ)
        d²F/dθ² = Σ_t [ Jᵀ_y Π̃_y J_y + Jᵀ_x Π̃_x J_x ] + Π_θ I
        θ ← θ - (d²F/dθ²)⁻¹ dF/dθ

    Args:
        model: DEMModel specifying the generative model. Must have params set.
        kappa_p: Learning rate for gradient-descent parameter update (default 0.01).
        use_gauss_newton: If True, use Gauss-Newton (default). If False, use
            gradient descent. Gauss-Newton matches spm_DEM.m and converges faster.
    """

    def __init__(
        self,
        model: DEMModel,
        kappa_p: float = 0.01,
        use_gauss_newton: bool = True,
    ) -> None:
        self.model = model
        self.kappa_p = kappa_p
        self.use_gauss_newton = use_gauss_newton

        # JIT-compile the gradient computation
        self._grad_fn = jax.jit(self._make_grad_fn())

        if use_gauss_newton:
            self._gn_fn = jax.jit(self._make_gn_fn())

    def _make_grad_fn(self) -> callable:
        """Create JIT-compilable gradient function for one time step."""
        model = self.model

        def grad_one_step(
            params: jnp.ndarray,
            mu_x_tilde: jnp.ndarray,
            mu_v_tilde: jnp.ndarray,
            y_tilde: jnp.ndarray,
        ) -> jnp.ndarray:
            return jax.grad(_compute_vfe_wrt_params, argnums=0)(
                params, mu_x_tilde, mu_v_tilde, y_tilde, model
            )

        return grad_one_step

    def _make_gn_fn(self) -> callable:
        """Create JIT-compilable Gauss-Newton gradient+curvature function for one step.

        Returns (dFdp_t, dFdpp_t) for a single time step, following spm_DEM.m:
            J_y = ∂ε_y/∂θ = -∂g̃/∂θ,  J_x = ∂ε_x/∂θ = -∂f̃/∂θ
            dFdp_t  = Jᵀ_y Π̃_y ε_y + Jᵀ_x Π̃_x ε_x    (gradient)
            dFdpp_t = Jᵀ_y Π̃_y J_y + Jᵀ_x Π̃_x J_x    (Gauss-Newton curvature)
        """
        model = self.model

        def gn_one_step(
            params: jnp.ndarray,
            mu_x_tilde: jnp.ndarray,
            mu_v_tilde: jnp.ndarray,
            y_tilde: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Prediction errors
            eps_y = y_tilde - model.g(mu_x_tilde, mu_v_tilde, params)
            eps_x = model.D @ mu_x_tilde - model.f(mu_x_tilde, mu_v_tilde, params)

            # Jacobians of prediction errors w.r.t. params
            # J[i, j] = d(eps[i]) / d(params[j])
            J_y = jax.jacobian(
                lambda p: y_tilde - model.g(mu_x_tilde, mu_v_tilde, p)
            )(params)
            J_x = jax.jacobian(
                lambda p: model.D @ mu_x_tilde - model.f(mu_x_tilde, mu_v_tilde, p)
            )(params)

            # Gradient: dFdp = J.T @ Pi @ eps
            dFdp = J_y.T @ model.tilde_Pi_y @ eps_y + J_x.T @ model.tilde_Pi_x @ eps_x

            # Gauss-Newton curvature: dFdpp = J.T @ Pi @ J
            dFdpp = J_y.T @ model.tilde_Pi_y @ J_y + J_x.T @ model.tilde_Pi_x @ J_x

            return dFdp, dFdpp

        return gn_one_step

    def accumulate_gradient(
        self,
        mu_x_tilde: jnp.ndarray,
        mu_v_tilde: jnp.ndarray,
        y_tilde: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute dF/dθ for a single time step (data-fit term only).

        Args:
            mu_x_tilde: Posterior state mean at this time step,
                shape (n_x*n_order,).
            mu_v_tilde: Posterior cause mean at this time step,
                shape (n_v*n_order,).
            y_tilde: Generalized observation vector, shape (n_y*n_order,).
            params: Current parameter vector θ, shape (n_params,).

        Returns:
            Gradient dF/dθ (data-fit part), shape (n_params,).
        """
        return self._grad_fn(params, mu_x_tilde, mu_v_tilde, y_tilde)

    def accumulate_gauss_newton(
        self,
        mu_x_tilde: jnp.ndarray,
        mu_v_tilde: jnp.ndarray,
        y_tilde: jnp.ndarray,
        params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Gauss-Newton gradient and curvature for one time step.

        Following spm_DEM.m E-step accumulation.

        Args:
            mu_x_tilde: Posterior state mean, shape (n_x*n_order,).
            mu_v_tilde: Posterior cause mean, shape (n_v*n_order,).
            y_tilde: Generalized observation, shape (n_y*n_order,).
            params: Current parameter vector θ, shape (n_params,).

        Returns:
            Tuple of (dFdp_t, dFdpp_t):
                dFdp_t:  gradient contribution, shape (n_params,).
                dFdpp_t: curvature contribution, shape (n_params, n_params).
        """
        return self._gn_fn(params, mu_x_tilde, mu_v_tilde, y_tilde)

    def update(
        self,
        params: jnp.ndarray,
        grad: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update θ using the accumulated gradient (gradient descent).

        Applies:
            prior_term = Π_θ (θ - μ_θ)
            total_grad = grad + prior_term
            θ ← θ - κ_θ · total_grad

        Args:
            params: Current parameter vector θ, shape (n_params,).
            grad: Accumulated data-fit gradient Σ_t dF/dθ, shape (n_params,).

        Returns:
            Updated parameter vector θ, shape (n_params,).
        """
        model = self.model

        if model.params_prior_mean is not None:
            prior_mean = jnp.asarray(model.params_prior_mean)
        else:
            prior_mean = jnp.asarray(model.params)

        prior_grad = model.params_prior_pi * (params - prior_mean)
        total_grad = grad + prior_grad

        return params - self.kappa_p * total_grad

    def update_gauss_newton(
        self,
        params: jnp.ndarray,
        dFdp: jnp.ndarray,
        dFdpp: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update θ using Gauss-Newton step (matching spm_DEM.m E-step).

        Adds prior contribution and solves the linear system:
            total_dFdp  = dFdp  + Π_θ (θ - μ_θ)
            total_dFdpp = dFdpp + Π_θ I
            dp = total_dFdpp⁻¹ total_dFdp
            θ ← θ - dp

        Args:
            params: Current parameter vector θ, shape (n_params,).
            dFdp: Accumulated gradient Σ_t dFdp_t, shape (n_params,).
            dFdpp: Accumulated curvature Σ_t dFdpp_t, shape (n_params, n_params).

        Returns:
            Updated parameter vector θ, shape (n_params,).
        """
        model = self.model
        n_params = params.shape[0]

        if model.params_prior_mean is not None:
            prior_mean = jnp.asarray(model.params_prior_mean)
        else:
            prior_mean = jnp.asarray(model.params)

        # Add prior contributions (regularize toward prior)
        prior_grad = model.params_prior_pi * (params - prior_mean)
        prior_curv = model.params_prior_pi * jnp.eye(n_params)

        total_dFdp = dFdp + prior_grad
        total_dFdpp = dFdpp + prior_curv

        # Gauss-Newton step: dp = dFdpp^{-1} * dFdp
        dp = jnp.linalg.solve(total_dFdpp, total_dFdp)

        return params - dp

    def compute_precision(
        self,
        mu_x_sequence: List[jnp.ndarray],
        mu_v_sequence: List[jnp.ndarray],
        y_sequence: List[jnp.ndarray],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute posterior precision P_theta from accumulated Gauss-Newton curvature.

        P_theta = Σ_t [ Jᵀ_y Π̃_y J_y + Jᵀ_x Π̃_x J_x ] + Π_θ I

        This is the inverse posterior covariance of θ (Fisher information + prior).
        Use this to track parameter uncertainty for the epistemic A-step.

        Args:
            mu_x_sequence: List of T posterior state means, each shape (n_x*n_order,).
            mu_v_sequence: List of T posterior cause means, each shape (n_v*n_order,).
            y_sequence: List of T generalized observations, each shape (n_y*n_order,).
            params: Current parameter vector θ, shape (n_params,).

        Returns:
            P_theta: Posterior precision matrix, shape (n_params, n_params).
        """
        n_params = params.shape[0]
        total_dFdpp = jnp.zeros((n_params, n_params))
        for mu_x, mu_v, y in zip(mu_x_sequence, mu_v_sequence, y_sequence):
            _, dFdpp_t = self.accumulate_gauss_newton(mu_x, mu_v, y, params)
            total_dFdpp = total_dFdpp + dFdpp_t
        prior_curv = self.model.params_prior_pi * jnp.eye(n_params)
        return total_dFdpp + prior_curv

    def run(
        self,
        mu_x_sequence: List[jnp.ndarray],
        mu_v_sequence: List[jnp.ndarray],
        y_sequence: List[jnp.ndarray],
        params: jnp.ndarray,
        n_iter: int = 1,
    ) -> jnp.ndarray:
        """Run E-step over all time steps for n_iter iterations.

        If use_gauss_newton=True (default): each iteration accumulates
        gradient and curvature over all T steps, then applies one
        Gauss-Newton update (matching spm_DEM.m).

        If use_gauss_newton=False: each iteration accumulates gradient
        over all T steps, then applies gradient descent update.

        Args:
            mu_x_sequence: List of T posterior state means,
                each shape (n_x*n_order,).
            mu_v_sequence: List of T posterior cause means,
                each shape (n_v*n_order,).
            y_sequence: List of T generalized observations,
                each shape (n_y*n_order,).
            params: Initial parameter vector θ, shape (n_params,).
            n_iter: Number of E-step iterations (default 1).

        Returns:
            Updated parameter vector θ, shape (n_params,).
        """
        if self.use_gauss_newton:
            for _ in range(n_iter):
                n_params = params.shape[0]
                total_dFdp = jnp.zeros(n_params)
                total_dFdpp = jnp.zeros((n_params, n_params))
                for mu_x, mu_v, y in zip(mu_x_sequence, mu_v_sequence, y_sequence):
                    dFdp_t, dFdpp_t = self.accumulate_gauss_newton(
                        mu_x, mu_v, y, params
                    )
                    total_dFdp = total_dFdp + dFdp_t
                    total_dFdpp = total_dFdpp + dFdpp_t
                params = self.update_gauss_newton(params, total_dFdp, total_dFdpp)
        else:
            for _ in range(n_iter):
                total_grad = jnp.zeros_like(params)
                for mu_x, mu_v, y in zip(mu_x_sequence, mu_v_sequence, y_sequence):
                    total_grad = total_grad + self.accumulate_gradient(
                        mu_x, mu_v, y, params
                    )
                params = self.update(params, total_grad)

        return params
