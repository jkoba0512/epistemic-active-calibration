"""D-step: Variational Free Energy minimization for state inference.

Implements the D-step of DEM, which updates the posterior means over
generalized states by gradient descent on the Variational Free Energy (VFE).

Two integration modes are supported:

1. **Gradient descent** (``use_d_operator=False``): Standard gradient descent
   on VFE without the D (shift) operator term. Stable for arbitrary step sizes
   within the Lipschitz bound of the Hessian.

2. **Generalized motion** (``use_d_operator=True``): Full DEM D-step with the
   generalized motion term ``D * mu`` included. This matches Friston's original
   formulation but requires very small step sizes to remain stable, since the
   D operator couples higher-order derivatives. Uses exponential-Euler
   stabilization (Ozaki 1992 local linearization) when ``use_local_linearization``
   is True.

For practical use, gradient descent mode (default) is recommended.

References:
    Friston, K.J. et al. (2008). DEM: A variational treatment of dynamic
    systems. NeuroImage, 41(3), 849-885.

    Ozaki, T. (1992). A bridge between nonlinear time series models and
    nonlinear stochastic dynamical systems: A local linearization approach.
    Statistica Sinica, 2(1), 113-135.

    SPM spm_dx.m: https://github.com/spm/spm/blob/main/spm_dx.m
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from .model import DEMModel
from .utils import local_linearization_step


def compute_vfe(
    mu_x_tilde: jnp.ndarray,
    mu_v_tilde: jnp.ndarray,
    y_tilde: jnp.ndarray,
    model: DEMModel,
) -> jnp.ndarray:
    """Compute the Variational Free Energy (VFE).

    VFE = 0.5 * eps_y^T Pi_y eps_y + 0.5 * eps_x^T Pi_x eps_x

    where:
        eps_y = y_tilde - g(mu_x_tilde, mu_v_tilde)
        eps_x = D * mu_x_tilde - f(mu_x_tilde, mu_v_tilde)

    Args:
        mu_x_tilde: Posterior mean over generalized states, shape (n_x*n_order,).
        mu_v_tilde: Posterior mean over generalized causes, shape (n_v*n_order,).
        y_tilde: Generalized observation vector, shape (n_y*n_order,).
        model: DEMModel specifying f, g, and precision matrices.

    Returns:
        Scalar VFE value (non-negative when precision matrices are PD).
    """
    # Prediction errors
    g_pred = model.g(mu_x_tilde, mu_v_tilde, model.params)
    f_pred = model.f(mu_x_tilde, mu_v_tilde, model.params)

    eps_y = y_tilde - g_pred
    eps_x = model.D @ mu_x_tilde - f_pred

    # Quadratic forms with precision matrices
    vfe_y = 0.5 * eps_y @ model.tilde_Pi_y @ eps_y
    vfe_x = 0.5 * eps_x @ model.tilde_Pi_x @ eps_x

    return vfe_y + vfe_x


class DStep:
    """D-step: state inference via gradient descent on VFE.

    Implements the D-step update for posterior mean estimation.

    **Gradient descent mode** (``use_d_operator=False``, default):
        d(mu_x)/dt = -kappa_mu * dF/d(mu_x)
        d(mu_v)/dt = -kappa_mu * dF/d(mu_v)

    **Generalized motion mode** (``use_d_operator=True``):
        d(mu_x)/dt = D * mu_x - kappa_mu * dF/d(mu_x)
        d(mu_v)/dt = D * mu_v - kappa_mu * dF/d(mu_v)

    When ``use_d_operator=True`` and ``use_local_linearization=True``, the
    Ozaki (1992) local linearization is used instead of Euler integration.
    This is numerically stable even for large dt (e.g., dt=0.01) by computing
    the full Jacobian J = D - kappa * H(VFE) and integrating via matrix
    exponential.

    The generalized motion mode matches Friston's original DEM formulation
    (see spm_dx.m). The gradient descent mode is unconditionally stable for
    dt < 2 / L where L is the Lipschitz constant of the gradient.

    Args:
        model: DEMModel specifying the generative model.
        kappa_mu: Learning rate for state update (default 1.0).
        dt: Integration time step (default 0.001).
        n_iter: Number of integration steps per D-step call (default 32).
        use_d_operator: If True, include D*mu term (generalized motion).
                        If False, use plain gradient descent (default False).
        use_local_linearization: If True and use_d_operator=True, use Ozaki
                        local linearization for stable integration (default True).
        ll_damping: Regularization for Jacobian in local linearization (default 1e-6).
    """

    def __init__(
        self,
        model: DEMModel,
        kappa_mu: float = 1.0,
        dt: float = 0.001,
        n_iter: int = 32,
        use_d_operator: bool = False,
        use_local_linearization: bool = True,
        ll_damping: float = 1e-6,
    ) -> None:
        self.model = model
        self.kappa_mu = kappa_mu
        self.dt = dt
        self.n_iter = n_iter
        self.use_d_operator = use_d_operator
        self.use_local_linearization = use_local_linearization
        self.ll_damping = ll_damping

        # JIT-compile the single integration step
        self._euler_step = jax.jit(self._make_euler_step())

    def _make_euler_step(self) -> callable:
        """Create JIT-compilable single integration step.

        When use_d_operator=True and use_local_linearization=True, uses Ozaki
        local linearization (stable for large dt). Otherwise uses Euler method.
        """
        model = self.model
        kappa_mu = self.kappa_mu
        dt = self.dt
        use_d_operator = self.use_d_operator
        use_local_linearization = self.use_local_linearization
        ll_damping = self.ll_damping

        def euler_step(
            mu_x_tilde: jnp.ndarray,
            mu_v_tilde: jnp.ndarray,
            y_tilde: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single integration step for D-step update.

            Args:
                mu_x_tilde: Current state mean, shape (n_x*n_order,).
                mu_v_tilde: Current cause mean, shape (n_v*n_order,).
                y_tilde: Generalized observations, shape (n_y*n_order,).

            Returns:
                Tuple of updated (mu_x_tilde, mu_v_tilde).
            """
            # Compute gradients of VFE via automatic differentiation
            grad_x = jax.grad(compute_vfe, argnums=0)(
                mu_x_tilde, mu_v_tilde, y_tilde, model
            )
            grad_v = jax.grad(compute_vfe, argnums=1)(
                mu_x_tilde, mu_v_tilde, y_tilde, model
            )

            if use_d_operator and use_local_linearization:
                # Full DEM D-step with Ozaki (1992) local linearization.
                # This is numerically stable even for large dt.
                #
                # Dynamics: F(mu_x) = D*mu_x - kappa * grad_x
                # Jacobian: J_x = D - kappa * H_x(VFE)
                # Update:   Δmu_x = local_linearization_step(F, J_x, dt)

                # Hessian of VFE w.r.t. mu_x_tilde
                H_x = jax.hessian(compute_vfe, argnums=0)(
                    mu_x_tilde, mu_v_tilde, y_tilde, model
                )
                H_v = jax.hessian(compute_vfe, argnums=1)(
                    mu_x_tilde, mu_v_tilde, y_tilde, model
                )

                # Full Jacobian of the D-step dynamics
                J_x = model.D - kappa_mu * H_x
                J_v = -kappa_mu * H_v  # D matrix for causes is zero

                # Current dynamics value F(mu)
                F_x = model.D @ mu_x_tilde - kappa_mu * grad_x
                F_v = -kappa_mu * grad_v

                # Ozaki local linearization step
                delta_x = local_linearization_step(F_x, J_x, dt, ll_damping)
                delta_v = local_linearization_step(F_v, J_v, dt, ll_damping)

                new_mu_x = mu_x_tilde + delta_x
                new_mu_v = mu_v_tilde + delta_v

            elif use_d_operator:
                # Full DEM D-step with Euler integration (may be unstable for large dt)
                # d_mu/dt = D*mu - kappa * grad_F
                d_mu_x = model.D @ mu_x_tilde - kappa_mu * grad_x
                d_mu_v = model.D @ mu_v_tilde - kappa_mu * grad_v

                new_mu_x = mu_x_tilde + dt * d_mu_x
                new_mu_v = mu_v_tilde + dt * d_mu_v

            else:
                # Stable gradient descent: d_mu/dt = -kappa * grad_F
                d_mu_x = -kappa_mu * grad_x
                d_mu_v = -kappa_mu * grad_v

                new_mu_x = mu_x_tilde + dt * d_mu_x
                new_mu_v = mu_v_tilde + dt * d_mu_v

            return new_mu_x, new_mu_v

        return euler_step

    def run(
        self,
        mu_x_tilde: jnp.ndarray,
        mu_v_tilde: jnp.ndarray,
        y_tilde: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Run n_iter Euler steps of the D-step.

        Args:
            mu_x_tilde: Initial state mean, shape (n_x*n_order,).
            mu_v_tilde: Initial cause mean, shape (n_v*n_order,).
            y_tilde: Generalized observations, shape (n_y*n_order,).

        Returns:
            Tuple of (updated mu_x_tilde, updated mu_v_tilde, final VFE value).
        """
        for _ in range(self.n_iter):
            mu_x_tilde, mu_v_tilde = self._euler_step(mu_x_tilde, mu_v_tilde, y_tilde)

        vfe = compute_vfe(mu_x_tilde, mu_v_tilde, y_tilde, self.model)
        return mu_x_tilde, mu_v_tilde, float(vfe)

    def run_single_step(
        self,
        mu_x_tilde: jnp.ndarray,
        mu_v_tilde: jnp.ndarray,
        y_tilde: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run a single Euler step of the D-step.

        Args:
            mu_x_tilde: Current state mean, shape (n_x*n_order,).
            mu_v_tilde: Current cause mean, shape (n_v*n_order,).
            y_tilde: Generalized observations, shape (n_y*n_order,).

        Returns:
            Tuple of (updated mu_x_tilde, updated mu_v_tilde).
        """
        return self._euler_step(mu_x_tilde, mu_v_tilde, y_tilde)
