"""Action update for ADEM (Active Dynamic Expectation Maximization).

Implements the action update equation:
    da/dt = -kappa_a * (dg/da)^T Pi_y eps_y

Actions affect observations through the generative process G,
allowing the agent to minimize prediction errors through action.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.inference import compute_vfe


class ActionUpdate:
    """Action update module for ADEM.

    Implements:
        da/dt = -kappa_a * dF/da

    where F is the VFE and a affects the observations via the generative process.

    The gradient is computed as:
        dF/da = (dg/da)^T * Pi_y * eps_y

    Args:
        model: DEMModel specifying the generative model.
        g_action: Function mapping (x_tilde, v_tilde, a, params) -> y_tilde,
                  the observation function that takes action into account.
                  If None, uses model.g (action-independent observations).
        kappa_a: Learning rate for action update (default 1.0).
        dt: Integration time step (default 0.01).
    """

    def __init__(
        self,
        model: DEMModel,
        g_action: Callable | None = None,
        kappa_a: float = 1.0,
        dt: float = 0.01,
    ) -> None:
        self.model = model
        self.g_action = g_action
        self.kappa_a = kappa_a
        self.dt = dt

        # JIT-compile the action gradient computation
        self._action_grad = jax.jit(self._make_action_grad())

    def _make_action_grad(self) -> Callable:
        """Create JIT-compilable action gradient function."""
        model = self.model
        g_action = self.g_action

        def vfe_with_action(
            a: jnp.ndarray,
            mu_x_tilde: jnp.ndarray,
            mu_v_tilde: jnp.ndarray,
            y_tilde: jnp.ndarray,
        ) -> jnp.ndarray:
            """VFE as a function of action a.

            Args:
                a: Action vector, shape (n_a,).
                mu_x_tilde: State mean, shape (n_x*n_order,).
                mu_v_tilde: Cause mean, shape (n_v*n_order,).
                y_tilde: Generalized observations, shape (n_y*n_order,).

            Returns:
                Scalar VFE value.
            """
            if g_action is not None:
                # Observation function that includes action
                g_pred = g_action(mu_x_tilde, mu_v_tilde, a, model.params)
            else:
                # Default: action does not affect g (for testing)
                g_pred = model.g(mu_x_tilde, mu_v_tilde, model.params)

            f_pred = model.f(mu_x_tilde, mu_v_tilde, model.params)

            eps_y = y_tilde - g_pred
            eps_x = model.D @ mu_x_tilde - f_pred

            vfe_y = 0.5 * eps_y @ model.tilde_Pi_y @ eps_y
            vfe_x = 0.5 * eps_x @ model.tilde_Pi_x @ eps_x

            return vfe_y + vfe_x

        def action_grad(
            a: jnp.ndarray,
            mu_x_tilde: jnp.ndarray,
            mu_v_tilde: jnp.ndarray,
            y_tilde: jnp.ndarray,
        ) -> jnp.ndarray:
            """Compute gradient of VFE w.r.t. action.

            Args:
                a: Current action, shape (n_a,).
                mu_x_tilde: State mean, shape (n_x*n_order,).
                mu_v_tilde: Cause mean, shape (n_v*n_order,).
                y_tilde: Generalized observations, shape (n_y*n_order,).

            Returns:
                Gradient dF/da, shape (n_a,).
            """
            return jax.grad(vfe_with_action, argnums=0)(
                a, mu_x_tilde, mu_v_tilde, y_tilde
            )

        return action_grad

    def step(
        self,
        a: jnp.ndarray,
        mu_x_tilde: jnp.ndarray,
        mu_v_tilde: jnp.ndarray,
        y_tilde: jnp.ndarray,
    ) -> jnp.ndarray:
        """Perform one Euler step of the action update.

        Args:
            a: Current action, shape (n_a,).
            mu_x_tilde: State mean, shape (n_x*n_order,).
            mu_v_tilde: Cause mean, shape (n_v*n_order,).
            y_tilde: Generalized observations, shape (n_y*n_order,).

        Returns:
            Updated action, shape (n_a,).
        """
        grad_a = self._action_grad(a, mu_x_tilde, mu_v_tilde, y_tilde)
        da = -self.kappa_a * grad_a
        return a + self.dt * da
