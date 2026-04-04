"""Generative model definitions for DEM/ADEM.

Provides:
- DEMModel: dataclass specifying the generative model
- LinearDEMModel: factory for linear state-space models
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import jax
import jax.numpy as jnp

from src.dem.core import make_D_matrix, make_tilde_precision


@dataclass
class DEMModel:
    """Generative model specification for DEM/ADEM.

    The model defines:
        D * tilde_x = tilde_f(tilde_x, tilde_v) + noise_x
        tilde_y = tilde_g(tilde_x, tilde_v) + noise_y

    where tilde denotes generalized coordinates.

    Attributes:
        f: State transition function f(x_tilde, v_tilde, params) -> dx_dt_tilde.
            In generalized coordinates, this maps (n_x*n_order,) x (n_v*n_order,)
            -> (n_x*n_order,).
        g: Observation function g(x_tilde, v_tilde, params) -> y_tilde.
            Maps (n_x*n_order,) x (n_v*n_order,) -> (n_y*n_order,).
        n_x: State dimension.
        n_v: Cause (input) dimension.
        n_y: Observation dimension.
        n_order: Embedding order (default 4).
        pi_y: Observation precision scalar.
        pi_x: State noise precision scalar.
        s_y: Smoothness parameter for observation noise.
        s_x: Smoothness parameter for state noise.
        params: Model parameters passed to f and g.
    """

    f: Callable
    g: Callable
    n_x: int
    n_v: int
    n_y: int
    n_order: int = 4
    pi_y: float = 1.0
    pi_x: float = 1.0
    s_y: float = 1.0
    s_x: float = 1.0
    params: Any = None

    def __post_init__(self) -> None:
        """Precompute derived matrices."""
        self._D = make_D_matrix(self.n_order, self.n_x)
        self._tilde_Pi_y = make_tilde_precision(
            self.pi_y, self.n_order, self.s_y, self.n_y
        )
        self._tilde_Pi_x = make_tilde_precision(
            self.pi_x, self.n_order, self.s_x, self.n_x
        )

    @property
    def D(self) -> jnp.ndarray:
        """D (shift) operator matrix, shape (n_x*n_order, n_x*n_order)."""
        return self._D

    @property
    def tilde_Pi_y(self) -> jnp.ndarray:
        """Generalized observation precision, shape (n_y*n_order, n_y*n_order)."""
        return self._tilde_Pi_y

    @property
    def tilde_Pi_x(self) -> jnp.ndarray:
        """Generalized state precision, shape (n_x*n_order, n_x*n_order)."""
        return self._tilde_Pi_x

    @property
    def dim_x_tilde(self) -> int:
        """Total dimension of generalized state."""
        return self.n_order * self.n_x

    @property
    def dim_v_tilde(self) -> int:
        """Total dimension of generalized causes."""
        return self.n_order * self.n_v

    @property
    def dim_y_tilde(self) -> int:
        """Total dimension of generalized observations."""
        return self.n_order * self.n_y


def _make_linear_f(A: jnp.ndarray, n_order: int) -> Callable:
    """Create generalized f for linear dynamics dx/dt = A x + v.

    In generalized coordinates, f applies A at each order independently.
    The generalized f maps:
        [x, x', x'', ...] -> [Ax, Ax', Ax'', ...]  (state part)
    with causes added at the zeroth order.

    Args:
        A: System matrix of shape (n_x, n_x).
        n_order: Embedding order.

    Returns:
        JAX-compatible function f(x_tilde, v_tilde, params).
    """
    n_x = A.shape[0]

    def f(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: Any) -> jnp.ndarray:
        """Generalized linear state transition.

        Args:
            x_tilde: Generalized state, shape (n_x * n_order,).
            v_tilde: Generalized causes, shape (n_v * n_order,).
            params: Unused for linear model.

        Returns:
            f(x_tilde, v_tilde), shape (n_x * n_order,).
        """
        result = []
        for i in range(n_order):
            x_i = x_tilde[i * n_x : (i + 1) * n_x]
            Ax_i = A @ x_i
            # Add cause only at zeroth order
            if i == 0:
                n_v = v_tilde.shape[0] // n_order
                v_0 = v_tilde[:n_v]
                result.append(Ax_i + v_0)
            else:
                # Higher orders: A applied to higher derivatives
                v_i = v_tilde[i * n_x : (i + 1) * n_x] if i < n_order else jnp.zeros(n_x)
                result.append(Ax_i + v_i)
        return jnp.concatenate(result)

    return f


def _make_linear_g(C: jnp.ndarray, n_order: int) -> Callable:
    """Create generalized g for linear observations y = C x.

    In generalized coordinates, g applies C at each order independently:
        [y, y', y'', ...] = [Cx, Cx', Cx'', ...]

    Args:
        C: Observation matrix of shape (n_y, n_x).
        n_order: Embedding order.

    Returns:
        JAX-compatible function g(x_tilde, v_tilde, params).
    """
    n_x = C.shape[1]
    n_y = C.shape[0]

    def g(x_tilde: jnp.ndarray, v_tilde: jnp.ndarray, params: Any) -> jnp.ndarray:
        """Generalized linear observation function.

        Args:
            x_tilde: Generalized state, shape (n_x * n_order,).
            v_tilde: Generalized causes, shape (n_v * n_order,) - unused.
            params: Unused for linear model.

        Returns:
            g(x_tilde, v_tilde), shape (n_y * n_order,).
        """
        result = []
        for i in range(n_order):
            x_i = x_tilde[i * n_x : (i + 1) * n_x]
            result.append(C @ x_i)
        return jnp.concatenate(result)

    return g


def LinearDEMModel(
    A: jnp.ndarray,
    C: jnp.ndarray,
    n_order: int = 4,
    pi_y: float = 1.0,
    pi_x: float = 1.0,
    s_y: float = 1.0,
    s_x: float = 1.0,
) -> DEMModel:
    """Factory function for a linear DEM generative model.

    Creates a model with dynamics:
        dx/dt = A x + v
        y = C x

    Args:
        A: System matrix, shape (n_x, n_x).
        C: Observation matrix, shape (n_y, n_x).
        n_order: Embedding order (default 4).
        pi_y: Observation precision.
        pi_x: State noise precision.
        s_y: Observation noise smoothness.
        s_x: State noise smoothness.

    Returns:
        DEMModel configured for the linear system.

    Example:
        >>> A = jnp.array([[-1.0]])
        >>> C = jnp.array([[1.0]])
        >>> model = LinearDEMModel(A, C)
    """
    n_x = A.shape[0]
    n_y = C.shape[0]
    n_v = n_x  # causes have same dimension as state

    f = _make_linear_f(A, n_order)
    g = _make_linear_g(C, n_order)

    return DEMModel(
        f=f,
        g=g,
        n_x=n_x,
        n_v=n_v,
        n_y=n_y,
        n_order=n_order,
        pi_y=pi_y,
        pi_x=pi_x,
        s_y=s_y,
        s_x=s_x,
        params=None,
    )
