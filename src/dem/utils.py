"""Numerical utilities for DEM/ADEM.

Implements:
- Ozaki (1992) local linearization step for stable ODE integration
"""

from typing import Optional
import jax
import jax.numpy as jnp
import jax.scipy.linalg


def local_linearization_step(
    f_val: jnp.ndarray,
    J: jnp.ndarray,
    dt: float,
    damping: float = 1e-6,
) -> jnp.ndarray:
    """Single integration step using Ozaki (1992) local linearization.

    Integrates dx/dt = f(x) analytically by linearizing around x₀:
        dx/dt ≈ J·(x - x₀) + f(x₀)

    Analytical solution:
        Δx = J⁻¹(e^(JΔt) - I) f(x₀)

    Implemented via the augmented matrix method (equivalent to SPM spm_dx.m):
        A = [[J, f(x₀)], [0, 0]] * dt
        Δx = expm(A)[:n, n]

    This provides exponentially stable integration even when the Jacobian J
    has large eigenvalues. Euler integration requires dt < 2 / max_eigenvalue,
    whereas this method is unconditionally stable.

    Args:
        f_val: Value of f(x₀), shape (n,). Dynamics at the current point.
        J: Jacobian matrix ∂f/∂x, shape (n, n).
        dt: Integration time step.
        damping: Regularization constant for the Jacobian (numerical stability).

    Returns:
        Δx: State update vector, shape (n,).

    References:
        Ozaki, T. (1992). A bridge between nonlinear time series models and
        nonlinear stochastic dynamical systems: A local linearization approach.
        Statistica Sinica, 2(1), 113-135.

        Friston, K.J. et al. spm_dx.m, SPM12.
        https://github.com/spm/spm/blob/main/spm_dx.m

    Example:
        >>> # Linear system: dx/dt = A*x, J = A
        >>> A = jnp.array([[-1.0]])
        >>> x0 = jnp.array([1.0])
        >>> f_val = A @ x0
        >>> delta_x = local_linearization_step(f_val, A, dt=0.1)
    """
    n = f_val.shape[0]

    # Regularize Jacobian for numerical stability
    J_reg = J - damping * jnp.eye(n)

    # Build augmented matrix A of shape (n+1, n+1):
    # A = [[J_reg * dt,  f_val * dt],
    #      [0,           0         ]]
    A_top = jnp.concatenate([J_reg * dt, (f_val * dt)[:, None]], axis=1)
    A_bot = jnp.zeros((1, n + 1))
    A_ext = jnp.concatenate([A_top, A_bot], axis=0)

    # Compute matrix exponential
    expm_A = jax.scipy.linalg.expm(A_ext)

    # Δx is the upper-right block (first n elements of the last column)
    delta_x = expm_A[:n, n]

    return delta_x


def local_linearization_step_combined(
    f_val: jnp.ndarray,
    J: jnp.ndarray,
    dt: float,
    damping: float = 1e-6,
) -> jnp.ndarray:
    """Alias for local_linearization_step (kept for backward compatibility).

    Args:
        f_val: Value of f(x₀), shape (n,).
        J: Jacobian matrix, shape (n, n).
        dt: Integration time step.
        damping: Regularization constant for the Jacobian.

    Returns:
        Δx: State update vector, shape (n,).
    """
    return local_linearization_step(f_val, J, dt, damping)
