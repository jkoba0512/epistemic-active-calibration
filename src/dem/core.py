"""Mathematical foundations for DEM/ADEM.

Implements:
- D operator (shift operator) for generalized coordinates
- R matrix (precision matrix for smooth random fluctuations)
- Utility functions for generalized coordinates
"""

from typing import Union
import jax
import jax.numpy as jnp


def make_D_matrix(n_order: int, n_dim: int) -> jnp.ndarray:
    """Construct the D (shift/derivative) operator matrix.

    The D operator acts on generalized coordinates tilde_x = [x, x', x'', ...]
    and performs a one-step upward shift:
        D * tilde_x = [x', x'', ..., 0]

    Args:
        n_order: Embedding order p (number of generalized coordinate levels).
        n_dim: Dimension d of the state at each order.

    Returns:
        D matrix of shape (n_order * n_dim, n_order * n_dim).

    Example:
        >>> D = make_D_matrix(3, 2)
        >>> D.shape
        (6, 6)
    """
    total_dim = n_order * n_dim
    D = jnp.zeros((total_dim, total_dim))

    # Place identity blocks on the super-diagonal
    for i in range(n_order - 1):
        row_start = i * n_dim
        col_start = (i + 1) * n_dim
        D = D.at[row_start : row_start + n_dim, col_start : col_start + n_dim].set(
            jnp.eye(n_dim)
        )

    return D


def _double_factorial(n: int) -> float:
    """Compute double factorial n!! = n * (n-2) * (n-4) * ... * 1 (or 2).

    By convention, (-1)!! = 1 and 0!! = 1.

    Args:
        n: Non-negative integer.

    Returns:
        Double factorial as a float.
    """
    if n <= 0:
        return 1.0
    result = 1.0
    while n > 0:
        result *= n
        n -= 2
    return result


def make_R_matrix(n_order: int, s: float = 1.0) -> jnp.ndarray:
    """Construct the R matrix (precision matrix for smooth random fluctuations).

    R is derived from the covariance of generalized Gaussian fluctuations.
    For a Gaussian correlation function exp(-tau^2 / (2*s^2)), the (i,j) entry
    of R is the (i+j)-th derivative of the correlation function at tau=0,
    multiplied by (-1)^i to account for the generalized inner product:

        R[i,j] = (-1)^i * (1/s)^(i+j) * (-1)^((i+j)/2) * (i+j-1)!!
                  if (i+j) is even, else 0

    where n!! denotes the double factorial and (-1)!! = 1 by convention.

    This yields (for n_order=4, s=1):
        R = [[1,  0, -1,  0],
             [0,  1,  0, -3],
             [-1, 0,  3,  0],
             [0, -3,  0, 15]]

    Args:
        n_order: Size of the matrix (n x n).
        s: Smoothness parameter (s > 0). Larger s -> smoother fluctuations.

    Returns:
        R matrix of shape (n_order, n_order).

    Example:
        >>> R = make_R_matrix(4, s=1.0)
        # R = [[1, 0, -1, 0], [0, 1, 0, -3], [-1, 0, 3, 0], [0, -3, 0, 15]]
    """
    R = jnp.zeros((n_order, n_order))

    for i in range(n_order):
        for j in range(n_order):
            k = i + j
            if k % 2 == 1:
                # Odd total order: covariance is zero
                pass
            else:
                # R[i,j] = (-1)^i * s^(-k) * (-1)^(k/2) * (k-1)!!
                sign = ((-1) ** i) * ((-1) ** (k // 2))
                val = sign * (1.0 / (s ** k)) * _double_factorial(k - 1)
                R = R.at[i, j].set(val)

    return R


def make_tilde_precision(
    pi: float, n_order: int, s: float, n_dim: int = 1
) -> jnp.ndarray:
    """Construct the generalized precision matrix via Kronecker product.

    tilde_Pi = Pi * R (kron product), where:
        Pi = pi * I_{n_dim} (base precision matrix)
        R = make_R_matrix(n_order, s)

    The full precision matrix has shape (n_order * n_dim, n_order * n_dim).

    Args:
        pi: Base precision scalar.
        n_order: Embedding order.
        s: Smoothness parameter for R matrix.
        n_dim: Dimension of state at each order.

    Returns:
        Generalized precision matrix of shape (n_order * n_dim, n_order * n_dim).
    """
    Pi_base = pi * jnp.eye(n_dim)
    R = make_R_matrix(n_order, s)
    # Kronecker product: tilde_Pi = R ⊗ Pi_base  (order-first layout)
    # D matrix and generalized functions use order-first layout:
    #   x_tilde = [x(0), x(1), ..., x(p-1)]  where x(k) ∈ R^n_dim
    # kron(R, Pi_base) gives block (i*n_dim : (i+1)*n_dim, j*n_dim : (j+1)*n_dim) = R[i,j] * Pi_base
    # which correctly couples orders i and j within each independent dimension.
    # For n_dim=1 this is identical to kron(Pi_base, R).
    tilde_Pi = jnp.kron(R, Pi_base)
    return tilde_Pi


def shift_operator(tilde_x: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    """Apply the D shift operator to generalized coordinates.

    Args:
        tilde_x: Generalized state vector of shape (n_order * n_dim,).
        D: D matrix of shape (n_order * n_dim, n_order * n_dim).

    Returns:
        D @ tilde_x of shape (n_order * n_dim,).
    """
    return D @ tilde_x


def generalized_coordinates(
    x_trajectory: jnp.ndarray, dt: float, n_order: int
) -> jnp.ndarray:
    """Approximate generalized coordinates from a trajectory using finite differences.

    Args:
        x_trajectory: State trajectory of shape (T, n_dim).
        dt: Time step.
        n_order: Number of generalized coordinate orders.

    Returns:
        Generalized coordinates of shape (n_order * n_dim,) at time t=0.
    """
    n_dim = x_trajectory.shape[1]
    result = []
    current = x_trajectory

    for _ in range(n_order):
        result.append(current[0])
        if len(result) < n_order:
            # Finite difference approximation of derivative
            diff = jnp.diff(current, axis=0) / dt
            current = diff

    return jnp.concatenate(result)
