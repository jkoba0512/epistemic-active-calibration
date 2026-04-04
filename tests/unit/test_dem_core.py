"""Unit tests for src/dem/core.py.

Tests the mathematical foundations of DEM:
- D (shift) operator matrix
- R (precision) matrix
- Generalized precision (Kronecker product)
"""

import pytest
import jax.numpy as jnp
import numpy as np

from src.dem.core import make_D_matrix, make_R_matrix, make_tilde_precision, shift_operator


class TestDMatrix:
    """Tests for the D (shift) operator matrix."""

    def test_D_matrix_shape(self):
        """D matrix should have shape (n_order*n_dim, n_order*n_dim)."""
        for n_order in [2, 3, 4]:
            for n_dim in [1, 2, 3]:
                D = make_D_matrix(n_order, n_dim)
                expected_size = n_order * n_dim
                assert D.shape == (expected_size, expected_size), (
                    f"D.shape={D.shape} for n_order={n_order}, n_dim={n_dim}"
                )

    def test_D_matrix_shift(self):
        """D * mu_tilde should correctly shift generalized coordinates.

        For n_order=3, n_dim=1, tilde_x = [x, x', x'']:
        D * tilde_x = [x', x'', 0]
        """
        n_order = 3
        n_dim = 1
        D = make_D_matrix(n_order, n_dim)

        # tilde_x = [x0, x1, x2] representing [x, x', x'']
        tilde_x = jnp.array([1.0, 2.0, 3.0])
        shifted = D @ tilde_x

        expected = jnp.array([2.0, 3.0, 0.0])
        np.testing.assert_allclose(shifted, expected, atol=1e-6,
                                   err_msg="D matrix shift failed for n_order=3, n_dim=1")

    def test_D_matrix_shift_multidim(self):
        """D shift should work correctly for multi-dimensional states.

        For n_order=2, n_dim=2, tilde_x = [x0a, x0b, x1a, x1b]:
        D * tilde_x = [x1a, x1b, 0, 0]
        """
        n_order = 2
        n_dim = 2
        D = make_D_matrix(n_order, n_dim)

        tilde_x = jnp.array([1.0, 2.0, 3.0, 4.0])
        shifted = D @ tilde_x

        expected = jnp.array([3.0, 4.0, 0.0, 0.0])
        np.testing.assert_allclose(shifted, expected, atol=1e-6,
                                   err_msg="D matrix shift failed for n_order=2, n_dim=2")

    def test_D_matrix_last_block_zero(self):
        """The last block row of D should be all zeros."""
        n_order = 4
        n_dim = 2
        D = make_D_matrix(n_order, n_dim)
        total = n_order * n_dim
        last_block = D[(n_order - 1) * n_dim : total, :]
        np.testing.assert_allclose(last_block, 0.0, atol=1e-10,
                                   err_msg="Last block of D should be zero")

    def test_D_matrix_superdiagonal_identity(self):
        """D should have identity blocks on the super-diagonal."""
        n_order = 3
        n_dim = 2
        D = make_D_matrix(n_order, n_dim)

        # Check first super-diagonal identity block
        block_01 = D[:n_dim, n_dim : 2 * n_dim]
        np.testing.assert_allclose(block_01, jnp.eye(n_dim), atol=1e-10,
                                   err_msg="Super-diagonal block not identity")


class TestRMatrix:
    """Tests for the R (precision) matrix."""

    def test_R_matrix_n4_s1(self):
        """R with n=4, s=1 should match known analytical values.

        Expected:
        R = [[1,  0, -1,  0],
             [0,  1,  0, -3],
             [-1, 0,  3,  0],
             [0, -3,  0, 15]]
        """
        R = make_R_matrix(4, s=1.0)
        expected = jnp.array([
            [1.0,  0.0, -1.0,  0.0],
            [0.0,  1.0,  0.0, -3.0],
            [-1.0, 0.0,  3.0,  0.0],
            [0.0, -3.0,  0.0, 15.0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-6,
                                   err_msg="R(n=4, s=1) does not match expected values")

    def test_R_matrix_shape(self):
        """R matrix should be square with size (n_order, n_order)."""
        for n in [2, 3, 4, 5]:
            R = make_R_matrix(n, s=1.0)
            assert R.shape == (n, n), f"R.shape={R.shape} for n={n}"

    def test_R_matrix_symmetric(self):
        """R matrix should be symmetric."""
        for n in [2, 3, 4]:
            for s in [0.5, 1.0, 2.0]:
                R = make_R_matrix(n, s=s)
                np.testing.assert_allclose(R, R.T, atol=1e-10,
                                           err_msg=f"R not symmetric for n={n}, s={s}")

    def test_R_matrix_positive_definite(self):
        """R matrix should be positive definite (all eigenvalues > 0)."""
        for n in [2, 3, 4]:
            for s in [0.5, 1.0, 2.0]:
                R = make_R_matrix(n, s=s)
                eigenvalues = jnp.linalg.eigvalsh(R)
                assert jnp.all(eigenvalues > 0), (
                    f"R not positive definite for n={n}, s={s}. "
                    f"Eigenvalues: {eigenvalues}"
                )

    def test_R_matrix_diagonal_is_one_for_n1(self):
        """R(1,1) should be [[1]] for any s."""
        R = make_R_matrix(1, s=2.0)
        assert R.shape == (1, 1)
        np.testing.assert_allclose(R[0, 0], 1.0, atol=1e-10)

    def test_R_matrix_odd_entries_zero(self):
        """Off-diagonal entries with odd index difference should be zero."""
        R = make_R_matrix(4, s=1.0)
        for i in range(4):
            for j in range(4):
                if abs(i - j) % 2 == 1:
                    np.testing.assert_allclose(
                        R[i, j], 0.0, atol=1e-10,
                        err_msg=f"R[{i},{j}] should be 0 (odd difference)"
                    )

    def test_R_matrix_smoothness_scaling(self):
        """Larger s should produce smaller off-diagonal entries."""
        R1 = make_R_matrix(4, s=1.0)
        R2 = make_R_matrix(4, s=2.0)
        # The (0,2) entry: -1/s^2 -> should be smaller in magnitude for larger s
        assert abs(R2[0, 2]) < abs(R1[0, 2]), \
            "Larger s should reduce off-diagonal magnitude"


class TestTildePrecision:
    """Tests for the generalized precision matrix (Kronecker product)."""

    def test_tilde_precision_shape(self):
        """Generalized precision should have shape (n_order*n_dim, n_order*n_dim)."""
        for n_order in [2, 3, 4]:
            for n_dim in [1, 2]:
                Pi_tilde = make_tilde_precision(1.0, n_order, 1.0, n_dim)
                expected_size = n_order * n_dim
                assert Pi_tilde.shape == (expected_size, expected_size), (
                    f"Shape mismatch: {Pi_tilde.shape} vs {(expected_size, expected_size)}"
                )

    def test_tilde_precision_kronecker_structure(self):
        """Generalized precision should equal kron(pi*I_n_dim, R)."""
        pi = 2.0
        n_order = 3
        s = 1.0
        n_dim = 2

        Pi_tilde = make_tilde_precision(pi, n_order, s, n_dim)

        # Expected: kron(pi * I_n_dim, R)
        R = make_R_matrix(n_order, s)
        Pi_base = pi * jnp.eye(n_dim)
        expected = jnp.kron(Pi_base, R)

        np.testing.assert_allclose(Pi_tilde, expected, atol=1e-6,
                                   err_msg="Kronecker structure mismatch")

    def test_tilde_precision_scalar_case(self):
        """For n_dim=1, tilde_Pi should equal pi * R."""
        pi = 3.0
        n_order = 4
        s = 1.0

        Pi_tilde = make_tilde_precision(pi, n_order, s, n_dim=1)
        R = make_R_matrix(n_order, s)
        expected = pi * R

        np.testing.assert_allclose(Pi_tilde, expected, atol=1e-6,
                                   err_msg="Scalar case: tilde_Pi != pi*R")

    def test_tilde_precision_positive_definite(self):
        """Generalized precision matrix should be positive definite."""
        Pi_tilde = make_tilde_precision(1.0, 4, 1.0, n_dim=2)
        eigenvalues = jnp.linalg.eigvalsh(Pi_tilde)
        assert jnp.all(eigenvalues > 0), \
            f"Generalized precision not positive definite. Eigenvalues: {eigenvalues}"


class TestShiftOperator:
    """Tests for the shift_operator convenience function."""

    def test_shift_operator_equals_D_matmul(self):
        """shift_operator(x, D) should equal D @ x."""
        n_order = 3
        n_dim = 2
        D = make_D_matrix(n_order, n_dim)
        x = jnp.arange(n_order * n_dim, dtype=float)

        result = shift_operator(x, D)
        expected = D @ x

        np.testing.assert_allclose(result, expected, atol=1e-10)
