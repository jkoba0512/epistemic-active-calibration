"""
Unit tests for PrecisionManager.

Core claim being tested:
  c_visual < theta  =>  Pi_visual < 1.0 AND Pi_tactile > 1.0
  c_visual >= theta =>  Pi_visual = 1.0 AND Pi_tactile = 1.0
  Pi values must always be positive (negative precision is physically meaningless)
"""

import copy
import numpy as np
import pytest

from aif_occlusion.core.precision.precision_manager import PrecisionManager


@pytest.fixture
def pm():
    return PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)


class TestPrecisionWeights:

    def test_no_occlusion_returns_unity(self, pm):
        w = pm.compute_weights(c_visual=1.0)
        assert w.visual == pytest.approx(1.0)
        assert w.tactile == pytest.approx(1.0)

    def test_at_threshold_returns_unity(self, pm):
        w = pm.compute_weights(c_visual=0.4)
        assert w.visual == pytest.approx(1.0)
        assert w.tactile == pytest.approx(1.0)

    def test_occlusion_reduces_visual(self, pm):
        w = pm.compute_weights(c_visual=0.2)
        assert w.visual < 1.0

    def test_occlusion_increases_tactile(self, pm):
        w = pm.compute_weights(c_visual=0.2)
        assert w.tactile > 1.0

    def test_full_occlusion_hits_extremes(self, pm):
        w = pm.compute_weights(c_visual=0.0)
        assert w.visual == pytest.approx(pm.pi_visual_min)
        assert w.tactile == pytest.approx(pm.pi_tactile_max)

    def test_precision_always_positive(self, pm):
        for c in np.linspace(0, 1, 21):
            w = pm.compute_weights(c)
            assert w.visual > 0, f"Pi_visual <= 0 at c_visual={c}"
            assert w.tactile > 0, f"Pi_tactile <= 0 at c_visual={c}"

    def test_proprio_always_one(self, pm):
        for c in [0.0, 0.2, 0.5, 1.0]:
            assert pm.compute_weights(c).proprio == pytest.approx(1.0)

    def test_clamps_c_visual_above_one(self, pm):
        w = pm.compute_weights(c_visual=1.5)
        assert w.visual == pytest.approx(1.0)

    def test_clamps_c_visual_below_zero(self, pm):
        w = pm.compute_weights(c_visual=-0.5)
        assert w.visual == pytest.approx(pm.pi_visual_min)
        assert w.tactile == pytest.approx(pm.pi_tactile_max)

    def test_visual_monotone_decreasing(self, pm):
        c_vals = np.linspace(0, pm.theta, 10)
        pi_vals = [pm.compute_weights(c).visual for c in c_vals]
        assert all(a <= b for a, b in zip(pi_vals, pi_vals[1:]))

    def test_tactile_monotone_increasing(self, pm):
        c_vals = np.linspace(0, pm.theta, 10)
        pi_vals = [pm.compute_weights(c).tactile for c in c_vals]
        assert all(a >= b for a, b in zip(pi_vals, pi_vals[1:]))


class TestApplyToA:

    @pytest.fixture
    def simple_A(self):
        """3-modality A list: visual(4x3), tactile(2x3), other(3x3)."""
        A = [
            np.eye(3, 4).T,          # shape (4, 3) — identity-ish
            np.array([[1,1,0],[0,0,1]], dtype=float),  # shape (2, 3)
            np.eye(3),               # shape (3, 3)
        ]
        # Normalize columns
        for i in range(len(A)):
            A[i] = A[i] / A[i].sum(axis=0, keepdims=True)
        return A

    def test_no_occlusion_leaves_A_unchanged(self, pm, simple_A):
        A_out = pm.apply_to_A(simple_A, c_visual=1.0)
        np.testing.assert_allclose(A_out[0], simple_A[0])

    def test_full_occlusion_applies_correct_noise(self, pm, simple_A):
        # At c_visual=0.0: noise = 1 - pi_visual_min  (not necessarily 1.0)
        A_out = pm.apply_to_A(simple_A, c_visual=0.0)
        noise = pm.noise_level(0.0)
        n_obs = simple_A[0].shape[0]
        uniform = np.ones_like(simple_A[0]) / n_obs
        expected = (1 - noise) * simple_A[0] + noise * uniform
        np.testing.assert_allclose(A_out[0], expected)

    def test_pm_with_zero_min_makes_fully_uniform(self, simple_A):
        pm_zero = PrecisionManager(theta=0.4, pi_visual_min=0.0)
        A_out = pm_zero.apply_to_A(simple_A, c_visual=0.0)
        n_obs = simple_A[0].shape[0]
        expected = np.ones_like(simple_A[0]) / n_obs
        np.testing.assert_allclose(A_out[0], expected, atol=1e-10)

    def test_tactile_not_modified(self, pm, simple_A):
        A_out = pm.apply_to_A(simple_A, c_visual=0.0)
        np.testing.assert_array_equal(A_out[1], simple_A[1])

    def test_output_columns_sum_to_one(self, pm, simple_A):
        for c in [0.0, 0.2, 0.5, 1.0]:
            A_out = pm.apply_to_A(simple_A, c)
            col_sums = A_out[0].sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_does_not_modify_original(self, pm, simple_A):
        A_original_visual = simple_A[0].copy()
        pm.apply_to_A(simple_A, c_visual=0.0)
        np.testing.assert_array_equal(simple_A[0], A_original_visual)
