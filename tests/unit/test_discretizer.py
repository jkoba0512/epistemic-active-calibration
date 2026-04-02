"""
Unit tests for JointDiscretizer and ContactDiscretizer.

Core claim: encode(decode(x)) ≈ x within quantization error.
"""

import numpy as np
import pytest

from aif_occlusion.utils.discretizer import JointDiscretizer, ContactDiscretizer


class TestJointDiscretizer:

    @pytest.fixture
    def jd(self):
        return JointDiscretizer(n_bins=20, angle_min=-np.pi / 2, angle_max=np.pi / 2)

    def test_encode_min_returns_zero(self, jd):
        assert jd.encode(-np.pi / 2) == 0

    def test_encode_max_returns_last_bin(self, jd):
        assert jd.encode(np.pi / 2) == jd.n_bins - 1

    def test_encode_clamps_below_min(self, jd):
        assert jd.encode(-999.0) == 0

    def test_encode_clamps_above_max(self, jd):
        assert jd.encode(999.0) == jd.n_bins - 1

    def test_decode_returns_centre_of_bin(self, jd):
        for idx in range(jd.n_bins):
            angle = jd.decode(idx)
            assert jd.angle_min <= angle <= jd.angle_max

    def test_roundtrip_within_quantization_error(self, jd):
        step = (jd.angle_max - jd.angle_min) / jd.n_bins
        for angle in np.linspace(jd.angle_min, jd.angle_max, 50):
            recovered = jd.decode(jd.encode(angle))
            assert abs(recovered - angle) <= step, (
                f"Roundtrip error {abs(recovered - angle):.4f} > step {step:.4f}"
            )

    def test_decode_delta_stay_is_zero(self, jd):
        assert jd.decode_delta(1, n_actions=3) == pytest.approx(0.0)

    def test_decode_delta_left_is_negative(self, jd):
        assert jd.decode_delta(0, n_actions=3) < 0

    def test_decode_delta_right_is_positive(self, jd):
        assert jd.decode_delta(2, n_actions=3) > 0

    def test_n_bins_respected(self, jd):
        assert jd.n_bins == 20


class TestContactDiscretizer:

    @pytest.fixture
    def cd(self):
        return ContactDiscretizer(thresholds=[0.3, 0.7])

    def test_below_first_threshold_is_level_zero(self, cd):
        assert cd.encode(0.0) == 0
        assert cd.encode(0.29) == 0

    def test_between_thresholds_is_level_one(self, cd):
        assert cd.encode(0.3) == 1
        assert cd.encode(0.5) == 1
        assert cd.encode(0.69) == 1

    def test_above_last_threshold_is_last_level(self, cd):
        assert cd.encode(0.7) == 2
        assert cd.encode(1.0) == 2
        assert cd.encode(999.0) == 2

    def test_n_levels_equals_thresholds_plus_one(self, cd):
        assert cd.n_levels == 3

    def test_label_returns_string(self, cd):
        assert isinstance(cd.label(0), str)
        assert isinstance(cd.label(2), str)
