"""
Unit tests for OcclusionManipulatorEnv.

Core claims:
  - reset() returns valid observation with correct obj_loc_idx
  - step() advances step_count and returns bounded observations
  - c_visual matches occlusion_mode
  - visual_obs_idx is in valid range
  - tactile_obs_idx is 0 or 1
  - arm_pos_idx is within [0, n_arm_positions)
  - contact detected when arm aligned with object
"""

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from aif_occlusion.simulation.mujoco_env import (
    OcclusionManipulatorEnv,
    N_VISUAL_OBS,
    _CONTACT_ANGLES,
    _SHOULDER_MIN,
    _SHOULDER_MAX,
)


@pytest.fixture
def env_no_occlusion():
    return OcclusionManipulatorEnv(occlusion_mode="none", max_steps=30)


@pytest.fixture
def env_full_occlusion():
    return OcclusionManipulatorEnv(occlusion_mode="full", max_steps=30)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------
class TestReset:

    def test_returns_env_obs(self, env_no_occlusion):
        obs = env_no_occlusion.reset(obj_loc_idx=1, seed=0)
        assert obs is not None

    def test_obj_loc_idx_respected(self, env_no_occlusion):
        for loc in range(3):
            obs = env_no_occlusion.reset(obj_loc_idx=loc)
            assert obs.obj_loc_idx == loc

    def test_random_obj_loc_within_range(self, env_no_occlusion):
        for _ in range(10):
            obs = env_no_occlusion.reset()
            assert 0 <= obs.obj_loc_idx < 3

    def test_step_count_zero_after_reset(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0)
        assert env_no_occlusion._step_count == 0

    def test_arm_starts_at_min_angle(self, env_no_occlusion):
        obs = env_no_occlusion.reset(obj_loc_idx=0, seed=42)
        assert abs(obs.arm_angle - _SHOULDER_MIN) < 0.05

    def test_seeded_reset_reproducible(self, env_no_occlusion):
        obs_a = env_no_occlusion.reset(seed=7)
        obs_b = env_no_occlusion.reset(seed=7)
        assert obs_a.obj_loc_idx == obs_b.obj_loc_idx


# ---------------------------------------------------------------------------
# c_visual / occlusion_mode
# ---------------------------------------------------------------------------
class TestOcclusionMode:

    def test_none_gives_c_visual_one(self, env_no_occlusion):
        assert env_no_occlusion.c_visual == pytest.approx(1.0)

    def test_full_gives_c_visual_zero(self, env_full_occlusion):
        assert env_full_occlusion.c_visual == pytest.approx(0.0)

    def test_partial_gives_c_visual_half(self):
        env = OcclusionManipulatorEnv(occlusion_mode="partial")
        assert env.c_visual == pytest.approx(0.5)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="occlusion_mode"):
            OcclusionManipulatorEnv(occlusion_mode="invisible")


# ---------------------------------------------------------------------------
# Visual observation
# ---------------------------------------------------------------------------
class TestVisualObs:

    def test_no_occlusion_visual_equals_obj_loc(self, env_no_occlusion):
        for loc in range(3):
            obs = env_no_occlusion.reset(obj_loc_idx=loc, seed=0)
            assert obs.visual_obs_idx == loc, (
                f"Expected visual_obs={loc} for obj_loc={loc}, "
                f"got {obs.visual_obs_idx}"
            )

    def test_full_occlusion_visual_is_ambiguous(self, env_full_occlusion):
        for loc in range(3):
            obs = env_full_occlusion.reset(obj_loc_idx=loc, seed=0)
            # Ambiguous index = N_VISUAL_OBS - 1 = 3
            assert obs.visual_obs_idx == N_VISUAL_OBS - 1

    def test_visual_obs_always_in_range(self, env_no_occlusion):
        for loc in range(3):
            obs = env_no_occlusion.reset(obj_loc_idx=loc)
            assert 0 <= obs.visual_obs_idx < N_VISUAL_OBS


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------
class TestStep:

    def test_step_increments_step_count(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=1, seed=0)
        result = env_no_occlusion.step(1)   # stay
        assert env_no_occlusion._step_count == 1

    def test_invalid_action_raises(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0)
        with pytest.raises(ValueError, match="action"):
            env_no_occlusion.step(3)

    def test_arm_moves_right_on_action_2(self, env_no_occlusion):
        obs0 = env_no_occlusion.reset(obj_loc_idx=0, seed=0)
        result = env_no_occlusion.step(2)   # move right
        assert result.obs.arm_angle > obs0.arm_angle - 1e-6

    def test_arm_angle_stays_bounded(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0, seed=0)
        for _ in range(15):
            result = env_no_occlusion.step(2)   # keep moving right
            assert _SHOULDER_MIN - 0.05 <= result.obs.arm_angle <= _SHOULDER_MAX + 0.05

    def test_tactile_obs_is_binary(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0, seed=0)
        for _ in range(5):
            result = env_no_occlusion.step(1)
            assert result.obs.tactile_obs_idx in (0, 1)

    def test_arm_pos_idx_in_range(self, env_no_occlusion):
        n = env_no_occlusion.n_arm_positions
        env_no_occlusion.reset(obj_loc_idx=1, seed=0)
        for _ in range(10):
            result = env_no_occlusion.step(2)
            assert 0 <= result.obs.arm_pos_idx < n

    def test_truncated_at_max_steps(self):
        env = OcclusionManipulatorEnv(occlusion_mode="none", max_steps=3)
        env.reset(obj_loc_idx=0, seed=0)
        for i in range(3):
            result = env.step(1)   # stay — no contact, no done
        assert result.truncated

    def test_done_when_contact(self, env_no_occlusion):
        """Move arm to object position and check for done signal."""
        # Object at center (loc=1), contact angle ≈ 0 rad
        env_no_occlusion.reset(obj_loc_idx=1, seed=0)
        # Arm starts at _SHOULDER_MIN ≈ -0.5, needs to move right to 0.0
        done = False
        for _ in range(30):
            result = env_no_occlusion.step(2)   # keep moving right
            if result.done:
                done = True
                break
        assert done, "Expected contact (done=True) when arm sweeps to center object"


# ---------------------------------------------------------------------------
# get_pymdp_obs()
# ---------------------------------------------------------------------------
class TestPymdpObs:

    def test_returns_two_element_list(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0)
        obs = env_no_occlusion.get_pymdp_obs()
        assert len(obs) == 2

    def test_visual_index_first(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=2, seed=0)
        obs = env_no_occlusion.get_pymdp_obs()
        assert 0 <= obs[0] < N_VISUAL_OBS

    def test_tactile_index_second(self, env_no_occlusion):
        env_no_occlusion.reset(obj_loc_idx=0, seed=0)
        obs = env_no_occlusion.get_pymdp_obs()
        assert obs[1] in (0, 1)
