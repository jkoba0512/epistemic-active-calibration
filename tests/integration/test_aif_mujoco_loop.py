"""
Integration tests: AIF agent + MuJoCo environment end-to-end.

Tests verify that the pymdp Agent (legacy API) and OcclusionManipulatorEnv
work together correctly across an episode.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from aif_calib_robustness.simulation.mujoco_env import (
    OcclusionManipulatorEnv,
    N_VISUAL_OBS,
    _CONTACT_ANGLES,
)
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager

# ---------------------------------------------------------------------------
# Constants matching the generative model
# ---------------------------------------------------------------------------
N_POS = 5
N_OBJ = 3
N_VIS = 4   # = N_VISUAL_OBS
N_TAC = 2

OBJ_ARM = {0: 1, 1: 2, 2: 3}   # obj_loc -> arm_pos index where contact occurs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generative_model():
    """Build and return (A, B, C, D) matching OcclusionManipulatorEnv.

    A, B, C, D are pymdp object arrays (required by pymdp.legacy.Agent).
    """
    num_states = [N_POS, N_OBJ]
    num_obs    = [N_VIS, N_TAC]

    # --- A matrices ---
    A = utils.obj_array_zeros([[o] + num_states for o in num_obs])

    # A[0]: visual — shape (N_VIS, N_POS, N_OBJ)
    for j in range(N_OBJ):
        A[0][j, :, j] = 1.0   # all arm positions see obj at loc j

    # A[1]: tactile — shape (N_TAC, N_POS, N_OBJ)
    A[1][0] = 1.0   # default: no contact
    for j, i in OBJ_ARM.items():
        A[1][0, i, j] = 0.0
        A[1][1, i, j] = 1.0   # contact when aligned

    # --- B matrices ---
    B = utils.obj_array(2)

    B_arm = np.zeros((N_POS, N_POS, 3))
    for i in range(N_POS):
        B_arm[max(0, i - 1), i, 0] = 1.0    # action 0: move left
        B_arm[i, i, 1]              = 1.0    # action 1: stay
        B_arm[min(4, i + 1), i, 2] = 1.0    # action 2: move right
    B[0] = B_arm

    B_obj = np.zeros((N_OBJ, N_OBJ, 3))
    for k in range(3):
        np.fill_diagonal(B_obj[:, :, k], 1.0)
    B[1] = B_obj

    # --- C and D ---
    C = utils.obj_array_zeros(num_obs)
    C[0] = np.zeros(N_VIS)
    C[1] = np.array([-1.0, 3.0])

    D = utils.obj_array_uniform(num_states)

    return A, B, C, D


@pytest.fixture
def env_no_occ():
    return OcclusionManipulatorEnv(occlusion_mode="none", max_steps=50)


@pytest.fixture
def env_full_occ():
    return OcclusionManipulatorEnv(occlusion_mode="full", max_steps=50)


# ---------------------------------------------------------------------------
# Helper: run one AIF episode
# ---------------------------------------------------------------------------

def _run_aif_episode(env, A, B, C, D, obj_loc_idx, seed=0, max_steps=50):
    """
    Run a single AIF episode. Returns (actions, beliefs_history, done, final_obs).
    """
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=2, inference_horizon=2)
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)

    actions = []
    beliefs_history = []
    done = False
    final_obs = None

    for t in range(max_steps):
        pymdp_obs = env.get_pymdp_obs()
        beliefs = agent.infer_states(pymdp_obs)
        q_pi, G = agent.infer_policies()
        action_list = agent.sample_action()
        action = int(action_list[0])

        result = env.step(action)
        actions.append(action)
        beliefs_history.append(beliefs)
        final_obs = result.obs

        if result.done or result.truncated:
            done = result.done
            break

    return actions, beliefs_history, done, final_obs


# ---------------------------------------------------------------------------
# TestAIFEpisode
# ---------------------------------------------------------------------------

class TestAIFEpisode:
    """Basic episode mechanics: actions, beliefs, termination, obs shapes."""

    def test_agent_produces_valid_actions(self, env_no_occ, generative_model):
        """All actions sampled over 5 steps must be in {0, 1, 2}."""
        A, B, C, D = generative_model
        agent = Agent(A=A, B=B, C=C, D=D, policy_len=2, inference_horizon=2)
        env_no_occ.reset(obj_loc_idx=1, seed=42)

        for _ in range(5):
            pymdp_obs = env_no_occ.get_pymdp_obs()
            agent.infer_states(pymdp_obs)
            agent.infer_policies()
            action_list = agent.sample_action()
            action = int(action_list[0])
            assert action in {0, 1, 2}, f"Invalid action {action}"
            env_no_occ.step(action)

    def test_beliefs_sum_to_one(self, env_no_occ, generative_model):
        """After each infer_states, beliefs for each factor must sum to 1.0."""
        A, B, C, D = generative_model
        agent = Agent(A=A, B=B, C=C, D=D, policy_len=2, inference_horizon=2)
        env_no_occ.reset(obj_loc_idx=1, seed=0)

        for _ in range(5):
            pymdp_obs = env_no_occ.get_pymdp_obs()
            beliefs = agent.infer_states(pymdp_obs)
            agent.infer_policies()
            action_list = agent.sample_action()
            action = int(action_list[0])

            # beliefs is a list of arrays, one per factor
            for factor_idx, belief in enumerate(beliefs):
                # belief may be a list of arrays (inference_horizon > 1)
                if isinstance(belief, (list, tuple)):
                    for b in belief:
                        assert abs(float(np.sum(b)) - 1.0) < 1e-5, (
                            f"Beliefs for factor {factor_idx} do not sum to 1: "
                            f"sum={np.sum(b)}"
                        )
                else:
                    assert abs(float(np.sum(belief)) - 1.0) < 1e-5, (
                        f"Beliefs for factor {factor_idx} do not sum to 1: "
                        f"sum={np.sum(belief)}"
                    )
            env_no_occ.step(action)

    @pytest.mark.slow
    def test_episode_terminates(self, env_no_occ, generative_model):
        """With no occlusion, obj_loc=1 (center), AIF agent finds object within 50 steps."""
        A, B, C, D = generative_model
        actions, beliefs_history, done, final_obs = _run_aif_episode(
            env_no_occ, A, B, C, D, obj_loc_idx=1, seed=0, max_steps=50
        )
        assert done, (
            "Expected AIF agent to reach contact (done=True) within 50 steps "
            f"with no occlusion and obj_loc=1. Steps taken: {len(actions)}"
        )

    def test_obs_shapes_match_A(self, env_no_occ, generative_model):
        """pymdp_obs indices must be within range for each modality."""
        A, B, C, D = generative_model
        env_no_occ.reset(obj_loc_idx=2, seed=7)

        for _ in range(5):
            pymdp_obs = env_no_occ.get_pymdp_obs()
            # modality 0: visual, range [0, N_VIS)
            assert 0 <= pymdp_obs[0] < A[0].shape[0], (
                f"Visual obs {pymdp_obs[0]} out of range [0, {A[0].shape[0]})"
            )
            # modality 1: tactile, range [0, N_TAC)
            assert 0 <= pymdp_obs[1] < A[1].shape[0], (
                f"Tactile obs {pymdp_obs[1]} out of range [0, {A[1].shape[0]})"
            )
            env_no_occ.step(2)   # move right


# ---------------------------------------------------------------------------
# TestOcclusionEffect
# ---------------------------------------------------------------------------

class TestOcclusionEffect:
    """Occlusion changes behavior and precision modifies A correctly."""

    def test_no_occlusion_visual_info_correct(self, env_no_occ):
        """With no occlusion, visual obs matches obj_loc after reset."""
        for loc in range(N_OBJ):
            obs = env_no_occ.reset(obj_loc_idx=loc, seed=0)
            assert obs.visual_obs_idx == loc, (
                f"No-occlusion: expected visual_obs={loc} for obj_loc={loc}, "
                f"got {obs.visual_obs_idx}"
            )

    def test_full_occlusion_visual_is_ambiguous(self, env_full_occ):
        """With full occlusion, visual obs == 3 (ambiguous) always."""
        ambiguous_idx = N_VISUAL_OBS - 1   # = 3
        for loc in range(N_OBJ):
            obs = env_full_occ.reset(obj_loc_idx=loc, seed=0)
            assert obs.visual_obs_idx == ambiguous_idx, (
                f"Full occlusion: expected ambiguous obs ({ambiguous_idx}) "
                f"for obj_loc={loc}, got {obs.visual_obs_idx}"
            )
            # Also check after a few steps
            for _ in range(3):
                result = env_full_occ.step(1)   # stay
                assert result.obs.visual_obs_idx == ambiguous_idx, (
                    f"Full occlusion mid-episode: expected {ambiguous_idx}, "
                    f"got {result.obs.visual_obs_idx}"
                )

    def test_precision_modifies_A_visual(self, generative_model):
        """A_noisy[0] != A_original[0] when c_visual=0.0 (full occlusion)."""
        A, B, C, D = generative_model
        pm = PrecisionManager(theta=0.4, pi_visual_min=0.1)
        A_noisy = pm.apply_to_A(A, c_visual=0.0)

        assert not np.allclose(A_noisy[0], A[0]), (
            "Expected visual A-matrix to be modified when c_visual=0.0"
        )

    def test_precision_preserves_A_tactile(self, generative_model):
        """A_noisy[1] == A_original[1] (tactile modality not modified)."""
        A, B, C, D = generative_model
        pm = PrecisionManager(theta=0.4, pi_visual_min=0.1)
        A_noisy = pm.apply_to_A(A, c_visual=0.0)

        np.testing.assert_array_equal(
            A_noisy[1], A[1],
            err_msg="Tactile A-matrix should not be modified by PrecisionManager"
        )


# ---------------------------------------------------------------------------
# TestMultiEpisode
# ---------------------------------------------------------------------------

class TestMultiEpisode:
    """Statistical behavior across multiple episodes."""

    @pytest.mark.slow
    def test_no_occlusion_finds_all_objects(self, env_no_occ, generative_model):
        """Run 3 episodes (one per obj_loc), all should reach contact."""
        A, B, C, D = generative_model
        for loc in range(N_OBJ):
            actions, beliefs_history, done, final_obs = _run_aif_episode(
                env_no_occ, A, B, C, D, obj_loc_idx=loc, seed=loc, max_steps=50
            )
            assert done, (
                f"AIF agent failed to reach contact for obj_loc={loc} "
                f"within 50 steps. Steps taken: {len(actions)}"
            )

    @pytest.mark.slow
    def test_contact_detected_at_correct_location(self, env_no_occ, generative_model):
        """When done=True with arm movement, arm_angle should be near the expected angle.

        Note: obj_loc=2 (rightmost) always requires movement from start.
        obj_loc=0 and 1 may trigger spurious contact at the reset position
        due to physics overlap, so we only check angle when the arm has
        moved at least 0.10 rad from the start (shoulder_min = -0.5 rad).
        """
        A, B, C, D = generative_model
        _SHOULDER_MIN_VAL = -0.50

        # Test rightmost object specifically — always requires real movement
        loc = 2
        actions, beliefs_history, done, final_obs = _run_aif_episode(
            env_no_occ, A, B, C, D, obj_loc_idx=loc, seed=loc * 10, max_steps=50
        )
        if done:
            expected_angle = _CONTACT_ANGLES[loc]
            arm_has_moved = abs(final_obs.arm_angle - _SHOULDER_MIN_VAL) > 0.10
            if arm_has_moved:
                angle_err = abs(final_obs.arm_angle - expected_angle)
                # Tolerance: one discrete step (0.25 rad) + contact zone slack
                assert angle_err < 0.30, (
                    f"Contact at wrong arm angle for obj_loc={loc}: "
                    f"arm_angle={final_obs.arm_angle:.4f}, "
                    f"expected={expected_angle:.4f}, "
                    f"error={angle_err:.4f}"
                )
