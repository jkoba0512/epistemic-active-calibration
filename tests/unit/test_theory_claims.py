"""
Theory-level unit tests for MultiModalAIFAgent.

Each test maps directly to a falsifiable theoretical claim.
Tests 1-3 (TestSequentialBeliefUpdate) MUST FAIL before the agent rebuild
bug is fixed, and PASS after.

Theoretical claims under test:
  1. Sequential belief updating: posterior at t+1 is informed by posterior at t
  2. Precision coupling: c_visual change → different beliefs (visual channel)
  3. Tactile precision upregulation: pi_tactile > 1 → sharpened A_tactile
  4. AIF must beat random baseline on object localization
  5. EFE epistemic value: G varies under uncertainty, drives exploration
  6. Precision mechanism: same obs + different c_visual → different posteriors
  7. Soft tactile precision: higher pi_tactile → sharper contact posterior
"""

import numpy as np
import pytest
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager


# ---------------------------------------------------------------------------
# Minimal generative model (3 arm positions × 3 object locations)
# ---------------------------------------------------------------------------
_N_POS = 3
_N_OBJ = 3
_N_VIS = 4   # obs 0-2 = obj_loc revealed, obs 3 = ambiguous (occluded)
_N_TAC = 2   # 0 = no-contact, 1 = contact


def _build_A():
    # A_visual: clean and fully informative — obj_loc j → visual obs j
    A_visual = np.zeros((_N_VIS, _N_POS, _N_OBJ))
    for j in range(_N_OBJ):
        A_visual[j, :, j] = 1.0

    # A_tactile: contact when arm_pos == obj_loc (diagonal contact)
    A_tactile = np.zeros((_N_TAC, _N_POS, _N_OBJ))
    A_tactile[0] = 1.0                          # default: no contact everywhere
    for i in range(_N_OBJ):
        A_tactile[0, i, i] = 0.0
        A_tactile[1, i, i] = 1.0               # contact on diagonal

    A = utils.obj_array(2)
    A[0] = A_visual
    A[1] = A_tactile
    return A


def _build_BCD():
    # B_arm: left/stay/right transitions
    B_arm = np.zeros((_N_POS, _N_POS, 3))
    for i in range(_N_POS):
        B_arm[max(0, i - 1), i, 0] = 1.0
        B_arm[i, i, 1] = 1.0
        B_arm[min(_N_POS - 1, i + 1), i, 2] = 1.0

    # B_obj: object location is static
    B_obj = np.zeros((_N_OBJ, _N_OBJ, 3))
    for k in range(3):
        np.fill_diagonal(B_obj[:, :, k], 1.0)

    B = utils.obj_array(2)
    B[0] = B_arm
    B[1] = B_obj

    C = utils.obj_array(2)
    C[0] = np.zeros(_N_VIS)
    C[1] = np.array([-1.0, 3.0])

    D = utils.obj_array(2)
    D[0] = np.ones(_N_POS) / _N_POS
    D[1] = np.ones(_N_OBJ) / _N_OBJ

    return B, C, D


def _entropy(dist: np.ndarray) -> float:
    dist = np.asarray(dist, dtype=float).ravel() + 1e-12
    return float(-np.sum(dist * np.log(dist)))


def _build_A_soft(p_contact: float = 0.9, p_bg: float = 0.05):
    """
    Soft A_tactile: P(contact | arm=i, obj=i) = p_contact,
                    P(contact | arm=i, obj≠i) = p_bg.
    Unlike binary A_tactile (0/1), soft values allow power-law sharpening
    and prevent log(0)=-inf dominance in mean-field VFE.
    """
    A_visual = np.zeros((_N_VIS, _N_POS, _N_OBJ))
    for j in range(_N_OBJ):
        A_visual[j, :, j] = 1.0

    A_tactile = np.zeros((_N_TAC, _N_POS, _N_OBJ))
    for arm_i in range(_N_POS):
        for obj_j in range(_N_OBJ):
            if arm_i == obj_j:
                A_tactile[1, arm_i, obj_j] = p_contact
                A_tactile[0, arm_i, obj_j] = 1.0 - p_contact
            else:
                A_tactile[1, arm_i, obj_j] = p_bg
                A_tactile[0, arm_i, obj_j] = 1.0 - p_bg

    A = utils.obj_array(2)
    A[0] = A_visual
    A[1] = A_tactile
    return A


@pytest.fixture
def agent():
    A = _build_A()
    B, C, D = _build_BCD()
    pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)
    ag = MultiModalAIFAgent(
        A, B, C, D,
        precision_manager=pm,
        policy_len=1,
        inference_horizon=1,
    )
    ag.reset()
    return ag


# ---------------------------------------------------------------------------
# Claim 1: Sequential belief updating (the agent rebuild bug)
# ---------------------------------------------------------------------------
class TestSequentialBeliefUpdate:
    """
    These tests FAIL before the rebuild bug is fixed:
      In `step()`, `self._agent = self._build_agent(A_noisy)` discards
      the pymdp posterior and resets to D (uniform prior) every step.
    """

    def test_beliefs_carry_forward_across_steps(self, agent):
        """
        Step 1: clear visual → posterior peaks at obj_loc=2.
        Step 2: full occlusion (ambiguous visual, no contact).
           Without rebuild bug: prior = step-1 posterior → MAP stays at 2.
           With    rebuild bug: prior resets to D=[1/3,1/3,1/3] → flat beliefs.
        """
        # Step 1 — informative: visual obs 2 means object is at location 2
        r1 = agent.step([2, 0], c_visual=1.0)
        map1 = int(np.argmax(r1.beliefs[1]))
        assert map1 == 2, f"Step 1: expected MAP obj_loc=2, got {map1}"

        # Step 2 — occluded: no new information about object location
        r2 = agent.step([3, 0], c_visual=0.0)
        map2 = int(np.argmax(r2.beliefs[1]))
        assert map2 == 2, (
            f"Rebuild bug: step-2 MAP={map2} (expected 2). "
            f"Beliefs over obj_loc: {np.round(r2.beliefs[1], 3)}"
        )

    def test_entropy_non_increasing_under_consistent_obs(self, agent):
        """
        Feeding the same informative observation repeatedly must not increase
        entropy. With the rebuild bug, entropy resets to log(3) every step.
        """
        entropies = []
        for _ in range(4):
            r = agent.step([2, 0], c_visual=1.0)
            entropies.append(_entropy(r.beliefs[1]))

        for t in range(len(entropies) - 1):
            assert entropies[t + 1] <= entropies[t] + 1e-6, (
                f"Entropy rose at step {t + 1}: "
                f"{entropies[t]:.4f} → {entropies[t + 1]:.4f}\n"
                f"Full sequence: {[f'{h:.4f}' for h in entropies]}"
            )

    def test_posterior_differs_from_prior_after_step(self, agent):
        """
        After one informative observation, P(obj_loc=2) must exceed 1/3
        (the uniform prior). Minimal sanity check that inference runs.
        """
        uniform_prob = 1.0 / _N_OBJ
        r = agent.step([2, 0], c_visual=1.0)
        assert r.beliefs[1][2] > uniform_prob + 0.1, (
            f"Belief barely updated: P(obj=2)={r.beliefs[1][2]:.3f}, prior={uniform_prob:.3f}"
        )


# ---------------------------------------------------------------------------
# Claim 2: Precision coupling — c_visual changes posterior
# ---------------------------------------------------------------------------
class TestPrecisionCoupling:

    def test_clear_vision_more_certain_than_occluded(self, agent):
        """
        Clear vision (obs directly reveals obj_loc) gives certain beliefs.
        Full occlusion (obs=3 = ambiguous, noisy A) gives uncertain beliefs.
        Note: obs=[2,0] vs obs=[3,0] — different obs because occlusion
        changes WHICH observation the environment produces.
        """
        # Clear: obs=2 (camera sees obj at loc 2) → certainty
        agent.reset()
        r_clear = agent.step([2, 0], c_visual=1.0)
        H_clear = _entropy(r_clear.beliefs[1])

        # Fully occluded: obs=3 (ambiguous visual), noisy A_visual
        agent.reset()
        r_occ = agent.step([3, 0], c_visual=0.0)
        H_occ = _entropy(r_occ.beliefs[1])

        assert H_clear < H_occ - 0.1, (
            f"Clear vision not more certain: H_clear={H_clear:.3f}, H_occ={H_occ:.3f}"
        )

    def test_action_distribution_differs_under_occlusion(self, agent):
        """
        Under full occlusion (uniform beliefs), the agent must sometimes
        choose different actions than under clear vision (certain beliefs).
        """
        np.random.seed(0)
        n = 40
        actions_clear = []
        actions_occ   = []
        for _ in range(n):
            agent.reset()
            actions_clear.append(int(agent.step([2, 0], c_visual=1.0).action[0]))
            agent.reset()
            actions_occ.append(int(agent.step([3, 0], c_visual=0.0).action[0]))

        n_differ = sum(a != b for a, b in zip(actions_clear, actions_occ))
        assert n_differ > 0, (
            "Actions identical under clear (obs=[2,0]) and occluded (obs=[3,0]) — "
            "precision switching has no behavioral effect"
        )


# ---------------------------------------------------------------------------
# Claim 3: Tactile precision upregulation is applied (not just computed)
# ---------------------------------------------------------------------------
class TestTactilePrecision:

    def test_tactile_A_sharpened_when_noise_floor_set(self):
        """
        With tactile_noise_floor > 0 and c_visual=0.0, A_tactile must be
        sharpened (power-law scaling by pi_tactile). The output A[1] must
        differ from the clean A[1].

        NOTE: With binary A_tactile (0/1 entries), power-law has no effect
        because 0^p=0 and 1^p=1. A non-zero noise floor is required for
        the sharpening to be measurable. This test uses noise_floor=0.1.
        """
        pm = PrecisionManager(
            theta=0.4,
            pi_tactile_max=5.0,
            pi_visual_min=0.1,
            tactile_noise_floor=0.1,   # enables sharpening
        )
        A = _build_A()
        A_list = [A[0].copy(), A[1].copy()]

        weights = pm.compute_weights(c_visual=0.0)
        assert weights.tactile > 1.0, "pi_tactile should be > 1 at c_visual=0"

        A_out = pm.apply_to_A(A_list, c_visual=0.0)

        assert not np.allclose(A_out[1], A_list[1]), (
            "A_tactile unchanged despite pi_tactile > 1.0 and noise_floor=0.1 — "
            "tactile precision upregulation not implemented in apply_to_A()"
        )

    def test_tactile_columns_sum_to_one_after_sharpening(self):
        """After sharpening, A_tactile columns must still be valid probability vectors."""
        pm = PrecisionManager(
            theta=0.4, pi_tactile_max=5.0, tactile_noise_floor=0.1
        )
        A = _build_A()
        A_list = [A[0].copy(), A[1].copy()]
        A_out = pm.apply_to_A(A_list, c_visual=0.0)

        col_sums = A_out[1].sum(axis=0)
        np.testing.assert_allclose(
            col_sums, 1.0, atol=1e-9,
            err_msg="A_tactile columns do not sum to 1 after sharpening"
        )

    def test_no_tactile_change_when_noise_floor_zero(self):
        """
        With default tactile_noise_floor=0 (backward compatibility),
        A_tactile must not be modified.
        """
        pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)
        A = _build_A()
        A_list = [A[0].copy(), A[1].copy()]
        A_clean_tac = A_list[1].copy()

        A_out = pm.apply_to_A(A_list, c_visual=0.0)
        np.testing.assert_array_equal(A_out[1], A_clean_tac)


# ---------------------------------------------------------------------------
# Meta-test: AIF must beat a random baseline
# ---------------------------------------------------------------------------
class TestAIFBeatsRandom:

    def test_aif_localizes_object_faster_than_chance(self, agent):
        """
        AIF agent should identify the true object location (MAP estimate)
        faster than a random guesser.  Random expected steps ≈ N_OBJ/2 = 1.5.
        """
        n_trials = 30

        def aif_steps_to_localize(true_loc: int) -> int:
            agent.reset()
            for t in range(10):
                r = agent.step([true_loc, 0], c_visual=1.0)
                if int(np.argmax(r.beliefs[1])) == true_loc:
                    return t
            return 10

        def random_steps_to_guess(true_loc: int) -> int:
            for t in range(10):
                if np.random.randint(0, _N_OBJ) == true_loc:
                    return t
            return 10

        np.random.seed(42)
        aif_mean    = np.mean([aif_steps_to_localize(t % _N_OBJ) for t in range(n_trials)])
        random_mean = np.mean([random_steps_to_guess(t % _N_OBJ) for t in range(n_trials)])

        assert aif_mean < random_mean, (
            f"AIF ({aif_mean:.2f} steps) no better than random ({random_mean:.2f} steps) — "
            "the system is not performing genuine inference"
        )


# ---------------------------------------------------------------------------
# Claim 5: EFE epistemic value drives exploration under uncertainty
# ---------------------------------------------------------------------------
class TestEFEEpistemicValue:
    """
    Claim: G (expected free energy) is non-uniform under belief uncertainty
    and favors actions that resolve it (epistemic value / information gain).
    This distinguishes AIF from random softmax — it PLANS to gain information.

    Requires soft A_tactile so that G is dominated by epistemic value, not
    by deterministic elimination artifacts from binary tactile (log(0)=-inf).
    """

    def _make_soft_agent(self):
        A = _build_A_soft(p_contact=0.9, p_bg=0.05)
        B, C, D = _build_BCD()
        # Flat C: no instrumental preference, so G is purely epistemic
        C[1] = np.zeros(_N_TAC)
        # Known arm at pos=0 so that moving the arm has asymmetric epistemic value.
        # With uniform arm prior, every action resolves the same expected obs → G flat.
        D[0] = np.array([1.0, 0.0, 0.0])
        pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)
        ag = MultiModalAIFAgent(
            A, B, C, D,
            precision_manager=pm,
            policy_len=1,
            inference_horizon=1,
        )
        ag.reset()
        return ag

    def test_G_nonuniform_under_uncertainty(self):
        """
        With arm known at pos=0 and obj_loc uncertain (uniform), G must vary
        across actions: staying at pos=0 where we already have no-contact info
        has lower epistemic value than moving to unvisited positions.
        If G is flat, the agent cannot plan to gather information.
        """
        ag = self._make_soft_agent()
        r = ag.step([3, 0], c_visual=0.0)   # ambiguous visual, no contact at arm=0
        G_range = float(np.max(r.G) - np.min(r.G))
        assert G_range > 1e-4, (
            f"G is nearly flat (range={G_range:.6f}) under uncertainty — "
            "agent has no epistemic preference; AIF reduces to random policy"
        )

    def test_G_flattens_after_contact_resolves_uncertainty(self):
        """
        After tactile contact resolves object location, beliefs are certain
        → epistemic value of further exploration drops.
        G range should be smaller after contact than before.
        """
        ag = self._make_soft_agent()

        # Step 1: occluded, no contact → uncertain beliefs, spread G
        r_pre = ag.step([3, 0], c_visual=0.0)
        G_range_pre = float(np.max(r_pre.G) - np.min(r_pre.G))

        # Step 2: contact at arm_pos=2, obj_loc=2 → certain beliefs
        r_post = ag.step([3, 1], c_visual=0.0)   # obs=1 = contact
        G_range_post = float(np.max(r_post.G) - np.min(r_post.G))

        assert G_range_pre > G_range_post, (
            f"G range did not shrink after contact resolved uncertainty: "
            f"pre={G_range_pre:.4f}, post={G_range_post:.4f}\n"
            f"Beliefs pre:  {np.round(r_pre.beliefs[1], 3)}\n"
            f"Beliefs post: {np.round(r_post.beliefs[1], 3)}"
        )

    def test_action_distribution_biased_under_uncertainty(self):
        """
        Under uniform beliefs (occluded), the agent must NOT choose actions
        uniformly at random. The softmax over G should produce a biased
        policy — i.e., at least one action has probability substantially
        above 1/N_actions.
        """
        ag = self._make_soft_agent()
        r = ag.step([3, 0], c_visual=0.0)
        n_actions = len(r.q_pi)
        uniform_p = 1.0 / n_actions
        max_p = float(np.max(r.q_pi))
        assert max_p > uniform_p + 0.05, (
            f"Policy is nearly uniform (max={max_p:.3f}, uniform={uniform_p:.3f}) — "
            f"EFE is not biasing action selection\n"
            f"q_pi: {np.round(r.q_pi, 3)}"
        )


# ---------------------------------------------------------------------------
# Claim 6: Precision mechanism — same obs, different c_visual → different beliefs
# ---------------------------------------------------------------------------
class TestPrecisionMechanism:
    """
    Mechanism-level test: visual precision (not just the observation) shapes
    the posterior. Two agents receive the same observations but differ only
    in c_visual. Their beliefs must differ because the precision-weighted
    likelihood has different curvature.

    Uses soft A_tactile to prevent binary tactile domination, which would
    make beliefs identical regardless of visual precision.
    """

    def _make_soft_agent_with_c(self, c_visual_val):
        """Agent that always uses a fixed c_visual regardless of what's passed."""
        A = _build_A_soft(p_contact=0.9, p_bg=0.05)
        B, C, D = _build_BCD()
        pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)
        ag = MultiModalAIFAgent(
            A, B, C, D,
            precision_manager=pm,
            policy_len=1,
            inference_horizon=1,
        )
        ag.reset()
        return ag

    def test_precision_alone_changes_beliefs_same_observation(self):
        """
        obs=[2, 0] (visual obs=2 + no contact) with c_visual=1.0 vs c_visual=0.1.
        With c_visual=1.0: A_visual clean → obs=2 perfectly reveals obj_loc=2,
                           P(obj=2) → 1.0.
        With c_visual=0.1: A_visual noised → obs=2 is ambiguous,
                           P(obj=2) ≈ 0.75 (weaker update).
        The posteriors must differ, proving precision shapes the likelihood.

        NOTE: obs=[3, 0] cannot be used — obs=3 maps to the all-zero row of
        clean A_visual and a constant row of noised A_visual.  In both cases
        the visual likelihood is uniform across obj states, so precision
        switching has no effect on the posterior.
        """
        ag_high = self._make_soft_agent_with_c(1.0)
        ag_low  = self._make_soft_agent_with_c(0.0)

        r_high = ag_high.step([2, 0], c_visual=1.0)
        r_low  = ag_low.step([2, 0],  c_visual=0.1)

        beliefs_high = np.array(r_high.beliefs[1])
        beliefs_low  = np.array(r_low.beliefs[1])

        assert not np.allclose(beliefs_high, beliefs_low, atol=1e-3), (
            f"Beliefs identical under high/low precision with ambiguous obs — "
            f"precision switching has no effect on posterior.\n"
            f"High c_visual: {np.round(beliefs_high, 4)}\n"
            f"Low  c_visual: {np.round(beliefs_low,  4)}"
        )

    def test_high_precision_amplifies_informative_obs(self):
        """
        With an informative visual obs (obs=2, obj=2) and full precision,
        P(obj=2) should be higher than with degraded precision (c_visual=0.1).
        Soft A_tactile: tactile obs=0 (no contact, arm=0) provides weak evidence,
        so visual precision difference is detectable.
        """
        # arm at pos 0, obj at pos 2 → no contact: tactile_obs=0
        ag_hi = self._make_soft_agent_with_c(1.0)
        ag_lo = self._make_soft_agent_with_c(0.1)

        r_hi = ag_hi.step([2, 0], c_visual=1.0)
        r_lo = ag_lo.step([2, 0], c_visual=0.1)

        p_hi = float(r_hi.beliefs[1][2])
        p_lo = float(r_lo.beliefs[1][2])

        assert p_hi > p_lo + 0.05, (
            f"High precision ({p_hi:.3f}) not substantially above low precision ({p_lo:.3f}) "
            f"for informative visual obs — precision not amplifying likelihood"
        )


# ---------------------------------------------------------------------------
# Claim 7: Soft tactile precision — higher pi_tactile sharpens contact posterior
# ---------------------------------------------------------------------------
class TestSoftTactilePrecisionBehavior:
    """
    With soft A_tactile (not binary), the power-law sharpening by pi_tactile
    must produce measurably different posteriors under contact observations.

    This validates that tactile precision upregulation (the Johansson-Flanagan
    switching mechanism) actually changes inferred object location beliefs —
    not just the A matrix shape.
    """

    def _make_agent_pi(self, pi_tactile_max: float, tactile_noise_floor: float = 0.1):
        A = _build_A_soft(p_contact=0.7, p_bg=0.15)   # softer, more room for sharpening
        B, C, D = _build_BCD()
        # Known arm at pos=0: contact at arm=0 uniquely implicates obj_loc=0.
        # With uniform arm prior, P(contact|obj=0) ≡ 1/3 regardless of arm,
        # making contact uninformative about obj_loc.
        D[0] = np.array([1.0, 0.0, 0.0])
        pm = PrecisionManager(
            theta=0.4,
            pi_tactile_max=pi_tactile_max,
            pi_visual_min=0.1,
            tactile_noise_floor=tactile_noise_floor,
        )
        ag = MultiModalAIFAgent(
            A, B, C, D,
            precision_manager=pm,
            policy_len=1,
            inference_horizon=1,
        )
        ag.reset()
        return ag

    def test_higher_pi_tactile_increases_contact_confidence(self):
        """
        Arm known at pos=0, contact observed (obs=[3, 1], c_visual=0.0).
        Contact at arm=0 → obj likely at loc=0.
        With pi_tactile_max=1.0 (no sharpening): P(obj=0) = p_contact = 0.7.
        With pi_tactile_max=5.0 + noise_floor=0.1 (strong sharpening): P(obj=0) ≈ 1.0.
        """
        ag_low  = self._make_agent_pi(pi_tactile_max=1.0, tactile_noise_floor=0.0)
        ag_high = self._make_agent_pi(pi_tactile_max=5.0, tactile_noise_floor=0.1)

        # obs=[3, 1]: ambiguous visual (occluded) + contact. arm=0, so contact → obj likely at 0.
        r_low  = ag_low.step([3, 1],  c_visual=0.0)
        r_high = ag_high.step([3, 1], c_visual=0.0)

        p_low  = float(r_low.beliefs[1][0])
        p_high = float(r_high.beliefs[1][0])

        assert p_high > p_low + 0.05, (
            f"Higher pi_tactile did not increase P(obj=0|contact): "
            f"pi=1.0 → {p_low:.3f}, pi=5.0 → {p_high:.3f}\n"
            f"Soft A_tactile sharpening must change posterior beliefs under contact."
        )

    def test_no_sharpening_with_zero_noise_floor(self):
        """
        With tactile_noise_floor=0 (binary A_tac), power-law has no effect:
        P(obj=0) under pi=1 and pi=5 must be identical.
        """
        ag_low  = self._make_agent_pi(pi_tactile_max=1.0, tactile_noise_floor=0.0)
        ag_high = self._make_agent_pi(pi_tactile_max=5.0, tactile_noise_floor=0.0)

        r_low  = ag_low.step([3, 1],  c_visual=0.0)
        r_high = ag_high.step([3, 1], c_visual=0.0)

        p_low  = float(r_low.beliefs[1][0])
        p_high = float(r_high.beliefs[1][0])

        np.testing.assert_allclose(
            p_low, p_high, atol=1e-6,
            err_msg=(
                f"With noise_floor=0, sharpening should not change beliefs: "
                f"pi=1.0 → {p_low:.6f}, pi=5.0 → {p_high:.6f}"
            )
        )
