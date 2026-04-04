"""
MultiModalAIFAgent — wraps pymdp.legacy.Agent with cross-modal precision switching.

This is the central class of the AIF Occlusion Manipulator project.
It manages:
  - The generative model (A, B, C, D matrices)
  - Per-step precision updates via PrecisionManager
  - The infer → act loop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymdp.legacy.agent import Agent

from aif_calib_robustness.core.precision.precision_manager import PrecisionManager, PrecisionWeights


@dataclass
class StepResult:
    """Return value of MultiModalAIFAgent.step()."""
    beliefs:         list              # posterior q(s) per factor
    action:          np.ndarray        # sampled action indices
    q_pi:            np.ndarray        # policy distribution
    G:               np.ndarray        # expected free energy per policy
    precision:       PrecisionWeights  # precision weights used this step
    c_visual:        float             # visual confidence that triggered switching


class MultiModalAIFAgent:
    """
    Cross-modal active inference agent with dynamic precision switching.

    Parameters
    ----------
    A, B, C, D : pymdp matrix lists
        Generative model components (must be built externally).
    precision_manager : PrecisionManager
        Handles c_visual → Pi conversion and A_visual noise injection.
    policy_len : int
        Planning horizon in steps.
    inference_horizon : int
        Temporal depth for belief updates.
    """

    def __init__(
        self,
        A: list,
        B: list,
        C: list,
        D: list,
        precision_manager: PrecisionManager | None = None,
        policy_len: int = 2,
        inference_horizon: int = 2,
    ) -> None:
        self._A_clean         = A
        self._B               = B
        self._C               = C
        self._D               = D
        self.precision_manager = precision_manager or PrecisionManager()
        self._policy_len       = policy_len
        self._inference_horizon = inference_horizon

        # Build the initial agent (clean A)
        self._agent = self._build_agent(A)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent(self, A: list) -> Agent:
        return Agent(
            A=A,
            B=self._B,
            C=self._C,
            D=self._D,
            policy_len=self._policy_len,
            inference_horizon=self._inference_horizon,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, observations: list[int], c_visual: float = 1.0) -> StepResult:
        """
        Run one AIF inference-action cycle.

        Parameters
        ----------
        observations : list[int]
            Discrete observation index per modality.
        c_visual : float
            Visual confidence score ∈ [0, 1].
            Values below theta trigger precision switching.

        Returns
        -------
        StepResult

        Notes
        -----
        The pymdp Agent instance is kept alive across the episode (created
        once in reset()).  Precision-weighted A matrices are updated in-place
        so that the previous posterior (stored in agent.qs) serves as the
        prior for the next inference step via the B-matrix transition.
        Rebuilding the Agent every step would discard the posterior and
        restart from D — breaking sequential Bayesian updating.
        """
        # 1. Compute precision weights
        weights = self.precision_manager.compute_weights(c_visual)

        # 2. Update A in-place — DO NOT rebuild the Agent
        #    This preserves agent.qs (posterior from t-1) as the prior for t.
        #    Pass tactile_obs so contact_triggered mode can gate sharpening.
        m_tac = self.precision_manager.tactile_modality_idx
        tactile_obs = int(observations[m_tac]) if m_tac < len(observations) else None
        A_noisy = self.precision_manager.apply_to_A(
            self._A_clean, c_visual, tactile_obs=tactile_obs
        )
        for m in range(len(A_noisy)):
            self._agent.A[m] = A_noisy[m]

        # 3. Belief update (VFE minimisation using previous posterior as prior)
        beliefs = self._agent.infer_states(observations)

        # 4. Policy inference (EFE computation)
        q_pi, G = self._agent.infer_policies()

        # 5. Action sampling (also stores action so next infer_states uses B-prior)
        action = self._agent.sample_action()

        return StepResult(
            beliefs=beliefs,
            action=action,
            q_pi=q_pi,
            G=G,
            precision=weights,
            c_visual=c_visual,
        )

    def reset(self) -> None:
        """Rebuild agent from clean A (resets internal belief history)."""
        self._agent = self._build_agent(self._A_clean)

    @property
    def A_clean(self) -> list:
        return self._A_clean

    @property
    def n_modalities(self) -> int:
        return len(self._A_clean)

    def __repr__(self) -> str:
        return (
            f"MultiModalAIFAgent("
            f"modalities={self.n_modalities}, "
            f"policy_len={self._policy_len}, "
            f"precision_manager={self.precision_manager})"
        )
