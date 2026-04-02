"""
PrecisionManager — dynamic precision switching for cross-modal AIF.

Maps visual confidence score c_visual ∈ [0, 1] onto per-modality
precision weights (Pi_visual, Pi_tactile, Pi_proprio) and degrades
the visual A-matrix accordingly.

This implements the Johansson-Flanagan type switching:
    c_visual < theta  =>  Pi_visual ↓,  Pi_tactile ↑
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PrecisionWeights:
    visual:   float
    tactile:  float
    proprio:  float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {"visual": self.visual, "tactile": self.tactile, "proprio": self.proprio}


class PrecisionManager:
    """
    Converts visual confidence c_visual into precision weights and
    applies noise to the visual A-matrix.

    Parameters
    ----------
    theta : float
        Occlusion threshold. Below this, precision switching activates.
    pi_tactile_max : float
        Maximum tactile precision weight (reached at c_visual = 0).
    pi_visual_min : float
        Minimum visual precision weight (floor, at c_visual = 0).
    visual_modality_idx : int
        Index of the visual modality in the A list (default 0).
    """

    def __init__(
        self,
        theta: float = 0.4,
        pi_tactile_max: float = 5.0,
        pi_visual_min: float = 0.1,
        visual_modality_idx: int = 0,
        tactile_modality_idx: int = 1,
        tactile_noise_floor: float = 0.0,
        contact_triggered: bool = False,
    ) -> None:
        self.theta = theta
        self.pi_tactile_max = pi_tactile_max
        self.pi_visual_min = pi_visual_min
        self.visual_modality_idx = visual_modality_idx
        self.tactile_modality_idx = tactile_modality_idx
        self.tactile_noise_floor = tactile_noise_floor
        self.contact_triggered = contact_triggered

    # ------------------------------------------------------------------
    # Precision weight computation
    # ------------------------------------------------------------------

    def compute_weights(self, c_visual: float) -> PrecisionWeights:
        """
        Compute Pi_visual and Pi_tactile from c_visual.

        c_visual >= theta  →  both at 1.0 (no switching)
        c_visual <  theta  →  Pi_visual ↓ linearly to pi_visual_min
                               Pi_tactile ↑ linearly to pi_tactile_max
        """
        c_visual = float(np.clip(c_visual, 0.0, 1.0))

        if c_visual >= self.theta:
            return PrecisionWeights(visual=1.0, tactile=1.0)

        ratio = (self.theta - c_visual) / self.theta   # 0 → 1 as c_visual → 0

        pi_visual  = max(self.pi_visual_min, 1.0 - ratio * (1.0 - self.pi_visual_min))
        pi_tactile = min(self.pi_tactile_max,
                         1.0 + ratio * (self.pi_tactile_max - 1.0))

        return PrecisionWeights(visual=pi_visual, tactile=pi_tactile)

    # ------------------------------------------------------------------
    # A-matrix degradation
    # ------------------------------------------------------------------

    def apply_to_A(
        self, A: list, c_visual: float, tactile_obs: int | None = None
    ) -> list:
        """
        Return a copy of A with precision-adjusted likelihoods.

        Visual modality (idx = visual_modality_idx):
            Degraded by noise injection toward uniform.
            noise = 1 - Pi_visual  (clamped to [0, 1])
            A_visual_out = (1 - noise) * A_visual_clean + noise * uniform

        Tactile modality (idx = tactile_modality_idx), only when
        tactile_noise_floor > 0:
            First softened by noise floor:
                A_tac_soft = (1 - floor) * A_tac + floor * uniform
            Then sharpened via power-law (temperature scaling) by pi_tactile:
                A_tac_sharp = A_tac_soft ** pi_tactile  (column-wise renorm)
            This operationalises Π_tactile upregulation: higher pi_tactile
            → sharper tactile likelihood → stronger VFE gradient from touch.
            With tactile_noise_floor=0 (default), A_tactile is unchanged.

        contact_triggered mode (contact_triggered=True):
            Sharpening is applied ONLY when tactile_obs == 1 (contact detected).
            This implements Johansson-Flanagan semantics: precision rises *at*
            the contact event rather than throughout the occlusion period,
            preventing no-contact observations from being over-interpreted.
            tactile_obs must be provided when contact_triggered=True.
        """
        weights = self.compute_weights(c_visual)
        A_out   = copy.deepcopy(A)

        # --- Visual: noise injection ---
        m_vis  = self.visual_modality_idx
        noise  = np.clip(1.0 - weights.visual, 0.0, 1.0)
        n_obs  = A[m_vis].shape[0]
        uniform_vis = np.ones_like(A[m_vis]) / n_obs
        A_out[m_vis] = (1.0 - noise) * A[m_vis] + noise * uniform_vis

        # --- Tactile: power-law sharpening ---
        # Condition: noise floor > 0 AND precision is elevated AND
        #   (not contact_triggered  OR  contact actually detected this step)
        sharpening_allowed = (
            self.tactile_noise_floor > 0.0
            and weights.tactile > 1.0
            and (not self.contact_triggered or tactile_obs == 1)
        )
        if sharpening_allowed:
            m_tac   = self.tactile_modality_idx
            n_t_obs = A[m_tac].shape[0]
            uniform_tac = np.ones_like(A[m_tac]) / n_t_obs

            # Soften with noise floor (prevents log(0) in power-law)
            A_soft = (1.0 - self.tactile_noise_floor) * A[m_tac] \
                     + self.tactile_noise_floor * uniform_tac

            # Sharpen via power-law (higher pi_tactile → sharper columns)
            A_sharp = A_soft ** weights.tactile
            col_sums = A_sharp.sum(axis=0, keepdims=True)
            A_out[m_tac] = A_sharp / col_sums

        return A_out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def noise_level(self, c_visual: float) -> float:
        """Return the noise level corresponding to c_visual."""
        weights = self.compute_weights(c_visual)
        return float(np.clip(1.0 - weights.visual, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"PrecisionManager(theta={self.theta}, "
            f"pi_tactile_max={self.pi_tactile_max}, "
            f"pi_visual_min={self.pi_visual_min}, "
            f"contact_triggered={self.contact_triggered})"
        )
