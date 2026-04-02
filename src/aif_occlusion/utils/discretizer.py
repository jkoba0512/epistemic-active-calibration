"""
Discretizer — converts continuous sensor values to/from discrete indices
used by pymdp.legacy.

Used as the bridge between LeRobot (continuous joint angles, currents)
and the pymdp Agent (integer observation / action indices).
"""

from __future__ import annotations

import numpy as np


class JointDiscretizer:
    """
    Discretize a single joint angle into N uniform bins.

    Parameters
    ----------
    n_bins : int
        Number of discrete bins.
    angle_min, angle_max : float
        Joint angle range in radians.
    """

    def __init__(
        self,
        n_bins: int = 20,
        angle_min: float = -np.pi / 2,
        angle_max: float = np.pi / 2,
    ) -> None:
        self.n_bins    = n_bins
        self.angle_min = angle_min
        self.angle_max = angle_max
        self._edges    = np.linspace(angle_min, angle_max, n_bins + 1)
        self._centers  = 0.5 * (self._edges[:-1] + self._edges[1:])

    def encode(self, angle: float) -> int:
        """Continuous angle → discrete bin index (clipped to [0, n_bins-1])."""
        idx = int(np.digitize(angle, self._edges) - 1)
        return int(np.clip(idx, 0, self.n_bins - 1))

    def decode(self, idx: int) -> float:
        """Discrete bin index → centre angle of the bin."""
        idx = int(np.clip(idx, 0, self.n_bins - 1))
        return float(self._centers[idx])

    def decode_delta(self, action_idx: int, n_actions: int = 3) -> float:
        """
        Convert a discrete action index to a continuous joint delta.

        Convention (n_actions=3):
            0 → move negative (−step)
            1 → stay (0)
            2 → move positive (+step)
        """
        step = (self.angle_max - self.angle_min) / (self.n_bins - 1)
        return float((action_idx - (n_actions // 2)) * step)

    def __repr__(self) -> str:
        return (
            f"JointDiscretizer(n_bins={self.n_bins}, "
            f"range=[{self.angle_min:.2f}, {self.angle_max:.2f}])"
        )


class ContactDiscretizer:
    """
    Discretize servo current (or AnySkin magnitude) into contact levels.

    Parameters
    ----------
    thresholds : list[float]
        Boundaries between contact levels (ascending).
        E.g. [0.3, 0.7] → 3 levels: [0, 0.3), [0.3, 0.7), [0.7, ∞)
    labels : list[str]
        Human-readable names for each level.
    """

    def __init__(
        self,
        thresholds: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self.thresholds = thresholds or [0.3, 0.7]
        self.labels     = labels or ["no_contact", "light_contact", "strong_contact"]
        assert len(self.labels) == len(self.thresholds) + 1

    @property
    def n_levels(self) -> int:
        return len(self.labels)

    def encode(self, value: float) -> int:
        """Continuous value → discrete contact level index."""
        for i, t in enumerate(self.thresholds):
            if value < t:
                return i
        return len(self.thresholds)

    def label(self, idx: int) -> str:
        return self.labels[int(np.clip(idx, 0, self.n_levels - 1))]

    def __repr__(self) -> str:
        return f"ContactDiscretizer(thresholds={self.thresholds})"
