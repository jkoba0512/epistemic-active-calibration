"""
Generative model factory for the 1-DoF proxy task.

Builds pymdp-compatible A, B, C, D matrices for the toy
arm-sweep / object-localization environment:
  - s[0]: arm_pos  ∈ {0, …, N_POS-1}
  - s[1]: obj_loc  ∈ {0, …, N_OBJ-1}
  - o[0]: visual   ∈ {0, …, N_OBJ-1, N_OBJ}  (N_OBJ = ambiguous/occluded)
  - o[1]: tactile  ∈ {0=no-contact, 1=contact}
"""

from __future__ import annotations

import numpy as np
from pymdp.legacy import utils


# Default object-to-contact-arm-position mapping used by OcclusionManipulatorEnv.
# obj_loc j -> arm_pos where fingertip touches object j.
DEFAULT_OBJ_ARM: dict[int, int] = {0: 1, 1: 2, 2: 3}


def build_A(
    n_pos: int,
    n_obj: int,
    *,
    p_contact: float = 1.0,
    p_bg: float = 0.0,
    obj_arm: dict[int, int] | None = None,
    with_proprio: bool = False,
    proprio_accuracy: float = 1.0,
) -> np.ndarray:
    """
    Build A (likelihood) matrices.

    Parameters
    ----------
    n_pos : int
        Number of discrete arm positions.
    n_obj : int
        Number of object locations (= number of informative visual obs).
    p_contact : float
        P(contact | arm == contact_arm_for_obj).  1.0 → binary; <1.0 → soft.
    p_bg : float
        P(contact | arm != contact_arm_for_obj).  0.0 → binary; >0.0 → soft.
    obj_arm : dict, optional
        Mapping obj_loc → arm_pos that produces contact.
        Defaults to DEFAULT_OBJ_ARM.
    with_proprio : bool
        If True, add a 3rd proprioceptive modality A[2] over arm_pos.
        This resolves the uniform arm-prior problem: tactile observations
        can directly inform obj_loc beliefs once arm_pos is known.
    proprio_accuracy : float
        Accuracy of proprioceptive readout ∈ (0, 1].
        1.0 → perfect (identity matrix); <1.0 → mixed with uniform,
        modelling encoder noise or discretisation errors.
        A_proprio = acc * I + (1-acc) * (1/n_pos) * ones

    Returns
    -------
    A : object array, shape (2,) or (3,)
        A[0] = A_visual   shape (n_obj+1, n_pos, n_obj)
        A[1] = A_tactile  shape (2,       n_pos, n_obj)
        A[2] = A_proprio  shape (n_pos,   n_pos, n_obj)  [only if with_proprio]
    """
    if obj_arm is None:
        obj_arm = DEFAULT_OBJ_ARM

    n_vis = n_obj + 1   # 0..n_obj-1 informative, n_obj = ambiguous

    A_visual = np.zeros((n_vis, n_pos, n_obj))
    for obj_loc in range(n_obj):
        A_visual[obj_loc, :, obj_loc] = 1.0   # visual obs = obj_loc when visible
    # Row n_obj (ambiguous) stays at 0; noise injection by PrecisionManager fills it.

    A_tactile = np.zeros((2, n_pos, n_obj))
    for arm_pos in range(n_pos):
        for obj_loc in range(n_obj):
            contact_arm = obj_arm.get(obj_loc, -1)
            if arm_pos == contact_arm:
                A_tactile[1, arm_pos, obj_loc] = p_contact
                A_tactile[0, arm_pos, obj_loc] = 1.0 - p_contact
            else:
                A_tactile[1, arm_pos, obj_loc] = p_bg
                A_tactile[0, arm_pos, obj_loc] = 1.0 - p_bg

    if with_proprio:
        # A_proprio[obs_arm, state_arm, obj_loc]
        # proprio_accuracy=1.0: perfect readout (identity)
        # proprio_accuracy<1.0: mixed with uniform (encoder noise / discretisation error)
        proprio_accuracy = float(np.clip(proprio_accuracy, 1e-6, 1.0))
        uniform_2d = np.ones((n_pos, n_pos)) / n_pos
        A_proprio_2d = proprio_accuracy * np.eye(n_pos) + (1.0 - proprio_accuracy) * uniform_2d
        A_proprio = np.stack([A_proprio_2d] * n_obj, axis=2)  # (n_pos, n_pos, n_obj)
        A = utils.obj_array(3)
        A[0] = A_visual
        A[1] = A_tactile
        A[2] = A_proprio
    else:
        A = utils.obj_array(2)
        A[0] = A_visual
        A[1] = A_tactile
    return A


def build_B(n_pos: int, n_obj: int, n_actions: int = 3) -> np.ndarray:
    """
    Build B (transition) matrices.

    arm_pos transitions: action 0=left, 1=stay, 2=right (clipped at boundaries).
    obj_loc:             static (identity for all actions).
    """
    B_arm = np.zeros((n_pos, n_pos, n_actions))
    for p in range(n_pos):
        B_arm[max(0, p - 1), p, 0] = 1.0
        B_arm[p,             p, 1] = 1.0
        B_arm[min(n_pos - 1, p + 1), p, 2] = 1.0

    B_obj = np.zeros((n_obj, n_obj, n_actions))
    for a in range(n_actions):
        np.fill_diagonal(B_obj[:, :, a], 1.0)

    B = utils.obj_array(2)
    B[0] = B_arm
    B[1] = B_obj
    return B


def build_C(
    n_vis: int,
    n_tac: int,
    *,
    contact_preference: float = 3.0,
    n_proprio: int = 0,
) -> np.ndarray:
    """
    Build C (prior preference) vectors.

    Visual: no preference (zeros).
    Tactile: prefer contact (positive reward for obs=1).
    Proprio: no preference (zeros); only included when n_proprio > 0.
    """
    if n_proprio > 0:
        C = utils.obj_array(3)
        C[0] = np.zeros(n_vis)
        C[1] = np.array([-contact_preference, contact_preference])
        C[2] = np.zeros(n_proprio)
    else:
        C = utils.obj_array(2)
        C[0] = np.zeros(n_vis)
        C[1] = np.array([-contact_preference, contact_preference])
    return C


def build_D(n_pos: int, n_obj: int) -> np.ndarray:
    """Uniform prior over arm positions and object locations."""
    D = utils.obj_array(2)
    D[0] = np.ones(n_pos) / n_pos
    D[1] = np.ones(n_obj) / n_obj
    return D
