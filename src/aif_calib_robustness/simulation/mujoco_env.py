"""
MuJoCo simulation environment for AIF occlusion manipulation research.

Architecture
------------
An L-shaped arm (shoulder joint only, sweeping in XY plane) attempts to
reach a hidden object placed at one of N_OBJ positions on a table.

State space matches the generative model in notebooks/02_generative_model_design.ipynb:
  - arm_pos  : N_ARM_POSITIONS discrete bins of shoulder angle
  - obj_loc  : N_OBJ_LOCATIONS discrete positions (hidden — not observable directly)

Observations:
  - visual_obs  : index 0–(N_OBJ-1) when camera sees object clearly,
                  index N_OBJ (= "ambiguous") under full occlusion
  - tactile_obs : 0 = no contact, 1 = contact

Actions:
  - 0: move left  (shoulder –)
  - 1: stay
  - 2: move right (shoulder +)

Occlusion modes
---------------
  'none'    : c_visual = 1.0  — camera sees object clearly
  'partial' : c_visual = 0.5  — stochastic visibility
  'full'    : c_visual = 0.0  — occluder blocks camera entirely
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from aif_calib_robustness.utils.discretizer import JointDiscretizer, ContactDiscretizer

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
_SCENE_XML = Path(__file__).parent / "assets" / "occlusion_scene.xml"

# Shoulder angle range (must match occlusion_scene.xml joint range)
_SHOULDER_MIN = -0.50   # rad
_SHOULDER_MAX =  0.50   # rad

# Object Y positions on the table (arm points in +X at z=0.30, sweeps ±Y)
# These are the exact Y values where contact occurs at each obj_loc.
_OBJ_Y_POSITIONS = [-0.08, 0.00, 0.08]  # left, center, right

# Shoulder angles that align fingertip with each object (arcsin(Y / arm_length))
# arm_length (upper arm X) = 0.30 m
_CONTACT_ANGLES = [float(np.arcsin(y / 0.30)) for y in _OBJ_Y_POSITIONS]
# ≈ [-0.267, 0.000, +0.267] rad

# Occluder positions for each mode: (x, y, z)
_OCCLUDER_POSITIONS = {
    "none":    (5.0, 0.0, 0.0),    # far away — not visible
    "partial": (0.15, 0.0, 0.15),  # partially blocks left-right view
    "full":    (0.15, 0.0, 0.25),  # directly between camera and table
}

# Number of visual observation categories (N_OBJ locations + 1 "ambiguous")
N_VISUAL_OBS = len(_OBJ_Y_POSITIONS) + 1   # = 4; index 3 = "cannot see"
_AMBIGUOUS_OBS = N_VISUAL_OBS - 1           # = 3


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class EnvObs:
    """All observations returned by reset() and step()."""
    arm_angle: float       # continuous shoulder angle (rad)
    arm_pos_idx: int       # discretized shoulder angle index
    touch_force: float     # raw fingertip touch sensor value (N)
    tactile_obs_idx: int   # discrete tactile obs (0 = no-contact, 1 = contact)
    visual_obs_idx: int    # discrete visual obs (0-2 = obj loc, 3 = ambiguous)
    c_visual: float        # visual confidence score [0, 1]
    obj_loc_idx: int       # TRUE object location index (for evaluation only!)


@dataclass
class StepResult:
    """Output of step()."""
    obs: EnvObs
    reward: float
    done: bool
    truncated: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class OcclusionManipulatorEnv:
    """
    MuJoCo 1-DoF reaching environment with occlusion control.

    Parameters
    ----------
    scene_xml_path : Path or str, optional
        Path to the MuJoCo XML scene file. Defaults to the bundled scene.
    n_arm_positions : int
        Number of discrete arm-position bins (default 5).
    occlusion_mode : {'none', 'partial', 'full'}
        Sets visual confidence c_visual for the episode.
    max_steps : int
        Episode horizon (default 50).
    """

    def __init__(
        self,
        scene_xml_path: Optional[Path] = None,
        n_arm_positions: int = 5,
        occlusion_mode: str = "none",
        max_steps: int = 50,
    ):
        if not HAS_MUJOCO:
            raise ImportError(
                "mujoco package is required. Install with: uv add mujoco"
            )

        if occlusion_mode not in _OCCLUDER_POSITIONS:
            raise ValueError(
                f"occlusion_mode must be one of {list(_OCCLUDER_POSITIONS)}; "
                f"got {occlusion_mode!r}"
            )

        self.occlusion_mode = occlusion_mode
        self.max_steps = max_steps
        self.n_arm_positions = n_arm_positions
        self.n_obj_locations = len(_OBJ_Y_POSITIONS)

        # Discretizers
        self.joint_disc = JointDiscretizer(
            n_bins=n_arm_positions,
            angle_min=_SHOULDER_MIN,
            angle_max=_SHOULDER_MAX,
        )
        self.contact_disc = ContactDiscretizer(
            thresholds=[0.5],        # 0 N = no contact; > 0.5 N = contact
            labels=["no_contact", "contact"],
        )

        xml_path = Path(scene_xml_path) if scene_xml_path else _SCENE_XML
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data  = mujoco.MjData(self._model)

        # Cache MuJoCo IDs
        self._shoulder_qposadr = self._model.jnt_qposadr[
            self._model.joint("shoulder").id
        ]
        self._touch_sensor_adr = self._model.sensor_adr[
            self._model.sensor("fingertip_touch").id
        ]
        self._obj_mocap_id  = self._model.body_mocapid[self._model.body("object").id]
        self._occ_mocap_id  = self._model.body_mocapid[self._model.body("occluder").id]

        # Episode state
        self._obj_loc_idx: int = 0
        self._step_count: int  = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def c_visual(self) -> float:
        """Visual confidence score for current occlusion_mode."""
        return {"none": 1.0, "partial": 0.5, "full": 0.0}[self.occlusion_mode]

    def reset(
        self,
        obj_loc_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> EnvObs:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        obj_loc_idx : int, optional
            Object location (0 = left, 1 = center, 2 = right).
            Sampled uniformly if None.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        EnvObs
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if obj_loc_idx is None:
            obj_loc_idx = int(self._rng.integers(self.n_obj_locations))
        self._obj_loc_idx = obj_loc_idx
        self._step_count  = 0

        mujoco.mj_resetData(self._model, self._data)

        # Start arm at left-most position (away from all objects initially)
        self._data.qpos[self._shoulder_qposadr] = _SHOULDER_MIN
        self._data.ctrl[0] = _SHOULDER_MIN

        # Place object and occluder
        self._place_object(obj_loc_idx)
        self._place_occluder(self.occlusion_mode)

        # Run forward dynamics to settle
        mujoco.mj_forward(self._model, self._data)
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        """
        Execute one action.

        Parameters
        ----------
        action : int
            0 = move left, 1 = stay, 2 = move right.

        Returns
        -------
        StepResult
        """
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0, 1, or 2; got {action}")

        # Update target shoulder angle
        delta     = self.joint_disc.decode_delta(action, n_actions=3)
        curr      = float(self._data.qpos[self._shoulder_qposadr])
        new_angle = float(np.clip(curr + delta, _SHOULDER_MIN, _SHOULDER_MAX))
        self._data.ctrl[0] = new_angle

        # Simulate (50 steps × 2 ms = 100 ms per action)
        for _ in range(50):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs      = self._get_obs()
        reward   = self._compute_reward(obs)
        done     = obs.tactile_obs_idx > 0   # contact achieved
        truncated = self._step_count >= self.max_steps
        return StepResult(obs=obs, reward=reward, done=done,
                          truncated=truncated, info={})

    def get_pymdp_obs(self) -> list[int]:
        """
        Return [visual_obs_idx, tactile_obs_idx] for the pymdp AIF agent.

        This matches the observation modality order in the generative model:
          modality 0 = visual  (shape N_VISUAL_OBS × N_ARM × N_OBJ)
          modality 1 = tactile (shape 2 × N_ARM × N_OBJ)
        """
        obs = self._get_obs()
        return [obs.visual_obs_idx, obs.tactile_obs_idx]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> EnvObs:
        arm_angle   = float(self._data.qpos[self._shoulder_qposadr])
        touch_force = float(self._data.sensordata[self._touch_sensor_adr])

        arm_pos_idx    = self.joint_disc.encode(arm_angle)
        tactile_obs_idx = self.contact_disc.encode(touch_force)
        visual_obs_idx  = self._compute_visual_obs()

        return EnvObs(
            arm_angle=arm_angle,
            arm_pos_idx=arm_pos_idx,
            touch_force=touch_force,
            tactile_obs_idx=tactile_obs_idx,
            visual_obs_idx=visual_obs_idx,
            c_visual=self.c_visual,
            obj_loc_idx=self._obj_loc_idx,
        )

    def _compute_visual_obs(self) -> int:
        """
        Map object location + occlusion to a visual observation index.

        No occlusion  : returns obj_loc_idx  (camera directly sees object)
        Full occlusion: returns N_OBJ        (ambiguous — "cannot see")
        Partial       : stochastic mix
        """
        c = self.c_visual
        if c >= 0.99:
            return self._obj_loc_idx
        if c <= 0.01:
            return _AMBIGUOUS_OBS
        # Partial occlusion: Bernoulli draw
        return (
            self._obj_loc_idx
            if self._rng.random() < c
            else _AMBIGUOUS_OBS
        )

    def _compute_reward(self, obs: EnvObs) -> float:
        """
        Reward function:
          +1.0 if contact with correct object (arm aligned with obj_loc)
          -0.1 per step otherwise
        """
        if obs.tactile_obs_idx > 0:
            obj_angle = _CONTACT_ANGLES[self._obj_loc_idx]
            angle_err = abs(obs.arm_angle - obj_angle)
            if angle_err < 0.15:
                return 1.0
        return -0.1

    def _place_object(self, obj_loc_idx: int) -> None:
        """Move the mocap object to the chosen Y position."""
        y = _OBJ_Y_POSITIONS[obj_loc_idx]
        self._data.mocap_pos[self._obj_mocap_id] = [0.30, y, 0.147]

    def _place_occluder(self, mode: str) -> None:
        """Move the occluder mocap body for the given occlusion mode."""
        pos = _OCCLUDER_POSITIONS[mode]
        self._data.mocap_pos[self._occ_mocap_id] = list(pos)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def render_text(self, obs: Optional[EnvObs] = None) -> str:
        """Return a one-line ASCII summary of current state."""
        if obs is None:
            obs = self._get_obs()
        contact_str = "CONTACT" if obs.tactile_obs_idx > 0 else "no-contact"
        occl_str    = self.occlusion_mode
        return (
            f"[step={self._step_count:3d}] "
            f"shoulder={obs.arm_angle:+.3f}rad  pos_idx={obs.arm_pos_idx}  "
            f"vis={obs.visual_obs_idx}  tac={contact_str}  "
            f"occlusion={occl_str}  obj_loc={obs.obj_loc_idx}"
        )
