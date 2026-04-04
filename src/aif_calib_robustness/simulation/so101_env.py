"""
SO-101 Occlusion Manipulator Environment.

Wraps the SO-101 MuJoCo model with the same API as OcclusionManipulatorEnv,
so all existing AIF agents and experiment scripts work unchanged.

Key differences from the toy L-arm environment:
  - Uses the real SO-101 MJCF model (yellow arm with gripper)
  - shoulder_pan is the only actuated joint; others are locked by equality constraints
  - Touch is detected via two jaw-tip sensors (fixed + moving jaw)
  - The arm sweeps ±0.40 rad around shoulder_pan = 0
  - Object table position adjusted to match gripper tip height (z≈0.147)
"""

from __future__ import annotations

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
_SCENE_XML = Path(__file__).parent / "assets" / "so101_occlusion_scene.xml"
_SO101_XML = Path(__file__).parent / "assets" / "so101" / "so101.xml"
_STL_DIR   = Path(__file__).parent / "assets" / "so101" / "assets"

# shoulder_pan range: SO-101 supports ±1.92 rad, but ±0.40 is enough to reach
# the 3 object positions at Y = -0.08, 0.00, +0.08 m
_SHOULDER_MIN = -0.40   # rad
_SHOULDER_MAX =  0.40   # rad

# Object Y positions — matched to actual gripper tip Y at each arm bin.
# Measured by sweeping shoulder_pan with all other joints locked:
#   bin 1 (pan=-0.20):  y≈+0.050   →  obj_loc=0 (left)
#   bin 2 (pan= 0.00):  y≈ 0.000   →  obj_loc=1 (center)
#   bin 3 (pan=+0.20):  y≈-0.049   →  obj_loc=2 (right)
_OBJ_Y_POSITIONS = [+0.050, 0.000, -0.049]   # left, center, right

# shoulder_pan angles corresponding to each object (bin centers of ±0.40 range)
_CONTACT_ANGLES = [-0.20, 0.00, +0.20]   # rad (bins 1, 2, 3)

# Occluder positions (same as original)
_OCCLUDER_POSITIONS = {
    "none":    (5.0, 0.0, 0.0),
    "partial": (0.15, 0.0, 0.15),
    "full":    (0.15, 0.0, 0.25),
}

N_VISUAL_OBS  = len(_OBJ_Y_POSITIONS) + 1
_AMBIGUOUS_OBS = N_VISUAL_OBS - 1   # = 3

# Initial joint configuration (locked joints; shoulder_pan actuated)
_INIT_QPOS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -0.70,
    "elbow_flex":     0.80,
    "wrist_flex":     0.40,
    "wrist_roll":     0.00,
    "gripper":        0.50,
}


# ---------------------------------------------------------------------------
# Data containers (identical to OcclusionManipulatorEnv)
# ---------------------------------------------------------------------------
@dataclass
class EnvObs:
    arm_angle:       float
    arm_pos_idx:     int
    touch_force:     float
    tactile_obs_idx: int
    visual_obs_idx:  int
    c_visual:        float
    obj_loc_idx:     int


@dataclass
class StepResult:
    obs:       EnvObs
    reward:    float
    done:      bool
    truncated: bool
    info:      dict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class SO101OcclusionEnv:
    """
    SO-101 gripper simulation with occlusion, matching the OcclusionManipulatorEnv API.

    Parameters
    ----------
    n_arm_positions : int
        Discrete bins for shoulder_pan (default 5).
    occlusion_mode : {'none', 'partial', 'full'}
    max_steps : int
    """

    def __init__(
        self,
        n_arm_positions: int = 5,
        occlusion_mode: str = "none",
        max_steps: int = 50,
        calib_offset: float = 0.0,
    ):
        if not HAS_MUJOCO:
            raise ImportError("mujoco package required.")
        if occlusion_mode not in _OCCLUDER_POSITIONS:
            raise ValueError(f"occlusion_mode must be one of {list(_OCCLUDER_POSITIONS)}")

        self.occlusion_mode  = occlusion_mode
        self.max_steps       = max_steps
        self.n_arm_positions = n_arm_positions
        self.n_obj_locations = len(_OBJ_Y_POSITIONS)
        # Systematic calibration offset: agent reports (actual_angle + offset).
        # Models the mismatch between training calibration and deployment calibration.
        self.calib_offset    = calib_offset

        self.joint_disc = JointDiscretizer(
            n_bins=n_arm_positions,
            angle_min=_SHOULDER_MIN,
            angle_max=_SHOULDER_MAX,
        )
        self.contact_disc = ContactDiscretizer(
            thresholds=[0.5],
            labels=["no_contact", "contact"],
        )

        # Load model with STL assets
        assets = {"so101.xml": _SO101_XML.read_bytes()}
        for stl in _STL_DIR.glob("*.stl"):
            assets[stl.name] = stl.read_bytes()
        self._model = mujoco.MjModel.from_xml_path(str(_SCENE_XML), assets=assets)
        self._data  = mujoco.MjData(self._model)

        # Cache MuJoCo IDs
        self._pan_qposadr = self._model.jnt_qposadr[
            self._model.joint("shoulder_pan").id
        ]
        self._pan_ctrl_id = self._model.actuator("shoulder_pan").id
        # Physical touch sensors on the gripper jaw tips.
        # The object (thin cylinder, r=5mm, contype=2) collides only with
        # collision_gripper geoms (contype=2/conaffinity=2, fingertip spheres),
        # not with arm body geoms (contype=1/conaffinity=1).
        _sid_fixed  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_touch_fixed")
        _sid_moving = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_touch_moving")
        self._touch_fixed_adr  = int(self._model.sensor_adr[_sid_fixed])
        self._touch_moving_adr = int(self._model.sensor_adr[_sid_moving])
        self._obj_mocap_id = self._model.body_mocapid[self._model.body("object").id]
        self._occ_mocap_id = self._model.body_mocapid[self._model.body("occluder").id]

        self._obj_loc_idx: int = 0
        self._step_count:  int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API (identical to OcclusionManipulatorEnv)
    # ------------------------------------------------------------------

    @property
    def c_visual(self) -> float:
        return {"none": 1.0, "partial": 0.5, "full": 0.0}[self.occlusion_mode]

    def reset(
        self,
        obj_loc_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> EnvObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if obj_loc_idx is None:
            obj_loc_idx = int(self._rng.integers(self.n_obj_locations))
        self._obj_loc_idx = obj_loc_idx
        self._step_count  = 0

        mujoco.mj_resetData(self._model, self._data)

        # Set all joint positions AND ctrl targets to their locked values.
        # Without matching ctrl values, the sts3215 actuators (kp=998) will
        # fight the equality constraints and move joints away from target.
        for name, val in _INIT_QPOS.items():
            jid  = self._model.joint(name).id
            aid  = self._model.actuator(name).id   # actuator has same name
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[aid] = val

        # Override shoulder_pan to start at leftmost position
        self._data.qpos[self._pan_qposadr] = _SHOULDER_MIN
        self._data.ctrl[self._model.actuator("shoulder_pan").id] = _SHOULDER_MIN

        self._place_object(obj_loc_idx)
        self._place_occluder(self.occlusion_mode)

        mujoco.mj_forward(self._model, self._data)
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0, 1, or 2; got {action}")

        delta     = self.joint_disc.decode_delta(action, n_actions=3)
        curr      = float(self._data.qpos[self._pan_qposadr])
        new_angle = float(np.clip(curr + delta, _SHOULDER_MIN, _SHOULDER_MAX))
        self._data.ctrl[self._pan_ctrl_id] = new_angle

        for _ in range(100):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs       = self._get_obs()
        reward    = self._compute_reward(obs)
        done      = obs.tactile_obs_idx > 0
        truncated = self._step_count >= self.max_steps
        return StepResult(obs=obs, reward=reward, done=done,
                          truncated=truncated, info={})

    def get_pymdp_obs(self) -> list[int]:
        obs = self._get_obs()
        return [obs.visual_obs_idx, obs.tactile_obs_idx]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_touch_force(self) -> float:
        """Physical touch: max force [N] from the two jaw-tip touch sensors."""
        f_fixed  = float(self._data.sensordata[self._touch_fixed_adr])
        f_moving = float(self._data.sensordata[self._touch_moving_adr])
        return max(f_fixed, f_moving)

    def _get_obs(self) -> EnvObs:
        arm_angle    = float(self._data.qpos[self._pan_qposadr])
        touch_force  = self._get_touch_force()
        # Apply calibration offset: agent sees a biased arm position.
        # The physical world (touch sensor, actual motion) is unaffected.
        reported_angle  = float(np.clip(
            arm_angle + self.calib_offset, _SHOULDER_MIN, _SHOULDER_MAX
        ))
        arm_pos_idx     = self.joint_disc.encode(reported_angle)
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
        c = self.c_visual
        if c >= 0.99:
            return self._obj_loc_idx
        if c <= 0.01:
            return _AMBIGUOUS_OBS
        return (self._obj_loc_idx if self._rng.random() < c else _AMBIGUOUS_OBS)

    def _compute_reward(self, obs: EnvObs) -> float:
        if obs.tactile_obs_idx > 0:
            obj_angle = _CONTACT_ANGLES[self._obj_loc_idx]
            if abs(obs.arm_angle - obj_angle) < 0.15:
                return 1.0
        return -0.1

    def _place_object(self, obj_loc_idx: int) -> None:
        y = _OBJ_Y_POSITIONS[obj_loc_idx]
        # x=0.284: fingertip X at center pan angle
        # z=0.115: cylinder center; half-height=40mm covers fingertip Z range
        self._data.mocap_pos[self._obj_mocap_id] = [0.284, y, 0.115]

    def _place_occluder(self, mode: str) -> None:
        self._data.mocap_pos[self._occ_mocap_id] = list(_OCCLUDER_POSITIONS[mode])
