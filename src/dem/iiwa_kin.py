"""Differentiable JAX forward kinematics for the KUKA LBR iiwa 14 R820.

Parses the vendored URDF (``assets/kuka_iiwa/model.urdf``) into per-joint fixed
transforms and rotation axes, then exposes a JAX-differentiable forward
kinematics that maps joint angles ``q`` (7,) and an unknown tool offset
``delta`` (3,) to the end-effector pose.  This mirrors the role of
``fk(q, theta)`` in the planar 4-DoF experiment so the existing E-step / IG /
probe / rollout-risk pipeline transfers by swapping the kinematics.

The model is vendored into the repository (see ``assets/kuka_iiwa/README.md``)
so the project is fully self-contained; nothing here depends on an external
PyBullet / robotics-toolbox install.  MuJoCo (project ``iiwa`` dependency group)
is used only as an independent reference for validation and for rendering.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import jax
import jax.numpy as jnp

URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "kuka_iiwa" / "model.urdf"
EE_LINK = "lbr_iiwa_link_7"
N_DOF = 7

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# URDF parsing (numpy, done once at import) -> fixed joint transforms + axes
# ---------------------------------------------------------------------------

def _rpy_to_matrix(rpy):
    """URDF fixed-axis roll-pitch-yaw (X-Y-Z) to a 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _parse_urdf(path: Path):
    """Return (origins_T (n,4,4), axes (n,3), joint_limits (n,2)) for revolute joints
    along the chain to EE_LINK, in kinematic order."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        axis = np.array([float(v) for v in j.find("axis").get("xyz", "0 0 1").split()])
        lim = j.find("limit")
        lo = float(lim.get("lower")) if lim is not None else -np.pi
        hi = float(lim.get("upper")) if lim is not None else np.pi
        joints[parent] = dict(child=child, xyz=xyz, rpy=rpy, axis=axis, lim=(lo, hi))

    # Walk the chain from the root link down to EE_LINK.
    children = {v["child"] for v in joints.values()}
    roots = [p for p in joints if p not in children]
    assert len(roots) == 1, f"expected single root, got {roots}"
    link = roots[0]
    origins, axes, limits = [], [], []
    while link in joints:
        j = joints[link]
        T = np.eye(4)
        T[:3, :3] = _rpy_to_matrix(j["rpy"])
        T[:3, 3] = j["xyz"]
        origins.append(T)
        axes.append(j["axis"] / np.linalg.norm(j["axis"]))
        limits.append(j["lim"])
        if j["child"] == EE_LINK:
            break
        link = j["child"]
    return np.stack(origins), np.stack(axes), np.array(limits)


_ORIGINS_NP, _AXES_NP, _LIMITS_NP = _parse_urdf(URDF_PATH)
assert _ORIGINS_NP.shape[0] == N_DOF, f"expected {N_DOF} joints, got {_ORIGINS_NP.shape[0]}"

ORIGINS = jnp.asarray(_ORIGINS_NP)          # (7, 4, 4) fixed parent->joint transforms
AXES = jnp.asarray(_AXES_NP)                # (7, 3) rotation axes
JOINT_LIMITS = jnp.asarray(_LIMITS_NP)     # (7, 2) lower/upper
Q_HOME = jnp.zeros(N_DOF)
# A comfortable, non-singular nominal posture (elbow bent) for posture control.
Q_NOMINAL = jnp.array([0.0, 0.5, 0.0, -1.2, 0.0, 0.8, 0.0])


# ---------------------------------------------------------------------------
# JAX forward kinematics
# ---------------------------------------------------------------------------

def _axis_angle_matrix(axis, angle):
    """Rodrigues rotation about a unit axis by angle (JAX)."""
    x, y, z = axis
    c, s = jnp.cos(angle), jnp.sin(angle)
    C = 1.0 - c
    return jnp.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def ee_transform(q):
    """Homogeneous transform world -> EE link frame for joint vector q (7,)."""
    T = jnp.eye(4)
    for i in range(N_DOF):
        Rj = _axis_angle_matrix(AXES[i], q[i])
        Tj = jnp.eye(4).at[:3, :3].set(Rj)
        T = T @ ORIGINS[i] @ Tj
    return T


def fk(q, delta=jnp.zeros(3)):
    """End-effector position (3,) for joints q (7,) and tool offset delta (3,).

    ``delta`` is the unknown body/tool parameter: a translational offset applied
    in the EE link frame, i.e. ``p_ee = R_ee(q) @ delta + p_ee_link(q)``.
    With ``delta = 0`` this returns the link-7 frame origin (matches MuJoCo).
    """
    T = ee_transform(q)
    return T[:3, :3] @ delta + T[:3, 3]


def fk_pose(q, delta=jnp.zeros(3)):
    """Return (position (3,), rotation (3,3)) of the EE for q and tool offset."""
    T = ee_transform(q)
    return T[:3, :3] @ delta + T[:3, 3], T[:3, :3]


# Convenience differentiable Jacobians (reused by the calibration pipeline).
jac_q = jax.jacobian(lambda q, delta: fk(q, delta), argnums=0)      # d p_ee / d q   (3,7)
jac_delta = jax.jacobian(lambda q, delta: fk(q, delta), argnums=1)  # d p_ee / d delta (3,3)
