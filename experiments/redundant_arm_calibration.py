"""Redundant arm calibration: null-space epistemic vs dual-control vs VFE-only.

Demonstrates that for a redundant robot arm (4-DOF, 2D EE task space),
epistemic actions can be projected into the null space of the task Jacobian
J_ee = d(fk)/d(q), reducing first-order end-effector interference while
improving parameter identifiability.

System:
    q  = [q1, q2, q3, q4]   joint angles (4-DOF planar arm)
    u  = [dq1,...,dq4]       velocity command
    θ  = [l1, l2, l3, l4]   unknown link lengths
    y  = [x_ee, y_ee]        2D end-effector position  (task-space dim = 2)

Null-space decomposition (proposed):
    J_ee   = ∂fk(q, θ)/∂q   ∈ R^{2×4}   (EE task Jacobian)
    N      = I - J_ee⁺ J_ee ∈ R^{4×4}   (null-space projector; rank varies)
    u_task = J_ee⁺ v_task                 (drives EE toward goal)
    u_epis = α · N · ∇_q IG(q; θ, P_θ)  (null-space IG ascent)
    u      = u_task + u_epis              (first-order EE interference suppressed)

Control conditions:
    vfe_only      Pure task-space control (J_ee⁺ v_task), no epistemic
    dual_lambda   Gradient descent on J = VFE - λ·IG (generalised 2-DOF approach)
    null_space    Proposed: u_task + α·N·∇_q IG  (λ-free, task-compatible)

Two-phase experiment:
    Phase 1 (steps 0…149):  EE held near Y_GOAL_HOLD = fk(Q0, θ_true)
                             Null-space epistemics explore with small EE drift.
    Phase 2 (steps 150…199): EE to Y_GOAL_TASK; success requires correct θ.

Degenerate start (Q0 = [0,0,0,0]):
    At full extension, J_θ = ∂fk/∂θ has rank 1 (all rows collinear → can only
    identify total length).  vfe_only is stuck here — no task error to drive
    exploration, no null-space motion.  null_space moves in the Jacobian null space,
bending the arm → FIM rank increases → E-step converges.

Usage:
    .venv/bin/python experiments/redundant_arm_calibration.py

Output:
    results/redundant_arm_calibration.png
    results/redundant_arm_calibration_summary.json
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
N_DOF = 4                                          # joints
THETA_TRUE = jnp.array([0.3, 0.3, 0.3, 0.3])     # true link lengths (1.2 m total)
THETA_INIT = jnp.array([0.6, 0.1, 0.3, 0.3])     # badly wrong (l1, l2 swapped)

# Degenerate start: arm fully extended in x-direction
Q0 = jnp.zeros(N_DOF)

# Phase-2 target configuration (non-degenerate, requires correct θ)
Q_TARGET_2 = jnp.array([jnp.pi / 4, jnp.pi / 3, -jnp.pi / 6, jnp.pi / 6])

DT = 0.05
N_STEPS = 200
CHANGE_STEP = 150       # Phase 2 starts here (Phase 1 is longer to calibrate)
N_SEEDS = 50

U_MAX = 1.0
K_TASK = 5.0            # EE task-space PD gain

SIGMA_OBS = 0.02
PI_Y = 1.0 / SIGMA_OBS ** 2
PI_X = 1.0
PARAMS_PRIOR_PI = 1.0
KAPPA_P = 0.5
N_ESTEP_ITER = 5        # more Gauss-Newton iterations per update

# Dual-control parameters
LAMBDA_DUAL = 5.0       # stronger λ to help escape degenerate manifold
LR_ACTION = 0.05
N_ACTION_ITER = 50      # more iterations for gradient descent to escape flat region

# Null-space epistemic gain
ALPHA_NS = 1.2          # moderate: balance exploration vs EE hold accuracy
ESTEP_FREQ = 3          # run E-step every 3 steps (was 5)

# Soft task-compatible probing parameters.
# PROBE_EPS_Q is the finite q displacement used to score candidate probes.
# PROBE_U_GAIN is the velocity magnitude for the blended epistemic direction.
PROBE_DAMPING = 1e-3
PROBE_U_GAIN = 0.2
PROBE_EPS_Q = DT * PROBE_U_GAIN
PROBE_GRAD_EPS = 1e-12
PROBE_TASK_PENALTY = 1e-3
PROBE_RANDOM_SEED = 20260421
PROBE_RANDOM_CANDIDATES = 64

PARAM_RMSE_FAIL = 0.08
TASK_ERR_FAIL = 0.10    # 4× the null-space median; 1.6× the vfe_only median

CONDITIONS = [
    "vfe_only",
    "dual_lambda",
    "null_space",
    "null_space_probe",
    "null_space_probe_recovery",
    "null_space_probe_posture",
    "null_space_probe_recovery_posture",
    "null_space_recovery",
    "null_space_posture",
]

PROBE_FIXED_CANDIDATES = jnp.array([
    [1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
    [1.0, -2.0, 1.0, 0.0],
    [0.0, 1.0, -2.0, 1.0],
], dtype=jnp.float64)

_probe_rng = np.random.default_rng(PROBE_RANDOM_SEED)
_probe_random_candidates = _probe_rng.normal(size=(PROBE_RANDOM_CANDIDATES, N_DOF))
_probe_random_candidates /= np.linalg.norm(
    _probe_random_candidates, axis=1, keepdims=True
)
PROBE_CANDIDATES = jnp.asarray(
    np.vstack([np.asarray(PROBE_FIXED_CANDIDATES), _probe_random_candidates]),
    dtype=jnp.float64,
)

# Recovery phase parameters (null_space_recovery only)
# Steps [RECOVERY_START, CHANGE_STEP): stop epistemic, blend EE target toward task goal
RECOVERY_START = CHANGE_STEP - 20  # = 130

# Posture condition parameters (null_space_posture only)
# Phase 2: 200 steps (vs 50) + null-space posture regularization
N_STEPS_POSTURE = CHANGE_STEP + 200  # = 350
POSTURE_GAIN = 1.0                   # gain for null-space posture term: N @ (-k * q)

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def fk(q, theta):
    """4-DOF planar arm → 2D EE position."""
    q_cum = jnp.cumsum(q)
    x_ee = jnp.sum(theta * jnp.cos(q_cum))
    y_ee = jnp.sum(theta * jnp.sin(q_cum))
    return jnp.array([x_ee, y_ee])


# Phase-1 hold goal: true EE position at Q0
Y_GOAL_HOLD = fk(Q0, THETA_TRUE)      # = [1.2, 0.0]

# Phase-2 task goal: true EE position at Q_TARGET_2
Y_GOAL_TASK = fk(Q_TARGET_2, THETA_TRUE)

# ---------------------------------------------------------------------------
# Information-theoretic utilities
# ---------------------------------------------------------------------------
def _compute_J_theta(q, theta):
    """J_θ = ∂fk(q, θ)/∂θ  ∈ R^{2×4}."""
    return jax.jacfwd(lambda th: fk(q, th))(theta)


def _compute_J_ee(q, theta):
    """J_ee = ∂fk(q, θ)/∂q  ∈ R^{2×4}."""
    return jax.jacfwd(lambda qi: fk(qi, theta))(q)


def _compute_fim_local(q, theta):
    """FIM = J_θᵀ Π_y J_θ  (single-step, local configuration)."""
    J = _compute_J_theta(q, theta)          # (2, 4)
    return J.T @ (PI_Y * jnp.eye(2)) @ J   # (4, 4)


def _compute_ig(P_theta, fim):
    """IG = ½ [log|P + FIM| – log|P|]."""
    _, ld1 = jnp.linalg.slogdet(P_theta + fim)
    _, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)


def _ig_at_q(q, theta, P_theta):
    """Differentiable IG(q; θ, P_θ) — used to obtain ∇_q IG."""
    fim = _compute_fim_local(q, theta)
    return _compute_ig(P_theta, fim)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def rollout_step(q, u):
    return jnp.clip(q + u * DT, -jnp.pi, jnp.pi)


# ---------------------------------------------------------------------------
# Action computation
# ---------------------------------------------------------------------------
PI_Y_TASK = 1.0 / 0.05 ** 2


def _damped_pseudoinverse(J, damping=PROBE_DAMPING):
    task_dim = J.shape[0]
    return J.T @ jnp.linalg.inv(J @ J.T + damping * jnp.eye(task_dim))


@jax.jit
def compute_null_space_action(q, theta_est, P_theta, y_goal):
    """Null-space epistemic action (proposed).

    u = J_ee⁺ v_task + α · N · ∇_q IG

    The second term lives entirely in null(J_ee), so it cannot perturb the EE
    (to first order).  No λ trade-off is needed.
    """
    # Task-space Jacobian and pseudo-inverse
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)     # (2, 4)
    J_pinv = jnp.linalg.pinv(J_ee)                          # (4, 2)

    # Task part: drive EE toward goal
    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)                      # (2,)
    u_task = J_pinv @ v_task                                 # (4,)

    # Null-space projector and epistemic gradient
    N_mat = jnp.eye(N_DOF) - J_pinv @ J_ee                  # (4, 4)
    ig_grad = jax.grad(lambda qi: _ig_at_q(qi, theta_est, P_theta))(q)  # (4,)
    u_epis = ALPHA_NS * N_mat @ ig_grad                      # (4,) in null(J_ee)

    return jnp.clip(u_task + u_epis, -U_MAX, U_MAX)


@jax.jit
def compute_null_space_probe_action(q, theta_est, P_theta, y_goal):
    """Soft null-space IG with finite-step probing.

    alpha_N is the confidence weight on the first-order null-space IG
    direction.  When the local information gradient is absent or mostly
    task-disruptive, alpha_N is small and finite-step probing dominates.
    """
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    J_hash = _damped_pseudoinverse(J_ee)

    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)
    u_task = J_hash @ v_task

    N_mat = jnp.eye(N_DOF) - J_hash @ J_ee
    ig_grad = jax.grad(lambda qi: _ig_at_q(qi, theta_est, P_theta))(q)
    n_grad = N_mat @ ig_grad
    ig_grad_norm = jnp.linalg.norm(ig_grad)
    n_grad_norm = jnp.linalg.norm(n_grad)
    alpha_N = jnp.where(
        ig_grad_norm > PROBE_GRAD_EPS,
        jnp.clip(n_grad_norm / (ig_grad_norm + PROBE_GRAD_EPS), 0.0, 1.0),
        0.0,
    )

    svals = jnp.linalg.svd(J_ee, compute_uv=False)
    sigma_min = jnp.min(svals)

    def score_candidate(raw_d):
        projected = N_mat @ raw_d
        norm = jnp.linalg.norm(projected)
        d = jnp.where(
            norm > PROBE_GRAD_EPS, projected / norm, jnp.zeros_like(projected)
        )
        ig_gain = _ig_at_q(q + PROBE_EPS_Q * d, theta_est, P_theta) - _ig_at_q(
            q, theta_est, P_theta
        )
        task_norm = jnp.linalg.norm(J_ee @ d)
        return ig_gain - PROBE_TASK_PENALTY * task_norm, ig_gain, d

    scores, gains, directions = jax.vmap(score_candidate)(PROBE_CANDIDATES)
    best_idx = jnp.argmax(scores)
    d_probe = directions[best_idx]
    best_gain = gains[best_idx]
    d_ig = jnp.where(
        n_grad_norm > PROBE_GRAD_EPS, n_grad / n_grad_norm, jnp.zeros_like(n_grad)
    )
    d_soft_raw = alpha_N * d_ig + (1.0 - alpha_N) * d_probe
    d_soft_norm = jnp.linalg.norm(d_soft_raw)
    d_soft = jnp.where(
        d_soft_norm > PROBE_GRAD_EPS,
        d_soft_raw / d_soft_norm,
        jnp.zeros_like(d_soft_raw),
    )

    u_epis = PROBE_U_GAIN * d_soft
    u = jnp.clip(u_task + u_epis, -U_MAX, U_MAX)
    probe_weight = 1.0 - alpha_N
    return u, probe_weight, best_gain, n_grad_norm, sigma_min


@jax.jit
def compute_posture_task_action(q, theta_est, y_goal):
    """Phase 2 action with null-space posture regularization.

    u = J_ee⁺ v_task + N · (−k · q)

    The posture term pulls joints via the null space, helping the arm escape
    near-singular configurations (q≈0) where pure task control converges slowly.
    """
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    J_pinv = jnp.linalg.pinv(J_ee)
    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)
    u_task = J_pinv @ v_task
    N_mat = jnp.eye(N_DOF) - J_pinv @ J_ee
    u_posture = N_mat @ (-POSTURE_GAIN * q)
    return jnp.clip(u_task + u_posture, -U_MAX, U_MAX)


@jax.jit
def compute_vfe_only_action(q, theta_est, y_goal):
    """Pure task-space action: u = J_ee⁺ v_task."""
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    J_pinv = jnp.linalg.pinv(J_ee)
    y_ee = fk(q, theta_est)
    v_task = -K_TASK * (y_ee - y_goal)
    return jnp.clip(J_pinv @ v_task, -U_MAX, U_MAX)


def _compute_fim_rollout(q, u, theta, n_roll=5):
    """FIM over a short rollout (for dual-control IG estimate)."""
    def y_future_fn(th):
        qi = q
        ys = []
        for _ in range(n_roll):
            qi = rollout_step(qi, u)
            ys.append(fk(qi, th))
        return jnp.concatenate(ys)     # (2*n_roll,)
    J = jax.jacfwd(y_future_fn)(theta) # (2*n_roll, 4)
    R_inv = jnp.eye(2 * n_roll) * PI_Y
    return J.T @ R_inv @ J              # (4, 4)


@jax.jit
def optimize_dual_action(q, theta_est, P_theta, y_goal, u_init):
    """Gradient descent on J = VFE – λ·IG (generalised dual-control baseline)."""
    def objective(u):
        q_pred = rollout_step(q, u)
        y_pred = fk(q_pred, theta_est)
        vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)
        fim = _compute_fim_rollout(q, u, theta_est)
        ig = _compute_ig(P_theta, fim)
        return vfe - LAMBDA_DUAL * ig

    def step(u, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(step, u_init, None, length=N_ACTION_ITER)
    return u_opt


# ---------------------------------------------------------------------------
# DEM E-step
# ---------------------------------------------------------------------------
def _build_estep(theta_init):
    model = DEMModel(
        f=lambda x, v, p: jnp.zeros(N_DOF),
        g=lambda x, v, p: fk(x, p),
        n_x=N_DOF, n_v=1, n_y=2, n_order=1,
        pi_y=PI_Y, pi_x=PI_X,
        params=theta_init,
        params_prior_pi=PARAMS_PRIOR_PI,
    )
    return EStep(model, kappa_p=KAPPA_P, use_gauss_newton=True)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
def _matrix_diag(mat, eps=1e-6):
    eigvals = np.linalg.eigvalsh(np.asarray(mat, dtype=float))
    eigvals = np.maximum(eigvals, 0.0)
    rank = int(np.sum(eigvals > eps))
    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])
    cond = float(max_eig / max(min_eig, eps))
    mat_j = np.asarray(mat, dtype=float) + eps * np.eye(mat.shape[0])
    _, logdet = np.linalg.slogdet(mat_j)
    return {"rank": rank, "min_eig": min_eig, "logdet": float(logdet), "cond": cond}


def _summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(seed, condition,
            phase2_horizon_override=None,
            phase2_controller_override=None):
    """Run one seed of the redundant-arm calibration experiment.

    phase2_horizon_override: int, overrides the number of Phase 2 steps.
    phase2_controller_override: "plain" or "posture", overrides Phase 2 controller.
    Both default to None (condition-based behaviour unchanged).
    """
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(N_DOF)
    estep = _build_estep(theta_est)

    q = Q0.copy()
    u = jnp.zeros(N_DOF)

    q_change = None   # q at start of Phase 2 (after Phase 1 final rollout)
    rmse_hist = []
    task_err_hist = []     # Phase 2: true EE distance to Y_GOAL_TASK
    p_theta_rank_hist = [] # rank of accumulated posterior precision P_θ
    p_theta_logdet_hist = []
    ee_hold_err_hist = []  # Phase 1: true EE distance from Y_GOAL_HOLD
    probe_mode_hist = []
    probe_gain_hist = []
    probe_u_ig_norm_hist = []
    probe_sigma_min_hist = []
    # θ is frozen at end of Phase 1 to cleanly measure calibration quality
    theta_frozen = None

    q_hist, v_hist, y_hist = [], [], []

    # Conditions that use the posture controller in Phase 2
    _posture_ctrl = frozenset([
        "null_space_posture",
        "null_space_probe_posture",
        "null_space_probe_recovery_posture",
        "null_space_probe_posture_50",   # ablation: posture ctrl, short horizon
    ])
    # Conditions that use the longer Phase 2 horizon (200 steps instead of 50)
    _long_horizon = frozenset([
        "null_space_posture",
        "null_space_probe_posture",
        "null_space_probe_recovery_posture",
        "null_space_probe_plain_200",    # ablation: plain ctrl, long horizon
    ])
    posture_conditions = _posture_ctrl | _long_horizon  # kept for legacy n_total check
    if phase2_horizon_override is not None:
        n_total = CHANGE_STEP + phase2_horizon_override
    else:
        n_total = N_STEPS_POSTURE if condition in _long_horizon else N_STEPS
    for t in range(n_total):
        probe_mode = 0.0
        probe_gain = 0.0
        probe_u_ig_norm = 0.0
        probe_sigma_min = 0.0

        # ---- Record metrics ----
        rmse_hist.append(float(jnp.sqrt(jnp.mean((theta_est - THETA_TRUE) ** 2))))
        ee_true = fk(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_TASK) ** 2))))
        ee_hold_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_HOLD) ** 2))))

        # Rank of accumulated data FIM = P_theta - prior * I
        # (prior is always rank-4; we want to track the DATA contribution)
        data_fim = P_theta - PARAMS_PRIOR_PI * jnp.eye(N_DOF)
        data_diag = _matrix_diag(data_fim, eps=1e-3)   # threshold above numerical noise
        p_theta_rank_hist.append(data_diag["rank"])
        p_theta_logdet_hist.append(_matrix_diag(P_theta)["logdet"])

        # ---- Select action ----
        if t < CHANGE_STEP:
            # Phase 1: calibration while holding EE near Y_GOAL_HOLD
            if condition == "vfe_only":
                u = compute_vfe_only_action(q, theta_est, Y_GOAL_HOLD)
            elif condition == "dual_lambda":
                u = optimize_dual_action(q, theta_est, P_theta, Y_GOAL_HOLD, u)
            elif condition == "null_space":
                u = compute_null_space_action(q, theta_est, P_theta, Y_GOAL_HOLD)
            elif condition == "null_space_probe":
                u, probe_mode, probe_gain, probe_u_ig_norm, probe_sigma_min = (
                    compute_null_space_probe_action(q, theta_est, P_theta, Y_GOAL_HOLD)
                )
            elif condition == "null_space_probe_recovery":
                if t < RECOVERY_START:
                    u, probe_mode, probe_gain, probe_u_ig_norm, probe_sigma_min = (
                        compute_null_space_probe_action(q, theta_est, P_theta, Y_GOAL_HOLD)
                    )
                else:
                    blend = (t - RECOVERY_START) / (CHANGE_STEP - RECOVERY_START)
                    y_blend = (1.0 - blend) * Y_GOAL_HOLD + blend * Y_GOAL_TASK
                    u = compute_vfe_only_action(q, theta_est, y_blend)
            elif condition in (
                "null_space_probe_posture",
                "null_space_probe_recovery_posture",
                "null_space_probe_plain_200",   # ablation
                "null_space_probe_posture_50",  # ablation
            ):
                if condition == "null_space_probe_recovery_posture" and t >= RECOVERY_START:
                    blend = (t - RECOVERY_START) / (CHANGE_STEP - RECOVERY_START)
                    y_blend = (1.0 - blend) * Y_GOAL_HOLD + blend * Y_GOAL_TASK
                    u = compute_vfe_only_action(q, theta_est, y_blend)
                else:
                    u, probe_mode, probe_gain, probe_u_ig_norm, probe_sigma_min = (
                        compute_null_space_probe_action(q, theta_est, P_theta, Y_GOAL_HOLD)
                    )
            elif condition == "null_space_recovery":
                if t < RECOVERY_START:
                    u = compute_null_space_action(q, theta_est, P_theta, Y_GOAL_HOLD)
                else:
                    blend = (t - RECOVERY_START) / (CHANGE_STEP - RECOVERY_START)
                    y_blend = (1.0 - blend) * Y_GOAL_HOLD + blend * Y_GOAL_TASK
                    u = compute_vfe_only_action(q, theta_est, y_blend)
            else:  # null_space_posture: same Phase 1 as null_space
                u = compute_null_space_action(q, theta_est, P_theta, Y_GOAL_HOLD)
        else:
            # Phase 2: task execution with FROZEN θ (no further calibration)
            if theta_frozen is None:
                theta_frozen = theta_est
                q_change = np.array(q)
            use_posture = (
                phase2_controller_override == "posture"
                or (phase2_controller_override is None and condition in _posture_ctrl)
            )
            if use_posture:
                u = compute_posture_task_action(q, theta_frozen, Y_GOAL_TASK)
            else:
                u = compute_vfe_only_action(q, theta_frozen, Y_GOAL_TASK)

        probe_mode_hist.append(float(probe_mode))
        probe_gain_hist.append(float(probe_gain))
        probe_u_ig_norm_hist.append(float(probe_u_ig_norm))
        probe_sigma_min_hist.append(float(probe_sigma_min))

        # ---- Apply action and observe ----
        q = rollout_step(q, u)
        y_obs = fk(q, THETA_TRUE) + rng.normal(0, SIGMA_OBS, size=(2,))
        y_obs = jnp.array(y_obs)

        q_hist.append(q)
        v_hist.append(jnp.zeros(1))
        y_hist.append(y_obs)

        # ---- E-step every ESTEP_FREQ steps (Phase 1 only) ----
        if t < CHANGE_STEP and t > 5 and t % ESTEP_FREQ == 0:
            theta_est = estep.run(q_hist, v_hist, y_hist, theta_est,
                                  n_iter=N_ESTEP_ITER)
            theta_est = jnp.clip(theta_est, 0.05, 2.0)
            P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
            estep = _build_estep(theta_est)

    theta_end_ph1 = theta_frozen if theta_frozen is not None else theta_est
    q_end_ph1 = q_change if q_change is not None else np.array(q)
    # Final P_theta diagnostics (for failure analysis)
    p_final = _matrix_diag(P_theta)
    return {
        "rmse": np.array(rmse_hist),
        "task_err": np.array(task_err_hist),
        "p_theta_rank": np.array(p_theta_rank_hist),
        "p_theta_logdet": np.array(p_theta_logdet_hist),
        "ee_hold_err": np.array(ee_hold_err_hist),
        "probe_mode": np.array(probe_mode_hist),
        "probe_gain": np.array(probe_gain_hist),
        "probe_u_ig_norm": np.array(probe_u_ig_norm_hist),
        "probe_sigma_min": np.array(probe_sigma_min_hist),
        "theta_final_ph1": np.array(theta_end_ph1),
        "q_change": q_end_ph1,
        # Per-seed Phase-1-end diagnostics for failure analysis
        "rmse_at_change": float(np.array(rmse_hist)[CHANGE_STEP - 1]),
        "task_err_final": float(np.array(task_err_hist)[-1]),
        "ee_hold_mean": float(np.mean(np.abs(np.array(ee_hold_err_hist)[:CHANGE_STEP]))),
        "probe_weight_mean_phase1": float(
            np.mean(np.array(probe_mode_hist)[:CHANGE_STEP])
        ),
        "probe_gain_max_phase1": float(np.max(np.array(probe_gain_hist)[:CHANGE_STEP])),
        "p_theta_rank_at_change": int(np.array(p_theta_rank_hist)[CHANGE_STEP - 1]),
        "p_theta_min_eig": p_final["min_eig"],
        "p_theta_cond": p_final["cond"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {c: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        print(f"Running '{cond}'", end="", flush=True)
        for seed in range(N_SEEDS):
            r = run_one(seed, cond)
            results[cond].append(r)
            print(".", end="", flush=True)
        print(
            f"  done.  "
            f"RMSE@{CHANGE_STEP}={np.median([r['rmse'][CHANGE_STEP-1] for r in results[cond]]):.4f}  "
            f"TaskErr@{N_STEPS}={np.median([r['task_err'][-1] for r in results[cond]]):.4f}  "
            f"EEhold={np.median([np.mean(r['ee_hold_err'][:CHANGE_STEP]) for r in results[cond]]):.4f}"
        )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    COLORS = {"vfe_only": "C3", "dual_lambda": "C0", "null_space": "C2",
              "null_space_probe": "C5", "null_space_probe_recovery": "C6",
              "null_space_probe_posture": "C7",
              "null_space_probe_recovery_posture": "C8",
              "null_space_recovery": "C4", "null_space_posture": "C1"}
    LABELS = {
        "vfe_only":             "VFE only (task, no epistemic)",
        "dual_lambda":          f"Dual control (λ={LAMBDA_DUAL})",
        "null_space":           "Null-space epistemic (proposed)",
        "null_space_probe":     "Null-space + finite-step probing",
        "null_space_probe_recovery": "Probe + recovery",
        "null_space_probe_posture":  "Probe + posture Ph2",
        "null_space_probe_recovery_posture": "Probe + recovery + posture Ph2",
        "null_space_recovery":  "Null-space + recovery (proposed+)",
        "null_space_posture":   f"Null-space + posture Ph2 ×4 steps (proposed++)",
    }

    summary = {}
    print(
        f"\n{'Condition':12s}  {f'RMSE@{CHANGE_STEP}':9s}  {f'TaskErr@{N_STEPS}':11s}  "
        f"{'EEhold Ph1':10s}  {'Prank@Ph1':9s}  {'failRMSE':8s}  {'failTask':8s}"
    )
    for cond in CONDITIONS:
        rmse_at_ch = np.array([r["rmse"][CHANGE_STEP - 1] for r in results[cond]])
        task_final = np.array([r["task_err"][-1] for r in results[cond]])
        ee_hold = np.array([np.mean(r["ee_hold_err"][:CHANGE_STEP]) for r in results[cond]])
        prank_at_ch = np.array([r["p_theta_rank"][CHANGE_STEP - 1] for r in results[cond]])
        fail_rmse = float(np.mean(rmse_at_ch > PARAM_RMSE_FAIL))
        fail_task = float(np.mean(task_final > TASK_ERR_FAIL))

        summary[cond] = {
            "n_seeds": N_SEEDS,
            "rmse_at_change": _summarize(rmse_at_ch),
            "task_err_final": _summarize(task_final),
            "ee_hold_err_phase1": _summarize(ee_hold),
            "p_theta_rank_at_change": _summarize(prank_at_ch),
            "probe_weight_mean_phase1": _summarize(
                [r["probe_weight_mean_phase1"] for r in results[cond]]
            ),
            "probe_gain_max_phase1": _summarize(
                [r["probe_gain_max_phase1"] for r in results[cond]]
            ),
            "rmse_failure_rate": fail_rmse,
            "task_failure_rate": fail_task,
        }
        print(
            f"  {cond:10s}  {np.median(rmse_at_ch):.4f}    "
            f"{np.median(task_final):.4f}       "
            f"{np.median(ee_hold):.4f}      "
            f"{np.median(prank_at_ch):.1f}        "
            f"{fail_rmse:.2f}      {fail_task:.2f}"
        )

    out_json = project_root / "results" / "redundant_arm_calibration_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {out_json}")

    # -----------------------------------------------------------------------
    # Failure analysis for null_space condition
    # -----------------------------------------------------------------------
    if "null_space" in results:
        ns_results = results["null_space"]
        failure_analysis = {"n_seeds": N_SEEDS, "conditions": {}}

        for cond in CONDITIONS:
            per_seed = []
            for seed_idx, r in enumerate(results[cond]):
                rmse_ch = r["rmse_at_change"]
                task_f  = r["task_err_final"]
                ee_hold = r["ee_hold_mean"]
                prank   = r["p_theta_rank_at_change"]

                # Failure classification
                if rmse_ch > PARAM_RMSE_FAIL:
                    failure_mode = "estimation_failure"   # theta never converged
                elif task_f > TASK_ERR_FAIL:
                    if prank < N_DOF:
                        failure_mode = "incomplete_identifiability"  # FIM rank < 4
                    elif ee_hold > 0.05:
                        failure_mode = "ee_drift"        # null-space projection broke
                    else:
                        failure_mode = "task_control"    # theta ok, task control failed
                else:
                    failure_mode = "success"

                per_seed.append({
                    "seed": seed_idx,
                    "rmse_at_change": float(rmse_ch),
                    "task_err_final": float(task_f),
                    "ee_hold_mean": float(ee_hold),
                    "probe_weight_mean_phase1": float(r["probe_weight_mean_phase1"]),
                    "probe_gain_max_phase1": float(r["probe_gain_max_phase1"]),
                    "p_theta_rank_at_change": int(prank),
                    "p_theta_min_eig": float(r["p_theta_min_eig"]),
                    "p_theta_cond": float(r["p_theta_cond"]),
                    "theta_final_ph1": [float(x) for x in r["theta_final_ph1"]],
                    "failure_mode": failure_mode,
                })

            modes = [s["failure_mode"] for s in per_seed]
            mode_counts = {m: modes.count(m) for m in set(modes)}
            failure_analysis["conditions"][cond] = {
                "mode_counts": mode_counts,
                "per_seed": per_seed,
            }
            print(f"\n  [{cond}] failure modes: {mode_counts}")

        out_fa = project_root / "results" / "redundant_arm_failure_analysis.json"
        with open(out_fa, "w") as f:
            json.dump(failure_analysis, f, indent=2)
        print(f"Saved failure analysis → {out_fa}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    # For plotting, truncate/pad all trajectories to N_STEPS for visual comparison
    # (null_space_posture runs to N_STEPS_POSTURE but we show first N_STEPS here)
    steps = np.arange(N_STEPS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Redundant arm (4-DOF, 2D task)  —  null-space epistemic vs dual-control vs VFE-only\n"
        r"$\theta_{init}=[0.6,0.1,0.3,0.3]$, $\theta_{true}=[0.3,0.3,0.3,0.3]$  |  "
        f"Q0=[0,0,0,0] (degenerate)  |  "
        f"Phase 1: 0–{CHANGE_STEP-1}  |  Phase 2: {CHANGE_STEP}–{N_STEPS-1}",
        fontsize=9,
    )

    def shade(ax):
        ax.axvspan(CHANGE_STEP, N_STEPS, color="gray", alpha=0.10, label="Phase 2 (task)")
        ax.axvline(CHANGE_STEP, color="gray", linestyle="--", linewidth=0.8)

    def plot_med_iqr(ax, cond, key, transform=lambda x: x):
        # Truncate to N_STEPS for uniform comparison across conditions
        mat = np.array([transform(r[key])[:N_STEPS] for r in results[cond]])
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(steps, med, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(steps, q25, q75, color=COLORS[cond], alpha=0.2)

    # (A) Parameter RMSE
    ax = axes[0, 0]
    for cond in CONDITIONS:
        plot_med_iqr(ax, cond, "rmse")
    shade(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE(θ)")
    ax.set_title("(A) Parameter RMSE  [lower = better calibration]")
    ax.legend(fontsize=8)

    # (B) True 2D task error (Phase 2 performance)
    ax = axes[0, 1]
    for cond in CONDITIONS:
        plot_med_iqr(ax, cond, "task_err")
    shade(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("EE distance (m)")
    ax.set_title("(B) True 2D task error  [lower = arm reached goal]")
    ax.legend(fontsize=8)

    # (C) Accumulated posterior precision rank (P_theta) over time
    ax = axes[1, 0]
    for cond in CONDITIONS:
        plot_med_iqr(ax, cond, "p_theta_rank")
    shade(ax)
    ax.axhline(N_DOF, color="k", linestyle=":", linewidth=0.8, label=f"Full rank ({N_DOF})")
    ax.set_xlabel("Step")
    ax.set_ylabel("rank(P_θ)")
    ax.set_title(
        f"(C) Data FIM rank = rank(P_θ − π_prior·I)  [full rank={N_DOF} → all θ_i identifiable]\n"
        "null-space reaches full rank; VFE-only / dual-λ stuck at rank 1"
    )
    ax.legend(fontsize=8)

    # (D) EE hold error (Phase 1) — key metric: task interference
    ax = axes[1, 1]
    for cond in CONDITIONS:
        mat = np.array([r["ee_hold_err"][:CHANGE_STEP] for r in results[cond]])  # Phase 1 only
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(steps[:CHANGE_STEP], med, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(steps[:CHANGE_STEP], q25, q75, color=COLORS[cond], alpha=0.2)
    ax.set_xlabel("Step (Phase 1 only)")
    ax.set_ylabel("EE hold error (m)")
    ax.set_title(
        "(D) EE displacement from hold goal (Phase 1)\n"
        "null-space: near zero (epistemic motion orthogonal to EE);\n"
        "dual-λ: non-zero trade-off; VFE-only: constant offset (θ wrong)"
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_fig = project_root / "results" / "redundant_arm_calibration.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
