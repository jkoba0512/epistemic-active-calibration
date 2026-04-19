"""Dual-control with 1D observation: demonstrates clear epistemic advantage.

The key insight: when observation is 1D (x_ee only) and arm starts near extension,
VFE-only control drives the arm to a fixed (wrong) target and stays there.
The fixed target provides non-informative observations → E-step stalls.
Dual-control with epistemic bonus actively moves q2 to improve FIM rank.

Conditions:
  vfe_only      λ = 0.0   pure task (drives to wrong target, calibration stalls)
  random        u ~ U(-U_MAX, U_MAX) during Phase 1 (task-free excitation)
  scripted_q2   hand-coded q2 excitation during Phase 1 (strong heuristic baseline)
  fim_greedy    D-optimal greedy action maximizing logdet(FIM_future)
  dual_no_precision_feedback
                λ = 3.0   epistemic objective uses fixed prior precision only
  dual_weak     λ = 0.5   mild epistemic
  dual_strong   λ = 3.0   strong epistemic (exploration-heavy)
  dual_adaptive λ = f(P)  high when uncertain, low when calibrated

System:
  q = [q1, q2]   joint angles
  u = [dq1, dq2] velocity command
  θ = [l1, l2]   unknown link lengths
  y = [x_ee]     1D observation (x component only)
  y_goal = x_ee(q_target, θ_true)   scalar target

Usage:
    .venv/bin/python experiments/dual_control_1d_obs.py

Output:
    results/dual_control_1d_obs.png
    results/dual_control_1d_obs_diagnostics.png
    results/dual_control_1d_obs_summary.json
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
THETA_TRUE = jnp.array([0.5, 0.5])    # true link lengths
THETA_INIT = jnp.array([0.9, 0.1])    # initial estimate (badly wrong)
# Q0=[π/3, 0]: with theta_init=[0.9,0.1], x_ee_est = 0.9*cos(π/3)+0.1*cos(π/3) = 0.5 = x_goal
# → VFE-only thinks it's ALREADY AT the goal, never moves → E-step stalls (rank-1 FIM)
# → Epistemic A-step MUST move q2 to break degeneracy
Q0 = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET = jnp.array([jnp.pi / 3, 0.0])   # reachable with true theta at different q config

DT = 0.05
N_STEPS = 100
CHANGE_STEP = 50    # Phase 2 starts here: 2D goal introduced
N_SEEDS = 50
U_MAX = 0.8
LR_ACTION = 0.05
N_ACTION_ITER = 30

# E-step parameters
SIGMA_OBS = 0.02
PI_Y = 1.0 / SIGMA_OBS**2
PI_X = 1.0
PARAMS_PRIOR_PI = 1.0   # weak prior → aggressive calibration
KAPPA_P = 0.5            # Gauss-Newton step size
N_ESTEP_ITER = 3

# Dual-control lambda settings
LAMBDA_0 = 3.0
LAMBDA_SCALE = 0.5
LR_EPISTEMIC = 0.05
N_EPISTEMIC_ITER = 30

CONDITIONS = [
    "vfe_only",
    "random",
    "scripted_q2",
    "fim_greedy",
    "dual_no_precision_feedback",
    "dual_weak",
    "dual_strong",
    "dual_adaptive",
]
LAMBDA_FIXED = {
    "vfe_only": 0.0,
    "random": 0.0,
    "scripted_q2": 0.0,
    "fim_greedy": 0.0,
    "dual_no_precision_feedback": 3.0,
    "dual_weak": 0.5,
    "dual_strong": 3.0,
    "dual_adaptive": None,
}
SCRIPTED_Q2_AMP = 0.75
SCRIPTED_Q2_PERIOD = 18
FIM_GREEDY_ENERGY = 0.05
PARAM_RMSE_FAILURE_THRESHOLD = 0.10
TASK_ERR_FAILURE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Kinematics (2-DOF planar)
# ---------------------------------------------------------------------------
def fk_2d(q, theta):
    """Full 2D EE position."""
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    y = l1 * jnp.sin(q[0]) + l2 * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])

def fk_1d(q, theta):
    """1D observation: x_ee only."""
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    return jnp.array([x])

# Phase 1: 1D calibration goal
Y_GOAL = fk_1d(Q_TARGET, THETA_TRUE)   # shape (1,)

# Phase 2: new 2D task goal (requires correct theta to reach)
# Q_TARGET_2 = [π/6, π/3] is well away from the degenerate q2=0 manifold
Q_TARGET_2 = jnp.array([jnp.pi / 6, jnp.pi / 3])
Y_GOAL_2D = fk_2d(Q_TARGET_2, THETA_TRUE)   # shape (2,)

# ---------------------------------------------------------------------------
# Rollout for FIM computation
# ---------------------------------------------------------------------------
def rollout_step(q, u):
    return jnp.clip(q + u * DT, -jnp.pi, jnp.pi)

def rollout(q0, u, n_steps=5):
    """Rollout n_steps with constant u."""
    def step(q, _):
        q_next = rollout_step(q, u)
        return q_next, q_next
    _, qs = jax.lax.scan(step, q0, None, length=n_steps)
    return qs   # (n_steps, 2)

def compute_fim(q, u, theta):
    """FIM = J.T @ R_inv @ J, J = d(y_future)/d(theta)."""
    def y_future_fn(th):
        qs = rollout(q, u, n_steps=5)
        return jnp.concatenate([fk_1d(qi, th) for qi in qs])
    J = jax.jacfwd(y_future_fn)(theta)   # (5, 2)
    R_inv = jnp.eye(5) * PI_Y
    return J.T @ R_inv @ J

def compute_info_gain(P_theta, fim):
    """IG = 0.5 * (logdet(P + FIM) - logdet(P))."""
    n = P_theta.shape[0]
    sign1, ld1 = jnp.linalg.slogdet(P_theta + fim)
    sign0, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)

def _matrix_diagnostics(mat, threshold=1e-6):
    """Return rank/eigenvalue diagnostics for a small symmetric matrix."""
    eigvals = np.linalg.eigvalsh(np.asarray(mat, dtype=float))
    eigvals = np.maximum(eigvals, 0.0)
    rank = int(np.sum(eigvals > threshold))
    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])
    condition = float(max_eig / max(min_eig, threshold))
    mat_jittered = np.asarray(mat, dtype=float) + threshold * np.eye(mat.shape[0])
    sign, logdet = np.linalg.slogdet(mat_jittered)
    return {
        "rank": rank,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "condition": condition,
        "logdet": float(logdet if sign > 0 else np.nan),
    }

def _summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "iqr25": float(np.percentile(arr, 25)),
        "iqr75": float(np.percentile(arr, 75)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }

def scripted_q2_action(t):
    """Deterministic q2 excitation baseline with no epistemic feedback."""
    phase = 2.0 * np.pi * (t + 1) / SCRIPTED_Q2_PERIOD
    return jnp.array([0.0, SCRIPTED_Q2_AMP * np.sin(phase)])

def optimize_fim_greedy_action(q, theta_est):
    """D-optimal greedy baseline: choose action maximizing logdet(FIM_future).

    This is intentionally not an Active Inference controller. It is a simple
    OED-style comparison that asks whether a direct FIM-greedy heuristic is
    sufficient to break the 1D observation degeneracy.
    """
    grid = np.linspace(-U_MAX, U_MAX, 5)
    best_score = -np.inf
    best_u = np.zeros(2)
    for u1 in grid:
        for u2 in grid:
            u_cand = jnp.array([u1, u2])
            fim = compute_fim(q, u_cand, theta_est)
            sign, logdet = jnp.linalg.slogdet(fim + 1e-6 * jnp.eye(2))
            score = float(logdet - FIM_GREEDY_ENERGY * jnp.sum(u_cand**2))
            if sign > 0 and score > best_score:
                best_score = score
                best_u = np.array([u1, u2])
    return jnp.array(best_u)

# ---------------------------------------------------------------------------
# Action optimisation
# ---------------------------------------------------------------------------
PI_Y_TASK = 1.0 / 0.05**2   # task-goal precision

@jax.jit
def optimize_action(q, theta_est, P_theta, y_goal, lambda_eff, u_init):
    """Gradient descent on J = VFE - lambda_eff * IG."""
    def objective(u):
        q_pred = rollout_step(q, u)
        y_pred = fk_1d(q_pred, theta_est)
        vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)
        fim = compute_fim(q, u, theta_est)
        ig = compute_info_gain(P_theta, fim)
        return vfe - lambda_eff * ig

    def descent_step(u, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
    return u_opt

@jax.jit
def optimize_task_action_2d(q, theta_est, y_goal_2d, u_init):
    """Phase 2: pure 2D task control with calibrated theta_est."""
    def objective(u):
        q_pred = rollout_step(q, u)
        y_pred = fk_2d(q_pred, theta_est)
        return 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal_2d) ** 2)

    def descent_step(u, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
    return u_opt

# ---------------------------------------------------------------------------
# DEM E-step setup
# ---------------------------------------------------------------------------
def _build_model(theta_init):
    return DEMModel(
        f=lambda x, v, p: jnp.zeros(2),
        g=lambda x, v, p: fk_1d(x, p),
        n_x=2, n_v=1, n_y=1, n_order=1,
        pi_y=PI_Y, pi_x=PI_X,
        params=theta_init,
        params_prior_pi=PARAMS_PRIOR_PI,
    )

def _build_estep(theta_init):
    model = _build_model(theta_init)
    return EStep(model, kappa_p=KAPPA_P, use_gauss_newton=True)

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(seed, condition):
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(2)

    estep = _build_estep(theta_est)

    q = Q0.copy()
    u = jnp.zeros(2)

    rmse_hist = []
    task_err_hist = []   # Phase 2: true 2D task error
    q2_hist = []
    lambda_hist = []
    action_energy_hist = []
    fim_rank_hist = []
    fim_logdet_hist = []
    fim_min_eig_hist = []
    precision_rank_hist = []
    precision_logdet_hist = []
    precision_min_eig_hist = []
    precision_condition_hist = []
    data_rank_hist = []
    data_logdet_hist = []
    data_min_eig_hist = []
    info_gain_hist = []

    q_hist = []
    v_hist = []
    y_hist = []

    for t in range(N_STEPS):
        # --- Record metrics ---
        err = theta_est - THETA_TRUE
        rmse_hist.append(float(jnp.sqrt(jnp.mean(err**2))))
        q2_hist.append(float(q[1]))

        # True 2D task error (uses ground-truth theta)
        ee_true = fk_2d(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_2D) ** 2))))

        # --- Phase 1 (t < CHANGE_STEP): 1D calibration / exploration ---
        if t < CHANGE_STEP:
            if condition == "random":
                lambda_eff = 0.0
                u = jnp.array(rng.uniform(-U_MAX, U_MAX, size=(2,)))
            elif condition == "scripted_q2":
                lambda_eff = 0.0
                u = scripted_q2_action(t)
            elif condition == "fim_greedy":
                lambda_eff = 0.0
                u = optimize_fim_greedy_action(q, theta_est)
            else:
                lf = LAMBDA_FIXED[condition]
                if lf is None:
                    lam_min = float(jnp.linalg.eigvalsh(P_theta)[0])
                    lambda_eff = LAMBDA_0 / (1.0 + lam_min / LAMBDA_SCALE)
                else:
                    lambda_eff = lf
                P_for_action = (
                    PARAMS_PRIOR_PI * jnp.eye(2)
                    if condition == "dual_no_precision_feedback"
                    else P_theta
                )
                u = optimize_action(q, theta_est, P_for_action, Y_GOAL, lambda_eff, u)
            lambda_hist.append(float(lambda_eff))
            action_energy_hist.append(float(jnp.sum(u**2)))

            fim_action = compute_fim(q, u, theta_est)
            ig_action = compute_info_gain(P_theta, fim_action)
            fim_diag = _matrix_diagnostics(fim_action)
            precision_diag = _matrix_diagnostics(P_theta)
            data_info = P_theta - PARAMS_PRIOR_PI * jnp.eye(2)
            data_diag = _matrix_diagnostics(data_info)
            fim_rank_hist.append(fim_diag["rank"])
            fim_logdet_hist.append(fim_diag["logdet"])
            fim_min_eig_hist.append(fim_diag["min_eig"])
            precision_rank_hist.append(precision_diag["rank"])
            precision_logdet_hist.append(precision_diag["logdet"])
            precision_min_eig_hist.append(precision_diag["min_eig"])
            precision_condition_hist.append(precision_diag["condition"])
            data_rank_hist.append(data_diag["rank"])
            data_logdet_hist.append(data_diag["logdet"])
            data_min_eig_hist.append(data_diag["min_eig"])
            info_gain_hist.append(float(ig_action))

            q = rollout_step(q, u)
            y_obs = fk_1d(q, THETA_TRUE) + rng.normal(0, SIGMA_OBS, size=(1,))
            y_obs = jnp.array(y_obs)

            q_hist.append(q)
            v_hist.append(jnp.zeros(1))
            y_hist.append(y_obs)

            # E-step (every 5 steps after warmup)
            if t > 5 and t % 5 == 0:
                theta_est = estep.run(q_hist, v_hist, y_hist, theta_est, n_iter=N_ESTEP_ITER)
                theta_est = jnp.clip(theta_est, 0.05, 2.0)
                P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
                estep = _build_estep(theta_est)

        # --- Phase 2 (t >= CHANGE_STEP): 2D task with calibrated theta ---
        else:
            lambda_hist.append(0.0)
            u = optimize_task_action_2d(q, theta_est, Y_GOAL_2D, u)
            action_energy_hist.append(float(jnp.sum(u**2)))
            fim_action = compute_fim(q, u, theta_est)
            ig_action = compute_info_gain(P_theta, fim_action)
            fim_diag = _matrix_diagnostics(fim_action)
            precision_diag = _matrix_diagnostics(P_theta)
            data_info = P_theta - PARAMS_PRIOR_PI * jnp.eye(2)
            data_diag = _matrix_diagnostics(data_info)
            fim_rank_hist.append(fim_diag["rank"])
            fim_logdet_hist.append(fim_diag["logdet"])
            fim_min_eig_hist.append(fim_diag["min_eig"])
            precision_rank_hist.append(precision_diag["rank"])
            precision_logdet_hist.append(precision_diag["logdet"])
            precision_min_eig_hist.append(precision_diag["min_eig"])
            precision_condition_hist.append(precision_diag["condition"])
            data_rank_hist.append(data_diag["rank"])
            data_logdet_hist.append(data_diag["logdet"])
            data_min_eig_hist.append(data_diag["min_eig"])
            info_gain_hist.append(float(ig_action))
            q = rollout_step(q, u)

    return {
        "rmse": np.array(rmse_hist),
        "task_err": np.array(task_err_hist),
        "q2": np.array(q2_hist),
        "lambda": np.array(lambda_hist),
        "action_energy": np.array(action_energy_hist),
        "fim_rank": np.array(fim_rank_hist),
        "fim_logdet": np.array(fim_logdet_hist),
        "fim_min_eig": np.array(fim_min_eig_hist),
        "precision_rank": np.array(precision_rank_hist),
        "precision_logdet": np.array(precision_logdet_hist),
        "precision_min_eig": np.array(precision_min_eig_hist),
        "precision_condition": np.array(precision_condition_hist),
        "data_rank": np.array(data_rank_hist),
        "data_logdet": np.array(data_logdet_hist),
        "data_min_eig": np.array(data_min_eig_hist),
        "info_gain": np.array(info_gain_hist),
        "theta_final": np.array(theta_est),
        "frac_near_zero_q2": float(np.mean(np.abs(q2_hist[:CHANGE_STEP]) < 0.1)),
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
        print(f"Running {cond} ...")
        for seed in range(N_SEEDS):
            r = run_one(seed, cond)
            results[cond].append(r)
            print(f"  seed={seed:2d}  RMSE_final={r['rmse'][-1]:.4f}"
                  f"  mean|q2|={np.mean(np.abs(r['q2'])):.3f}"
                  f"  theta_final={r['theta_final']}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(
        f"  {'Condition':15s}  {'RMSE@50':8s}  {'TaskErr@100':11s}  "
        f"{'mean|q2|Ph1':11s}  {'EnergyPh1':9s}  {'failRMSE':8s}  {'failTask':8s}"
    )
    summary = {}
    for cond in CONDITIONS:
        rmse_at_change = np.array([r["rmse"][CHANGE_STEP - 1] for r in results[cond]])
        task_err_final = np.array([r["task_err"][-1] for r in results[cond]])
        q2_means = np.array([np.mean(np.abs(r["q2"][:CHANGE_STEP])) for r in results[cond]])
        energy_ph1 = np.array([np.mean(r["action_energy"][:CHANGE_STEP]) for r in results[cond]])
        frac_zero = np.array([r["frac_near_zero_q2"] for r in results[cond]])
        data_rank_final = np.array([r["data_rank"][CHANGE_STEP - 1] for r in results[cond]])
        info_gain_ph1 = np.array([np.mean(r["info_gain"][:CHANGE_STEP]) for r in results[cond]])
        precision_cond_final = np.array([r["precision_condition"][CHANGE_STEP - 1] for r in results[cond]])
        rmse_failure_rate = float(np.mean(rmse_at_change > PARAM_RMSE_FAILURE_THRESHOLD))
        task_failure_rate = float(np.mean(task_err_final > TASK_ERR_FAILURE_THRESHOLD))
        summary[cond] = {
            "n_seeds": N_SEEDS,
            "rmse_at_change": _summarize(rmse_at_change),
            "task_err_final": _summarize(task_err_final),
            "mean_abs_q2_phase1": _summarize(q2_means),
            "action_energy_phase1": _summarize(energy_ph1),
            "frac_near_zero_q2": _summarize(frac_zero),
            "data_rank_at_change": _summarize(data_rank_final),
            "info_gain_phase1": _summarize(info_gain_ph1),
            "precision_condition_at_change": _summarize(precision_cond_final),
            "rmse_failure_rate": rmse_failure_rate,
            "task_failure_rate": task_failure_rate,
            # Backward-compatible flat medians for README/tutorial snippets.
            "rmse_at_change_median": float(np.median(rmse_at_change)),
            "task_err_final_median": float(np.median(task_err_final)),
            "mean_abs_q2_phase1_median": float(np.median(q2_means)),
            "action_energy_phase1_median": float(np.median(energy_ph1)),
            "frac_near_zero_q2_median": float(np.median(frac_zero)),
        }
        print(f"  {cond:15s}  {np.median(rmse_at_change):.4f}    "
              f"{np.median(task_err_final):.4f}       "
              f"{np.median(q2_means):.3f}        "
              f"{np.median(energy_ph1):.3f}      "
              f"{rmse_failure_rate:.2f}      {task_failure_rate:.2f}")

    out_json = project_root / "results" / "dual_control_1d_obs_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Dual-control: 1D calibration (x_ee) → 2D task execution\n"
        r"$\theta_{init}=[0.9,0.1]$, $\theta_{true}=[0.5,0.5]$  |  "
        f"Phase 1: steps 0–{CHANGE_STEP-1}  |  Phase 2: steps {CHANGE_STEP}–{N_STEPS-1}",
        fontsize=10,
    )

    COLORS = {
        "vfe_only": "C3",
        "random": "C4",
        "scripted_q2": "C5",
        "fim_greedy": "C7",
        "dual_no_precision_feedback": "C6",
        "dual_weak": "C1",
        "dual_strong": "C0",
        "dual_adaptive": "C2",
    }
    LABELS = {
        "vfe_only": "VFE only (λ=0)",
        "random": "Random excitation",
        "scripted_q2": "Scripted q2",
        "fim_greedy": "FIM greedy",
        "dual_no_precision_feedback": "Dual no precision feedback",
        "dual_weak": "Dual weak (λ=0.5)",
        "dual_strong": "Dual strong (λ=3.0)",
        "dual_adaptive": "Dual adaptive",
    }

    steps = np.arange(N_STEPS)

    def shade_phase2(ax):
        ax.axvspan(CHANGE_STEP, N_STEPS, color="gray", alpha=0.10, label="Phase 2 (2D task)")
        ax.axvline(CHANGE_STEP, color="gray", linestyle="--", linewidth=0.8)

    # --- (A) RMSE curves ---
    ax = axes[0, 0]
    for cond in CONDITIONS:
        rmse_mat = np.array([r["rmse"] for r in results[cond]])
        med = np.median(rmse_mat, axis=0)
        q25 = np.percentile(rmse_mat, 25, axis=0)
        q75 = np.percentile(rmse_mat, 75, axis=0)
        ax.plot(steps, med, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(steps, q25, q75, color=COLORS[cond], alpha=0.2)
    shade_phase2(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE (θ)")
    ax.set_title("(A) Parameter RMSE  [lower = better calibration]")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # --- (B) True 2D task error ---
    ax = axes[0, 1]
    for cond in CONDITIONS:
        te_mat = np.array([r["task_err"] for r in results[cond]])
        med = np.median(te_mat, axis=0)
        q25 = np.percentile(te_mat, 25, axis=0)
        q75 = np.percentile(te_mat, 75, axis=0)
        ax.plot(steps, med, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(steps, q25, q75, color=COLORS[cond], alpha=0.2)
    shade_phase2(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("EE distance (m)")
    ax.set_title("(B) True 2D task error  [lower = better task achievement]")
    ax.legend(fontsize=7)

    # --- (C) Elbow angle q2 ---
    ax = axes[1, 0]
    for cond in CONDITIONS:
        q2_mat = np.array([r["q2"] for r in results[cond]])
        med = np.median(q2_mat, axis=0)
        ax.plot(steps, med, color=COLORS[cond], label=LABELS[cond])
    shade_phase2(ax)
    ax.axhline(0, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("q2 (rad)")
    ax.set_title("(C) Elbow angle q2  [near-zero → rank-1 FIM → calibration fails]")
    ax.legend(fontsize=7)

    # --- (D) Task error boxplot at final step ---
    ax = axes[1, 1]
    lam_labels = {
        "vfe_only": "0",
        "random": "rand",
        "scripted_q2": "script",
        "fim_greedy": "fim",
        "dual_no_precision_feedback": "3/noP",
        "dual_weak": "0.5",
        "dual_strong": "3.0",
        "dual_adaptive": "adpt",
    }
    task_final = [[r["task_err"][-1] for r in results[cond]] for cond in CONDITIONS]
    bp = ax.boxplot(task_final, tick_labels=CONDITIONS, patch_artist=True)
    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.7)
    ax.set_ylabel("Final task error (m)")
    ax.set_title("(D) Final 2D task error  [lower = arm reached true goal]")
    ax.set_xticklabels([f"{c}\n(λ={lam_labels[c]})" for c in CONDITIONS], fontsize=8)

    plt.tight_layout()
    out = project_root / "results" / "dual_control_1d_obs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out}")

    # -----------------------------------------------------------------------
    # Mechanism diagnostics: does excitation precede precision/rank/reaching?
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Mechanism diagnostics: excitation → information rank/precision → calibration → task",
        fontsize=11,
    )

    def plot_median_iqr(ax, cond, key, transform=lambda x: x):
        mat = np.array([transform(r[key]) for r in results[cond]])
        med = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        ax.plot(steps, med, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(steps, q25, q75, color=COLORS[cond], alpha=0.12)

    diagnostic_specs = [
        (axes[0, 0], "q2", lambda x: np.abs(x), "abs(q2) (rad)",
         "(A) Elbow excitation"),
        (axes[0, 1], "fim_rank", lambda x: x, "rank(FIM_future(action))",
         "(B) Selected-action FIM rank"),
        (axes[1, 0], "precision_min_eig", lambda x: x, "min eig(P_theta)",
         "(C) Weakest posterior precision direction"),
        (axes[1, 1], "info_gain", lambda x: x, "IG of selected action",
         "(D) Action information gain"),
        (axes[2, 0], "rmse", lambda x: x, "Parameter RMSE",
         "(E) Calibration error"),
        (axes[2, 1], "task_err", lambda x: x, "2D task error (m)",
         "(F) Downstream task error"),
    ]

    for ax, key, transform, ylabel, title in diagnostic_specs:
        for cond in CONDITIONS:
            plot_median_iqr(ax, cond, key, transform)
        shade_phase2(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3)
    axes[2, 0].set_yscale("log")
    for ax in axes[2, :]:
        ax.set_xlabel("Step")
    axes[0, 0].legend(fontsize=6, ncols=2)

    plt.tight_layout()
    out_diag = project_root / "results" / "dual_control_1d_obs_diagnostics.png"
    plt.savefig(out_diag, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostics figure → {out_diag}")


if __name__ == "__main__":
    main()
