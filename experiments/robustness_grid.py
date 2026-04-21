"""Robustness grid sweep: 3 axes × 3 levels × 20 seeds = 540 runs.

Axes:
  sigma_obs         : {0.01, 0.02, 0.05}   observation noise std
  theta_init_scale  : {0.5, 1.0, 2.0}      scale of initial parameter error
  condition         : {vfe_only, dual_weak, fim_greedy_cost_matched}

theta_init definition:
  delta = [0.4, -0.4]   (base offset from theta_true=[0.5,0.5])
  theta_init(s) = theta_true + s * delta  = [0.5+0.4s, 0.5-0.4s]
  clipped to [0.05, 2.0]

Output:
  results/robustness_grid.json
  results/robustness_grid.png

Usage:
    .venv/bin/python experiments/robustness_grid.py
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
# Fixed system parameters (identical to dual_control_1d_obs.py)
# ---------------------------------------------------------------------------
THETA_TRUE = jnp.array([0.5, 0.5])
Q0 = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET_2 = jnp.array([jnp.pi / 6, jnp.pi / 3])

DT = 0.05
N_STEPS = 100
CHANGE_STEP = 50
N_SEEDS = 20
U_MAX = 0.8
LR_ACTION = 0.05
N_ACTION_ITER = 30

PI_X = 1.0
PARAMS_PRIOR_PI = 1.0
KAPPA_P = 0.5
N_ESTEP_ITER = 3

LAMBDA_WEAK = 0.5
LAMBDA_0 = 3.0
LAMBDA_SCALE = 0.5
FIM_GREEDY_CM_TARGET = 0.667

PARAM_RMSE_FAILURE_THRESHOLD = 0.10
TASK_ERR_FAILURE_THRESHOLD = 0.05

# Grid axes
SIGMA_OBS_VALUES = [0.01, 0.02, 0.05]
THETA_INIT_SCALES = [0.5, 1.0, 2.0]
CONDITIONS = ["vfe_only", "dual_weak", "fim_greedy_cost_matched"]

# Base offset for theta_init: theta_init(s) = theta_true + s * delta.
#
# This direction preserves l1 + l2 for scales 0.5 and 1.0, matching the
# deliberately hard degeneracy used in dual_control_1d_obs.py. In this 1D
# observation model, the FIM-greedy action score depends on d x_ee / d theta,
# which is a function of q but not of theta itself. As a result, the
# cost-matched FIM baseline can produce nearly identical trajectories across
# theta_init_scale values; that is an expected consequence of this stress-test
# geometry rather than an accidental reuse of results.
THETA_INIT_DELTA = jnp.array([0.4, -0.4])

# Precompute the 60 direction candidates for fim_greedy_cost_matched
_N_ANGLES = 60
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _N_ANGLES, endpoint=False)
_TARGET_NORM = float(np.sqrt(FIM_GREEDY_CM_TARGET))
# shape (_N_ANGLES, 2)
_U_CANDIDATES = jnp.array(
    np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1) * _TARGET_NORM
).clip(-U_MAX, U_MAX)

# ---------------------------------------------------------------------------
# Kinematics (2-DOF planar) — copied from dual_control_1d_obs.py
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

# Derived constants
Y_GOAL = fk_1d(Q_TARGET, THETA_TRUE)       # shape (1,)
Y_GOAL_2D = fk_2d(Q_TARGET_2, THETA_TRUE)  # shape (2,)

# ---------------------------------------------------------------------------
# Rollout and FIM helpers
# ---------------------------------------------------------------------------
def rollout_step(q, u):
    return jnp.clip(q + u * DT, -jnp.pi, jnp.pi)

def rollout(q0, u, n_steps=5):
    def step(q, _):
        q_next = rollout_step(q, u)
        return q_next, q_next
    _, qs = jax.lax.scan(step, q0, None, length=n_steps)
    return qs  # (n_steps, 2)

def compute_fim(q, u, theta, pi_y):
    """FIM = J.T @ R_inv @ J, J = d(y_future)/d(theta)."""
    def y_future_fn(th):
        qs = rollout(q, u, n_steps=5)
        return jnp.concatenate([fk_1d(qi, th) for qi in qs])
    J = jax.jacfwd(y_future_fn)(theta)   # (5, 2)
    R_inv = jnp.eye(5) * pi_y
    return J.T @ R_inv @ J

def compute_info_gain(P_theta, fim):
    """IG = 0.5 * (logdet(P + FIM) - logdet(P))."""
    sign1, ld1 = jnp.linalg.slogdet(P_theta + fim)
    sign0, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)

# ---------------------------------------------------------------------------
# Batched FIM-greedy cost-matched (vectorized over all candidate directions)
# ---------------------------------------------------------------------------
def _make_fim_greedy_cm_jit(pi_y):
    """Return a JIT-compiled function that picks the best direction at fixed energy."""
    @jax.jit
    def _fim_greedy_cm(q, theta, u_candidates):
        """Score all candidates and return the best one.

        u_candidates: shape (N, 2)
        Returns: u* shape (2,)
        """
        def score_one(u_cand):
            fim = compute_fim(q, u_cand, theta, pi_y)
            sign, logdet = jnp.linalg.slogdet(fim + 1e-6 * jnp.eye(2))
            return jnp.where(sign > 0, logdet, -1e9)

        scores = jax.vmap(score_one)(u_candidates)   # (N,)
        best_idx = jnp.argmax(scores)
        return u_candidates[best_idx]

    return _fim_greedy_cm

# ---------------------------------------------------------------------------
# Task action optimisers
# ---------------------------------------------------------------------------
PI_Y_TASK = 1.0 / 0.05**2  # task-goal precision (fixed)

def make_optimize_action(pi_y):
    """Return a jit-compiled optimize_action closure for given pi_y."""
    @jax.jit
    def optimize_action(q, theta_est, P_theta, y_goal, lambda_eff, u_init):
        def objective(u):
            q_pred = rollout_step(q, u)
            y_pred = fk_1d(q_pred, theta_est)
            vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)
            fim = compute_fim(q, u, theta_est, pi_y)
            ig = compute_info_gain(P_theta, fim)
            return vfe - lambda_eff * ig

        def descent_step(u, _):
            g = jax.grad(objective)(u)
            return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None

        u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
        return u_opt
    return optimize_action

@jax.jit
def optimize_task_action_2d(q, theta_est, y_goal_2d, u_init):
    """Phase 2: pure 2D task control."""
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
# DEM E-step
# ---------------------------------------------------------------------------
def _build_estep(theta_init, pi_y):
    model = DEMModel(
        f=lambda x, v, p: jnp.zeros(2),
        g=lambda x, v, p: fk_1d(x, p),
        n_x=2, n_v=1, n_y=1, n_order=1,
        pi_y=pi_y, pi_x=PI_X,
        params=theta_init,
        params_prior_pi=PARAMS_PRIOR_PI,
    )
    return EStep(model, kappa_p=KAPPA_P, use_gauss_newton=True)

# ---------------------------------------------------------------------------
# Per (sigma_obs, condition) runner: caches JIT-compiled functions
# ---------------------------------------------------------------------------
class ConditionRunner:
    """Holds JIT-compiled functions for a specific (sigma_obs, condition) pair."""

    def __init__(self, sigma_obs, condition):
        self.sigma_obs = sigma_obs
        self.condition = condition
        pi_y = 1.0 / sigma_obs**2
        self.pi_y = pi_y

        if condition in ("vfe_only", "dual_weak"):
            self.optimize_action = make_optimize_action(pi_y)
            self._fim_greedy_cm = None
        elif condition == "fim_greedy_cost_matched":
            self.optimize_action = None
            self._fim_greedy_cm = _make_fim_greedy_cm_jit(pi_y)
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def run_one(self, seed, theta_init_scale):
        sigma_obs = self.sigma_obs
        condition = self.condition
        pi_y = self.pi_y

        theta_init = jnp.clip(THETA_TRUE + theta_init_scale * THETA_INIT_DELTA, 0.05, 2.0)

        rng = np.random.default_rng(seed)
        theta_est = theta_init.copy()
        P_theta = PARAMS_PRIOR_PI * jnp.eye(2)

        estep = _build_estep(theta_est, pi_y)

        q = Q0.copy()
        u = jnp.zeros(2)

        rmse_hist = []
        task_err_hist = []
        action_energy_phase1 = []

        q_hist = []
        v_hist = []
        y_hist = []

        for t in range(N_STEPS):
            err = theta_est - THETA_TRUE
            rmse_hist.append(float(jnp.sqrt(jnp.mean(err**2))))
            ee_true = fk_2d(q, THETA_TRUE)
            task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_2D) ** 2))))

            if t < CHANGE_STEP:
                if condition == "vfe_only":
                    lambda_eff = 0.0
                    u = self.optimize_action(q, theta_est, P_theta, Y_GOAL, lambda_eff, u)
                elif condition == "dual_weak":
                    lambda_eff = LAMBDA_WEAK
                    u = self.optimize_action(q, theta_est, P_theta, Y_GOAL, lambda_eff, u)
                elif condition == "fim_greedy_cost_matched":
                    u = self._fim_greedy_cm(q, theta_est, _U_CANDIDATES)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                action_energy_phase1.append(float(jnp.sum(u**2)))

                q = rollout_step(q, u)
                y_obs = fk_1d(q, THETA_TRUE) + rng.normal(0, sigma_obs, size=(1,))
                y_obs = jnp.array(y_obs)

                q_hist.append(q)
                v_hist.append(jnp.zeros(1))
                y_hist.append(y_obs)

                if t > 5 and t % 5 == 0:
                    theta_est = estep.run(q_hist, v_hist, y_hist, theta_est, n_iter=N_ESTEP_ITER)
                    theta_est = jnp.clip(theta_est, 0.05, 2.0)
                    P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
                    estep = _build_estep(theta_est, pi_y)
            else:
                u = optimize_task_action_2d(q, theta_est, Y_GOAL_2D, u)
                q = rollout_step(q, u)

        rmse_at_change = rmse_hist[CHANGE_STEP - 1]
        task_err_final = task_err_hist[-1]
        energy_median = float(np.median(action_energy_phase1)) if action_energy_phase1 else 0.0

        return {
            "rmse_at_change": rmse_at_change,
            "task_err_final": task_err_final,
            "action_energy_phase1": energy_median,
        }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # Run grid
    # -----------------------------------------------------------------------
    # results_raw[sigma_obs][scale][cond] = list of per-seed dicts
    results_raw = {}

    for sigma_obs in SIGMA_OBS_VALUES:
        results_raw[sigma_obs] = {}
        for scale in THETA_INIT_SCALES:
            results_raw[sigma_obs][scale] = {}
            for cond in CONDITIONS:
                runner = ConditionRunner(sigma_obs, cond)
                results_raw[sigma_obs][scale][cond] = []
                for seed in range(N_SEEDS):
                    print(f"sigma={sigma_obs:.2f} scale={scale:.1f} {cond} seed={seed}", flush=True)
                    r = runner.run_one(seed, scale)
                    results_raw[sigma_obs][scale][cond].append(r)

    # -----------------------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------------------
    summary = {}
    for sigma_obs in SIGMA_OBS_VALUES:
        summary[sigma_obs] = {}
        for scale in THETA_INIT_SCALES:
            summary[sigma_obs][scale] = {}
            for cond in CONDITIONS:
                runs = results_raw[sigma_obs][scale][cond]
                rmse_vals = np.array([r["rmse_at_change"] for r in runs])
                task_vals = np.array([r["task_err_final"] for r in runs])
                energy_vals = np.array([r["action_energy_phase1"] for r in runs])

                summary[sigma_obs][scale][cond] = {
                    "rmse_at_change_median": float(np.median(rmse_vals)),
                    "task_err_final_median": float(np.median(task_vals)),
                    "rmse_failure_rate": float(np.mean(rmse_vals > PARAM_RMSE_FAILURE_THRESHOLD)),
                    "task_failure_rate": float(np.mean(task_vals > TASK_ERR_FAILURE_THRESHOLD)),
                    "action_energy_phase1_median": float(np.median(energy_vals)),
                }

    # Serialize with string keys for JSON
    def str_key_summary(s):
        out = {}
        for sigma_obs in SIGMA_OBS_VALUES:
            sigma_key = f"{sigma_obs:.2f}"
            out[sigma_key] = {}
            for scale in THETA_INIT_SCALES:
                scale_key = f"{scale:.1f}"
                out[sigma_key][scale_key] = {}
                for cond in CONDITIONS:
                    out[sigma_key][scale_key][cond] = s[sigma_obs][scale][cond]
        return out

    out_json = project_root / "results" / "robustness_grid.json"
    with open(out_json, "w") as f:
        json.dump(str_key_summary(summary), f, indent=2)
    print(f"\nSaved JSON → {out_json}")

    # -----------------------------------------------------------------------
    # Plot: 3×3 grid of subplots, rows=sigma_obs, cols=theta_init_scale
    # Each cell: grouped bar chart (3 conditions), task_err_final_median
    # Failure rate as text annotation on each bar
    # -----------------------------------------------------------------------
    COND_COLORS = {
        "vfe_only": "C3",
        "dual_weak": "C1",
        "fim_greedy_cost_matched": "C0",
    }
    COND_LABELS = {
        "vfe_only": "VFE only",
        "dual_weak": "Dual weak\n(λ=0.5)",
        "fim_greedy_cost_matched": "FIM greedy\n(cost-matched)",
    }

    n_rows = len(SIGMA_OBS_VALUES)
    n_cols = len(THETA_INIT_SCALES)
    n_conds = len(CONDITIONS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 9), sharey=False)
    fig.suptitle(
        "Robustness grid: task error (Phase 2 final)\n"
        "Rows = σ_obs, Columns = θ_init scale; numbers = task failure rate",
        fontsize=10,
    )

    for ri, sigma_obs in enumerate(SIGMA_OBS_VALUES):
        for ci, scale in enumerate(THETA_INIT_SCALES):
            ax = axes[ri][ci]
            for bi, cond in enumerate(CONDITIONS):
                cell = summary[sigma_obs][scale][cond]
                height = cell["task_err_final_median"]
                fail_rate = cell["task_failure_rate"]
                ax.bar(
                    bi, height,
                    color=COND_COLORS[cond],
                    alpha=0.8,
                    width=0.6,
                    label=COND_LABELS[cond] if ri == 0 and ci == 0 else "_nolegend_",
                )
                ax.text(
                    bi, height + 0.002,
                    f"{fail_rate:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

            ax.set_xticks(range(n_conds))
            ax.set_xticklabels(
                [COND_LABELS[c] for c in CONDITIONS],
                fontsize=7,
            )
            if ci == 0:
                ax.set_ylabel(f"σ={sigma_obs:.2f}\ntask err (m)", fontsize=8)
            if ri == 0:
                ax.set_title(f"scale={scale:.1f}", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.3)

    # Legend in top-left cell
    axes[0][0].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_png = project_root / "results" / "robustness_grid.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_png}")


if __name__ == "__main__":
    main()
