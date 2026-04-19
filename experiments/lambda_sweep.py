"""Lambda sweep: epistemic weight λ vs calibration and task performance.

Sweeps λ ∈ {0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0} plus adaptive λ,
using the same degenerate 1D-observation scenario as dual_control_1d_obs.py.

Degenerate scenario:
  Q0 = Q_TARGET = [π/3, 0], theta_init=[0.9,0.1], theta_true=[0.5,0.5]
  At Q0, x_ee_est = x_ee_true = 0.5 = x_goal → VFE=0 → arm never moves (λ=0 case)

Two-phase protocol:
  Phase 1 (t=0..CHANGE_STEP-1): 1D VFE + λ·IG calibration
  Phase 2 (t=CHANGE_STEP..N_STEPS-1): 2D task execution with calibrated θ_est

Metrics:
  - RMSE(θ) at step CHANGE_STEP  (calibration quality)
  - Task error at final step      (task achievement after calibration)
  - AUC_RMSE over Phase 1         (cumulative calibration speed)

Usage:
    .venv/bin/python experiments/lambda_sweep.py

Output:
    results/lambda_sweep.png
    results/lambda_sweep.json
"""

import sys, json
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
# System parameters (identical to dual_control_1d_obs.py)
# ---------------------------------------------------------------------------
THETA_TRUE = jnp.array([0.5, 0.5])
THETA_INIT = jnp.array([0.9, 0.1])
Q0         = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET   = jnp.array([jnp.pi / 3, 0.0])
Q_TARGET_2 = jnp.array([jnp.pi / 6, jnp.pi / 3])

DT          = 0.05
N_STEPS     = 100
CHANGE_STEP = 50
N_SEEDS     = 20
U_MAX       = 0.8
LR_ACTION   = 0.05
N_ACTION_ITER = 30

SIGMA_OBS      = 0.02
PI_Y           = 1.0 / SIGMA_OBS**2
PI_X           = 1.0
PARAMS_PRIOR_PI = 1.0
KAPPA_P        = 0.5
N_ESTEP_ITER   = 3

LAMBDA_0     = 3.0
LAMBDA_SCALE = 0.5

PI_Y_TASK = 1.0 / 0.05**2

# λ sweep values (fixed) + adaptive
LAMBDA_FIXED_VALS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
LABEL_ADAPTIVE    = "adaptive"

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def fk_2d(q, theta):
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    y = l1 * jnp.sin(q[0]) + l2 * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])

def fk_1d(q, theta):
    l1, l2 = theta[0], theta[1]
    x = l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1])
    return jnp.array([x])

Y_GOAL    = fk_1d(Q_TARGET,   THETA_TRUE)
Y_GOAL_2D = fk_2d(Q_TARGET_2, THETA_TRUE)

# ---------------------------------------------------------------------------
# Rollout / FIM / IG
# ---------------------------------------------------------------------------
def rollout_step(q, u):
    return jnp.clip(q + u * DT, -jnp.pi, jnp.pi)

def rollout(q0, u, n_steps=5):
    def step(q, _):
        q_next = rollout_step(q, u)
        return q_next, q_next
    _, qs = jax.lax.scan(step, q0, None, length=n_steps)
    return qs

def compute_fim(q, u, theta):
    def y_future_fn(th):
        qs = rollout(q, u, n_steps=5)
        return jnp.concatenate([fk_1d(qi, th) for qi in qs])
    J = jax.jacfwd(y_future_fn)(theta)
    R_inv = jnp.eye(5) * PI_Y
    return J.T @ R_inv @ J

def compute_info_gain(P_theta, fim):
    _, ld1 = jnp.linalg.slogdet(P_theta + fim)
    _, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)

# ---------------------------------------------------------------------------
# Action optimizers
# ---------------------------------------------------------------------------
@jax.jit
def optimize_action(q, theta_est, P_theta, y_goal, lambda_eff, u_init):
    def objective(u):
        q_pred = rollout_step(q, u)
        y_pred = fk_1d(q_pred, theta_est)
        vfe = 0.5 * PI_Y_TASK * jnp.sum((y_pred - y_goal) ** 2)
        fim = compute_fim(q, u, theta_est)
        ig  = compute_info_gain(P_theta, fim)
        return vfe - lambda_eff * ig
    def descent_step(u, _):
        g = jax.grad(objective)(u)
        return jnp.clip(u - LR_ACTION * g, -U_MAX, U_MAX), None
    u_opt, _ = jax.lax.scan(descent_step, u_init, None, length=N_ACTION_ITER)
    return u_opt

@jax.jit
def optimize_task_action_2d(q, theta_est, y_goal_2d, u_init):
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
# E-step
# ---------------------------------------------------------------------------
def _build_estep(theta_est):
    model = DEMModel(
        f=lambda x, v, p: jnp.zeros(2),
        g=lambda x, v, p: fk_1d(x, p),
        n_x=2, n_v=1, n_y=1, n_order=1,
        pi_y=PI_Y, pi_x=PI_X,
        params=theta_est,
        params_prior_pi=PARAMS_PRIOR_PI,
    )
    return EStep(model, kappa_p=KAPPA_P, use_gauss_newton=True)

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(seed, lambda_val):
    """lambda_val: float (fixed) or None (adaptive)."""
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta   = PARAMS_PRIOR_PI * jnp.eye(2)
    estep     = _build_estep(theta_est)
    q         = Q0.copy()
    u         = jnp.zeros(2)

    rmse_hist    = []
    task_err_hist = []
    q2_hist      = []
    lambda_hist  = []
    q_hist, v_hist, y_hist = [], [], []

    for t in range(N_STEPS):
        err = theta_est - THETA_TRUE
        rmse_hist.append(float(jnp.sqrt(jnp.mean(err**2))))
        ee_true = fk_2d(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_2D)**2))))
        q2_hist.append(float(q[1]))

        if t < CHANGE_STEP:
            # --- Phase 1: 1D VFE + epistemic ---
            if lambda_val is None:
                lam_min    = float(jnp.linalg.eigvalsh(P_theta)[0])
                lambda_eff = LAMBDA_0 / (1.0 + lam_min / LAMBDA_SCALE)
            else:
                lambda_eff = float(lambda_val)
            lambda_hist.append(lambda_eff)

            u = optimize_action(q, theta_est, P_theta, Y_GOAL, lambda_eff, u)
            q = rollout_step(q, u)
            y_obs = fk_1d(q, THETA_TRUE) + rng.normal(0, SIGMA_OBS, size=(1,))
            y_obs = jnp.array(y_obs)

            q_hist.append(q); v_hist.append(jnp.zeros(1)); y_hist.append(y_obs)

            if t > 5 and t % 5 == 0:
                theta_est = estep.run(q_hist, v_hist, y_hist, theta_est,
                                      n_iter=N_ESTEP_ITER)
                theta_est = jnp.clip(theta_est, 0.05, 2.0)
                P_theta   = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
                estep     = _build_estep(theta_est)
        else:
            # --- Phase 2: 2D task with calibrated θ ---
            lambda_hist.append(0.0)
            u = optimize_task_action_2d(q, theta_est, Y_GOAL_2D, u)
            q = rollout_step(q, u)

    return {
        "rmse":          np.array(rmse_hist),
        "task_err":      np.array(task_err_hist),
        "q2":            np.array(q2_hist),
        "lambda_traj":   np.array(lambda_hist),
        "theta_final":   np.array(theta_est),
        "rmse_at_change":  float(rmse_hist[CHANGE_STEP - 1]),
        "task_err_final":  float(task_err_hist[-1]),
        "auc_rmse_ph1":    float(np.mean(rmse_hist[:CHANGE_STEP])),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_lambda_vals  = LAMBDA_FIXED_VALS + [None]   # None = adaptive
    all_lambda_labels = [str(lv) for lv in LAMBDA_FIXED_VALS] + [LABEL_ADAPTIVE]

    results = {}

    for lv, label in zip(all_lambda_vals, all_lambda_labels):
        print(f"λ = {label:8s} ", end="", flush=True)
        runs = []
        for seed in range(N_SEEDS):
            r = run_one(seed, lv)
            runs.append(r)
            print(".", end="", flush=True)
        results[label] = runs
        med_rmse = np.median([r["rmse_at_change"]  for r in runs])
        med_task = np.median([r["task_err_final"]  for r in runs])
        med_auc  = np.median([r["auc_rmse_ph1"]    for r in runs])
        print(f"  RMSE@{CHANGE_STEP}={med_rmse:.4f}  TaskErr@final={med_task:.4f}  AUC={med_auc:.4f}")

    # -----------------------------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------------------------
    summary = {}
    for label in all_lambda_labels:
        runs = results[label]
        summary[label] = {
            "rmse_at_change_median": float(np.median([r["rmse_at_change"] for r in runs])),
            "rmse_at_change_q25":    float(np.percentile([r["rmse_at_change"] for r in runs], 25)),
            "rmse_at_change_q75":    float(np.percentile([r["rmse_at_change"] for r in runs], 75)),
            "task_err_final_median": float(np.median([r["task_err_final"] for r in runs])),
            "task_err_final_q25":    float(np.percentile([r["task_err_final"] for r in runs], 25)),
            "task_err_final_q75":    float(np.percentile([r["task_err_final"] for r in runs], 75)),
            "auc_rmse_ph1_median":   float(np.median([r["auc_rmse_ph1"] for r in runs])),
        }
    out_json = project_root / "results" / "lambda_sweep.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON → {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fixed_labels = [str(lv) for lv in LAMBDA_FIXED_VALS]
    fixed_x      = LAMBDA_FIXED_VALS

    def extract(metric, labels=fixed_labels):
        med = [np.median([r[metric] for r in results[lb]]) for lb in labels]
        q25 = [np.percentile([r[metric] for r in results[lb]], 25) for lb in labels]
        q75 = [np.percentile([r[metric] for r in results[lb]], 75) for lb in labels]
        return np.array(med), np.array(q25), np.array(q75)

    adpt_rmse = np.median([r["rmse_at_change"]  for r in results[LABEL_ADAPTIVE]])
    adpt_task = np.median([r["task_err_final"]  for r in results[LABEL_ADAPTIVE]])
    adpt_auc  = np.median([r["auc_rmse_ph1"]    for r in results[LABEL_ADAPTIVE]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "λ sweep: epistemic weight vs performance\n"
        r"1D obs, degenerate scenario ($Q_0=[\pi/3,0]$, $\theta_{init}=[0.9,0.1]$, $\theta_{true}=[0.5,0.5]$)",
        fontsize=10,
    )

    ADPT_COLOR = "C2"
    FIXED_COLOR = "C0"

    # --- (A) RMSE at CHANGE_STEP ---
    ax = axes[0]
    med, q25, q75 = extract("rmse_at_change")
    ax.plot(fixed_x, med, "o-", color=FIXED_COLOR, label="Fixed λ")
    ax.fill_between(fixed_x, q25, q75, color=FIXED_COLOR, alpha=0.2)
    ax.axhline(adpt_rmse, color=ADPT_COLOR, linestyle="--", label=f"Adaptive λ (median)")
    ax.set_xlabel("λ (fixed)")
    ax.set_ylabel(f"RMSE(θ) at step {CHANGE_STEP}")
    ax.set_title(f"(A) Calibration quality at step {CHANGE_STEP}")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # --- (B) Task error at final step ---
    ax = axes[1]
    med, q25, q75 = extract("task_err_final")
    ax.plot(fixed_x, med, "o-", color=FIXED_COLOR, label="Fixed λ")
    ax.fill_between(fixed_x, q25, q75, color=FIXED_COLOR, alpha=0.2)
    ax.axhline(adpt_task, color=ADPT_COLOR, linestyle="--", label=f"Adaptive λ (median)")
    ax.set_xlabel("λ (fixed)")
    ax.set_ylabel("2D task error (m) at final step")
    ax.set_title("(B) Task achievement at final step")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # --- (C) AUC RMSE Phase 1 ---
    ax = axes[2]
    med, q25, q75 = extract("auc_rmse_ph1")
    ax.plot(fixed_x, med, "o-", color=FIXED_COLOR, label="Fixed λ")
    ax.fill_between(fixed_x, q25, q75, color=FIXED_COLOR, alpha=0.2)
    ax.axhline(adpt_auc, color=ADPT_COLOR, linestyle="--", label=f"Adaptive λ (median)")
    ax.set_xlabel("λ (fixed)")
    ax.set_ylabel("Mean RMSE(θ) over Phase 1")
    ax.set_title("(C) Cumulative calibration speed (AUC)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_png = project_root / "results" / "lambda_sweep.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_png}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n=== Summary table ===")
    print(f"  {'λ':12s}  {'RMSE@'+str(CHANGE_STEP):10s}  {'TaskErr@final':13s}  {'AUC_RMSE':8s}")
    for label in all_lambda_labels:
        s = summary[label]
        print(f"  {label:12s}  {s['rmse_at_change_median']:.4f}      "
              f"{s['task_err_final_median']:.4f}         "
              f"{s['auc_rmse_ph1_median']:.4f}")


if __name__ == "__main__":
    main()
