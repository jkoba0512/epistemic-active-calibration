"""Terminal control risk diagnostics.

For each condition and seed, runs Phase 1 to collect q_change and theta_hat_change,
then computes geometric risk metrics at the phase transition.  Compares how well
theta_rmse vs terminal risk metrics separate task failures.

Output:
    results/terminal_control_risk_diagnostics.json
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (
    run_one,
    fk,
    compute_vfe_only_action,
    compute_posture_task_action,
    rollout_step,
    THETA_TRUE,
    Y_GOAL_TASK,
    CHANGE_STEP,
    N_SEEDS,
    PARAM_RMSE_FAIL,
    TASK_ERR_FAIL,
    N_DOF,
)

DIAG_CONDITIONS = [
    "null_space",
    "null_space_probe",
    "null_space_probe_posture",
    "null_space_probe_recovery_posture",
    "null_space_posture",
    "vfe_only",
    "dual_lambda",
    "null_space_recovery",
    "null_space_probe_recovery",
]

ROLLOUT_HORIZONS = [20, 50, 200]
DAMPING = 1e-3
EPS = 1e-6


# ---------------------------------------------------------------------------
# Risk metric computation
# ---------------------------------------------------------------------------

def _damped_pseudoinverse(J):
    task_dim = J.shape[0]
    return J.T @ jnp.linalg.inv(J @ J.T + DAMPING * jnp.eye(task_dim))


def compute_risk_metrics(q, theta):
    """Geometric terminal control risk metrics at (q, theta)."""
    J = jax.jacfwd(lambda qi: fk(qi, theta))(q)   # (2, 4)
    J_hash = _damped_pseudoinverse(J)

    svals = jnp.linalg.svd(J, compute_uv=False)
    sigma_min = float(jnp.min(svals))
    sigma_max = float(jnp.max(svals))
    manip = float(jnp.prod(svals))
    cond = sigma_max / max(sigma_min, EPS)

    r_sing = 1.0 / (sigma_min + EPS)
    r_gain = float(jnp.linalg.norm(J_hash, ord=2))

    e_task = Y_GOAL_TASK - fk(q, theta)
    r_step = float(jnp.linalg.norm(J_hash @ e_task))

    return {
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": float(cond),
        "manipulability": float(manip),
        "r_sing": float(r_sing),
        "r_gain": float(r_gain),
        "r_step": float(r_step),
    }


def rollout_risk(q_start, theta_control, theta_eval, ctrl_fn, horizon):
    """Noiseless K-step rollout; returns final EE distance to Y_GOAL_TASK."""
    q = jnp.array(q_start)
    for _ in range(horizon):
        u = ctrl_fn(q, theta_control)
        q = rollout_step(q, u)
    return float(jnp.linalg.norm(fk(q, theta_eval) - Y_GOAL_TASK))


def _add_rollout_risks(risk_dict, q, theta_control, theta_eval):
    plain_fn = lambda q, th: compute_vfe_only_action(q, th, Y_GOAL_TASK)
    posture_fn = lambda q, th: compute_posture_task_action(q, th, Y_GOAL_TASK)
    for h in ROLLOUT_HORIZONS:
        risk_dict[f"r_rollout_{h}_plain"] = rollout_risk(
            q, theta_control, theta_eval, plain_fn, h
        )
        risk_dict[f"r_rollout_{h}_posture"] = rollout_risk(
            q, theta_control, theta_eval, posture_fn, h
        )


# ---------------------------------------------------------------------------
# AUC / separation statistics
# ---------------------------------------------------------------------------

def rank_auc(scores, labels):
    """AUC = P(score_pos > score_neg) via Mann-Whitney rank statistic."""
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    u = float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    return u / (len(pos) * len(neg))


def pearson_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _separation_stats(per_seed, task_failed_arr):
    """Compute correlation and AUC for each risk key vs task_failed."""
    theta_rmse = np.array([s["theta_rmse_at_change"] for s in per_seed])
    labels = task_failed_arr.astype(int)

    risk_keys_hat = list(per_seed[0]["risk_hat"].keys())
    risk_keys_true = list(per_seed[0]["risk_true"].keys())

    sep = {}

    def _add(name, scores):
        sep[name] = {
            "corr": pearson_corr(scores, labels),
            "auc": rank_auc(np.asarray(scores, dtype=float), labels),
            "success_median": float(np.median(scores[~task_failed_arr])) if np.any(~task_failed_arr) else float("nan"),
            "failed_median": float(np.median(scores[task_failed_arr])) if np.any(task_failed_arr) else float("nan"),
        }

    _add("theta_rmse", theta_rmse)
    for k in risk_keys_hat:
        scores = np.array([s["risk_hat"][k] for s in per_seed])
        _add(f"hat_{k}", scores)
    for k in risk_keys_true:
        scores = np.array([s["risk_true"][k] for s in per_seed])
        _add(f"true_{k}", scores)

    return sep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "settings": {
            "n_seeds": N_SEEDS,
            "change_step": CHANGE_STEP,
            "task_err_fail": TASK_ERR_FAIL,
            "param_rmse_fail": PARAM_RMSE_FAIL,
            "rollout_horizons": ROLLOUT_HORIZONS,
            "damping": DAMPING,
        },
        "conditions": {},
    }

    for cond in DIAG_CONDITIONS:
        print(f"\nRunning '{cond}' ", end="", flush=True)
        per_seed = []

        for seed in range(N_SEEDS):
            r = run_one(seed, cond)

            q_change = jnp.array(r["q_change"])
            theta_hat = jnp.array(r["theta_final_ph1"])

            task_failed = bool(r["task_err_final"] > TASK_ERR_FAIL)
            rmse_failed = bool(r["rmse_at_change"] > PARAM_RMSE_FAIL)

            if rmse_failed:
                failure_mode = "estimation_failure"
            elif task_failed:
                failure_mode = "task_control"
            else:
                failure_mode = "success"

            # Geometric risk at phase transition
            risk_hat = compute_risk_metrics(q_change, theta_hat)
            risk_true = compute_risk_metrics(q_change, THETA_TRUE)

            # K-step rollout risk: estimated theta control, evaluated at true theta
            _add_rollout_risks(risk_hat, q_change, theta_hat, THETA_TRUE)
            # Oracle: true theta for both control and evaluation
            _add_rollout_risks(risk_true, q_change, THETA_TRUE, THETA_TRUE)

            per_seed.append({
                "seed": seed,
                "task_failed": task_failed,
                "rmse_failed": rmse_failed,
                "failure_mode": failure_mode,
                "theta_rmse_at_change": float(r["rmse_at_change"]),
                "task_err_final": float(r["task_err_final"]),
                "ee_hold_err_phase1": float(r["ee_hold_mean"]),
                "q_change": [float(x) for x in q_change],
                "theta_hat_change": [float(x) for x in theta_hat],
                "risk_hat": risk_hat,
                "risk_true": risk_true,
            })
            print(".", end="", flush=True)

        task_failed_arr = np.array([s["task_failed"] for s in per_seed], dtype=bool)
        n_success = int(np.sum(~task_failed_arr))
        n_failed = int(np.sum(task_failed_arr))
        mode_counts = {}
        for s in per_seed:
            mode_counts[s["failure_mode"]] = mode_counts.get(s["failure_mode"], 0) + 1

        sep = _separation_stats(per_seed, task_failed_arr)

        print(
            f"  success={n_success} failed={n_failed}  "
            f"auc(theta_rmse)={sep['theta_rmse']['auc']:.3f}  "
            f"auc(hat_r_rollout_50_plain)={sep.get('hat_r_rollout_50_plain', {}).get('auc', float('nan')):.3f}"
        )

        output["conditions"][cond] = {
            "mode_counts": mode_counts,
            "n_success": n_success,
            "n_failed": n_failed,
            "separation": sep,
            "per_seed": per_seed,
        }

    out_path = results_dir / "terminal_control_risk_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print summary table
    print("\n--- Separation AUC summary (auc > 0.5 means risk predicts failure) ---")
    key_metrics = [
        "theta_rmse",
        "hat_r_sing", "hat_r_gain", "hat_r_step",
        "hat_r_rollout_50_plain", "hat_r_rollout_50_posture",
        "true_r_rollout_50_plain", "true_r_rollout_50_posture",
        "hat_r_rollout_200_plain", "hat_r_rollout_200_posture",
    ]
    header = f"{'condition':35s}" + "".join(f"{k:30s}" for k in key_metrics)
    print(header)
    for cond in DIAG_CONDITIONS:
        sep = output["conditions"][cond]["separation"]
        row = f"{cond:35s}"
        for k in key_metrics:
            v = sep.get(k, {}).get("auc", float("nan"))
            row += f"{v:30.3f}"
        print(row)


if __name__ == "__main__":
    main()
