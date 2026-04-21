"""Phase 2 control causality study for redundant arm.

Isolates the cause of the 28% task failure in null_space condition by fixing
Phase 1 end states and varying only the Phase 2 controller.

Approach:
    1. Run Phase 1 (null_space) for all N_SEEDS → save q, theta_est at step 150.
    2. Re-run Phase 2 only with different controller variants.
    3. Compare failure rates to identify root cause.

Phase 2 conditions:
    estimated_theta       Baseline (current): use theta_est from Phase 1
    oracle_theta          Use THETA_TRUE: is estimation the bottleneck?
    longer_100            50→100 Phase 2 steps: is 50 steps too short?
    longer_200            50→200 Phase 2 steps: upper bound on convergence
    high_gain             K_TASK × 3: is gain insufficient?
    damped_ik             Damped pseudoinverse (μ=0.01): singular posture issue?
    posture_regularized   Null-space posture term −k·q: stuck near q≈0?

Output:
    results/redundant_arm_phase2_control_study.json
    results/redundant_arm_phase2_control_study_summary.json
    results/redundant_arm_phase2_control_study.png
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

from experiments.redundant_arm_calibration import (
    N_DOF, THETA_TRUE, THETA_INIT, Q0, DT, N_STEPS, CHANGE_STEP, N_SEEDS,
    U_MAX, K_TASK, SIGMA_OBS, PI_Y, PI_X, PARAMS_PRIOR_PI, KAPPA_P,
    N_ESTEP_ITER, ALPHA_NS, ESTEP_FREQ, PARAM_RMSE_FAIL, TASK_ERR_FAIL,
    fk, Y_GOAL_HOLD, Y_GOAL_TASK, _matrix_diag, _summarize,
    rollout_step, _build_estep, _ig_at_q,
    compute_null_space_action,
)

# ---------------------------------------------------------------------------
# Phase 2 condition definitions
# ---------------------------------------------------------------------------
# Each entry: (use_oracle, k_task_mult, p2_n_steps, damped_mu, posture_gain)
P2_CONDITIONS = {
    "estimated_theta":      dict(oracle=False, k_mult=1.0, p2_steps=50,  mu=0.0,  posture=0.0),
    "oracle_theta":         dict(oracle=True,  k_mult=1.0, p2_steps=50,  mu=0.0,  posture=0.0),
    "longer_100":           dict(oracle=False, k_mult=1.0, p2_steps=100, mu=0.0,  posture=0.0),
    "longer_200":           dict(oracle=False, k_mult=1.0, p2_steps=200, mu=0.0,  posture=0.0),
    "high_gain":            dict(oracle=False, k_mult=3.0, p2_steps=50,  mu=0.0,  posture=0.0),
    "damped_ik":            dict(oracle=False, k_mult=1.0, p2_steps=50,  mu=0.01, posture=0.0),
    "posture_regularized":  dict(oracle=False, k_mult=1.0, p2_steps=50,  mu=0.0,  posture=1.0),
    # Combinations: can 28% → <10% be achieved?
    "longer_200_oracle":    dict(oracle=True,  k_mult=1.0, p2_steps=200, mu=0.0,  posture=0.0),
    "longer_200_high_gain": dict(oracle=False, k_mult=3.0, p2_steps=200, mu=0.0,  posture=0.0),
    "longer_200_posture":   dict(oracle=False, k_mult=1.0, p2_steps=200, mu=0.0,  posture=1.0),
}

TASK_ERR_FAIL_P2 = TASK_ERR_FAIL  # same threshold


# ---------------------------------------------------------------------------
# Phase 2 controllers
# ---------------------------------------------------------------------------
def _compute_p2_action(q, theta, k_task, mu, posture_gain):
    """Phase 2 controller variants."""
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta))(q)

    if mu > 0.0:
        # Damped pseudoinverse: J^T (J J^T + μI)^{-1}
        J_pinv = J_ee.T @ jnp.linalg.inv(J_ee @ J_ee.T + mu * jnp.eye(2))
    else:
        J_pinv = jnp.linalg.pinv(J_ee)

    y_ee = fk(q, theta)
    v_task = -k_task * (y_ee - Y_GOAL_TASK)
    u_task = J_pinv @ v_task

    if posture_gain > 0.0:
        # Null-space posture regularization: pull joints toward q=0 to escape singularity
        N_mat = jnp.eye(N_DOF) - J_pinv @ J_ee
        u_posture = N_mat @ (-posture_gain * q)
        u = u_task + u_posture
    else:
        u = u_task

    return jnp.clip(u, -U_MAX, U_MAX)


# ---------------------------------------------------------------------------
# Phase 1: run null_space, save end states
# ---------------------------------------------------------------------------
def run_phase1(seed):
    """Run Phase 1 (null_space condition), return end state."""
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    P_theta = PARAMS_PRIOR_PI * jnp.eye(N_DOF)
    estep = _build_estep(theta_est)

    q = Q0.copy()
    u = jnp.zeros(N_DOF)
    q_hist, v_hist, y_hist = [], [], []

    for t in range(CHANGE_STEP):
        u = compute_null_space_action(q, theta_est, P_theta, Y_GOAL_HOLD)
        q = rollout_step(q, u)
        y_obs = fk(q, THETA_TRUE) + rng.normal(0, SIGMA_OBS, size=(2,))
        y_obs = jnp.array(y_obs)
        q_hist.append(q)
        v_hist.append(jnp.zeros(1))
        y_hist.append(y_obs)

        if t > 5 and t % ESTEP_FREQ == 0:
            theta_est = estep.run(q_hist, v_hist, y_hist, theta_est, n_iter=N_ESTEP_ITER)
            theta_est = jnp.clip(theta_est, 0.05, 2.0)
            P_theta = estep.compute_precision(q_hist, v_hist, y_hist, theta_est)
            estep = _build_estep(theta_est)

    rmse_ph1 = float(jnp.sqrt(jnp.mean((theta_est - THETA_TRUE) ** 2)))
    J_ee = jax.jacfwd(lambda qi: fk(qi, theta_est))(q)
    eigs = np.linalg.eigvalsh(np.array(J_ee @ J_ee.T, dtype=float))
    j_rank = int(np.sum(eigs > 1e-6))
    j_cond = float(eigs[-1] / max(eigs[0], 1e-12))

    return {
        "q": np.array(q),
        "theta_est": np.array(theta_est),
        "rmse_ph1": rmse_ph1,
        "j_rank_ph1_end": j_rank,
        "j_cond_ph1_end": j_cond,
        "task_err_ph2_start": float(jnp.sqrt(jnp.sum((fk(q, THETA_TRUE) - Y_GOAL_TASK) ** 2))),
    }


# ---------------------------------------------------------------------------
# Phase 2: run from saved state with a given controller config
# ---------------------------------------------------------------------------
def run_phase2(q_start, theta_est, cfg, rng):
    """Run Phase 2 from saved Phase 1 end state."""
    theta = jnp.array(THETA_TRUE) if cfg["oracle"] else jnp.array(theta_est)
    k_task = K_TASK * cfg["k_mult"]
    mu = cfg["mu"]
    posture = cfg["posture"]
    n_steps = cfg["p2_steps"]

    q = jnp.array(q_start)
    task_err_hist = []

    for _ in range(n_steps):
        ee_true = fk(q, THETA_TRUE)
        task_err_hist.append(float(jnp.sqrt(jnp.sum((ee_true - Y_GOAL_TASK) ** 2))))
        u = _compute_p2_action(q, theta, k_task, mu, posture)
        q = rollout_step(q, u)

    # Final task error
    ee_final = fk(q, THETA_TRUE)
    task_err_final = float(jnp.sqrt(jnp.sum((ee_final - Y_GOAL_TASK) ** 2)))
    task_err_hist.append(task_err_final)

    return {
        "task_err_final": task_err_final,
        "task_err_hist": task_err_hist,
        "converged_50": float(task_err_hist[min(49, len(task_err_hist)-1)]),
        "converged_100": float(task_err_hist[min(99, len(task_err_hist)-1)]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Phase 1: save end states for all seeds ---
    print("Running Phase 1 (null_space) for all seeds...", flush=True)
    phase1_states = []
    for seed in range(N_SEEDS):
        s = run_phase1(seed)
        phase1_states.append(s)
        print(".", end="", flush=True)
    print(" done.")

    # Summarize Phase 1 end states
    rmse_ph1 = np.array([s["rmse_ph1"] for s in phase1_states])
    fail_ph1 = float(np.mean(rmse_ph1 > PARAM_RMSE_FAIL))
    print(f"  Phase1 RMSE@150 median={np.median(rmse_ph1):.4f}  fail={fail_ph1:.2f}")

    # --- Phase 2: run each condition ---
    results = {cname: [] for cname in P2_CONDITIONS}

    for cname, cfg in P2_CONDITIONS.items():
        rng = np.random.default_rng(42)
        print(f"Phase 2 condition: '{cname}'", end="", flush=True)
        for seed, state in enumerate(phase1_states):
            r = run_phase2(state["q"], state["theta_est"], cfg, rng)
            r["seed"] = seed
            r["rmse_ph1"] = state["rmse_ph1"]
            r["task_err_ph2_start"] = state["task_err_ph2_start"]
            r["j_rank_ph1_end"] = state["j_rank_ph1_end"]
            r["j_cond_ph1_end"] = state["j_cond_ph1_end"]
            r["fail_ph1"] = state["rmse_ph1"] > PARAM_RMSE_FAIL
            results[cname].append(r)
            print(".", end="", flush=True)
        task_finals = np.array([r["task_err_final"] for r in results[cname]])
        fail_rate = float(np.mean(task_finals > TASK_ERR_FAIL_P2))
        print(f"  done.  median={np.median(task_finals):.4f}  failTask={fail_rate:.2f}", flush=True)

    # --- Summary ---
    print(f"\n{'Condition':22s}  {'median':7s}  {'mean':7s}  {'failTask':8s}  {'p90':7s}")
    summary = {}
    for cname in P2_CONDITIONS:
        task_finals = np.array([r["task_err_final"] for r in results[cname]])
        fail_rate = float(np.mean(task_finals > TASK_ERR_FAIL_P2))
        summary[cname] = {
            "task_err_final": _summarize(task_finals),
            "task_failure_rate": fail_rate,
            "p90": float(np.percentile(task_finals, 90)),
            "per_seed": results[cname],
        }
        print(
            f"  {cname:20s}  {np.median(task_finals):.4f}   {np.mean(task_finals):.4f}   "
            f"{fail_rate:.2f}      {np.percentile(task_finals, 90):.4f}"
        )

    out_json = project_root / "results" / "redundant_arm_phase2_control_study.json"
    with open(out_json, "w") as f:
        # per_seed has arrays — convert to lists
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(x) for x in obj]
            return obj
        json.dump(to_serializable(summary), f, indent=2)
    print(f"\nSaved → {out_json}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Phase 2 Control Causality Study\n"
        "Phase 1 fixed (null_space, N=50 seeds); Phase 2 controller varies",
        fontsize=10,
    )

    cnames = list(P2_CONDITIONS.keys())
    fail_rates = [summary[c]["task_failure_rate"] for c in cnames]
    medians = [summary[c]["task_err_final"]["median"] for c in cnames]
    p90s = [summary[c]["p90"] for c in cnames]

    colors = ["C3" if f > 0.20 else "C1" if f > 0.10 else "C2" for f in fail_rates]

    # (A) Failure rate
    ax = axes[0]
    bars = ax.bar(range(len(cnames)), fail_rates, color=colors)
    ax.axhline(0.10, color="k", linestyle="--", linewidth=0.8, label="10% target")
    ax.axhline(0.28, color="gray", linestyle=":", linewidth=0.8, label="baseline (28%)")
    ax.set_xticks(range(len(cnames)))
    ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Task failure rate")
    ax.set_title("(A) Task failure rate by Phase 2 condition")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # (B) Median task error
    ax = axes[1]
    ax.bar(range(len(cnames)), medians, color=colors)
    ax.axhline(TASK_ERR_FAIL_P2, color="k", linestyle="--", linewidth=0.8, label=f"fail threshold ({TASK_ERR_FAIL_P2})")
    ax.set_xticks(range(len(cnames)))
    ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TaskErr@final (median)")
    ax.set_title("(B) Median final task error")
    ax.legend(fontsize=8)

    # (C) Failure rate decomposed: ph1 fail vs ph2 fail
    ax = axes[2]
    ph1_fail_rates = [float(np.mean([r["fail_ph1"] for r in results[c]])) for c in cnames]
    ph2_only_rates = [max(0, f - ph1) for f, ph1 in zip(fail_rates, ph1_fail_rates)]
    ax.bar(range(len(cnames)), ph1_fail_rates, label="Phase 1 fail (RMSE)", color="C3")
    ax.bar(range(len(cnames)), ph2_only_rates, bottom=ph1_fail_rates, label="Phase 2 fail only", color="C1")
    ax.axhline(0.10, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(cnames)))
    ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Failure rate")
    ax.set_title("(C) Phase 1 vs Phase 2 failure decomposition")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_fig = project_root / "results" / "redundant_arm_phase2_control_study.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
