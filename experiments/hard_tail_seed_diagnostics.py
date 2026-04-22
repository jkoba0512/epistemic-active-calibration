"""Hard-tail seed analysis and short rollout risk AUC diagnostics.

Key questions:
1. Why do seeds 1, 3 always fail (plain/200)?
2. Why was seed 37 rescued by risk-aware probe but seed 22 broken?
3. Does K=5-10 short rollout risk predict task failure? (AUC analysis)

Outputs:
    results/hard_tail_seed_diagnostics.json
    results/hard_tail_seed_diagnostics.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.redundant_arm_calibration import (
    run_one,
    fk,
    compute_vfe_only_action,
    compute_posture_task_action,
    rollout_step,
    CHANGE_STEP,
    TASK_ERR_FAIL,
    N_SEEDS,
    Y_GOAL_TASK,
    THETA_TRUE,
    PROBE_DAMPING,
)

HARD_TAIL_BASELINE = [1, 3, 37]
HARD_TAIL_RISK_AWARE = [1, 3, 22]
ALL_HARD_TAIL = sorted(set(HARD_TAIL_BASELINE) | set(HARD_TAIL_RISK_AWARE))  # [1,3,22,37]
COMPARISON_SEEDS = [0, 2, 11, 16, 27, 29, 42, 47]
DETAIL_SEEDS = sorted(set(ALL_HARD_TAIL) | set(COMPARISON_SEEDS))

ROLLOUT_K_AUC = [1, 3, 5, 10, 20, 50]
ROLLOUT_K_DETAIL = [1, 3, 5, 10, 20, 50, 100, 150, 200]
HORIZONS = [50, 80, 100, 120, 150, 200]
PROBES = ["null_space_probe", "null_space_probe_risk"]
TRAJ_CONDITIONS = [("plain", 200), ("posture", 150), ("posture", 200)]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(score_failed > score_success), ties count as 0.5."""
    failed = scores[labels]
    success = scores[~labels]
    if len(failed) == 0 or len(success) == 0:
        return 0.5
    pairs = failed[:, None] - success[None, :]
    return float(np.mean(pairs > 0) + 0.5 * np.mean(pairs == 0))


def noiseless_rollout(q_start, theta_control, theta_eval, controller: str, K: int) -> float:
    """K-step noiseless Phase 2 rollout from q_start.

    Uses theta_control for action computation and theta_eval for EE evaluation.
    """
    q = jnp.array(q_start)
    theta_c = jnp.array(theta_control)
    theta_e = jnp.array(theta_eval)
    for _ in range(K):
        if controller == "plain":
            u = compute_vfe_only_action(q, theta_c, Y_GOAL_TASK)
        else:
            u = compute_posture_task_action(q, theta_c, Y_GOAL_TASK)
        q = rollout_step(q, u)
    ee = fk(q, theta_e)
    return float(jnp.sqrt(jnp.sum((ee - Y_GOAL_TASK) ** 2)))


def compute_static_metrics(q, theta_hat) -> dict:
    """Jacobian-based static geometry metrics at a configuration q."""
    q_j = jnp.array(q)
    th_j = jnp.array(theta_hat)

    J_hat = np.array(jax.jacfwd(lambda qi: fk(qi, th_j))(q_j))
    J_true = np.array(jax.jacfwd(lambda qi: fk(qi, THETA_TRUE))(q_j))

    svals_hat = np.linalg.svd(J_hat, compute_uv=False)
    svals_true = np.linalg.svd(J_true, compute_uv=False)

    J_hash_hat = J_hat.T @ np.linalg.inv(J_hat @ J_hat.T + PROBE_DAMPING * np.eye(2))
    J_hash_true = J_true.T @ np.linalg.inv(J_true @ J_true.T + PROBE_DAMPING * np.eye(2))

    e_hat = np.array(Y_GOAL_TASK - fk(q_j, th_j))
    e_true = np.array(Y_GOAL_TASK - fk(q_j, THETA_TRUE))

    manip_hat = float(np.sqrt(max(np.linalg.det(J_hat @ J_hat.T), 0.0)))
    manip_true = float(np.sqrt(max(np.linalg.det(J_true @ J_true.T), 0.0)))

    return {
        "sigma_min_hat": float(svals_hat[-1]),
        "sigma_max_hat": float(svals_hat[0]),
        "condition_number_hat": float(svals_hat[0] / max(svals_hat[-1], 1e-10)),
        "manipulability_hat": manip_hat,
        "sigma_min_true": float(svals_true[-1]),
        "manipulability_true": manip_true,
        "q_norm": float(np.linalg.norm(q)),
        "r_step_hat": float(np.linalg.norm(J_hash_hat @ e_hat)),
        "r_step_true": float(np.linalg.norm(J_hash_true @ e_true)),
    }


def run_phase2_trajectory(q_change, theta_hat, controller: str, H: int) -> dict:
    """Run Phase 2 from q_change with frozen theta_hat, return per-step metrics."""
    q = jnp.array(q_change)
    theta_c = jnp.array(theta_hat)
    keys = ["q_norm", "task_err_true", "task_err_hat", "sigma_min_hat", "r_step_hat"]
    traj = {k: [] for k in keys}

    for _ in range(H):
        if controller == "plain":
            u = compute_vfe_only_action(q, theta_c, Y_GOAL_TASK)
        else:
            u = compute_posture_task_action(q, theta_c, Y_GOAL_TASK)
        q = rollout_step(q, u)

        J_hat = np.array(jax.jacfwd(lambda qi: fk(qi, theta_c))(q))
        svals = np.linalg.svd(J_hat, compute_uv=False)
        J_hash = J_hat.T @ np.linalg.inv(J_hat @ J_hat.T + PROBE_DAMPING * np.eye(2))
        e_hat_vec = np.array(Y_GOAL_TASK - fk(q, theta_c))

        traj["q_norm"].append(float(jnp.linalg.norm(q)))
        traj["task_err_true"].append(float(jnp.sqrt(jnp.sum((fk(q, THETA_TRUE) - Y_GOAL_TASK) ** 2))))
        traj["task_err_hat"].append(float(jnp.sqrt(jnp.sum((fk(q, theta_c) - Y_GOAL_TASK) ** 2))))
        traj["sigma_min_hat"].append(float(svals[-1]))
        traj["r_step_hat"].append(float(np.linalg.norm(J_hash @ e_hat_vec)))

    return traj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Phase 1 runs for all N=50 seeds (both probes)
    # -----------------------------------------------------------------------
    print("=== Step 1: Phase 1 runs (all seeds, both probes) ===")
    phase1 = {probe: {} for probe in PROBES}

    for probe in PROBES:
        print(f"  probe={probe}", end="", flush=True)
        for seed in range(N_SEEDS):
            r = run_one(seed, probe, phase2_horizon_override=1)
            q_c = r["q_change"]
            th_c = r["theta_final_ph1"]
            phase1[probe][seed] = {
                "q_change": q_c.tolist(),
                "theta_hat": th_c.tolist(),
                "theta_rmse_at_change": float(r["rmse_at_change"]),
                "ee_hold_mean": float(r["ee_hold_mean"]),
                "p_theta_rank": int(r["p_theta_rank_at_change"]),
                **compute_static_metrics(q_c, th_c),
            }
            print(".", end="", flush=True)
        print()

    # -----------------------------------------------------------------------
    # Step 2: Load task_failed labels from sweep JSON files
    # phase2_horizon_sweep.json: results[ctrl][H] (probe=null_space_probe fixed)
    # phase4_risk_aware_probing.json: results[probe][ctrl][H] (both probes)
    # -----------------------------------------------------------------------
    print("=== Step 2: Load task_failed labels ===")
    sweep_path = results_dir / "phase2_horizon_sweep.json"
    phase4_path = results_dir / "phase4_risk_aware_probing.json"
    with open(sweep_path) as f:
        sweep_data = json.load(f)
    with open(phase4_path) as f:
        phase4_data = json.load(f)

    # task_failed_by_probe_H[probe][H] = bool array over N_SEEDS
    task_failed_by_probe_H = {}
    for probe in PROBES:
        task_failed_by_probe_H[probe] = {}
        for H in HORIZONS:
            if probe == "null_space_probe":
                # phase2_horizon_sweep has results[ctrl][H]
                failed_set = set(sweep_data["results"]["plain"][str(H)]["failed_seeds"])
            else:
                # phase4 has results[probe][ctrl][H]
                failed_set = set(phase4_data["results"][probe]["plain"][str(H)]["failed_seeds"])
            task_failed_by_probe_H[probe][H] = np.array(
                [s in failed_set for s in range(N_SEEDS)]
            )
            n_failed = int(np.sum(task_failed_by_probe_H[probe][H]))
            print(f"  {probe}/plain/H={H}: {n_failed} failed")

    # -----------------------------------------------------------------------
    # Step 3: Rollout risks for all 50 seeds (AUC analysis)
    # -----------------------------------------------------------------------
    print("=== Step 3: Rollout risks (all seeds, AUC Ks) ===")
    # rollout_risks[probe][ctrl][theta_type][K] = list of float (per seed)
    rollout_risks = {}
    for probe in PROBES:
        rollout_risks[probe] = {}
        for ctrl in ["plain", "posture"]:
            rollout_risks[probe][ctrl] = {
                "hat_true": {K: [] for K in ROLLOUT_K_AUC},
                "oracle":   {K: [] for K in ROLLOUT_K_AUC},
            }
        print(f"  probe={probe}", end="", flush=True)
        for seed in range(N_SEEDS):
            q_c = phase1[probe][seed]["q_change"]
            th_c = phase1[probe][seed]["theta_hat"]
            for ctrl in ["plain", "posture"]:
                for K in ROLLOUT_K_AUC:
                    r_hat = noiseless_rollout(q_c, th_c, THETA_TRUE, ctrl, K)
                    r_oracle = noiseless_rollout(q_c, THETA_TRUE, THETA_TRUE, ctrl, K)
                    rollout_risks[probe][ctrl]["hat_true"][K].append(r_hat)
                    rollout_risks[probe][ctrl]["oracle"][K].append(r_oracle)
            print(".", end="", flush=True)
        print()

    # -----------------------------------------------------------------------
    # Step 4: AUC computation
    # -----------------------------------------------------------------------
    print("=== Step 4: AUC computation ===")
    auc_results = {probe: {} for probe in PROBES}
    for probe in PROBES:
        for H in HORIZONS:
            auc_results[probe][H] = {}
            labels = task_failed_by_probe_H[probe][H]
            for ctrl in ["plain", "posture"]:
                auc_results[probe][H][ctrl] = {}
                for theta_type in ["hat_true", "oracle"]:
                    auc_results[probe][H][ctrl][theta_type] = {}
                    for K in ROLLOUT_K_AUC:
                        scores = np.array(rollout_risks[probe][ctrl][theta_type][K])
                        auc = compute_auc(scores, labels)
                        auc_results[probe][H][ctrl][theta_type][K] = round(auc, 4)

    # Print AUC summary for null_space_probe / plain / hat_true
    print("\n  AUC(R_rollout_K → task_failed_H) [null_space_probe / plain ctrl / hat_true]:")
    print(f"  {'H\\K':>8}", end="")
    for K in ROLLOUT_K_AUC:
        print(f"  K={K:>3}", end="")
    print()
    for H in HORIZONS:
        print(f"  H={H:>5}", end="")
        for K in ROLLOUT_K_AUC:
            auc = auc_results["null_space_probe"][H]["plain"]["hat_true"][K]
            print(f"  {auc:.3f}", end="")
        print()

    # -----------------------------------------------------------------------
    # Step 5: Per-seed detailed metrics for hard-tail + comparison seeds
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Detailed per-seed metrics ===")
    per_seed_detail = {probe: {} for probe in PROBES}

    for probe in PROBES:
        print(f"  probe={probe}", end="", flush=True)
        for seed in DETAIL_SEEDS:
            q_c = phase1[probe][seed]["q_change"]
            th_c = phase1[probe][seed]["theta_hat"]

            # Detailed rollout risks at more K values
            detail_risks = {}
            for ctrl in ["plain", "posture"]:
                detail_risks[ctrl] = {"hat_true": {}, "oracle": {}}
                for K in ROLLOUT_K_DETAIL:
                    r_hat = noiseless_rollout(q_c, th_c, THETA_TRUE, ctrl, K)
                    r_oracle = noiseless_rollout(q_c, THETA_TRUE, THETA_TRUE, ctrl, K)
                    detail_risks[ctrl]["hat_true"][K] = round(r_hat, 6)
                    detail_risks[ctrl]["oracle"][K] = round(r_oracle, 6)

            per_seed_detail[probe][seed] = {
                **phase1[probe][seed],
                "rollout_risk": detail_risks,
            }
            print(".", end="", flush=True)
        print()

    # -----------------------------------------------------------------------
    # Step 6: Phase 2 trajectories for hard-tail seeds
    # -----------------------------------------------------------------------
    print("=== Step 6: Phase 2 trajectories for hard-tail seeds ===")
    trajectories = {probe: {seed: {} for seed in ALL_HARD_TAIL} for probe in PROBES}

    for probe in PROBES:
        for seed in ALL_HARD_TAIL:
            q_c = phase1[probe][seed]["q_change"]
            th_c = phase1[probe][seed]["theta_hat"]
            for ctrl, H in TRAJ_CONDITIONS:
                label = f"{ctrl}/{H}"
                print(f"  probe={probe} seed={seed} {label}", flush=True)
                trajectories[probe][seed][label] = run_phase2_trajectory(q_c, th_c, ctrl, H)

    # -----------------------------------------------------------------------
    # Step 7: Assemble and save JSON
    # -----------------------------------------------------------------------
    print("=== Step 7: Saving JSON ===")

    # Convert rollout_risks for JSON (nested dict with int keys → str)
    def _to_json_risks(risks):
        out = {}
        for ctrl in risks:
            out[ctrl] = {}
            for tt in risks[ctrl]:
                out[ctrl][tt] = {str(K): v for K, v in risks[ctrl][tt].items()}
        return out

    output = {
        "settings": {
            "hard_tail_baseline": HARD_TAIL_BASELINE,
            "hard_tail_risk_aware": HARD_TAIL_RISK_AWARE,
            "all_hard_tail": ALL_HARD_TAIL,
            "comparison_seeds": COMPARISON_SEEDS,
            "detail_seeds": DETAIL_SEEDS,
            "rollout_k_auc": ROLLOUT_K_AUC,
            "rollout_k_detail": ROLLOUT_K_DETAIL,
            "horizons": HORIZONS,
            "probes": PROBES,
            "traj_conditions": [f"{ctrl}/{H}" for ctrl, H in TRAJ_CONDITIONS],
            "task_err_fail": float(TASK_ERR_FAIL),
        },
        "auc": {
            probe: {
                str(H): {
                    ctrl: {
                        tt: {str(K): v for K, v in auc_results[probe][H][ctrl][tt].items()}
                        for tt in ["hat_true", "oracle"]
                    }
                    for ctrl in ["plain", "posture"]
                }
                for H in HORIZONS
            }
            for probe in PROBES
        },
        "rollout_risks_all_seeds": {
            probe: {
                str(seed): _to_json_risks({
                    ctrl: {
                        tt: {str(K): rollout_risks[probe][ctrl][tt][K][seed] for K in ROLLOUT_K_AUC}
                        for tt in ["hat_true", "oracle"]
                    }
                    for ctrl in ["plain", "posture"]
                })
                for seed in range(N_SEEDS)
            }
            for probe in PROBES
        },
        "per_seed_detail": {
            probe: {
                str(seed): per_seed_detail[probe][seed]
                for seed in DETAIL_SEEDS
            }
            for probe in PROBES
        },
        "trajectories": {
            probe: {
                str(seed): trajectories[probe][seed]
                for seed in ALL_HARD_TAIL
            }
            for probe in PROBES
        },
    }

    out_path = results_dir / "hard_tail_seed_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")

    # -----------------------------------------------------------------------
    # Step 8: Print summary
    # -----------------------------------------------------------------------
    print("\n=== Summary: hard-tail seed static metrics ===")
    print(f"{'seed':>5}  {'probe':>26}  {'q_norm':>7}  {'sigma_min':>9}  "
          f"{'r_step_hat':>10}  {'rmse':>8}  {'status':>10}")
    for probe in PROBES:
        for seed in ALL_HARD_TAIL:
            d = per_seed_detail[probe][seed]
            is_ht_b = seed in HARD_TAIL_BASELINE and probe == "null_space_probe"
            is_ht_r = seed in HARD_TAIL_RISK_AWARE and probe == "null_space_probe_risk"
            status = "HARD-TAIL" if (is_ht_b or is_ht_r) else "rescued" if (
                (seed == 37 and probe == "null_space_probe_risk") or
                (seed == 22 and probe == "null_space_probe")
            ) else ""
            print(f"{seed:5d}  {probe:>26}  {d['q_norm']:7.4f}  "
                  f"{d['sigma_min_hat']:9.4f}  {d['r_step_hat']:10.4f}  "
                  f"{d['theta_rmse_at_change']:8.4f}  {status:>10}")

    print("\n=== Summary: rollout risk at K=50 for hard-tail seeds ===")
    for probe in PROBES:
        print(f"\n  probe={probe}:")
        for seed in ALL_HARD_TAIL:
            rr = per_seed_detail[probe][seed]["rollout_risk"]
            r50_plain = rr["plain"]["hat_true"].get(50, rr["plain"]["hat_true"].get("50", "N/A"))
            r50_posture = rr["posture"]["hat_true"].get(50, rr["posture"]["hat_true"].get("50", "N/A"))
            print(f"    seed={seed}  R50_plain={r50_plain:.4f}  R50_posture={r50_posture:.4f}")

    # -----------------------------------------------------------------------
    # Step 9: Plots
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Hard-tail seed diagnostics: rollout risk AUC and trajectory analysis",
            fontsize=11,
        )

        # Panel A: task_err(t) for hard-tail seeds, plain/200 vs posture/200
        ax = axes[0, 0]
        colors_seed = {1: "C0", 3: "C1", 22: "C2", 37: "C3"}
        probe = "null_space_probe"
        for seed in ALL_HARD_TAIL:
            for ctrl, H in [("plain", 200), ("posture", 200)]:
                lbl = f"{ctrl}/{H}"
                traj = trajectories[probe][seed][lbl]
                ts = np.arange(len(traj["task_err_true"]))
                ls = "-" if ctrl == "posture" else "--"
                ax.plot(ts, traj["task_err_true"], color=colors_seed[seed],
                        linestyle=ls, alpha=0.8,
                        label=f"s={seed} {ctrl}" if H == 200 else None)
        ax.axhline(TASK_ERR_FAIL, color="k", linestyle=":", linewidth=0.8, label="fail threshold")
        ax.set_xlabel("Phase 2 step")
        ax.set_ylabel("Task error (m)")
        ax.set_title("(A) Phase 2 task_err: hard-tail seeds\n(dashed=plain, solid=posture, horizon=200)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Panel B: R_rollout_K vs K for hard-tail and comparison seeds
        ax = axes[0, 1]
        probe = "null_space_probe"
        k_arr = np.array(ROLLOUT_K_DETAIL)
        for seed in ALL_HARD_TAIL:
            rr = per_seed_detail[probe][seed]["rollout_risk"]
            vals = [rr["plain"]["hat_true"][K] for K in ROLLOUT_K_DETAIL]
            ax.plot(k_arr, vals, "o-", color=colors_seed[seed], label=f"s={seed} (hard)")
        # A few comparison seeds
        for seed in COMPARISON_SEEDS[:3]:
            rr = per_seed_detail[probe][seed]["rollout_risk"]
            vals = [rr["plain"]["hat_true"][K] for K in ROLLOUT_K_DETAIL]
            ax.plot(k_arr, vals, "s--", color="gray", alpha=0.5, label=f"s={seed} (ok)" if seed == COMPARISON_SEEDS[0] else None)
        ax.axhline(TASK_ERR_FAIL, color="k", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Rollout steps K")
        ax.set_ylabel("Residual (m)")
        ax.set_title("(B) R_rollout_K vs K\n(null_space_probe / plain ctrl / hat_true)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Panel C: AUC vs K for various H
        ax = axes[1, 0]
        probe = "null_space_probe"
        k_arr_auc = np.array(ROLLOUT_K_AUC)
        cmap = plt.cm.viridis
        h_colors = {H: cmap(i / max(len(HORIZONS) - 1, 1)) for i, H in enumerate(HORIZONS)}
        for H in HORIZONS:
            auc_plain = [auc_results[probe][H]["plain"]["hat_true"][K] for K in ROLLOUT_K_AUC]
            auc_posture = [auc_results[probe][H]["posture"]["hat_true"][K] for K in ROLLOUT_K_AUC]
            ax.plot(k_arr_auc, auc_plain, "o-", color=h_colors[H], label=f"H={H} plain")
            ax.plot(k_arr_auc, auc_posture, "s--", color=h_colors[H], alpha=0.7)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, label="AUC=0.8 threshold")
        ax.axhline(0.5, color="k", linestyle=":", linewidth=0.5)
        ax.set_xlabel("Rollout steps K")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.05)
        ax.set_title("(C) AUC(R_rollout_K → task_failed_H) vs K\n(solid=plain ctrl, dashed=posture ctrl)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # Panel D: q_change comparison: baseline vs risk-aware for seeds 37 and 22
        ax = axes[1, 1]
        for seed, marker, name in [(37, "o", "seed 37"), (22, "s", "seed 22")]:
            for probe, color, label_sfx in [
                ("null_space_probe", "C0", "baseline"),
                ("null_space_probe_risk", "C1", "risk-aware"),
            ]:
                d = per_seed_detail[probe][seed]
                q = np.array(d["q_change"])
                ax.scatter(
                    np.arange(len(q)), q,
                    marker=marker, color=color, alpha=0.8, s=60,
                    label=f"{name} {label_sfx}",
                )
        ax.set_xlabel("Joint index")
        ax.set_ylabel("q_change (rad)")
        ax.set_title("(D) q_change: baseline vs risk-aware\n(seed 37 rescued, seed 22 broken)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        plt.tight_layout()
        fig_path = results_dir / "hard_tail_seed_diagnostics.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure → {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
