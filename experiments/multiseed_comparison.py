"""Step 1: Multi-seed statistical comparison of calibration policies.

20 seeds × 30 steps で Phase 1 が単一 seed に依存した結果かどうかを検証する。
同一条件（theta_init=[0.7,0.3], sigma_obs=0.01）で Phase 1 と直接比較可能。

Conditions:
  - random    : ランダム速度指令
  - sinusoidal: 正弦波速度指令
  - epistemic : 情報利得最大化 (FIM-only A-step)

Metrics:
  - RMSE per step       : mean ± 95% bootstrap CI
  - logdet(Σ_θ) per step: mean ± 95% bootstrap CI
  - Convergence step    : first t where RMSE(t) < CONV_THRESH (分布・中央値・IQR)
  - AUC RMSE            : Σ_t RMSE(t)  (総積算誤差、低いほど良い)

Statistical tests:
  - Mann-Whitney U : epistemic vs random, epistemic vs sinusoidal
                     on (a) final RMSE and (b) AUC RMSE

Usage:
    .venv/bin/python experiments/multiseed_comparison.py

Output:
    results/multiseed_comparison.png
    results/multiseed_comparison_stats.json
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import mannwhitneyu

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep


# ── Experiment parameters ─────────────────────────────────────────────────────

N_SEEDS  = 20
N_STEPS  = 30
CONV_THRESH = 0.05          # RMSE threshold for "converged" (m)

THETA_TRUE      = jnp.array([0.5, 0.5])
THETA_INIT      = jnp.array([0.7, 0.3])   # same as Phase 1
THETA_PRIOR_PI  = 0.5
Q0              = jnp.array([0.0, 0.5])
Q_CLIP          = 1.5
U_MAX           = 1.0
DT              = 0.05
SIGMA_OBS       = 0.01
R_OBS_INV       = jnp.eye(2) / (SIGMA_OBS ** 2)
DAMPING         = 1e-6

N_ESTEP_ITER    = 3
N_EPISTEMIC_ITER = 25
LR_EPISTEMIC    = 0.3
N_FUTURE_STEPS  = 8

SIN_AMP    = 0.8
SIN_OMEGA1 = 0.6
SIN_OMEGA2 = 0.4
SIN_PHASE2 = 1.1

CONDITIONS = ["random", "sinusoidal", "epistemic"]
COLORS     = {"random": "tab:blue", "sinusoidal": "tab:orange", "epistemic": "tab:green"}


# ── Kinematic model ───────────────────────────────────────────────────────────

def fk(q, theta):
    x = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    y = theta[0] * jnp.sin(q[0]) + theta[1] * jnp.sin(q[0] + q[1])
    return jnp.array([x, y])


def rollout_fn(q, u, dt, n_steps):
    def step(q_curr, _):
        return q_curr + u * dt, None
    q_f, _ = jax.lax.scan(step, q, None, length=n_steps)
    return q_f


def y_future_fn(q, u, theta, dt, n_steps):
    return fk(rollout_fn(q, u, dt, n_steps), theta)


def compute_fim(q, u, theta):
    J = jax.jacfwd(lambda t: y_future_fn(q, u, t, DT, N_FUTURE_STEPS))(theta)
    return J.T @ R_OBS_INV @ J


def compute_info_gain(P_theta, FIM_future):
    I = jnp.eye(P_theta.shape[0])
    _, ld0 = jnp.linalg.slogdet(P_theta + DAMPING * I)
    _, ld1 = jnp.linalg.slogdet(P_theta + FIM_future + DAMPING * I)
    return 0.5 * (ld1 - ld0)


# ── EStep (shared across seeds — stateless w.r.t. params) ────────────────────

def _build_estep():
    def f_zero(x, v, p): return jnp.zeros(2)
    def g_fk(x, v, p):   return fk(x, p)

    model = DEMModel(
        f=f_zero, g=g_fk,
        n_x=2, n_v=2, n_y=2, n_order=1,
        pi_y=1.0 / (SIGMA_OBS ** 2),
        pi_x=1.0,
        params=THETA_INIT,
        params_prior_mean=THETA_INIT,
        params_prior_pi=THETA_PRIOR_PI,
    )
    return EStep(model, use_gauss_newton=True)

ESTEP = _build_estep()   # one shared, JIT-compiled EStep


# ── Epistemic action optimizer ────────────────────────────────────────────────

@jax.jit
def optimize_epistemic_action(q, theta_est, P_theta, u_init):
    def ig_of_u(u):
        return compute_info_gain(P_theta, compute_fim(q, u, theta_est))

    def ascent_step(u, _):
        return jnp.clip(u + LR_EPISTEMIC * jax.grad(ig_of_u)(u), -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(ascent_step, u_init, None, length=N_EPISTEMIC_ITER)
    return u_opt


# ── Single seed simulation ────────────────────────────────────────────────────

def run_one_seed(condition: str, seed: int) -> dict:
    """Run one seed. Returns arrays of length N_STEPS+1."""
    rng = np.random.default_rng(seed)

    theta_est = THETA_INIT.copy()
    q = Q0.copy()
    P_theta = THETA_PRIOR_PI * jnp.eye(2)
    u_current = jnp.zeros(2)

    q_acc, v_acc, y_acc = [], [], []

    rmse_hist       = [float(jnp.linalg.norm(theta_est - THETA_TRUE))]
    logdet_sig_hist = [0.0]

    for t in range(N_STEPS):
        # ── Action ──────────────────────────────────────────────────────────
        if condition == "random":
            u = jnp.array(rng.standard_normal(2) * 0.6).clip(-U_MAX, U_MAX)

        elif condition == "sinusoidal":
            ts = t * DT
            u = jnp.array([
                SIN_AMP * np.sin(SIN_OMEGA1 * ts),
                SIN_AMP * np.sin(SIN_OMEGA2 * ts + SIN_PHASE2),
            ]).clip(-U_MAX, U_MAX)

        else:  # epistemic
            u = optimize_epistemic_action(q, theta_est, P_theta, u_current)
            u_current = u

        # ── Step ─────────────────────────────────────────────────────────────
        q_next = jnp.clip(q + u * DT, -Q_CLIP, Q_CLIP)
        y_obs  = fk(q_next, THETA_TRUE) + jnp.array(rng.standard_normal(2) * SIGMA_OBS)

        q_acc.append(q_next)
        v_acc.append(jnp.zeros(2))
        y_acc.append(y_obs)

        # ── E-step ───────────────────────────────────────────────────────────
        theta_est = ESTEP.run(q_acc, v_acc, y_acc, theta_est, n_iter=N_ESTEP_ITER)
        theta_est = jnp.clip(theta_est, 0.1, 1.5)

        P_theta = ESTEP.compute_precision(q_acc, v_acc, y_acc, theta_est)

        # ── Metrics ──────────────────────────────────────────────────────────
        rmse = float(jnp.linalg.norm(theta_est - THETA_TRUE))
        _, ld = jnp.linalg.slogdet(P_theta + DAMPING * jnp.eye(2))
        rmse_hist.append(rmse)
        logdet_sig_hist.append(float(-ld))

        q = q_next

    return {
        "rmse":        np.array(rmse_hist),        # (N_STEPS+1,)
        "logdet_sigma": np.array(logdet_sig_hist),  # (N_STEPS+1,)
    }


# ── Statistical helpers ───────────────────────────────────────────────────────

def bootstrap_ci(data2d: np.ndarray, n_boot: int = 2000, ci: float = 0.95):
    """Per-step bootstrap CI. data2d shape: (n_seeds, n_steps).

    Returns (mean, lo, hi) each shape (n_steps,).
    """
    n = data2d.shape[0]
    rng = np.random.default_rng(0)
    means = np.mean(data2d, axis=0)
    boots = np.stack([
        np.mean(data2d[rng.integers(0, n, n)], axis=0)
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boots, alpha * 100, axis=0)
    hi = np.percentile(boots, (1 - alpha) * 100, axis=0)
    return means, lo, hi


def convergence_step(rmse_arr: np.ndarray, thresh: float) -> int:
    """First t where RMSE < thresh. Returns N_STEPS+1 if never."""
    idxs = np.where(rmse_arr < thresh)[0]
    return int(idxs[0]) if len(idxs) > 0 else len(rmse_arr)


def auc_rmse(rmse_arr: np.ndarray) -> float:
    """Σ_t RMSE(t): total accumulated parameter error (step 0 excluded)."""
    return float(np.sum(rmse_arr[1:]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"Step 1: Multi-seed Comparison  ({N_SEEDS} seeds × {N_STEPS} steps)")
    print(f"  theta_true={list(THETA_TRUE)},  theta_init={list(THETA_INIT)}")
    print(f"  sigma_obs={SIGMA_OBS} m,  CONV_THRESH={CONV_THRESH} m")
    print("=" * 65)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Run all seeds ─────────────────────────────────────────────────────────
    all_rmse    = {}   # cond -> (N_SEEDS, N_STEPS+1)
    all_logdet  = {}
    conv_steps  = {}   # cond -> (N_SEEDS,)
    auc_values  = {}   # cond -> (N_SEEDS,)

    for cond in CONDITIONS:
        print(f"Running '{cond}' ({N_SEEDS} seeds)...", end="", flush=True)
        rmse_list, ld_list = [], []
        conv_list, auc_list = [], []

        for seed in range(N_SEEDS):
            r = run_one_seed(cond, seed=seed * 100 + 42)
            rmse_list.append(r["rmse"])
            ld_list.append(r["logdet_sigma"])
            conv_list.append(convergence_step(r["rmse"], CONV_THRESH))
            auc_list.append(auc_rmse(r["rmse"]))
            print(".", end="", flush=True)

        all_rmse[cond]   = np.stack(rmse_list)
        all_logdet[cond] = np.stack(ld_list)
        conv_steps[cond] = np.array(conv_list)
        auc_values[cond] = np.array(auc_list)
        print(f" done.  median conv={np.median(conv_list):.0f}  "
              f"AUC={np.mean(auc_list):.3f}")

    print()

    # ── Per-condition summary ─────────────────────────────────────────────────
    print("=" * 65)
    print("Summary Statistics")
    print(f"  {'Condition':<14} {'RMSE@30 mean':>13} {'AUC mean':>10} "
          f"{'Conv step':>12} {'Conv rate':>11}")
    print("-" * 65)
    for cond in CONDITIONS:
        final_rmse = all_rmse[cond][:, -1]
        cs = conv_steps[cond]
        conv_rate = np.mean(cs < N_STEPS + 1) * 100
        print(f"  {cond:<14} {np.mean(final_rmse):>13.4f} "
              f"{np.mean(auc_values[cond]):>10.3f} "
              f"{np.median(cs):>8.0f} (IQR {np.percentile(cs,25):.0f}–{np.percentile(cs,75):.0f})"
              f"  {conv_rate:>6.0f}%")
    print()

    # ── Mann-Whitney U tests ──────────────────────────────────────────────────
    print("Mann-Whitney U tests (one-sided: epistemic < other)")
    stats_out = {}
    for other in ["random", "sinusoidal"]:
        for metric_name, values in [("final_RMSE", all_rmse),
                                     ("AUC_RMSE",   auc_values)]:
            ep_vals  = values["epistemic"][:, -1] if metric_name == "final_RMSE" else values["epistemic"]
            oth_vals = values[other][:, -1]       if metric_name == "final_RMSE" else values[other]
            stat, p  = mannwhitneyu(ep_vals, oth_vals, alternative="less")
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
            print(f"  epistemic vs {other:<12} [{metric_name}]  "
                  f"U={stat:.0f}  p={p:.4f}  {sig}")
            stats_out[f"epistemic_vs_{other}_{metric_name}"] = {"U": float(stat), "p": float(p)}
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {}
    for cond in CONDITIONS:
        final_rmse = all_rmse[cond][:, -1]
        cs = conv_steps[cond]
        summary[cond] = {
            "final_rmse_mean":   float(np.mean(final_rmse)),
            "final_rmse_std":    float(np.std(final_rmse)),
            "auc_mean":          float(np.mean(auc_values[cond])),
            "auc_std":           float(np.std(auc_values[cond])),
            "conv_step_median":  float(np.median(cs)),
            "conv_step_q25":     float(np.percentile(cs, 25)),
            "conv_step_q75":     float(np.percentile(cs, 75)),
            "conv_rate":         float(np.mean(cs < N_STEPS + 1)),
        }
    json_path = results_dir / "multiseed_comparison_stats.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "mann_whitney": stats_out}, f, indent=2)
    print(f"Stats saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        steps = np.arange(N_STEPS + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: RMSE curves with 95% CI
        ax = axes[0, 0]
        for cond in CONDITIONS:
            mean, lo, hi = bootstrap_ci(all_rmse[cond])
            ax.plot(steps, mean, color=COLORS[cond], lw=2.0, label=cond)
            ax.fill_between(steps, lo, hi, color=COLORS[cond], alpha=0.18)
        ax.axhline(CONV_THRESH, color="k", lw=1.0, ls="--", alpha=0.5,
                   label=f"conv threshold ({CONV_THRESH} m)")
        ax.set_xlabel("Step")
        ax.set_ylabel("||θ_est − θ_true|| (m)")
        ax.set_title(f"Parameter RMSE  (mean ± 95% CI, {N_SEEDS} seeds)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: logdet(Σ) with 95% CI
        ax = axes[0, 1]
        for cond in CONDITIONS:
            mean, lo, hi = bootstrap_ci(all_logdet[cond])
            ax.plot(steps, mean, color=COLORS[cond], lw=2.0, label=cond)
            ax.fill_between(steps, lo, hi, color=COLORS[cond], alpha=0.18)
        ax.set_xlabel("Step")
        ax.set_ylabel("logdet(Σ_θ)  [lower = more certain]")
        ax.set_title(f"Posterior Uncertainty  (mean ± 95% CI, {N_SEEDS} seeds)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 3: Convergence step distribution (box plot)
        ax = axes[1, 0]
        data_conv = [conv_steps[c] for c in CONDITIONS]
        bp = ax.boxplot(data_conv, labels=CONDITIONS, patch_artist=True,
                        medianprops={"color": "black", "lw": 2})
        for patch, cond in zip(bp["boxes"], CONDITIONS):
            patch.set_facecolor(COLORS[cond])
            patch.set_alpha(0.7)
        ax.axhline(N_STEPS + 1, color="gray", ls=":", lw=1.0, alpha=0.6,
                   label="censored (never converged)")
        ax.set_ylabel("Step to RMSE < 0.05 m")
        ax.set_title(f"Convergence Speed Distribution  (thresh={CONV_THRESH} m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 4: AUC RMSE distribution (box plot)
        ax = axes[1, 1]
        data_auc = [auc_values[c] for c in CONDITIONS]
        bp2 = ax.boxplot(data_auc, labels=CONDITIONS, patch_artist=True,
                         medianprops={"color": "black", "lw": 2})
        for patch, cond in zip(bp2["boxes"], CONDITIONS):
            patch.set_facecolor(COLORS[cond])
            patch.set_alpha(0.7)

        # Annotate Mann-Whitney results
        def annot_mw(ax, x1, x2, y, p):
            sig = "p<.05 *" if p < 0.05 else (f"p={p:.2f}")
            ax.plot([x1, x1, x2, x2], [y, y * 1.03, y * 1.03, y], "k-", lw=0.8)
            ax.text((x1 + x2) / 2, y * 1.04, sig, ha="center", va="bottom", fontsize=8)

        y_top = max(np.max(auc_values[c]) for c in CONDITIONS) * 1.1
        for j, other in enumerate(["random", "sinusoidal"]):
            p_val = stats_out[f"epistemic_vs_{other}_AUC_RMSE"]["p"]
            annot_mw(ax, CONDITIONS.index("epistemic") + 1,
                     CONDITIONS.index(other) + 1,
                     y_top * (1.0 + j * 0.15), p_val)

        ax.set_ylabel("AUC RMSE  (Σ_t RMSE(t), lower = better)")
        ax.set_title("Total Accumulated Error  (Mann-Whitney annotation)")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Step 1: Multi-seed Statistical Comparison  "
            f"({N_SEEDS} seeds × {N_STEPS} steps)\n"
            f"θ_true=[0.5,0.5], θ_init=[0.7,0.3], σ_obs={SIGMA_OBS}m",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig_path = results_dir / "multiseed_comparison.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    print("Done.")


if __name__ == "__main__":
    main()
