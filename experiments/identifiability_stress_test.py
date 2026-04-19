"""Identifiability Stress Test: 1D EE observation (x_ee only).

Key insight (identifiability):
    With y = x_ee = l1·cos(q1) + l2·cos(q1+q2)  [scalar observation],
    the per-step Jacobian is:
        J(q) = [cos(q1), cos(q1+q2)]   shape (1, 2)
    FIM_step = J.T @ (1/σ²) @ J         shape (2, 2), rank ≤ 1

    When q2 ≈ 0: J = [cos(q1), cos(q1)] → both columns equal → rank 0 per step.
    Accumulated over T steps, FIM is rank-2 only if the arm visits
    configurations with diverse q2 values (q2_a ≠ q2_b ≠ 0).

    Failure mode: if the arm stays near q2 = 0 throughout,
    only l1 + l2 is identifiable; l1 and l2 individually remain uncertain.

Conditions compared:
    - random    : uniform velocity commands → q2 explored by chance
    - sinusoidal: fixed-pattern q2 oscillation (small amplitude for q2)
    - epistemic : gradient ascent on IG → actively seeks q2 ≠ 0

Start: q0 = [π/4, 0.0]  (arm fully extended → worst identifiability)
Obs:   y = x_ee  (1D, rank-deficient FIM per step)

Metrics tracked per step (mean ± 95% CI across N_SEEDS):
    - RMSE l1, l2 individually
    - σ_l1, σ_l2 = sqrt(diag(P_θ⁻¹))   per-parameter posterior std
    - FIM condition number  κ = λ_max/λ_min of P_θ
    - q2 trajectory

Usage:
    .venv/bin/python experiments/identifiability_stress_test.py

Output:
    results/identifiability_stress_test.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem.model import DEMModel
from src.dem.estep import EStep


# ── Parameters ────────────────────────────────────────────────────────────────

N_SEEDS  = 50
N_STEPS  = 60

THETA_TRUE     = jnp.array([0.5, 0.5])
THETA_INIT     = jnp.array([0.7, 0.3])
THETA_PRIOR_PI = 0.2              # weak prior → FIM dominates
Q0             = jnp.array([jnp.pi / 4, 0.0])   # arm extended (q2 = 0 = worst case)
Q_CLIP         = 1.5
U_MAX          = 0.5              # max velocity command
DT             = 0.05
SIGMA_OBS      = 0.02             # observation noise on x_ee
DAMPING        = 1e-8

N_ESTEP_ITER     = 5
N_EPISTEMIC_ITER = 30
LR_EPISTEMIC     = 0.4
N_FUTURE_STEPS   = 8

# Sinusoidal: large q1 excitation, small q2 (simulates limited q2 workspace)
SIN_AMP1   = 0.8
SIN_OMEGA1 = 0.6
SIN_AMP2   = 0.15              # deliberately small q2 component
SIN_OMEGA2 = 0.4
SIN_PHASE2 = 1.1

CONDITIONS = ["random", "sinusoidal", "epistemic"]
COLORS     = {"random": "tab:blue", "sinusoidal": "tab:orange", "epistemic": "tab:green"}


# ── Kinematic model (1D observation: x_ee only) ───────────────────────────────

def x_ee(q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """End-effector x-position (scalar, wrapped as 1D array)."""
    val = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    return jnp.array([val])


def fk_full(q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Full EE position (for computing true RMSE only)."""
    px = theta[0] * jnp.cos(q[0]) + theta[1] * jnp.cos(q[0] + q[1])
    py = theta[0] * jnp.sin(q[0]) + theta[1] * jnp.sin(q[0] + q[1])
    return jnp.array([px, py])


def rollout_fn(q: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Single-step Euler kinematic update."""
    def step(q_curr, _): return q_curr + u * DT, None
    q_f, _ = jax.lax.scan(step, q, None, length=N_FUTURE_STEPS)
    return q_f


def y_future_fn(q: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Future x_ee after rolling out action u."""
    return x_ee(rollout_fn(q, u), theta)


def compute_fim(q: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """FIM = J.T @ (1/σ²) @ J, J = d(x_ee)/d(theta), shape (2,2)."""
    J = jax.jacfwd(lambda t: y_future_fn(q, u, t))(theta)  # (1, 2)
    return J.T @ (jnp.eye(1) / SIGMA_OBS**2) @ J


def compute_info_gain(P_theta: jnp.ndarray, FIM_future: jnp.ndarray) -> jnp.ndarray:
    I = jnp.eye(2)
    _, ld0 = jnp.linalg.slogdet(P_theta + DAMPING * I)
    _, ld1 = jnp.linalg.slogdet(P_theta + FIM_future + DAMPING * I)
    return 0.5 * (ld1 - ld0)


# ── DEM E-step (shared, JIT-compiled once) ────────────────────────────────────

def _build_estep() -> EStep:
    def f_zero(x, v, p): return jnp.zeros(2)
    def g_xee(x, v, p):  return x_ee(x, p)   # 1D observation

    model = DEMModel(
        f=f_zero, g=g_xee,
        n_x=2, n_v=2, n_y=1, n_order=1,
        pi_y=1.0 / SIGMA_OBS**2,
        pi_x=1.0,
        params=THETA_INIT,
        params_prior_mean=THETA_INIT,
        params_prior_pi=THETA_PRIOR_PI,
    )
    return EStep(model, use_gauss_newton=True)

ESTEP = _build_estep()


# ── Epistemic action optimizer ────────────────────────────────────────────────

@jax.jit
def optimize_epistemic_action(
    q: jnp.ndarray,
    theta_est: jnp.ndarray,
    P_theta: jnp.ndarray,
    u_init: jnp.ndarray,
) -> jnp.ndarray:
    """Gradient ascent on IG w.r.t. action u."""
    def ig_of_u(u):
        return compute_info_gain(P_theta, compute_fim(q, u, theta_est))

    def ascent_step(u, _):
        g = jax.grad(ig_of_u)(u)
        return jnp.clip(u + LR_EPISTEMIC * g, -U_MAX, U_MAX), None

    u_opt, _ = jax.lax.scan(ascent_step, u_init, None, length=N_EPISTEMIC_ITER)
    return u_opt


# ── Metrics from P_theta ──────────────────────────────────────────────────────

def p_theta_metrics(P_theta: jnp.ndarray) -> dict:
    """Compute per-parameter std and condition number from P_theta."""
    eigs = jnp.linalg.eigvalsh(P_theta)
    eigs_safe = jnp.where(eigs > 1e-12, eigs, jnp.ones_like(eigs) * 1e-12)
    cond_num = eigs_safe[-1] / eigs_safe[0]
    Sigma = jnp.linalg.inv(P_theta + DAMPING * jnp.eye(2))
    sigma_l1 = float(jnp.sqrt(jnp.abs(Sigma[0, 0])))
    sigma_l2 = float(jnp.sqrt(jnp.abs(Sigma[1, 1])))
    return {
        "cond":     float(cond_num),
        "sigma_l1": sigma_l1,
        "sigma_l2": sigma_l2,
        "logdet_sigma": float(-jnp.linalg.slogdet(P_theta + DAMPING * jnp.eye(2))[1]),
    }


# ── Single-seed simulation ────────────────────────────────────────────────────

def run_one_seed(condition: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    theta_est = THETA_INIT.copy()
    q = Q0.copy()
    P_theta = THETA_PRIOR_PI * jnp.eye(2)
    u_current = jnp.zeros(2)
    q_acc, v_acc, y_acc = [], [], []

    rmse_hist    = [float(jnp.linalg.norm(theta_est - THETA_TRUE))]
    rmse_l1_hist = [float(jnp.abs(theta_est[0] - THETA_TRUE[0]))]
    rmse_l2_hist = [float(jnp.abs(theta_est[1] - THETA_TRUE[1]))]
    q2_hist      = [float(q[1])]
    m0 = p_theta_metrics(P_theta)
    cond_hist    = [m0["cond"]]
    sigma_l1_hist = [m0["sigma_l1"]]
    sigma_l2_hist = [m0["sigma_l2"]]

    for t in range(N_STEPS):
        # ── Action ──────────────────────────────────────────────────────────
        if condition == "random":
            u = jnp.array(rng.standard_normal(2) * 0.5).clip(-U_MAX, U_MAX)

        elif condition == "sinusoidal":
            ts = t * DT
            u = jnp.array([
                SIN_AMP1 * np.sin(SIN_OMEGA1 * ts),
                SIN_AMP2 * np.sin(SIN_OMEGA2 * ts + SIN_PHASE2),
            ]).clip(-U_MAX, U_MAX)

        else:  # epistemic
            u = optimize_epistemic_action(q, theta_est, P_theta, u_current)
            u_current = u

        # ── Step arm ─────────────────────────────────────────────────────────
        q_next = jnp.clip(q + u * DT, -Q_CLIP, Q_CLIP)
        y_obs  = x_ee(q_next, THETA_TRUE) + jnp.array(
            rng.standard_normal(1) * SIGMA_OBS
        )

        q_acc.append(q_next)
        v_acc.append(jnp.zeros(2))
        y_acc.append(y_obs)

        # ── E-step ───────────────────────────────────────────────────────────
        theta_est = ESTEP.run(q_acc, v_acc, y_acc, theta_est, n_iter=N_ESTEP_ITER)
        theta_est = jnp.clip(theta_est, 0.05, 1.8)
        P_theta   = ESTEP.compute_precision(q_acc, v_acc, y_acc, theta_est)

        # ── Log ──────────────────────────────────────────────────────────────
        m = p_theta_metrics(P_theta)
        rmse_hist.append(float(jnp.linalg.norm(theta_est - THETA_TRUE)))
        rmse_l1_hist.append(float(jnp.abs(theta_est[0] - THETA_TRUE[0])))
        rmse_l2_hist.append(float(jnp.abs(theta_est[1] - THETA_TRUE[1])))
        q2_hist.append(float(q_next[1]))
        cond_hist.append(m["cond"])
        sigma_l1_hist.append(m["sigma_l1"])
        sigma_l2_hist.append(m["sigma_l2"])
        q = q_next

    return {
        "rmse":      np.array(rmse_hist),
        "rmse_l1":   np.array(rmse_l1_hist),
        "rmse_l2":   np.array(rmse_l2_hist),
        "q2":        np.array(q2_hist),
        "cond":      np.array(cond_hist),
        "sigma_l1":  np.array(sigma_l1_hist),
        "sigma_l2":  np.array(sigma_l2_hist),
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(data2d: np.ndarray, n_boot: int = 1000, ci: float = 0.95):
    n = data2d.shape[0]
    rng = np.random.default_rng(0)
    means = np.mean(data2d, axis=0)
    boots = np.stack([
        np.mean(data2d[rng.integers(0, n, n)], axis=0)
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (1 - ci) / 2 * 100, axis=0)
    hi = np.percentile(boots, (1 + ci) / 2 * 100, axis=0)
    return means, lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Identifiability Stress Test")
    print("  Observation: y = x_ee only  (1D, rank-deficient FIM per step)")
    print(f"  Start: q0=[π/4, 0.0]  (arm extended → worst identifiability)")
    print(f"  theta_true={list(THETA_TRUE)}, theta_init={list(THETA_INIT)}")
    print(f"  sigma_obs={SIGMA_OBS}m, N_SEEDS={N_SEEDS}, N_STEPS={N_STEPS}")
    print("=" * 65)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    all_data = {}
    for cond in CONDITIONS:
        print(f"Running '{cond}'...", end="", flush=True)
        seed_results = []
        for seed in range(N_SEEDS):
            seed_results.append(run_one_seed(cond, seed=seed * 100 + 7))
            print(".", end="", flush=True)
        # Stack each metric across seeds
        all_data[cond] = {
            key: np.stack([r[key] for r in seed_results])
            for key in seed_results[0]
        }
        final_rmse   = all_data[cond]["rmse"][:, -1]
        final_sigma_l1 = all_data[cond]["sigma_l1"][:, -1]
        final_sigma_l2 = all_data[cond]["sigma_l2"][:, -1]
        final_q2_abs = np.abs(all_data[cond]["q2"]).mean(axis=1)  # mean |q2| per seed
        print(f" done.  RMSE={np.mean(final_rmse):.4f}  "
              f"σ_l1={np.mean(final_sigma_l1):.4f}  σ_l2={np.mean(final_sigma_l2):.4f}  "
              f"mean|q2|={np.mean(final_q2_abs):.3f}rad")

    print()
    print("=" * 65)
    print(f"{'Metric':<22}  {'random':>10} {'sinusoidal':>12} {'epistemic':>11}")
    print("-" * 65)
    for key, label in [
        ("rmse",     "RMSE (l1+l2)"),
        ("rmse_l1",  "|Δl1|"),
        ("rmse_l2",  "|Δl2|"),
        ("sigma_l1", "σ_l1 (m)"),
        ("sigma_l2", "σ_l2 (m)"),
        ("cond",     "FIM cond (P_θ)"),
    ]:
        row = {c: np.mean(all_data[c][key][:, -1]) for c in CONDITIONS}
        print(f"  {label:<20} {row['random']:>10.4f} "
              f"{row['sinusoidal']:>12.4f} {row['epistemic']:>11.4f}")
    print()

    # Mean |q2| explored per condition
    for cond in CONDITIONS:
        q2_mean = np.mean(np.abs(all_data[cond]["q2"]), axis=(0, 1))
        print(f"  Mean |q2| explored [{cond}]: {q2_mean:.4f} rad")
    print("=" * 65)

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(N_STEPS + 1)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        def plot_ci(ax, data2d, steps, color, label):
            mean, lo, hi = bootstrap_ci(data2d)
            ax.plot(steps, mean, color=color, lw=2.0, label=label)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18)

        # Panel 1: Total RMSE
        ax = axes[0, 0]
        for cond in CONDITIONS:
            plot_ci(ax, all_data[cond]["rmse"], steps, COLORS[cond], cond)
        ax.set_xlabel("Step"); ax.set_ylabel("||θ_est − θ_true|| (m)")
        ax.set_title("Parameter RMSE (mean±95%CI)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 2: σ_l1 (posterior std of l1)
        ax = axes[0, 1]
        for cond in CONDITIONS:
            plot_ci(ax, all_data[cond]["sigma_l1"], steps, COLORS[cond], cond)
        ax.set_xlabel("Step"); ax.set_ylabel("σ_l1 (m)  [lower = more certain]")
        ax.set_title("Posterior Std of l1")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 3: σ_l2 (posterior std of l2)
        ax = axes[0, 2]
        for cond in CONDITIONS:
            plot_ci(ax, all_data[cond]["sigma_l2"], steps, COLORS[cond], cond)
        ax.set_xlabel("Step"); ax.set_ylabel("σ_l2 (m)  [lower = more certain]")
        ax.set_title("Posterior Std of l2")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 4: FIM condition number (log scale)
        ax = axes[1, 0]
        for cond in CONDITIONS:
            mean, lo, hi = bootstrap_ci(all_data[cond]["cond"])
            ax.semilogy(steps, mean, color=COLORS[cond], lw=2.0, label=cond)
            ax.fill_between(steps, np.maximum(lo, 1e-1), hi,
                            color=COLORS[cond], alpha=0.15)
        ax.set_xlabel("Step"); ax.set_ylabel("κ(P_θ) = λ_max/λ_min  [log, lower = better]")
        ax.set_title("FIM Condition Number of P_θ")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

        # Panel 5: q2 trajectory (mean absolute value)
        ax = axes[1, 1]
        for cond in CONDITIONS:
            mean, lo, hi = bootstrap_ci(np.abs(all_data[cond]["q2"]))
            ax.plot(steps, mean, color=COLORS[cond], lw=2.0, label=cond)
            ax.fill_between(steps, lo, hi, color=COLORS[cond], alpha=0.18)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("Step"); ax.set_ylabel("|q2| (rad)")
        ax.set_title("q2 Explored (|q2| mean±CI)\n"
                     "← larger = better identifiability")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 6: |Δl1| and |Δl2| individually (final box plot)
        ax = axes[1, 2]
        x_pos = np.array([1, 2, 3])
        width = 0.3
        for i, (key, label, hatch) in enumerate([
            ("rmse_l1", "|Δl1|", ""),
            ("rmse_l2", "|Δl2|", "///"),
        ]):
            vals = [all_data[c][key][:, -1] for c in CONDITIONS]
            means = [np.mean(v) for v in vals]
            stds  = [np.std(v) for v in vals]
            bars = ax.bar(x_pos + (i - 0.5) * width, means,
                          width=width, yerr=stds,
                          color=[COLORS[c] for c in CONDITIONS],
                          alpha=0.7, hatch=hatch, label=label,
                          error_kw={"capsize": 4})

        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS, fontsize=9)
        ax.set_ylabel("Parameter error (m)")
        ax.set_title("Final |Δl1|, |Δl2|  (mean ± std)\n"
                     "solid = l1, hatched = l2")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Identifiability Stress Test: y = x_ee only  (1D, rank-deficient)\n"
            f"q0=[π/4, 0.0] (arm extended), σ_obs={SIGMA_OBS}m,  "
            f"{N_SEEDS} seeds × {N_STEPS} steps",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig_path = results_dir / "identifiability_stress_test.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    print("Done.")


if __name__ == "__main__":
    main()
