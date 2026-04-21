"""DEM-based physical parameter identification via windowed integral regression.

Demonstrates how DEM's D-step improves parameter identification of a damped
oscillator by providing smoother velocity estimates than finite differences.

System:
    Damped harmonic oscillator: θ'' = -k·θ - c·θ'
    True parameters: k=5.0 (spring), c=1.0 (damping)
    Observation: only noisy θ (position), noise σ=0.05 rad

Why raw double-FD regression fails:
    α_fd = Δ²θ_obs / dt²  amplifies noise: σ_α ≈ σ/dt² = 0.05/0.02² ≈ 125 rad/s²
    This is 25× the true signal (|α_true| ≤ k·|θ_max| ≈ 5 rad/s²).

Methods compared:
    A. Raw FD double-diff: α_fd = Δ²θ_obs/dt² → linear regression α = -k·θ - c·ω
       • Extremely noisy: k≈87 (25× true value)

    B. DEM D-step + windowed integral regression:
       • DEM kinematic filter: estimates smoother [θ_dem, ω_dem] from noisy θ_obs
       • Windowed Euler integral:
             Δω_dem(t, W) / W ≈ -k·θ̄ - c·ω̄   (average over window W)
         This uses only ONE finite difference on the DEM velocity (much less noisy
         than double FD of position), and averaging over W steps reduces noise.
       • Result: k≈6 (20% error), clearly better than raw FD

    C. FD + windowed integral (same approach using raw FD velocity):
       • Shows DEM denoising provides additional ~30% improvement over FD+windowed

Key insight:
    The windowed integral converts θ'' = -k·θ - c·θ' into:
        ω(t+W) - ω(t) = ∫_t^{t+W} (-k·θ - c·ω) dτ ≈ W·(-k·θ̄ - c·ω̄)
    This requires only a SINGLE velocity difference (not second derivative of position),
    reducing noise amplification from σ/dt² to ≈ σ_ω / W.
    DEM reduces σ_ω via Wiener filtering, giving a further advantage.

Usage:
    .venv/bin/python experiments/dem_estep_param_identification.py

Output:
    results/dem_estep_param_identification.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pin to CPU; workload is small-tensor / sequential

import jax.numpy as jnp

from src.dem.model import DEMModel
from src.dem.inference import DStep


# ============================================================
# System parameters
# ============================================================

K_TRUE  = 5.0   # spring constant (rad/s²  per rad)
C_TRUE  = 1.0   # viscous damping (rad/s² per rad/s)

THETA_0 = 1.0   # initial angle (rad)
OMEGA_0 = 0.0   # initial angular velocity (rad/s)

T_END   = 15.0
OBS_DT  = 0.02   # observation interval (s)
N_STEPS = int(T_END / OBS_DT)

NOISE_STD = 0.05  # encoder noise (rad)
SEED      = 42


# ============================================================
# DEM parameters
# ============================================================

N_X     = 2    # state: [θ, ω]
N_Y     = 1    # observe θ only
N_ORDER = 2    # generalized embedding order
N_V     = 1

PI_Y    = 8.0
PI_X    = 2.0
S_Y     = 1.0
S_X     = 1.0

N_ITER_D_STEP = 128
D_STEP_DT     = 0.01
KAPPA_MU      = 1.0

# Window sizes for windowed regression (in seconds)
WINDOWS_S = [0.2, 0.5, 1.0]


# ============================================================
# DEM kinematic model (f[0]=ω, rest=0)
# ============================================================

def build_kinematic_model() -> DEMModel:
    """DEM model with kinematic constraint dθ/dt = ω, rest smoothness prior.

    State x = [θ, ω].  x_tilde (order-first) = [θ, ω, dθ/dt, dω/dt].

    f[0] = ω  (kinematic: dθ/dt = ω)
    f[1:] = 0 (smoothness prior on dω/dt)

    g = [x_tilde[0], x_tilde[2]] = [θ, dθ/dt]  (observe θ at orders 0 and 1)
    y_tilde = [θ_obs, ω_fd]  (noisy position + single-FD velocity)
    """
    n_dim = N_X * N_ORDER

    def f(x_tilde, v_tilde, params):
        return jnp.zeros(n_dim).at[0].set(x_tilde[1])  # dθ/dt = ω

    def g(x_tilde, v_tilde, params):
        return jnp.array([x_tilde[0], x_tilde[2]])  # θ and dθ/dt (generalized obs)

    return DEMModel(
        f=f, g=g,
        n_x=N_X, n_v=N_V, n_y=N_Y,
        n_order=N_ORDER,
        pi_y=PI_Y, pi_x=PI_X,
        s_y=S_Y, s_x=S_X,
    )


# ============================================================
# Physics simulation
# ============================================================

def simulate(rng) -> tuple:
    """Simulate damped oscillator."""
    theta, omega = THETA_0, OMEGA_0
    theta_true_l = [theta]; omega_true_l = [omega]; t_l = [0.0]

    for k in range(N_STEPS):
        alpha     = -K_TRUE * theta - C_TRUE * omega
        omega    += OBS_DT * alpha
        theta    += OBS_DT * omega
        theta_true_l.append(theta); omega_true_l.append(omega)
        t_l.append((k + 1) * OBS_DT)

    t          = np.array(t_l)
    theta_true = np.array(theta_true_l)
    omega_true = np.array(omega_true_l)
    theta_obs  = theta_true + rng.normal(0.0, NOISE_STD, size=theta_true.shape)
    return t, theta_true, omega_true, theta_obs


# ============================================================
# Method A: Raw double-FD regression
# ============================================================

def estimate_double_fd(theta_obs: np.ndarray) -> tuple:
    """θ'' ≈ Δ²θ/dt²  →  regression α = -k·θ - c·ω."""
    omega_fd = np.gradient(theta_obs, OBS_DT)
    alpha_fd = np.gradient(omega_fd, OBS_DT)
    m = 3
    X = np.column_stack([theta_obs[m:-m], omega_fd[m:-m]])
    y = -alpha_fd[m:-m]
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[0]), float(c[1])


# ============================================================
# DEM D-step to get smooth [θ_dem, ω_dem]
# ============================================================

def run_dem_filter(theta_obs: np.ndarray) -> tuple:
    """Run DEM kinematic D-step to estimate smooth θ and ω.

    Returns (theta_dem, omega_dem) — both shape (N_STEPS,).
    """
    model  = build_kinematic_model()
    d_step = DStep(model, kappa_mu=KAPPA_MU, dt=D_STEP_DT,
                   n_iter=N_ITER_D_STEP, use_d_operator=False)

    mu_x      = jnp.zeros(N_X * N_ORDER).at[0].set(theta_obs[0])
    mu_v      = jnp.zeros(N_Y * N_ORDER)
    theta_dem = []
    omega_dem = []
    theta_prev = theta_obs[0]

    for k in range(1, len(theta_obs)):
        theta_o  = theta_obs[k]
        omega_fd = (theta_o - theta_prev) / OBS_DT
        y_tilde  = (jnp.zeros(N_Y * N_ORDER)
                    .at[0].set(theta_o)
                    .at[1].set(omega_fd))
        mu_x, mu_v, _ = d_step.run(mu_x, mu_v, y_tilde)
        theta_dem.append(float(mu_x[0]))
        omega_dem.append(float(mu_x[1]))
        theta_prev = theta_o

    return np.array(theta_dem), np.array(omega_dem)


# ============================================================
# Method B/C: Windowed integral regression
# ============================================================

def windowed_regression(theta_arr: np.ndarray,
                        omega_arr: np.ndarray,
                        W_steps: int) -> tuple:
    """Windowed Euler integral regression.

    Converts θ'' = -k·θ - c·θ' to:
        Δω(t, W) / W ≈ -k·θ̄ - c·ω̄
    using velocity difference over W steps and mean position/velocity.

    Returns (k_hat, c_hat).
    """
    X_list, y_list = [], []
    n = len(omega_arr)
    for i in range(n - W_steps):
        delta_omega = omega_arr[i + W_steps] - omega_arr[i]
        theta_bar   = theta_arr[i : i + W_steps].mean()
        omega_bar   = omega_arr[i : i + W_steps].mean()
        X_list.append([theta_bar, omega_bar])
        y_list.append(-delta_omega / (W_steps * OBS_DT))

    X = np.array(X_list)
    y = np.array(y_list)
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[0]), float(c[1])


# ============================================================
# Main
# ============================================================

def main():
    noise_in_ddot = NOISE_STD / (OBS_DT ** 2)
    print("=" * 60)
    print("DEM Parameter Identification via Windowed Integral Regression")
    print(f"  System: θ'' = -k·θ - c·θ'")
    print(f"  True: k={K_TRUE}, c={C_TRUE}")
    print(f"  Observation noise σ = {NOISE_STD} rad")
    print(f"  Raw FD noise in α:  σ/dt² ≈ {noise_in_ddot:.0f} rad/s²  "
          f"(signal ≈ {K_TRUE:.0f} rad/s²)")
    print("=" * 60)
    print()

    rng = np.random.default_rng(SEED)
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Simulate
    print("Step 1: Simulate")
    t, theta_true, omega_true, theta_obs = simulate(rng)
    print(f"  N={N_STEPS} steps, dt={OBS_DT}s, σ={NOISE_STD} rad")
    print()

    # DEM filter
    print("Step 2: DEM D-step (kinematic filter)")
    theta_dem, omega_dem = run_dem_filter(theta_obs)
    omega_true_inner = omega_true[1:]
    omega_fd_full    = np.diff(theta_obs) / OBS_DT
    omega_rms_fd     = float(np.sqrt(np.mean((omega_fd_full  - omega_true_inner) ** 2)))
    omega_rms_dem    = float(np.sqrt(np.mean((omega_dem      - omega_true_inner) ** 2)))
    print(f"  Velocity RMS error: FD={omega_rms_fd:.4f}  DEM={omega_rms_dem:.4f} "
          f"({(omega_rms_fd - omega_rms_dem) / omega_rms_fd * 100:.1f}% reduction)")
    print()

    # Method A: raw double-FD
    print("Step 3: Method A — Raw double-FD regression")
    k_fd2, c_fd2 = estimate_double_fd(theta_obs)
    print(f"  k={k_fd2:.3f} (Δk={abs(k_fd2-K_TRUE):.3f}),  c={c_fd2:.3f} (Δc={abs(c_fd2-C_TRUE):.3f})")
    print()

    # Method B: DEM + windowed
    print("Step 4: Method B — DEM + windowed integral regression")
    dem_results = {}
    for W_s in WINDOWS_S:
        W = int(W_s / OBS_DT)
        k_w, c_w = windowed_regression(theta_dem, omega_dem, W)
        dem_results[W_s] = (k_w, c_w)
        print(f"  W={W_s:.2f}s:  k={k_w:.3f} (Δk={abs(k_w-K_TRUE):.3f}),  "
              f"c={c_w:.3f} (Δc={abs(c_w-C_TRUE):.3f})")
    print()

    # Method C: FD + windowed (for comparison)
    print("Step 5: Method C — FD + windowed integral regression")
    fd_results = {}
    for W_s in WINDOWS_S:
        W = int(W_s / OBS_DT)
        # Align FD velocity array with theta_obs
        theta_for_fd = theta_obs[1:]   # align with omega_fd_full
        k_w, c_w = windowed_regression(theta_for_fd, omega_fd_full, W)
        fd_results[W_s] = (k_w, c_w)
        print(f"  W={W_s:.2f}s:  k={k_w:.3f} (Δk={abs(k_w-K_TRUE):.3f}),  "
              f"c={c_w:.3f} (Δc={abs(c_w-C_TRUE):.3f})")
    print()

    # Summary
    print("=" * 60)
    print("Summary (best windowed results at W=1.0s)")
    print("=" * 60)
    k_dem1, c_dem1 = dem_results[1.0]
    k_fd1,  c_fd1  = fd_results[1.0]
    print(f"  True:               k={K_TRUE:.3f}, c={C_TRUE:.3f}")
    print(f"  Raw double-FD:      k={k_fd2:.3f} (Δk={abs(k_fd2-K_TRUE):.3f}),  "
          f"c={c_fd2:.3f} (Δc={abs(c_fd2-C_TRUE):.3f})")
    print(f"  FD + windowed 1s:   k={k_fd1:.3f} (Δk={abs(k_fd1-K_TRUE):.3f}),  "
          f"c={c_fd1:.3f} (Δc={abs(c_fd1-C_TRUE):.3f})")
    print(f"  DEM + windowed 1s:  k={k_dem1:.3f} (Δk={abs(k_dem1-K_TRUE):.3f}),  "
          f"c={c_dem1:.3f} (Δc={abs(c_dem1-C_TRUE):.3f})")
    print("=" * 60)

    # Plot
    print()
    print("Step 6: Plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_dem = t[1:]   # DEM starts from step 1

        fig, axes = plt.subplots(3, 1, figsize=(12, 11))

        # Panel 1: angle + DEM estimate
        ax = axes[0]
        ax.plot(t, theta_true, 'k-', lw=1.5, label='True θ')
        ax.plot(t, theta_obs, 'gray', lw=0.6, alpha=0.5,
                label=f'Noisy obs (σ={NOISE_STD})')
        ax.plot(t_dem, theta_dem, 'b-', lw=1.5, label='DEM est θ')
        ax.set_ylabel('Angle θ (rad)')
        ax.set_title('Damped Oscillator: DEM Kinematic State Estimation')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 2: velocity (key for regression quality)
        ax = axes[1]
        ax.plot(t, omega_true, 'k-', lw=1.5, label='True ω')
        ax.plot(t_dem, omega_fd_full, 'r-', lw=0.8, alpha=0.5,
                label=f'FD ω (RMS err={omega_rms_fd:.3f})')
        ax.plot(t_dem, omega_dem, 'b-', lw=1.5,
                label=f'DEM est ω (RMS err={omega_rms_dem:.3f})')
        ax.set_ylabel('Angular velocity ω (rad/s)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 3: parameter identification results
        ax = axes[2]
        windows = WINDOWS_S
        k_dems = [dem_results[w][0] for w in windows]
        c_dems = [dem_results[w][1] for w in windows]
        k_fds  = [fd_results[w][0]  for w in windows]
        c_fds  = [fd_results[w][1]  for w in windows]

        x_pos = np.arange(len(windows))
        width = 0.2

        ax.axhline(K_TRUE, color='darkblue', ls='--', lw=1.5, alpha=0.7,
                   label=f'True k={K_TRUE}')
        ax.axhline(C_TRUE, color='darkred', ls='--', lw=1.5, alpha=0.7,
                   label=f'True c={C_TRUE}')
        ax.bar(x_pos - width, k_dems, width, color='blue', alpha=0.7,
               label='DEM k')
        ax.bar(x_pos,         k_fds,  width, color='red',  alpha=0.7,
               label='FD k')
        ax.bar(x_pos + width, c_dems, width, color='cyan', alpha=0.7,
               label='DEM c')
        ax.bar(x_pos + 2*width, c_fds, width, color='orange', alpha=0.7,
               label='FD c')
        ax.bar([-1.0], [k_fd2], 0.3, color='maroon', alpha=0.7,
               label=f'Double-FD k={k_fd2:.0f}')
        ax.set_xticks(x_pos); ax.set_xticklabels([f'W={w}s' for w in windows])
        ax.set_ylabel('Parameter estimate')
        ax.set_title('Parameter Identification: DEM vs FD (windowed integral regression)')
        ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim([-1.5, len(windows)])
        ax.set_ylim([0, max(12, k_fd2 * 0.1)])

        fig.suptitle(
            f'DEM Parameter ID: θ\'\'= -k·θ - c·θ\'   (true: k={K_TRUE}, c={C_TRUE})\n'
            f'σ={NOISE_STD}rad,  dt={OBS_DT}s  →  raw double-FD: k={k_fd2:.0f}  |  '
            f'DEM+windowed 1s: k={k_dem1:.2f}, c={c_dem1:.2f}',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()

        fig_path = results_dir / "dem_estep_param_identification.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found — skipping")


if __name__ == "__main__":
    main()
