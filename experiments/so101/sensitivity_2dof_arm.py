"""Sensitivity analysis: noise level and observation frequency.

Runs a systematic 2D sweep (noise_std x obs_freq) for four methods:
    1. FD (finite-difference velocity)
    2. DEM smooth  (f=0, smoothness prior)
    3. DEM gravity (physics-aware, diagonal precision)
    4. EKF         (Extended Kalman Filter with full physics model)

For each condition, reports angle RMSE and velocity RMSE averaged over both
joints. Results are saved as CSV and plotted as heat maps.

Research question:
    Under which conditions (high noise / low frequency) does DEM catch up to
    or outperform EKF? When is the physics prior in DEM actually valuable?

Usage:
    uv run --group so101 python experiments/so101/sensitivity_2dof_arm.py

Output:
    results/sensitivity_2dof_arm.png
    results/sensitivity_2dof_arm.csv
"""

import sys
import math
from pathlib import Path
from itertools import product

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import jax.numpy as jnp

# ── reuse physics constants and functions from the main demo ──────────────────
from experiments.so101.dem_mujoco_2dof_arm import (
    ARM_XML, Q0, DQ0,
    MASS, LINK_LEN, LC, G_ACCEL, DAMPING,
    N_JOINTS, N_ORDER,
    PI_Y, PI_X_SMOOTH, PI_X_GRAVITY, S_SMOOTH,
    N_ITER_SMOOTH, DT_SMOOTH, KAPPA_SMOOTH,
    N_ITER_GRAV, DT_GRAV, KAPPA_GRAV,
    EKF_Q_POS, EKF_Q_VEL,
    _arm_dynamics,
    build_dem_model_smoothness,
    build_dem_model_gravity,
    run_dem_smoothness,
    run_dem_gravity,
    run_ekf,
)
from src.dem.model import DEMModel

# ── sweep parameters ──────────────────────────────────────────────────────────

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10, 0.20]   # encoder noise std (rad)
OBS_FREQS    = [20, 10, 5, 2]                      # Hz

T_END   = 10.0
SIM_DT  = 0.002
N_SIM   = int(T_END / SIM_DT)

TORQUE_AMP1  = 2.5
TORQUE_FREQ1 = 0.6
TORQUE_AMP2  = 1.5
TORQUE_FREQ2 = 0.8

SEED = 42


# ── simulation ────────────────────────────────────────────────────────────────

def run_simulation(obs_dt: float, noise_std: float):
    """Run MuJoCo simulation and return observations at given OBS_DT/noise."""
    import mujoco

    obs_every = int(round(obs_dt / SIM_DT))
    n_obs     = int(T_END / obs_dt)

    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data  = mujoco.MjData(model)
    data.qpos[:2] = Q0
    data.qvel[:2] = DQ0

    rng = np.random.default_rng(SEED)

    t_obs   = np.zeros(n_obs)
    q_true  = np.zeros((n_obs, N_JOINTS))
    dq_true = np.zeros((n_obs, N_JOINTS))
    y_obs   = np.zeros((n_obs, N_JOINTS))
    tau_obs = np.zeros((n_obs, N_JOINTS))

    obs_idx = 0
    for step in range(N_SIM):
        t    = step * SIM_DT
        tau1 = TORQUE_AMP1 * math.sin(2.0 * math.pi * TORQUE_FREQ1 * t)
        tau2 = TORQUE_AMP2 * math.sin(2.0 * math.pi * TORQUE_FREQ2 * t)

        if step % obs_every == 0 and obs_idx < n_obs:
            t_obs[obs_idx]   = t
            q_true[obs_idx]  = data.qpos[:2].copy()
            dq_true[obs_idx] = data.qvel[:2].copy()
            y_obs[obs_idx]   = data.qpos[:2] + rng.normal(0.0, noise_std, 2)
            tau_obs[obs_idx] = [tau1, tau2]
            obs_idx += 1

        data.ctrl[0] = tau1
        data.ctrl[1] = tau2
        mujoco.mj_step(model, data)

    return q_true, dq_true, y_obs, tau_obs


def compute_fd(y_obs: np.ndarray, obs_dt: float) -> np.ndarray:
    N = len(y_obs)
    dq_fd = np.zeros_like(y_obs)
    for i in range(N):
        if 0 < i < N - 1:
            dq_fd[i] = (y_obs[i + 1] - y_obs[i - 1]) / (2.0 * obs_dt)
        elif i == 0:
            dq_fd[i] = (y_obs[1]  - y_obs[0])  / obs_dt
        else:
            dq_fd[i] = (y_obs[-1] - y_obs[-2]) / obs_dt
    return dq_fd


def build_y_tilde(y_obs: np.ndarray, dq_fd: np.ndarray) -> list:
    N = len(y_obs)
    out = []
    for i in range(N):
        yt = np.zeros(N_ORDER * N_JOINTS)
        yt[0] = y_obs[i, 0]
        yt[1] = y_obs[i, 1]
        yt[2] = dq_fd[i, 0]
        yt[3] = dq_fd[i, 1]
        out.append(jnp.array(yt))
    return out


# ── per-condition evaluation ──────────────────────────────────────────────────

def evaluate_condition(noise_std: float, obs_freq: float):
    """Run all four methods for one (noise, freq) condition.

    Returns dict with keys: fd_q, fd_v, smooth_q, smooth_v,
                             grav_q, grav_v, ekf_q, ekf_v
    (each is the mean RMSE over both joints, in rad or rad/s)
    """
    obs_dt = 1.0 / obs_freq

    q_true, dq_true, y_obs, tau_obs = run_simulation(obs_dt, noise_std)
    dq_fd = compute_fd(y_obs, obs_dt)

    # RMSE helper
    def rmse2(est, ref):
        return float(np.sqrt(np.mean((est - ref) ** 2)))

    # ── FD baseline ──
    fd_q = rmse2(y_obs,  q_true)
    fd_v = rmse2(dq_fd, dq_true)

    y_tilde_list = build_y_tilde(y_obs, dq_fd)

    # ── DEM smooth ──
    model_smooth = build_dem_model_smoothness()
    q_sm, dq_sm, _ = run_dem_smoothness(model_smooth, y_tilde_list, y_obs, dq_fd)
    sm_q = rmse2(q_sm,  q_true)
    sm_v = rmse2(dq_sm, dq_true)

    # ── DEM gravity ──
    model_grav = build_dem_model_gravity()
    q_gr, dq_gr, _ = run_dem_gravity(model_grav, y_tilde_list, tau_obs, y_obs, dq_fd)
    gr_q = rmse2(q_gr,  q_true)
    gr_v = rmse2(dq_gr, dq_true)

    # ── EKF ──
    # Pass obs_dt to EKF via a local wrapper (run_ekf uses SIM_DT internally)
    q_ek, dq_ek = _run_ekf_with_dt(y_obs, tau_obs, dq_fd, obs_dt, noise_std)
    ek_q = rmse2(q_ek,  q_true)
    ek_v = rmse2(dq_ek, dq_true)

    return dict(fd_q=fd_q, fd_v=fd_v,
                sm_q=sm_q, sm_v=sm_v,
                gr_q=gr_q, gr_v=gr_v,
                ek_q=ek_q, ek_v=ek_v)


def _run_ekf_with_dt(y_obs, tau_obs, dq_fd, obs_dt, noise_std):
    """EKF wrapper that accepts arbitrary obs_dt and noise_std."""
    import jax
    import jax.numpy as jnp

    n_x = 4
    n_y = 2
    N   = len(y_obs)

    ekf_r = noise_std ** 2
    Q = np.diag([EKF_Q_POS, EKF_Q_POS, EKF_Q_VEL, EKF_Q_VEL])
    R = np.diag([ekf_r, ekf_r])
    H = np.zeros((n_y, n_x)); H[0, 0] = H[1, 1] = 1.0
    I4 = np.eye(n_x)

    def _f_c(xj, uj):
        th1, th2, om1, om2 = xj; ta1, ta2 = uj
        al1, al2 = _arm_dynamics(th1, th2, om1, om2, ta1, ta2)
        return jnp.stack([om1, om2, al1, al2])

    def _f_rk4(xj, uj, dt_l):
        k1 = _f_c(xj,              uj)
        k2 = _f_c(xj + dt_l/2*k1, uj)
        k3 = _f_c(xj + dt_l/2*k2, uj)
        k4 = _f_c(xj + dt_l  *k3, uj)
        return xj + (dt_l / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    n_sub  = max(1, int(round(obs_dt / SIM_DT)))
    dt_sub = obs_dt / n_sub
    Q_sub  = Q / n_sub

    _f_sub   = jax.jit(lambda xj, uj: _f_rk4(xj, uj, dt_sub))
    _jac_sub = jax.jit(jax.jacobian(lambda xj, uj: _f_rk4(xj, uj, dt_sub),
                                     argnums=0))

    x = np.array([y_obs[0, 0], y_obs[0, 1], 0.0, 0.0])
    P = np.diag([ekf_r, ekf_r, 10.0, 10.0])

    q_est  = [[x[0], x[1]]]
    dq_est = [[x[2], x[3]]]

    for i in range(1, N):
        y  = y_obs[i]
        u  = tau_obs[i - 1].astype(float)
        uj = jnp.array(u)

        x_pred = x.copy()
        P_pred = P.copy()
        for _ in range(n_sub):
            xj     = jnp.array(x_pred)
            Fs     = np.array(_jac_sub(xj, uj))
            x_pred = np.array(_f_sub(xj, uj))
            P_pred = Fs @ P_pred @ Fs.T + Q_sub

        innov = y - H @ x_pred
        S     = H @ P_pred @ H.T + R
        K     = P_pred @ H.T @ np.linalg.inv(S)
        x     = x_pred + K @ innov
        I_KH  = I4 - K @ H
        P     = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        q_est.append([x[0], x[1]])
        dq_est.append([x[2], x[3]])

    return np.array(q_est), np.array(dq_est)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Sensitivity Analysis: noise x observation frequency")
    print(f"  Noise levels : {NOISE_LEVELS} rad")
    print(f"  Obs freqs    : {OBS_FREQS} Hz")
    print(f"  Methods      : FD, DEM smooth, DEM gravity, EKF")
    print("=" * 70)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    n_noise = len(NOISE_LEVELS)
    n_freq  = len(OBS_FREQS)
    total   = n_noise * n_freq

    # Storage: shape (n_noise, n_freq) for each metric
    keys = ["fd_q", "fd_v", "sm_q", "sm_v", "gr_q", "gr_v", "ek_q", "ek_v"]
    data = {k: np.full((n_noise, n_freq), np.nan) for k in keys}

    for ci, (ni, fi) in enumerate(product(range(n_noise), range(n_freq))):
        noise = NOISE_LEVELS[ni]
        freq  = OBS_FREQS[fi]
        print(f"  [{ci+1:2d}/{total}] noise={noise:.2f} rad, freq={freq:2d} Hz ...",
              flush=True, end=" ")
        res = evaluate_condition(noise, freq)
        for k in keys:
            data[k][ni, fi] = res[k]
        print(f"  EKF angle={res['ek_q']:.4f} vel={res['ek_v']:.4f}"
              f"  |  DEM-grav angle={res['gr_q']:.4f} vel={res['gr_v']:.4f}")

    # ── Save CSV ──
    csv_path = results_dir / "sensitivity_2dof_arm.csv"
    header = "noise_std,obs_freq_hz," + ",".join(keys)
    rows = []
    for ni, fi in product(range(n_noise), range(n_freq)):
        row = [NOISE_LEVELS[ni], OBS_FREQS[fi]] + [data[k][ni, fi] for k in keys]
        rows.append(",".join(f"{v:.6f}" for v in row))
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))
    print(f"\nCSV saved: {csv_path}")

    # ── Plot heat maps ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Relative improvement of DEM-gravity and EKF vs FD (for velocity)
        # and vs encoder (for angle) — positive = better than baseline
        def rel_improv(est, base):
            return (base - est) / (base + 1e-12) * 100.0

        imp_sm_q  = rel_improv(data["sm_q"],  data["fd_q"])
        imp_gr_q  = rel_improv(data["gr_q"],  data["fd_q"])
        imp_ek_q  = rel_improv(data["ek_q"],  data["fd_q"])
        imp_sm_v  = rel_improv(data["sm_v"],  data["fd_v"])
        imp_gr_v  = rel_improv(data["gr_v"],  data["fd_v"])
        imp_ek_v  = rel_improv(data["ek_v"],  data["fd_v"])

        xticklabels = [f"{f}Hz" for f in OBS_FREQS]
        yticklabels = [f"{n:.2f}" for n in NOISE_LEVELS]

        def heatmap(ax, mat, title, vmin=None, vmax=None, cmap="RdYlGn"):
            im = ax.imshow(mat, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, origin="upper")
            ax.set_xticks(range(n_freq))
            ax.set_xticklabels(xticklabels, fontsize=8)
            ax.set_yticks(range(n_noise))
            ax.set_yticklabels(yticklabels, fontsize=8)
            ax.set_xlabel("Obs frequency")
            ax.set_ylabel("Noise std (rad)")
            ax.set_title(title, fontsize=9)
            for ni in range(n_noise):
                for fi in range(n_freq):
                    ax.text(fi, ni, f"{mat[ni, fi]:.1f}%",
                            ha="center", va="center", fontsize=7,
                            color="black")
            return im

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        vmax_q = max(np.nanmax(imp_ek_q), 5)
        vmax_v = max(np.nanmax(imp_ek_v), 5)

        heatmap(axes[0, 0], imp_sm_q, "DEM smooth — Angle improvement vs FD (%)",
                vmin=-20, vmax=vmax_q)
        heatmap(axes[0, 1], imp_gr_q, "DEM gravity — Angle improvement vs FD (%)",
                vmin=-20, vmax=vmax_q)
        heatmap(axes[0, 2], imp_ek_q, "EKF — Angle improvement vs FD (%)",
                vmin=-20, vmax=vmax_q)

        heatmap(axes[1, 0], imp_sm_v, "DEM smooth — Vel improvement vs FD (%)",
                vmin=-20, vmax=vmax_v)
        heatmap(axes[1, 1], imp_gr_v, "DEM gravity — Vel improvement vs FD (%)",
                vmin=-20, vmax=vmax_v)
        heatmap(axes[1, 2], imp_ek_v, "EKF — Vel improvement vs FD (%)",
                vmin=-20, vmax=vmax_v)

        fig.suptitle(
            "Sensitivity: Noise level × Observation frequency\n"
            "2-DOF vertical arm, MuJoCo simulation\n"
            "Color = improvement over FD/encoder baseline (green=better)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fig_path = results_dir / "sensitivity_2dof_arm.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    # ── Print summary table ──
    print()
    print("=" * 70)
    print("Velocity RMSE (mean over joints, rad/s)")
    print("=" * 70)
    for method, key in [("FD", "fd_v"), ("DEM-smooth", "sm_v"),
                        ("DEM-grav", "gr_v"), ("EKF", "ek_v")]:
        print(f"\n  {method}")
        header_row = "  noise\\freq " + "".join(f"{f:>8}Hz" for f in OBS_FREQS)
        print(header_row)
        for ni, noise in enumerate(NOISE_LEVELS):
            row = f"  {noise:.2f}       " + "".join(
                f"{data[key][ni, fi]:>9.4f}" for fi in range(n_freq))
            print(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
