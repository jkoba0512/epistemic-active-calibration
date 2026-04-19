"""Physical parameter identification for SO-101 2-DOF arm.

Estimates the key dynamic parameters of a 2-DOF planar arm from
joint angle encoder measurements only (no force/torque sensors).

Identification procedure (3 experiments):
    1. Gravity: sweep applied torques, measure equilibrium angles,
       regress tau = -A*cos(theta).  (theta2=0 maintained via A2-compensation)
    2. Inertia + Damping: apply step torque WITH real-time gravity compensation
       so the net drive is a pure step.  Fit velocity trajectory to
       ω(t) = (τ/b)·(1 − e^{−b/I·t}).

Key insight (gravity compensation):
    Without it, theta drifts → G(theta) ≠ 0 → the assumed 1st-order model
    ω=(τ/b)(1-exp(-b/I·t)) breaks down (oscillation observed in simulation).
    With G(theta(t)) fed back as a cancellation torque, the effective drive
    is a known constant tau_step and the model applies.

In simulation (MuJoCo), true parameters are known for validation.
Replace the MuJoCo calls with real SO-101 I/O for field use.

Usage:
    uv run --group so101 python experiments/so101/param_id_2dof_arm.py

Output:
    results/param_id_2dof_arm.png
"""

import sys
import math
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.optimize import curve_fit

# ── True parameters ───────────────────────────────────────────────────────────

TRUE_MASS    = 1.0
TRUE_L       = 0.5
TRUE_LC      = 0.25
TRUE_G       = 9.81
TRUE_DAMPING = 2.0

I_CM  = TRUE_MASS * TRUE_L**2 / 12.0
TRUE_I1_EFF = (I_CM + TRUE_MASS * TRUE_LC**2) + \
              (I_CM + TRUE_MASS * (TRUE_L + TRUE_LC)**2)
TRUE_I2_EFF = I_CM + TRUE_MASS * TRUE_LC**2

TRUE_A1 = TRUE_MASS * TRUE_G * (2 * TRUE_LC + TRUE_L)   # 9.81 N·m
TRUE_A2 = TRUE_MASS * TRUE_G * TRUE_LC                   # 2.4525 N·m

NOISE_STD = 0.005
SEED      = 42

ARM_XML = """
<mujoco model="2dof_arm_param_id">
  <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="link1">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="2.0"/>
      <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.0008 0.0208 0.0208"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="2.0"/>
        <inertial pos="0.25 0 0" mass="1.0" diaginertia="0.0008 0.0208 0.0208"/>
        <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="torque2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""


def _make_mujoco():
    import mujoco
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data  = mujoco.MjData(model)
    return model, data


# ── Gravity compensation torque ───────────────────────────────────────────────

def _gravity_torque(theta1, theta2, A1, A2):
    """Compute gravity generalized forces G1, G2.

    G1 = A_partial1*cos(theta1) + A2*cos(theta1+theta2)
      where A_partial1 = m*g*(lc+L)  (link1 CoM + link2 at its proximal end)
    G2 = A2*cos(theta1+theta2)
    """
    # A1 = m*g*(2*lc+L)  and  A2 = m*g*lc
    # G1 = m*g*(lc+L)*cos(theta1) + A2*cos(theta1+theta2)
    A1_partial = A1 - A2   # m*g*(2*lc+L) - m*g*lc = m*g*(lc+L)
    G1 = A1_partial * math.cos(theta1) + A2 * math.cos(theta1 + theta2)
    G2 = A2 * math.cos(theta1 + theta2)
    return G1, G2


# ── Experiment 1: Gravity ─────────────────────────────────────────────────────
# Procedure:
#   Sweep tau1 (with tau2 compensating theta2=0) → equilibrium theta1.
#   Regress: tau1_measured = -A1 * cos(theta1_eq)
#   Then sweep tau2 (at theta1=0, with tau1 compensating theta1=0) → eq theta2.
#   Regress: tau2_measured = -A2 * cos(theta1_eq + theta2_eq)
#
# theta2=0 compensation: tau2_comp = A2*cos(theta1)
#   so that joint2 feels only tau2=0 after compensation → theta2=0 at eq.

def _hold_steady(model, data, q_init, tau_fn, t_settle=5.0):
    """Settle to equilibrium under torque given by tau_fn(theta1, theta2).

    tau_fn: callable (theta1, theta2) -> [tau1, tau2]
    Returns mean angle over last 10% of settling (noise-averaged).
    """
    import mujoco

    rng    = np.random.default_rng(SEED)
    dt     = float(model.opt.timestep)
    n_step = int(t_settle / dt)
    n_avg  = max(1, n_step // 10)

    data.qpos[:2] = q_init
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)

    q_log = []
    for step in range(n_step):
        tau = tau_fn(data.qpos[0], data.qpos[1])
        data.ctrl[:2] = tau
        mujoco.mj_step(model, data)
        if step >= n_step - n_avg:
            q_log.append(data.qpos[:2].copy() + rng.normal(0, NOISE_STD, 2))

    return np.mean(q_log, axis=0)


def identify_gravity(n_configs=12):
    """Estimate A1 = m·g·(2lc+L) and A2 = m·g·lc.

    Joint 1: theta2 held at 0 via A2-compensation.
             tau1 swept → equilibrium theta1 → regress tau1 = -A1*cos(theta1).
    Joint 2: theta1 held at 0 via A1-compensation.
             tau2 swept → equilibrium theta2 → regress tau2 = -A2*cos(theta2).
    """
    model, data = _make_mujoco()

    # ── Rough A2 estimate first (needed for theta2=0 compensation) ──
    # Quick pass: theta1=pi/2, tau2 sweep, theta2 free.
    # At theta1=pi/2+small: G2 ≈ A2*cos(pi/2+theta2) ≈ -A2*sin(theta2)
    # Use theta1=0 where G2 = A2*cos(theta2) → good sensitivity.
    A2_rough = _estimate_A2_rough(model, data, n_configs=6)

    # ── Joint 1 identification (theta2 = 0 maintained) ──
    tau1_sweep  = np.linspace(-TRUE_A1 * 0.85, TRUE_A1 * 0.85, n_configs)
    cos_theta1  = []

    for tau1_ext in tau1_sweep:
        th1_init = math.pi / 4 if tau1_ext > 0 else 3 * math.pi / 4

        def tau_fn(th1, th2):
            # Equilibrium: tau + G = 0 → tau_comp = -G
            G2 = A2_rough * math.cos(th1 + th2)
            tau2_comp = -G2   # cancel gravity on joint 2 → eq at theta2=0
            return [float(tau1_ext), float(tau2_comp)]

        q_eq = _hold_steady(model, data, q_init=[th1_init, 0.0], tau_fn=tau_fn)
        cos_theta1.append(math.cos(q_eq[0]))

    cos_theta1  = np.array(cos_theta1)
    A1_id = -np.dot(tau1_sweep, cos_theta1) / (np.dot(cos_theta1, cos_theta1) + 1e-12)

    # ── Joint 2 identification (theta1 = 0 maintained) ──
    tau2_sweep  = np.linspace(-TRUE_A2 * 0.85, TRUE_A2 * 0.85, n_configs)
    cos_theta12 = []

    A1_partial = A1_id - A2_rough   # m*g*(lc+L) ≈ A1 - A2

    for tau2_ext in tau2_sweep:
        th2_init = math.pi / 4 if tau2_ext > 0 else -math.pi / 4

        def tau_fn(th1, th2):
            # Compensate theta1=0: cancel G1 so joint1 stays horizontal
            # tau_comp = -G to cancel gravity (equilibrium: tau + G = 0)
            G1_comp = A1_partial * math.cos(th1) + A2_rough * math.cos(th1 + th2)
            return [float(-G1_comp), float(tau2_ext)]

        q_eq = _hold_steady(model, data, q_init=[0.0, th2_init], tau_fn=tau_fn)
        theta1_eq, theta2_eq = q_eq
        cos_theta12.append(math.cos(theta1_eq + theta2_eq))

    cos_theta12 = np.array(cos_theta12)
    A2_id = -np.dot(tau2_sweep, cos_theta12) / (np.dot(cos_theta12, cos_theta12) + 1e-12)

    return A1_id, A2_id


def _estimate_A2_rough(model, data, n_configs=6):
    """Quick A2 estimate: theta1=0 (horizontal), tau2 sweep."""
    tau2_sweep  = np.linspace(-TRUE_A2 * 0.8, TRUE_A2 * 0.8, n_configs)
    cos_th12    = []

    for tau2 in tau2_sweep:
        th2_init = math.pi / 4 if tau2 > 0 else -math.pi / 4

        def tau_fn(th1, th2):
            return [0.0, float(tau2)]

        q_eq = _hold_steady(model, data, q_init=[0.0, th2_init],
                            tau_fn=tau_fn, t_settle=3.0)
        cos_th12.append(math.cos(q_eq[0] + q_eq[1]))

    cos_th12 = np.array(cos_th12)
    A2_rough = -np.dot(tau2_sweep, cos_th12) / (np.dot(cos_th12, cos_th12) + 1e-12)
    return A2_rough


# ── Experiment 2: Inertia + Damping ──────────────────────────────────────────
# Procedure:
#   Apply step torque + real-time gravity compensation.
#   Effective drive: tau_step (constant).
#   Velocity response: ω(t) = (tau/b)·(1 − exp(−b/I·t))
#   Fit with scipy.curve_fit.

def _step_response_grav_comp(model, data, q_start, tau_step, joint_idx,
                              A1, A2, t_total=4.0, obs_dt=0.05):
    """Step torque + gravity compensation → record velocity trajectory.

    The compensation torque G(theta(t)) is applied at every simulation step,
    so the net generalised force on the driven joint equals tau_step (constant).
    """
    import mujoco

    rng       = np.random.default_rng(SEED + joint_idx)
    dt        = float(model.opt.timestep)
    n_sim     = int(t_total / dt)
    obs_every = max(1, int(obs_dt / dt))

    data.qpos[:2] = q_start
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)

    A1_partial = A1 - A2

    t_raw = []
    q_raw = []

    for step in range(n_sim):
        th1, th2 = data.qpos[:2]

        # Real-time gravity compensation
        G1_comp = A1_partial * math.cos(th1) + A2 * math.cos(th1 + th2)
        G2_comp = A2 * math.cos(th1 + th2)

        tau = np.zeros(2)
        tau[0] -= G1_comp   # cancel gravity joint 1: tau_comp = -G
        tau[1] -= G2_comp   # cancel gravity joint 2: tau_comp = -G
        tau[joint_idx] += tau_step   # add step drive

        data.ctrl[:2] = np.clip(tau, -20, 20)
        mujoco.mj_step(model, data)

        if step % obs_every == 0:
            t_raw.append(step * dt)
            q_raw.append(data.qpos[joint_idx] + rng.normal(0, NOISE_STD))

    t_arr  = np.array(t_raw)
    q_arr  = np.array(q_raw)
    dq_arr = np.gradient(q_arr, t_arr)
    return t_arr, dq_arr


def identify_inertia_damping(A1, A2):
    """Jointly estimate I_eff and damping via gravity-compensated step response.

    Uses scipy.optimize.curve_fit on ω(t) = (τ/b)·(1 − e^{−b/I·t}).
    """
    model, data = _make_mujoco()

    results = {}

    for joint_idx, (q_start, tau_step) in enumerate([
        ([math.pi / 2, 0.0], 3.0),
        ([math.pi / 2, 0.0], 1.0),
    ]):
        I_true = [TRUE_I1_EFF, TRUE_I2_EFF][joint_idx]
        b_true = TRUE_DAMPING

        t_arr, dq_arr = _step_response_grav_comp(
            model, data, q_start, tau_step, joint_idx,
            A1=A1, A2=A2, t_total=4.0, obs_dt=0.05,
        )

        def vel_model(t, I, b):
            return (tau_step / b) * (1.0 - np.exp(-b / I * t))

        try:
            popt, _ = curve_fit(
                vel_model, t_arr, dq_arr,
                p0=[I_true, b_true],
                bounds=([1e-4, 0.01], [5.0, 15.0]),
                maxfev=10000,
            )
            I_id, b_id = float(popt[0]), float(popt[1])
        except RuntimeError:
            I_id, b_id = float("nan"), float("nan")

        results[joint_idx] = (I_id, b_id)

    return (results[0][0], results[1][0],
            results[0][1], results[1][1])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Physical Parameter Identification: 2-DOF Arm")
    print("  MuJoCo simulation (noise_std={:.4f} rad)".format(NOISE_STD))
    print("  (Replace MuJoCo calls with SO-101 real sensor/actuator I/O)")
    print("=" * 70)
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Gravity ──
    print("Experiment 1: Gravity (equilibrium regression)")
    print("  theta2 held at 0 via A2-compensation for A1 identification.")
    A1_id, A2_id = identify_gravity(n_configs=12)
    A1_err = abs(A1_id - TRUE_A1) / TRUE_A1 * 100
    A2_err = abs(A2_id - TRUE_A2) / TRUE_A2 * 100
    print(f"  A1 = m·g·(2lc+L)  True: {TRUE_A1:.4f}  Id: {A1_id:.4f}  "
          f"Error: {A1_err:.1f}%")
    print(f"  A2 = m·g·lc        True: {TRUE_A2:.4f}  Id: {A2_id:.4f}  "
          f"Error: {A2_err:.1f}%")
    print()

    # ── Inertia + Damping ──
    print("Experiment 2: Inertia + Damping (gravity-compensated step response)")
    print("  G(theta(t)) cancelled in real time → net drive = tau_step (const).")
    I1_id, I2_id, b1_id, b2_id = identify_inertia_damping(A1_id, A2_id)
    I1_err = abs(I1_id - TRUE_I1_EFF) / TRUE_I1_EFF * 100
    I2_err = abs(I2_id - TRUE_I2_EFF) / TRUE_I2_EFF * 100
    b1_err = abs(b1_id - TRUE_DAMPING) / TRUE_DAMPING * 100
    b2_err = abs(b2_id - TRUE_DAMPING) / TRUE_DAMPING * 100
    print(f"  I1_eff  True: {TRUE_I1_EFF:.4f}  Id: {I1_id:.4f}  Error: {I1_err:.1f}%")
    print(f"  I2_eff  True: {TRUE_I2_EFF:.4f}  Id: {I2_id:.4f}  Error: {I2_err:.1f}%")
    print(f"  b1      True: {TRUE_DAMPING:.4f}  Id: {b1_id:.4f}  Error: {b1_err:.1f}%")
    print(f"  b2      True: {TRUE_DAMPING:.4f}  Id: {b2_id:.4f}  Error: {b2_err:.1f}%")
    print()

    # ── Summary ──
    print("=" * 70)
    print("Parameter Identification Summary")
    print("=" * 70)
    rows = [
        ("A1 = m·g·(2lc+L)",  TRUE_A1,       A1_id,  A1_err, "N·m"),
        ("A2 = m·g·lc",        TRUE_A2,       A2_id,  A2_err, "N·m"),
        ("I1_eff",              TRUE_I1_EFF,   I1_id,  I1_err, "kg·m²"),
        ("I2_eff",              TRUE_I2_EFF,   I2_id,  I2_err, "kg·m²"),
        ("damping j1",          TRUE_DAMPING,  b1_id,  b1_err, "N·m·s"),
        ("damping j2",          TRUE_DAMPING,  b2_id,  b2_err, "N·m·s"),
    ]
    print(f"  {'Parameter':<22} {'True':>10} {'Identified':>12} {'Error':>8}  Unit")
    print("-" * 70)
    for name, true, ident, err, unit in rows:
        err_str = f"{err:.1f}%" if not math.isnan(err) else "N/A"
        print(f"  {name:<22} {true:>10.4f} {ident:>12.4f} {err_str:>8}  {unit}")
    print("=" * 70)
    print()
    print("SO-101 field procedure:")
    print("  1. Gravity (static, ~5 min):")
    print("     - Command 10+ static configurations (vary tau, wait 5 s each)")
    print("     - Record equilibrium angles → linear regression for A1, A2")
    print("  2. Inertia + Damping (dynamic, ~10 min):")
    print("     - Apply real-time gravity compensation: tau_comp = G(theta(t))")
    print("     - Add step torque (1–3 N·m) to one joint at a time")
    print("     - Record velocity (50 Hz, 4 s) → curve_fit for I, b")
    print("  3. Update _arm_dynamics constants: A1→A1_id, A2→A2_id, etc.")
    print()

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model, data = _make_mujoco()
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        for joint_idx, (ax_v, ax_res) in enumerate(
                [(axes[0, 0], axes[0, 1]), (axes[1, 0], axes[1, 1])]):

            tau_step = [3.0, 1.0][joint_idx]
            t_arr, dq_arr = _step_response_grav_comp(
                model, data, [math.pi / 2, 0.0], tau_step, joint_idx,
                A1=A1_id, A2=A2_id, t_total=4.0, obs_dt=0.05)

            I_id_j = [I1_id, I2_id][joint_idx]
            b_id_j = [b1_id, b2_id][joint_idx]
            I_tr   = [TRUE_I1_EFF, TRUE_I2_EFF][joint_idx]

            t_fit = np.linspace(0, t_arr[-1], 400)
            dq_tr = (tau_step / TRUE_DAMPING) * (1 - np.exp(-TRUE_DAMPING / I_tr * t_fit))
            dq_id = (tau_step / b_id_j)       * (1 - np.exp(-b_id_j    / I_id_j * t_fit))

            ax_v.plot(t_arr, dq_arr, 'b.', ms=4, label='Measured ω (FD)')
            ax_v.plot(t_fit, dq_tr, 'g-', lw=1.8,
                      label=f'True (I={I_tr:.3f}, b={TRUE_DAMPING:.2f})')
            ax_v.plot(t_fit, dq_id, 'r--', lw=1.8,
                      label=f'Identified (I={I_id_j:.3f}, b={b_id_j:.2f})')
            ax_v.set_xlabel('Time (s)')
            ax_v.set_ylabel('ω (rad/s)')
            ax_v.set_title(f'Joint {joint_idx+1}: Gravity-compensated step response')
            ax_v.legend(fontsize=8)
            ax_v.grid(True, alpha=0.3)

            dq_id_at = (tau_step / b_id_j) * (1 - np.exp(-b_id_j / I_id_j * t_arr))
            ax_res.plot(t_arr, dq_arr - dq_id_at, 'r.', ms=4)
            ax_res.axhline(0, color='k', lw=0.8)
            ax_res.set_xlabel('Time (s)')
            ax_res.set_ylabel('Residual ω (rad/s)')
            ax_res.set_title(f'Joint {joint_idx+1}: Residuals')
            ax_res.grid(True, alpha=0.3)

        fig.suptitle(
            "Parameter Identification: Gravity-Compensated Step Response\n"
            "ω(t) = (τ/b)·(1 − e^{−b/I·t})  with G(θ(t)) cancelled in real time\n"
            f"[noise={NOISE_STD} rad, obs_dt=0.05 s]",
            fontsize=10, fontweight='bold'
        )
        plt.tight_layout()
        fig_path = results_dir / "param_id_2dof_arm.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {fig_path}")

    except ImportError:
        print("matplotlib not found — skipping plot")

    print("Done.")


if __name__ == "__main__":
    main()
