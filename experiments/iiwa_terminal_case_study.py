"""KUKA iiwa 7-DoF terminal-controllability case study (RA-L revision, task #1).

Confirmatory case study that the three-layer view of active calibration
(identifiability creation -> task-compatible exploration -> post-calibration
finite-horizon controllability) is not an artifact of the planar 4-DoF arm.

Self-contained: kinematics come from the vendored KUKA LBR iiwa model
(`src/dem/iiwa_kin.py`, validated against MuJoCo to machine precision).  The
unknown body parameters are distal link-length scale factors (theta), which —
unlike a tool offset — have configuration-dependent identifiability, matching
the planar experiment (see the FIM analysis in the design notes).

Two-phase protocol per seed:
  Phase 1 (calibration while holding):  hold EE at y_hold, estimate theta.
  Phase 2 (task execution, theta frozen): reach a new target y_task in H steps.

Controllers: plain (task only) / probe (null-space information gain) /
posture (null-space regularisation toward a comfortable nominal posture).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.dem import iiwa_kin as ik

N_DOF = ik.N_DOF                       # 7
P = ik.N_LINK_PARAMS                   # 3 unknown link scales
TASK_DIM = 3                           # 3D end-effector position

# --- Unknown body parameters (link-length scale factors) -------------------
THETA_TRUE = ik.THETA_LINK_TRUE                 # ones(3): nominal kinematics
THETA_INIT = jnp.array([1.30, 0.72, 1.20])      # badly wrong initial guess

# --- Time / control --------------------------------------------------------
DT = 0.05
CHANGE_STEP = 150          # Phase 2 starts here
U_MAX = 1.0
K_TASK = 5.0               # EE task-space PD gain

# --- Estimation ------------------------------------------------------------
SIGMA_OBS = 0.005          # 5 mm EE position noise
PI_Y = 1.0 / SIGMA_OBS ** 2
PRIOR_PI = 1.0             # prior precision on theta (around THETA_INIT)
N_ESTEP_ITER = 5
ESTEP_FREQ = 3
HIST_WINDOW = 60           # fixed-size E-step history (static shape -> compile once)

# --- Probing / posture -----------------------------------------------------
ALPHA_NS = 1.2             # null-space IG gain
POSTURE_GAIN = 1.0
Q_NOMINAL = ik.Q_NOMINAL   # comfortable elbow-bent posture
DAMP = 1e-3

# --- Start / goals ---------------------------------------------------------
# Degenerate-ish start: arm near-extended (small joint angles) toward the
# workspace boundary, where Phase-2 finite-horizon task control is hardest.
Q0 = jnp.array([0.0, 0.15, 0.0, 0.20, 0.0, 0.15, 0.0])
Q_TARGET = jnp.array([0.6, 0.5, -0.3, -1.0, 0.2, 0.7, 0.0])  # defines y_task

Y_HOLD = ik.fk_links(Q0, THETA_TRUE)
Y_TASK = ik.fk_links(Q_TARGET, THETA_TRUE)

# --- Failure thresholds (provisional; justified later in task #5) ----------
PARAM_RMSE_FAIL = 0.05     # link-scale RMSE
TASK_ERR_FAIL = 0.05       # 5 cm EE error at end of Phase 2


# ===========================================================================
# Kinematics-derived quantities (JAX)
# ===========================================================================

def fk(q, theta):
    return ik.fk_links(q, theta)


def J_q(q, theta):
    return jax.jacobian(lambda qi: fk(qi, theta))(q)        # (3, 7)


def J_theta(q, theta):
    return ik.jac_theta(q, theta)                            # (3, P)


def rollout_step(q, u):
    lo, hi = ik.JOINT_LIMITS[:, 0], ik.JOINT_LIMITS[:, 1]
    return jnp.clip(q + u * DT, lo, hi)


def _damped_pinv(J, damping=DAMP):
    return J.T @ jnp.linalg.inv(J @ J.T + damping * jnp.eye(J.shape[0]))


# ===========================================================================
# Self-contained Gauss-Newton E-step over theta (MAP with Gaussian prior)
# ===========================================================================

def _fixed_window(qs, ys):
    """Evenly-spaced fixed-size (HIST_WINDOW,) sample of the history -> static shape.

    Variable-length history would force XLA to recompile the E-step at every
    distinct length, accumulating compiled programs until host memory is
    exhausted.  Subsampling to a fixed count keeps coverage of the whole
    Phase-1 trajectory while compiling the E-step exactly once."""
    q = np.stack(qs); y = np.stack(ys)
    T = q.shape[0]
    if T >= HIST_WINDOW:
        idx = np.linspace(0, T - 1, HIST_WINDOW).astype(int)
    else:
        idx = np.concatenate([np.zeros(HIST_WINDOW - T, dtype=int), np.arange(T)])
    return jnp.asarray(q[idx]), jnp.asarray(y[idx])


@jax.jit
def estep_gauss_newton(qs, ys, theta):
    """MAP estimate of theta given fixed-size history, prior N(THETA_INIT, 1/PRIOR_PI)."""
    Lam_prior = PRIOR_PI * jnp.eye(P)

    def body(theta, _):
        def per_t(q_t, y_t):
            Jt = ik.jac_theta(q_t, theta)            # (3, P)
            r_t = y_t - fk(q_t, theta)               # (3,)
            return Jt.T @ (PI_Y * r_t), Jt.T @ (PI_Y * Jt)
        g, H = jax.vmap(per_t)(qs, ys)
        grad = jnp.sum(g, axis=0) - Lam_prior @ (theta - THETA_INIT)
        Hess = jnp.sum(H, axis=0) + Lam_prior
        dtheta = jnp.linalg.solve(Hess, grad)
        return theta + dtheta, None

    theta_new, _ = jax.lax.scan(body, theta, None, length=N_ESTEP_ITER)
    return theta_new


@jax.jit
def posterior_precision(qs, ys, theta):
    def per_t(q_t):
        Jt = ik.jac_theta(q_t, theta)
        return Jt.T @ (PI_Y * Jt)
    return jnp.sum(jax.vmap(per_t)(qs), axis=0) + PRIOR_PI * jnp.eye(P)


# ===========================================================================
# Information gain
# ===========================================================================

def _ig_at_q(q, theta, P_theta):
    Jt = ik.jac_theta(q, theta)
    fim = Jt.T @ (PI_Y * jnp.eye(TASK_DIM)) @ Jt
    _, ld1 = jnp.linalg.slogdet(P_theta + fim)
    _, ld0 = jnp.linalg.slogdet(P_theta)
    return 0.5 * (ld1 - ld0)


# ===========================================================================
# Controllers
# ===========================================================================

@jax.jit
def ctrl_plain(q, theta, y_goal):
    Jq = J_q(q, theta)
    v = -K_TASK * (fk(q, theta) - y_goal)
    return jnp.clip(_damped_pinv(Jq) @ v, -U_MAX, U_MAX)


@jax.jit
def ctrl_posture(q, theta, y_goal):
    Jq = J_q(q, theta)
    Jp = _damped_pinv(Jq)
    v = -K_TASK * (fk(q, theta) - y_goal)
    N = jnp.eye(N_DOF) - Jp @ Jq
    u = Jp @ v + N @ (-POSTURE_GAIN * (q - Q_NOMINAL))
    return jnp.clip(u, -U_MAX, U_MAX)


@jax.jit
def ctrl_probe(q, theta, P_theta, y_goal):
    """Null-space information-gain probing during Phase 1 hold."""
    Jq = J_q(q, theta)
    Jp = _damped_pinv(Jq)
    v = -K_TASK * (fk(q, theta) - y_goal)
    N = jnp.eye(N_DOF) - Jp @ Jq
    ig_grad = jax.grad(lambda qi: _ig_at_q(qi, theta, P_theta))(q)
    u = Jp @ v + ALPHA_NS * N @ ig_grad
    return jnp.clip(u, -U_MAX, U_MAX)


# ===========================================================================
# Short-horizon terminal rollout risk (predictor; bridges to the 4-DoF result)
# ===========================================================================
ROLLOUT_K = 20


@jax.jit
def rollout_risk(q_start, theta_ctrl, theta_eval, y_task):
    """K-step noiseless plain rollout from q_start; final true EE error to y_task.

    Control uses theta_ctrl (estimated); evaluation uses theta_eval (true for the
    estimated-risk variant, or true for both in the oracle variant)."""
    def step(q, _):
        u = ctrl_plain(q, theta_ctrl, y_task)
        return rollout_step(q, u), None
    q_end, _ = jax.lax.scan(step, q_start, None, length=ROLLOUT_K)
    return jnp.linalg.norm(ik.fk_links(q_end, theta_eval) - y_task)


@jax.jit
def sigma_min_Jq(q, theta):
    return jnp.min(jnp.linalg.svd(J_q(q, theta), compute_uv=False))


# ===========================================================================
# Single two-phase run
# ===========================================================================

def run_one(seed, phase1="probe", phase2="plain", horizon=50, q0=Q0, y_task=Y_TASK,
            verbose=False):
    rng = np.random.default_rng(seed)
    theta = THETA_INIT
    P_theta = PRIOR_PI * jnp.eye(P)
    q = jnp.array(q0)
    y_task = jnp.asarray(y_task)
    y_hold = ik.fk_links(jnp.array(q0), THETA_TRUE)

    qs, ys = [], []
    theta_frozen = None
    q_change = None
    rmse_at_change = None

    n_total = CHANGE_STEP + horizon

    for t in range(n_total):
        if t < CHANGE_STEP:
            if phase1 == "plain":
                u = ctrl_plain(q, theta, y_hold)
            elif phase1 == "probe":
                u = ctrl_probe(q, theta, P_theta, y_hold)
            else:
                raise ValueError(phase1)
        else:
            if theta_frozen is None:
                theta_frozen = theta
                q_change = np.array(q)
                rmse_at_change = float(jnp.sqrt(jnp.mean((theta - THETA_TRUE) ** 2)))
            if phase2 == "plain":
                u = ctrl_plain(q, theta_frozen, y_task)
            elif phase2 == "posture":
                u = ctrl_posture(q, theta_frozen, y_task)
            else:
                raise ValueError(phase2)

        q = rollout_step(q, u)
        y = ik.fk_links(q, THETA_TRUE) + jnp.array(rng.normal(0, SIGMA_OBS, size=TASK_DIM))
        qs.append(q)
        ys.append(y)

        if t < CHANGE_STEP and t > 5 and t % ESTEP_FREQ == 0:
            qfix, yfix = _fixed_window(qs, ys)
            theta = estep_gauss_newton(qfix, yfix, theta)
            theta = jnp.clip(theta, 0.3, 2.5)
            P_theta = posterior_precision(qfix, yfix, theta)

    ee_true = ik.fk_links(q, THETA_TRUE)
    task_err_final = float(jnp.linalg.norm(ee_true - y_task))
    q_chg = jnp.asarray(q_change)
    ee_hold = float(jnp.linalg.norm(ik.fk_links(q_chg, THETA_TRUE) - y_hold))

    # Terminal-state diagnostics at q_change (predictors of finite-horizon failure)
    r_rollout_hat = float(rollout_risk(q_chg, theta_frozen, THETA_TRUE, y_task))
    r_rollout_oracle = float(rollout_risk(q_chg, THETA_TRUE, THETA_TRUE, y_task))
    sig_min = float(sigma_min_Jq(q_chg, theta_frozen))

    return {
        "seed": seed,
        "rmse_at_change": rmse_at_change,
        "task_err_final": task_err_final,
        "theta_final": np.array(theta_frozen),
        "q_change": np.array(q_change),
        "ee_hold_at_change": ee_hold,
        "r_rollout_hat": r_rollout_hat,
        "r_rollout_oracle": r_rollout_oracle,
        "sigma_min_change": sig_min,
        "fail_task": task_err_final > TASK_ERR_FAIL,
        "fail_rmse": rmse_at_change > PARAM_RMSE_FAIL,
    }


# Per-seed geometry sampling: near-extended start jitter + reachable target jitter.
Q0_JITTER = 0.10     # rad
QT_JITTER = 0.25     # rad


def sample_geometry(seed):
    rng = np.random.default_rng(10_000 + seed)
    q0 = np.array(Q0) + rng.normal(0, Q0_JITTER, N_DOF)
    q0 = np.clip(q0, np.array(ik.JOINT_LIMITS[:, 0]), np.array(ik.JOINT_LIMITS[:, 1]))
    qt = np.array(Q_TARGET) + rng.normal(0, QT_JITTER, N_DOF)
    qt = np.clip(qt, np.array(ik.JOINT_LIMITS[:, 0]), np.array(ik.JOINT_LIMITS[:, 1]))
    y_task = np.array(ik.fk_links(jnp.array(qt), THETA_TRUE))
    return jnp.array(q0), jnp.array(y_task)


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    m = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return float(max(0, c-m)), float(min(1, c+m))


def _rank_auc(scores, labels):
    """AUC = P(score_pos > score_neg) via Mann-Whitney; nan if one class empty."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > 0) + 0.5*np.sum(d == 0)) / (len(pos)*len(neg)))


def horizon_sweep(seeds, horizons, conditions, vary_geometry=False):
    """failTask (+Wilson CI) and median taskErr vs horizon, per (phase1,phase2)."""
    out = {}
    for (p1, p2) in conditions:
        out[f"{p1}->{p2}"] = {}
        for H in horizons:
            rs = []
            for s in seeds:
                if vary_geometry:
                    q0, yt = sample_geometry(s)
                    rs.append(run_one(s, phase1=p1, phase2=p2, horizon=H, q0=q0, y_task=yt))
                else:
                    rs.append(run_one(s, phase1=p1, phase2=p2, horizon=H))
            task_errs = np.array([r["task_err_final"] for r in rs])
            rmses = np.array([r["rmse_at_change"] for r in rs])
            n_fail = int(np.sum(task_errs > TASK_ERR_FAIL))
            lo, hi = _wilson(n_fail, len(seeds))
            out[f"{p1}->{p2}"][H] = {
                "fail_task": float(n_fail / len(seeds)),
                "fail_task_ci": [lo, hi],
                "task_err_median": float(np.median(task_errs)),
                "task_err_iqr": [float(np.percentile(task_errs, 25)),
                                 float(np.percentile(task_errs, 75))],
                "rmse_median": float(np.median(rmses)),
                "n": len(seeds),
            }
    return out


def rollout_risk_auc(seeds, H_eval, phase1="probe", phase2="plain"):
    """Per-seed predictors vs failTask@H_eval; AUC of each predictor."""
    rows = []
    for s in seeds:
        q0, yt = sample_geometry(s)
        r = run_one(s, phase1=phase1, phase2=phase2, horizon=H_eval, q0=q0, y_task=yt)
        rows.append(r)
    fail = np.array([r["fail_task"] for r in rows], bool)
    preds = {
        "rmse_at_change": np.array([r["rmse_at_change"] for r in rows]),
        "inv_sigma_min": np.array([1.0/max(r["sigma_min_change"], 1e-9) for r in rows]),
        "rollout_risk_hat": np.array([r["r_rollout_hat"] for r in rows]),
        "rollout_risk_oracle": np.array([r["r_rollout_oracle"] for r in rows]),
    }
    return {
        "H_eval": H_eval,
        "n": len(seeds),
        "n_fail": int(fail.sum()),
        "auc": {k: _rank_auc(v, fail) for k, v in preds.items()},
        "per_seed": [
            {"seed": int(r["seed"]), "fail_task": bool(r["fail_task"]),
             "rmse_at_change": r["rmse_at_change"], "task_err_final": r["task_err_final"],
             "r_rollout_hat": r["r_rollout_hat"], "r_rollout_oracle": r["r_rollout_oracle"],
             "sigma_min_change": r["sigma_min_change"]}
            for r in rows
        ],
    }


if __name__ == "__main__":
    import json

    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    print(f"=== iiwa terminal-controllability ({mode}) ===")
    print(f"  N_DOF={N_DOF}  P(unknown link scales)={P}  task_dim={TASK_DIM}")
    print(f"  THETA_TRUE={np.array(THETA_TRUE)}  THETA_INIT={np.array(THETA_INIT)}")
    print(f"  Y_HOLD={np.array(Y_HOLD).round(3)}  Y_TASK={np.array(Y_TASK).round(3)}")
    print()

    if mode == "smoke":
        for phase1, phase2, H in [("probe", "plain", 50), ("probe", "posture", 50),
                                  ("plain", "plain", 50)]:
            r = run_one(0, phase1=phase1, phase2=phase2, horizon=H)
            print(f"  ph1={phase1:5s} ph2={phase2:7s} H={H}: "
                  f"rmse@change={r['rmse_at_change']:.4f} (fail={r['fail_rmse']})  "
                  f"taskErr={r['task_err_final']:.4f} (fail={r['fail_task']})  "
                  f"eeHold={r['ee_hold_at_change']:.4f}")
    elif mode in ("sweep", "varsweep"):
        seeds = list(range(int(sys.argv[2]))) if len(sys.argv) > 2 else list(range(8))
        horizons = [20, 50, 75, 100, 150, 200]
        conditions = [("probe", "plain"), ("probe", "posture"), ("plain", "plain")]
        vary = (mode == "varsweep")
        res = horizon_sweep(seeds, horizons, conditions, vary_geometry=vary)
        print(f"  mode={mode} (vary_geometry={vary})  seeds={len(seeds)}  horizons={horizons}\n")
        for cond, hd in res.items():
            print(f"  {cond}:")
            for H in horizons:
                d = hd[H]
                lo, hi = d["fail_task_ci"]
                print(f"    H={H:3d}: failTask={d['fail_task']:.2f} [{lo:.2f},{hi:.2f}]  "
                      f"taskErr(med)={d['task_err_median']:.4f}  rmse(med)={d['rmse_median']:.4f}")
        suffix = "_varied" if vary else ""
        out_path = project_root / "results" / f"iiwa_horizon_sweep{suffix}.json"
        with open(out_path, "w") as f:
            json.dump({"seeds": len(seeds), "horizons": horizons,
                       "vary_geometry": vary, "results": res}, f, indent=2)
        print(f"\nSaved -> {out_path}")
    elif mode == "auc":
        seeds = list(range(int(sys.argv[2]))) if len(sys.argv) > 2 else list(range(20))
        H_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 75
        res = rollout_risk_auc(seeds, H_eval)
        print(f"  rollout-risk AUC vs failTask@H={H_eval}  "
              f"(n={res['n']}, n_fail={res['n_fail']})\n")
        for k, v in res["auc"].items():
            print(f"    AUC({k:20s}) = {v:.3f}")
        out_path = project_root / "results" / "iiwa_rollout_risk_auc.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved -> {out_path}")
