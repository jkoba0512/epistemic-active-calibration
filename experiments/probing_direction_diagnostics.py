"""Local diagnostics for task-compatible finite-step probing.

This script checks the theoretical hinge point in
docs/theory_task_compatible_probing.md:

    at Q0, first-order null-space IG is zero, but finite-step candidate probes
    may still produce positive information gain with small task drift.

It does not change the redundant-arm controller or regenerate paper results.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from experiments.redundant_arm_calibration import (  # noqa: E402
    N_DOF,
    PARAMS_PRIOR_PI,
    Q0,
    THETA_INIT,
    THETA_TRUE,
    _compute_J_ee,
    _ig_at_q,
    fk,
)


DAMPING = 1e-3
GRAD_EPS = 1e-12
EPSILONS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
RNG_SEED = 20260421
NUM_RANDOM_DIRECTIONS = 128

CANDIDATE_DIRECTIONS = jnp.array(
    [
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
        [1.0, -2.0, 1.0, 0.0],
        [0.0, 1.0, -2.0, 1.0],
    ],
    dtype=jnp.float64,
)


def make_candidate_directions(seed=RNG_SEED, num_random=NUM_RANDOM_DIRECTIONS):
    rng = np.random.default_rng(seed)
    random_dirs = rng.normal(size=(num_random, N_DOF))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    candidates = np.vstack([np.asarray(CANDIDATE_DIRECTIONS), random_dirs])
    return jnp.asarray(candidates, dtype=jnp.float64)


def make_q_cases():
    return [
        ("q0", Q0),
        (
            "q0_noise_1e-4",
            Q0 + jnp.array([1.0, -0.5, 0.25, -0.75], dtype=jnp.float64) * 1e-4,
        ),
        (
            "q0_noise_1e-2",
            Q0 + jnp.array([1.0, -0.5, 0.25, -0.75], dtype=jnp.float64) * 1e-2,
        ),
        (
            "bent_reference",
            jnp.array([0.25, -0.35, 0.2, -0.15], dtype=jnp.float64),
        ),
    ]


def damped_pseudoinverse(J, damping):
    """Right damped pseudoinverse for a 2 x n task Jacobian."""
    task_dim = J.shape[0]
    return J.T @ jnp.linalg.inv(J @ J.T + damping * jnp.eye(task_dim))


def normalize_or_zero(v):
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > GRAD_EPS, v / norm, jnp.zeros_like(v)), norm


def matrix_rank_from_svals(svals, tol=1e-9):
    return int(np.sum(np.asarray(svals) > tol))


def evaluate_candidates(q, theta_est, P_theta, damping, eps, candidate_directions):
    J = _compute_J_ee(q, theta_est)
    J_hash = damped_pseudoinverse(J, damping)
    N_mu = jnp.eye(N_DOF) - J_hash @ J

    ig0 = _ig_at_q(q, theta_est, P_theta)
    y0_est = fk(q, theta_est)
    y0_true = fk(q, THETA_TRUE)

    rows = []
    for i, raw_d in enumerate(candidate_directions):
        projected = N_mu @ raw_d
        d, projected_norm = normalize_or_zero(projected)
        q_probe = q + eps * d
        ig_probe = _ig_at_q(q_probe, theta_est, P_theta)
        first_order_task = J @ d
        drift_est = fk(q_probe, theta_est) - y0_est
        drift_true = fk(q_probe, THETA_TRUE) - y0_true
        rows.append(
            {
                "candidate": int(i),
                "raw_direction": np.asarray(raw_d).tolist(),
                "direction": np.asarray(d).tolist(),
                "projected_norm": float(projected_norm),
                "first_order_task_norm": float(jnp.linalg.norm(first_order_task)),
                "ig_gain": float(ig_probe - ig0),
                "ig_probe": float(ig_probe),
                "task_drift_est_norm": float(jnp.linalg.norm(drift_est)),
                "task_drift_true_norm": float(jnp.linalg.norm(drift_true)),
            }
        )

    rows.sort(key=lambda r: r["ig_gain"], reverse=True)
    return rows


def summarize_rows(rows):
    gains = np.asarray([r["ig_gain"] for r in rows], dtype=float)
    drift_est = np.asarray([r["task_drift_est_norm"] for r in rows], dtype=float)
    drift_true = np.asarray([r["task_drift_true_norm"] for r in rows], dtype=float)
    return {
        "positive_gain_count": int(np.sum(gains > 0.0)),
        "total_candidates": int(gains.size),
        "gain_mean": float(np.mean(gains)),
        "gain_median": float(np.median(gains)),
        "gain_p90": float(np.quantile(gains, 0.9)),
        "gain_max": float(np.max(gains)),
        "task_drift_est_median": float(np.median(drift_est)),
        "task_drift_true_median": float(np.median(drift_true)),
    }


def evaluate_case(case_name, q, theta_est, P_theta, candidate_directions):
    J = _compute_J_ee(q, theta_est)
    J_pinv = jnp.linalg.pinv(J)
    N_exact = jnp.eye(N_DOF) - J_pinv @ J
    ig_grad = jax.grad(lambda qi: _ig_at_q(qi, theta_est, P_theta))(q)
    projected_grad = N_exact @ ig_grad
    ig_grad_norm = jnp.linalg.norm(ig_grad)
    projected_grad_norm = jnp.linalg.norm(projected_grad)
    alpha_N = jnp.where(
        ig_grad_norm > GRAD_EPS,
        projected_grad_norm / (ig_grad_norm + GRAD_EPS),
        0.0,
    )
    svals = jnp.linalg.svd(J, compute_uv=False)

    header = {
        "case": case_name,
        "q0": np.asarray(q).tolist(),
        "theta_est": np.asarray(theta_est).tolist(),
        "theta_true": np.asarray(THETA_TRUE).tolist(),
        "damping": DAMPING,
        "ig_at_q0": float(_ig_at_q(q, theta_est, P_theta)),
        "J_ee": np.asarray(J).tolist(),
        "J_ee_singular_values": np.asarray(svals).tolist(),
        "J_ee_rank_tol_1e-9": matrix_rank_from_svals(svals),
        "ig_grad": np.asarray(ig_grad).tolist(),
        "ig_grad_norm": float(ig_grad_norm),
        "exact_null_projected_ig_grad": np.asarray(projected_grad).tolist(),
        "exact_null_projected_ig_grad_norm": float(projected_grad_norm),
        "alpha_N": float(alpha_N),
    }

    by_epsilon = {}
    for eps in EPSILONS:
        rows = evaluate_candidates(
            q, theta_est, P_theta, DAMPING, eps, candidate_directions
        )
        by_epsilon[str(eps)] = {
            "best": rows[0],
            "stats": summarize_rows(rows),
            "all": rows,
        }

    return {
        "summary": header,
        "by_epsilon": by_epsilon,
    }


def main():
    theta_est = THETA_INIT
    P_theta = PARAMS_PRIOR_PI * jnp.eye(N_DOF)
    candidate_directions = make_candidate_directions()

    cases = {}
    for case_name, q in make_q_cases():
        cases[case_name] = evaluate_case(
            case_name, q, theta_est, P_theta, candidate_directions
        )

    out = {
        "settings": {
            "damping": DAMPING,
            "grad_eps": GRAD_EPS,
            "epsilons": EPSILONS,
            "rng_seed": RNG_SEED,
            "fixed_directions": int(CANDIDATE_DIRECTIONS.shape[0]),
            "random_directions": NUM_RANDOM_DIRECTIONS,
            "total_candidates": int(candidate_directions.shape[0]),
        },
        "cases": cases,
    }

    out_path = project_root / "results" / "probing_direction_diagnostics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for case_name, case in cases.items():
        header = case["summary"]
        print(f"=== {case_name} first-order diagnostics ===")
        print(f"q: {np.array(header['q0'])}")
        print(f"J singular values: {header['J_ee_singular_values']}")
        print(f"rank tol 1e-9: {header['J_ee_rank_tol_1e-9']}")
        print(f"IG(q): {header['ig_at_q0']:.12f}")
        print(f"||grad IG||: {header['ig_grad_norm']:.3e}")
        print(f"||N grad IG||: {header['exact_null_projected_ig_grad_norm']:.3e}")
        print(f"alpha_N: {header['alpha_N']:.3e}")
        print()
        print("=== Best finite-step probes ===")
        for eps in EPSILONS:
            result = case["by_epsilon"][str(eps)]
            best = result["best"]
            stats = result["stats"]
            print(
                f"eps={eps:g}  "
                f"best={best['candidate']}  "
                f"IG gain={best['ig_gain']:.6e}  "
                f"positive={stats['positive_gain_count']}/"
                f"{stats['total_candidates']}  "
                f"||Jd||={best['first_order_task_norm']:.3e}  "
                f"drift_est={best['task_drift_est_norm']:.3e}  "
                f"drift_true={best['task_drift_true_norm']:.3e}"
            )
        print()
    print()
    print(f"Saved diagnostics to {out_path}")


if __name__ == "__main__":
    main()
