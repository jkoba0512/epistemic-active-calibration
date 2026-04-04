"""
5-Condition comparison using the SO-101 gripper simulation.

Same conditions as four_condition_comparison.py but using SO101OcclusionEnv
instead of the toy L-arm.  Contact detection is oracle-based (angle tolerance)
since physical collision is disabled in the scene.

Usage
-----
    uv run python experiments/run_so101_comparison.py
    uv run python experiments/run_so101_comparison.py --episodes 50 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_calib_robustness.core.generative_model.model_builder import (
    DEFAULT_OBJ_ARM, build_A, build_B, build_C, build_D,
)
from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager
from aif_calib_robustness.simulation.so101_env import SO101OcclusionEnv

N_POS = 5
N_OBJ = 3
N_VIS = N_OBJ + 1
N_TAC = 2
N_PROPRIO = N_POS


def run_episode(
    env: SO101OcclusionEnv,
    A_clean, B, C, D,
    precision_manager: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    max_steps: int = 20,
    with_proprio: bool = False,
) -> int:
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)

    agent = MultiModalAIFAgent(
        A_clean, B, C, D,
        precision_manager=precision_manager,
        policy_len=2, inference_horizon=2,
    )
    agent.reset()

    for t in range(max_steps):
        env_obs = env._get_obs()
        obs = [env_obs.visual_obs_idx, env_obs.tactile_obs_idx]
        if with_proprio:
            obs.append(env_obs.arm_pos_idx)

        result = agent.step(obs, c_visual=env.c_visual)
        action = int(result.action[0])

        step_result = env.step(action)
        if step_result.obs.tactile_obs_idx > 0:
            return t + 1

    return max_steps


def run_condition(
    occlusion_mode: str,
    A_clean, precision_manager: PrecisionManager,
    n_episodes: int, seed_offset: int = 0,
    max_steps: int = 20, with_proprio: bool = False,
) -> list[int]:
    n_proprio = N_PROPRIO if with_proprio else 0
    env = SO101OcclusionEnv(
        occlusion_mode=occlusion_mode, n_arm_positions=N_POS, max_steps=max_steps + 5
    )
    return [
        run_episode(
            env, A_clean,
            build_B(N_POS, N_OBJ),
            build_C(N_VIS, N_TAC, n_proprio=n_proprio),
            build_D(N_POS, N_OBJ),
            precision_manager,
            obj_loc_idx=ep % N_OBJ,
            seed=seed_offset + ep,
            max_steps=max_steps,
            with_proprio=with_proprio,
        )
        for ep in range(n_episodes)
    ]


def _report(label: str, steps: list[int], max_steps: int) -> dict:
    contacted = sum(s < max_steps for s in steps)
    m = float(np.mean(steps))
    s = float(np.std(steps))
    print(f"  {label}: mean={m:.2f} ± {s:.2f}  contact={contacted}/{len(steps)}")
    return {"label": label, "mean": m, "std": s, "contacted": contacted,
            "n": len(steps), "steps": steps}


def _cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    pooled = ((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2)
    return float((np.mean(a) - np.mean(b)) / (np.sqrt(pooled) + 1e-12))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=30)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--max-steps", type=int,   default=20)
    parser.add_argument("--p-contact", type=float, default=0.9)
    parser.add_argument("--p-bg",      type=float, default=0.05)
    parser.add_argument("--pi-max",    type=float, default=5.0)
    parser.add_argument("--noise-floor", type=float, default=0.1)
    parser.add_argument("--proprio-accuracy", type=float, default=1.0)
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    A_binary       = build_A(N_POS, N_OBJ, p_contact=1.0, p_bg=0.0)
    A_soft         = build_A(N_POS, N_OBJ, p_contact=args.p_contact, p_bg=args.p_bg)
    A_soft_proprio = build_A(
        N_POS, N_OBJ, p_contact=args.p_contact, p_bg=args.p_bg,
        with_proprio=True, proprio_accuracy=args.proprio_accuracy,
    )

    pm_no  = PrecisionManager(theta=0.4, pi_tactile_max=args.pi_max,
                               pi_visual_min=0.1, tactile_noise_floor=0.0)
    pm_sw  = PrecisionManager(theta=0.4, pi_tactile_max=args.pi_max,
                               pi_visual_min=0.1, tactile_noise_floor=args.noise_floor)
    pm_ct  = PrecisionManager(theta=0.4, pi_tactile_max=args.pi_max,
                               pi_visual_min=0.1, tactile_noise_floor=args.noise_floor,
                               contact_triggered=True)

    conditions = [
        ("A: no-occ/soft",           "none", A_soft,         pm_no,  False),
        ("B: full-occ/binary",        "full", A_binary,       pm_no,  False),
        ("C: full-occ/soft",          "full", A_soft,         pm_no,  False),
        ("D: full-occ/sw",            "full", A_soft,         pm_sw,  False),
        ("E: full-occ/sw+proprio",    "full", A_soft_proprio, pm_ct,  True),
        ("F: proprio+always-on",      "full", A_soft_proprio, pm_sw,  True),
        ("G: CT-sw only",             "full", A_soft,         pm_ct,  False),
    ]

    print(f"SO-101 5-Condition Comparison  (n={args.episodes}, seed={args.seed})")
    print(f"  A_soft: p_contact={args.p_contact}, p_bg={args.p_bg}")
    print(f"  Switching: pi_max={args.pi_max}, noise_floor={args.noise_floor}")
    if args.proprio_accuracy < 1.0:
        print(f"  proprio_accuracy={args.proprio_accuracy}")
    print()

    results = []
    for label, occ, A, pm, proprio in conditions:
        steps = run_condition(occ, A, pm,
                              n_episodes=args.episodes,
                              seed_offset=args.seed,
                              max_steps=args.max_steps,
                              with_proprio=proprio)
        results.append(_report(label, steps, args.max_steps))

    # Statistics: C vs D, C vs E
    sc = results[2]["steps"]
    sd = results[3]["steps"]
    se = results[4]["steps"]
    t_cd, p_cd = scipy_stats.ttest_ind(sc, sd, equal_var=False)
    t_ce, p_ce = scipy_stats.ttest_ind(sc, se, equal_var=False)
    print(f"\n  C vs D: t={t_cd:.2f} p={p_cd:.3f} ({'*' if p_cd<0.05 else 'n.s.'})  "
          f"Δ={np.mean(sc)-np.mean(sd):+.2f}  d={_cohens_d(sc,sd):.2f}")
    print(f"  C vs E: t={t_ce:.2f} p={p_ce:.3f} ({'*' if p_ce<0.05 else 'n.s.'})  "
          f"Δ={np.mean(sc)-np.mean(se):+.2f}  d={_cohens_d(sc,se):.2f}"
          f"  ({'E faster' if np.mean(sc) > np.mean(se) else 'E slower or equal'})")

    # Ablation
    print(f"\n  Ablation:")
    c_m, f_m, g_m, e_m = [float(np.mean(results[i]["steps"])) for i in [2,5,6,4]]
    print(f"    F (proprio+always-on): {f_m:.2f}  proprio effect: {c_m-f_m:+.2f}")
    print(f"    G (CT-sw only):        {g_m:.2f}  CT effect:      {c_m-g_m:+.2f}")
    print(f"    E (both):              {e_m:.2f}  total effect:   {c_m-e_m:+.2f}")


if __name__ == "__main__":
    main()
