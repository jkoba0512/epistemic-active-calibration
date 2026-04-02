"""
5-Condition comparison: binary vs soft A_tactile × precision switching × proprioception.

Conditions
----------
A  no occlusion,   soft A_tac,   no switching,             no proprio  (visual baseline)
B  full occlusion, binary A_tac, no switching,             no proprio  (original)
C  full occlusion, soft A_tac,   no switching,             no proprio  (soft, control)
D  full occlusion, soft A_tac,   always-on switching,      no proprio  (original proposal)
E  full occlusion, soft A_tac,   contact-triggered switch, WITH proprio (full proposal)

Key comparisons:
  C vs D — same soft A_tac, switching always-on vs off (uniform prior baseline)
  C vs E — contact-triggered switching + proprioception vs no switching

Usage
-----
    uv run python experiments/four_condition_comparison.py
    uv run python experiments/four_condition_comparison.py --episodes 100 --seed 0
    uv run python experiments/four_condition_comparison.py --no-plot --out results/five_cond.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_occlusion.core.generative_model.model_builder import (
    DEFAULT_OBJ_ARM,
    build_A,
    build_B,
    build_C,
    build_D,
)
from aif_occlusion.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_occlusion.core.precision.precision_manager import PrecisionManager
from aif_occlusion.simulation.mujoco_env import OcclusionManipulatorEnv

# ── Model constants (must match mujoco_env) ───────────────────────────────────
N_POS = 5
N_OBJ = 3
N_VIS = N_OBJ + 1   # 0-2 informative, 3 = ambiguous/occluded
N_TAC = 2
N_PROPRIO = N_POS   # arm_pos directly observed
N_ACTIONS = 3
OBJ_ARM = DEFAULT_OBJ_ARM   # {0: 1, 1: 2, 2: 3}


# ── Contact oracle (model-based; bypasses MuJoCo sensor) ─────────────────────

def _contact(arm_pos: int, obj_loc: int) -> int:
    """1 if arm_pos == OBJ_ARM[obj_loc], else 0."""
    return int(arm_pos == OBJ_ARM.get(obj_loc, -1))


def _move(arm_pos: int, action: int) -> int:
    if action == 0:
        return max(0, arm_pos - 1)
    if action == 2:
        return min(N_POS - 1, arm_pos + 1)
    return arm_pos


# ── Single-episode runner ─────────────────────────────────────────────────────

def run_episode(
    env: OcclusionManipulatorEnv,
    A_clean: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    precision_manager: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    max_steps: int = 20,
    with_proprio: bool = False,
    use_oracle: bool = True,
) -> int:
    """
    Run one episode.  Returns steps-to-contact (max_steps if timeout).

    Parameters
    ----------
    use_oracle : bool
        True  (default): tactile and arm_pos come from the ground-truth oracle
                         (_contact() and model-tracked arm_pos). Matches prior
                         experiments.
        False           : tactile comes from the MuJoCo touch sensor; arm_pos
                         comes from JointDiscretizer applied to the actual joint
                         angle. Closer to real-robot conditions.
    """
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    arm_pos = 0   # model-side arm position tracker (used in oracle mode)

    agent = MultiModalAIFAgent(
        A_clean, B, C, D,
        precision_manager=precision_manager,
        policy_len=2,
        inference_horizon=2,
    )
    agent.reset()

    for t in range(max_steps):
        if use_oracle:
            visual_obs  = env.get_pymdp_obs()[0]
            tactile_obs = _contact(arm_pos, obj_loc_idx)
            obs = [visual_obs, tactile_obs]
            if with_proprio:
                obs.append(arm_pos)
        else:
            # Realistic mode: all observations from MuJoCo sensor readouts
            env_obs     = env._get_obs()
            visual_obs  = env_obs.visual_obs_idx
            tactile_obs = env_obs.tactile_obs_idx
            obs = [visual_obs, tactile_obs]
            if with_proprio:
                obs.append(env_obs.arm_pos_idx)

        result = agent.step(obs, c_visual=env.c_visual)
        action = int(result.action[0])

        step_result = env.step(action)

        if use_oracle:
            arm_pos = _move(arm_pos, action)
            if _contact(arm_pos, obj_loc_idx):
                return t + 1
        else:
            if step_result.obs.tactile_obs_idx > 0:
                return t + 1

    return max_steps


# ── Multi-episode runner ──────────────────────────────────────────────────────

def run_condition(
    occlusion_mode: str,
    A_clean: np.ndarray,
    precision_manager: PrecisionManager,
    n_episodes: int,
    seed_offset: int = 0,
    max_steps: int = 20,
    with_proprio: bool = False,
    use_oracle: bool = True,
) -> list[int]:
    env = OcclusionManipulatorEnv(
        occlusion_mode=occlusion_mode, n_arm_positions=N_POS, max_steps=50
    )
    n_proprio = N_PROPRIO if with_proprio else 0
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
            use_oracle=use_oracle,
        )
        for ep in range(n_episodes)
    ]


# ── Results reporting ─────────────────────────────────────────────────────────

def _report(label: str, steps: list[int], max_steps: int) -> dict:
    n = len(steps)
    contacted = sum(s < max_steps for s in steps)
    m = float(np.mean(steps))
    s = float(np.std(steps))
    print(f"  {label}: mean={m:.1f} ± {s:.1f}  contact={contacted}/{n}")
    return {"label": label, "mean": m, "std": s, "contacted": contacted, "n": n,
            "steps": steps}


def _cohens_d(a: list[int], b: list[int]) -> float:
    """Cohen's d effect size (pooled SD)."""
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    return float((np.mean(a) - np.mean(b)) / (np.sqrt(pooled_var) + 1e-12))


def _significance(
    steps_c: list[int],
    steps_d: list[int],
    steps_e: list[int],
) -> dict:
    # Welch's t-test (does not assume equal variance — C std=3.0 vs E std=0.8)
    t_cd, p_cd = scipy_stats.ttest_ind(steps_c, steps_d, equal_var=False)
    t_ce, p_ce = scipy_stats.ttest_ind(steps_c, steps_e, equal_var=False)
    sig_cd = "*" if p_cd < 0.05 else "n.s."
    sig_ce = "*" if p_ce < 0.05 else "n.s."
    delta_cd = float(np.mean(steps_c) - np.mean(steps_d))
    delta_ce = float(np.mean(steps_c) - np.mean(steps_e))
    d_cd = _cohens_d(steps_c, steps_d)
    d_ce = _cohens_d(steps_c, steps_e)
    print(f"\n  C vs D (always-on switch):          t={t_cd:.2f}, p={p_cd:.3f} ({sig_cd})"
          f"  Δ={delta_cd:+.1f}  d={d_cd:.2f}")
    print(f"  C vs E (contact-trigger + proprio):  t={t_ce:.2f}, p={p_ce:.3f} ({sig_ce})"
          f"  Δ={delta_ce:+.1f}  d={d_ce:.2f}"
          f"  ({'E faster' if delta_ce > 0 else 'E slower or equal'})")
    return {
        "c_vs_d": {"t": float(t_cd), "p": float(p_cd), "cohens_d": d_cd},
        "c_vs_e": {"t": float(t_ce), "p": float(p_ce), "cohens_d": d_ce},
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(results: list[dict], stats: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    all_steps = [r["steps"] for r in results]
    colors = ["steelblue", "tomato", "darkorange", "seagreen", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bp = axes[0].boxplot(
        all_steps, tick_labels=labels, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Steps to Contact")
    axes[0].set_title(
        "5-Condition Comparison\nbinary/soft A_tactile × switching × proprioception"
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([0, max(max(s) for s in all_steps) + 2])

    means = [r["mean"] for r in results]
    stds  = [r["std"]  for r in results]
    x = np.arange(len(labels))
    bars = axes[1].bar(x, means, yerr=stds, color=colors[:len(results)], alpha=0.7,
                       capsize=5, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Mean Steps to Contact")
    axes[1].set_title("Mean Steps ± SD")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim([0, max(means) * 1.5])
    for bar, mean in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    p_cd = stats["c_vs_d"]["p"]
    p_ce = stats["c_vs_e"]["p"]
    sig_str = (
        f"C vs D: p={p_cd:.3f} {'(*)' if p_cd < 0.05 else '(n.s.)'}\n"
        f"C vs E: p={p_ce:.3f} {'(*)' if p_ce < 0.05 else '(n.s.)'}"
    )
    axes[1].text(
        0.5, 0.95, sig_str, transform=axes[1].transAxes,
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per condition (default: 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed (default: 0)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max steps per episode (default: 20)")
    parser.add_argument("--p-contact", type=float, default=0.9,
                        help="Soft A_tactile contact probability (default: 0.9)")
    parser.add_argument("--p-bg", type=float, default=0.05,
                        help="Soft A_tactile background probability (default: 0.05)")
    parser.add_argument("--pi-max", type=float, default=5.0,
                        help="Max tactile precision (default: 5.0)")
    parser.add_argument("--noise-floor", type=float, default=0.1,
                        help="Tactile noise floor for sharpening (default: 0.1)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON output path (optional)")
    parser.add_argument("--realistic", action="store_true",
                        help="Use MuJoCo sensor values instead of oracle for tactile/proprio")
    parser.add_argument("--proprio-accuracy", type=float, default=1.0,
                        help="Proprioceptive accuracy ∈ (0,1] (default: 1.0 = perfect)")
    parser.add_argument("--fairness-check", action="store_true",
                        help="Add condition H (no-occ + proprio) for fair baseline comparison")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    # ── Build model variants ──────────────────────────────────────────────────
    A_binary      = build_A(N_POS, N_OBJ, p_contact=1.0, p_bg=0.0)
    A_soft        = build_A(N_POS, N_OBJ, p_contact=args.p_contact, p_bg=args.p_bg)
    A_soft_proprio = build_A(
        N_POS, N_OBJ, p_contact=args.p_contact, p_bg=args.p_bg,
        with_proprio=True, proprio_accuracy=args.proprio_accuracy,
    )
    use_oracle = not args.realistic

    pm_no_switch = PrecisionManager(
        theta=0.4, pi_tactile_max=args.pi_max,
        pi_visual_min=0.1, tactile_noise_floor=0.0,
    )
    pm_switch = PrecisionManager(
        theta=0.4, pi_tactile_max=args.pi_max,
        pi_visual_min=0.1, tactile_noise_floor=args.noise_floor,
    )
    pm_contact_trigger = PrecisionManager(
        theta=0.4, pi_tactile_max=args.pi_max,
        pi_visual_min=0.1, tactile_noise_floor=args.noise_floor,
        contact_triggered=True,
    )

    acc_str = f" acc={args.proprio_accuracy}" if args.proprio_accuracy < 1.0 else ""
    real_str = " [realistic]" if args.realistic else ""
    # (label, occlusion_mode, A_clean, precision_manager, with_proprio)
    conditions = [
        ("A: no-occ/soft",         "none", A_soft,         pm_no_switch,       False),
        ("B: full-occ/binary",     "full", A_binary,       pm_no_switch,       False),
        ("C: full-occ/soft",       "full", A_soft,         pm_no_switch,       False),
        ("D: full-occ/sw",         "full", A_soft,         pm_switch,          False),
        (f"E: full-occ/sw+proprio{acc_str}{real_str}", "full", A_soft_proprio, pm_contact_trigger, True),
        # Ablation: decompose proprio vs contact-triggered contributions
        (f"F: proprio+always-on{acc_str}", "full", A_soft_proprio, pm_switch,  True),
        ("G: CT-sw only",          "full", A_soft,         pm_contact_trigger, False),
    ]
    if args.fairness_check:
        # Condition H: fair baseline — same proprio as E but no occlusion
        conditions.append(
            (f"H: no-occ+proprio{acc_str}", "none", A_soft_proprio, pm_no_switch, True)
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    n_conditions = len(conditions)
    print(f"{n_conditions}-Condition Comparison  (n={args.episodes} episodes, seed={args.seed})")
    print(f"  A_soft: p_contact={args.p_contact}, p_bg={args.p_bg}")
    print(f"  Switching: pi_max={args.pi_max}, noise_floor={args.noise_floor}")
    if args.proprio_accuracy < 1.0:
        print(f"  proprio_accuracy={args.proprio_accuracy}")
    if args.realistic:
        print(f"  Mode: REALISTIC (MuJoCo sensor values)")
    print()

    results = []
    for label, occ_mode, A_clean, pm, proprio in conditions:
        steps = run_condition(occ_mode, A_clean, pm,
                              n_episodes=args.episodes,
                              seed_offset=args.seed,
                              max_steps=args.max_steps,
                              with_proprio=proprio,
                              use_oracle=use_oracle)
        results.append(_report(label, steps, args.max_steps))

    stats = _significance(results[2]["steps"], results[3]["steps"], results[4]["steps"])
    # Ablation summary
    if len(results) >= 7:
        f_mean = float(np.mean(results[5]["steps"]))
        g_mean = float(np.mean(results[6]["steps"]))
        e_mean = float(np.mean(results[4]["steps"]))
        c_mean = float(np.mean(results[2]["steps"]))
        print(f"\n  Ablation (proprio vs CT-switching):")
        print(f"    F (proprio+always-on): {f_mean:.1f}  — proprio効果: {c_mean - f_mean:+.1f}")
        print(f"    G (CT-sw only):        {g_mean:.1f}  — CT効果:     {c_mean - g_mean:+.1f}")
        print(f"    E (both):              {e_mean:.1f}  — 合計効果:    {c_mean - e_mean:+.1f}")

    # ── Output ────────────────────────────────────────────────────────────────
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args) | {"out": str(args.out)},
            "results": [{k: v for k, v in r.items() if k != "steps"} | {"steps": r["steps"]}
                        for r in results],
            "stats": stats,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"  Results saved → {args.out}")

    if not args.no_plot:
        plot_path = (args.out.parent / "five_condition_comparison.png"
                     if args.out else Path("results/five_condition_comparison.png"))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot(results, stats, plot_path)


if __name__ == "__main__":
    main()
