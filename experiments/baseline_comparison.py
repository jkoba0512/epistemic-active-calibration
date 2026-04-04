"""
Baseline comparison: AIF vs systematic-search policy
under varying proprioceptive noise levels.

Purpose
-------
Separate the contribution of:
  (a) information quantity  — having proprio at all
  (b) AIF belief updating   — handling noisy proprio gracefully

If AIF(noisy) >> systematic(noisy), it is AIF's uncertainty handling that matters.
If AIF(noisy) ≈ systematic(noisy), the benefit is just information quantity.

Conditions
----------
The same proprio_accuracy is given to both AIF and the systematic baseline,
so the only difference is *how* each policy uses that noisy information.

  AIF-perfect     AIF with proprio_accuracy=1.0  (existing condition E)
  AIF-noisy       AIF with proprio_accuracy=0.2
  SYS-perfect     Systematic scan with perfect proprio
  SYS-noisy       Systematic scan with noisy proprio (accuracy=0.2)
  RANDOM          Random action policy (lower bound)
  AIF-noproprio   AIF with no proprio (existing condition C)

Usage
-----
    uv run python experiments/baseline_comparison.py
    uv run python experiments/baseline_comparison.py --episodes 100
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
from aif_calib_robustness.simulation.mujoco_env import OcclusionManipulatorEnv

N_POS     = 5
N_OBJ     = 3
N_VIS     = N_OBJ + 1
N_TAC     = 2
N_PROPRIO = N_POS
N_ACTIONS = 3
OBJ_ARM   = DEFAULT_OBJ_ARM   # {0:1, 1:2, 2:3}


# ── Oracle helpers ────────────────────────────────────────────────────────────

def _contact(arm_pos: int, obj_loc: int) -> int:
    return int(arm_pos == OBJ_ARM.get(obj_loc, -1))

def _move(arm_pos: int, action: int) -> int:
    if action == 0: return max(0, arm_pos - 1)
    if action == 2: return min(N_POS - 1, arm_pos + 1)
    return arm_pos

def _noisy_proprio(true_pos: int, accuracy: float, rng: np.random.Generator) -> int:
    """Return a noisy proprioceptive observation of arm position."""
    if rng.random() < accuracy:
        return true_pos
    return int(rng.integers(0, N_POS))


# ── Baseline policies ─────────────────────────────────────────────────────────

class RandomPolicy:
    """Always choose a random action — lower bound."""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    def reset(self): pass
    def step(self, visual_obs, tactile_obs, proprio_obs=None) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


class SystematicPolicy:
    """
    Systematic left-to-right scan using noisy proprioception.

    Strategy: maintain a believed arm position from noisy proprio obs,
    then move right if no contact yet, left only if at boundary.
    This is the simplest sensible policy that *uses* proprio info.
    """
    def __init__(self):
        self.believed_pos = 0
        self.found_contact = False

    def reset(self):
        self.believed_pos = 0
        self.found_contact = False

    def step(self, visual_obs, tactile_obs, proprio_obs=None) -> int:
        if tactile_obs == 1:
            self.found_contact = True
            return 1  # stay

        # Update believed position from noisy proprio (if available)
        if proprio_obs is not None:
            self.believed_pos = proprio_obs

        # Systematic scan: move right unless at rightmost position
        if self.believed_pos < N_POS - 1:
            self.believed_pos = min(N_POS - 1, self.believed_pos + 1)
            return 2  # right
        else:
            self.believed_pos = max(0, self.believed_pos - 1)
            return 0  # left (hit boundary, reverse)


# ── Episode runners ───────────────────────────────────────────────────────────

def run_aif_episode(
    env: OcclusionManipulatorEnv,
    A_clean, pm: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    proprio_accuracy: float,
    max_steps: int = 20,
) -> int:
    """Run one AIF episode with specified proprio noise."""
    rng = np.random.default_rng(seed)
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    arm_pos = 0

    with_proprio = proprio_accuracy > 0.0
    n_proprio = N_PROPRIO if with_proprio else 0
    agent = MultiModalAIFAgent(
        A_clean,
        build_B(N_POS, N_OBJ),
        build_C(N_VIS, N_TAC, n_proprio=n_proprio),
        build_D(N_POS, N_OBJ),
        precision_manager=pm,
        policy_len=2,
        inference_horizon=2,
    )
    agent.reset()

    for t in range(max_steps):
        visual_obs  = env.get_pymdp_obs()[0]
        tactile_obs = _contact(arm_pos, obj_loc_idx)
        obs = [visual_obs, tactile_obs]
        if with_proprio:
            noisy_pos = _noisy_proprio(arm_pos, proprio_accuracy, rng)
            obs.append(noisy_pos)

        result = agent.step(obs, c_visual=env.c_visual)
        action = int(result.action[0])
        env.step(action)
        arm_pos = _move(arm_pos, action)

        if _contact(arm_pos, obj_loc_idx):
            return t + 1

    return max_steps


def run_baseline_episode(
    env: OcclusionManipulatorEnv,
    policy,
    obj_loc_idx: int,
    seed: int,
    proprio_accuracy: float,
    max_steps: int = 20,
) -> int:
    """Run one episode with a non-AIF baseline policy."""
    rng = np.random.default_rng(seed + 10000)
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    arm_pos = 0
    policy.reset()

    for t in range(max_steps):
        visual_obs  = env.get_pymdp_obs()[0]
        tactile_obs = _contact(arm_pos, obj_loc_idx)

        if proprio_accuracy > 0.0:
            noisy_pos = _noisy_proprio(arm_pos, proprio_accuracy, rng)
            action = policy.step(visual_obs, tactile_obs, noisy_pos)
        else:
            action = policy.step(visual_obs, tactile_obs)

        env.step(action)
        arm_pos = _move(arm_pos, action)

        if _contact(arm_pos, obj_loc_idx):
            return t + 1

    return max_steps


def run_condition_aif(
    A_clean, pm, proprio_accuracy, n_episodes, seed_offset, max_steps
) -> list[int]:
    env = OcclusionManipulatorEnv(occlusion_mode="full", n_arm_positions=N_POS, max_steps=50)
    return [
        run_aif_episode(env, A_clean, pm,
                        obj_loc_idx=ep % N_OBJ,
                        seed=seed_offset + ep,
                        proprio_accuracy=proprio_accuracy,
                        max_steps=max_steps)
        for ep in range(n_episodes)
    ]


def run_condition_baseline(
    policy_factory, proprio_accuracy, n_episodes, seed_offset, max_steps
) -> list[int]:
    env = OcclusionManipulatorEnv(occlusion_mode="full", n_arm_positions=N_POS, max_steps=50)
    rng_for_policy = np.random.default_rng(seed_offset)
    return [
        run_baseline_episode(env,
                             policy_factory(rng_for_policy),
                             obj_loc_idx=ep % N_OBJ,
                             seed=seed_offset + ep,
                             proprio_accuracy=proprio_accuracy,
                             max_steps=max_steps)
        for ep in range(n_episodes)
    ]


# ── Statistics & reporting ────────────────────────────────────────────────────

def _report(label: str, steps: list[int], max_steps: int) -> dict:
    m = float(np.mean(steps))
    s = float(np.std(steps))
    contacted = sum(x < max_steps for x in steps)
    print(f"  {label:35s}: mean={m:.2f} ± {s:.2f}  contact={contacted}/{len(steps)}")
    return {"label": label, "mean": m, "std": s, "steps": steps}


def _compare(a_label, a_steps, b_label, b_steps):
    t, p = scipy_stats.ttest_ind(a_steps, b_steps, equal_var=False)
    na, nb = len(a_steps), len(b_steps)
    pooled = ((na-1)*np.var(a_steps, ddof=1) + (nb-1)*np.var(b_steps, ddof=1)) / (na+nb-2)
    d = float((np.mean(a_steps) - np.mean(b_steps)) / (np.sqrt(pooled) + 1e-12))
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    delta = float(np.mean(a_steps) - np.mean(b_steps))
    print(f"  {a_label} vs {b_label}: Δ={delta:+.2f}  d={d:.2f}  p={p:.4f} {sig}")
    return {"delta": delta, "cohens_d": d, "p": p}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(results: list[dict], comparisons: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels    = [r["label"] for r in results]
    all_steps = [r["steps"] for r in results]
    means     = [r["mean"]  for r in results]
    stds      = [r["std"]   for r in results]

    # Color: blue=AIF, orange=systematic, grey=random
    colors = []
    for l in labels:
        if "AIF" in l:    colors.append("steelblue")
        elif "SYS" in l:  colors.append("darkorange")
        else:             colors.append("grey")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bp = axes[0].boxplot(all_steps, tick_labels=labels, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_ylabel("Steps to Contact")
    axes[0].set_title("AIF vs Systematic Baseline\nunder Proprioceptive Noise")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)

    x = np.arange(len(labels))
    bars = axes[1].bar(x, means, yerr=stds, color=colors, alpha=0.7,
                       capsize=5, edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Mean Steps to Contact")
    axes[1].set_title("Mean ± SD")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{mean:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Key comparison annotation
    key = comparisons.get("aif_noisy_vs_sys_noisy", {})
    if key:
        p, d = key.get("p", 1.0), key.get("cohens_d", 0.0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        axes[1].text(0.5, 0.95,
                     f"AIF-noisy vs SYS-noisy\np={p:.4f} {sig}  d={d:.2f}",
                     transform=axes[1].transAxes, ha="center", va="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes",    type=int,   default=60)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--max-steps",   type=int,   default=20)
    parser.add_argument("--noisy-acc",   type=float, default=0.2,
                        help="Proprio accuracy for noisy conditions (default: 0.2)")
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--out",         type=Path,  default=Path("results/baseline_comparison.png"))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    ACC_NOISY = args.noisy_acc

    # Shared AIF model components
    p_c, p_bg = 0.9, 0.05
    A_noproprio = build_A(N_POS, N_OBJ, p_contact=p_c, p_bg=p_bg)
    A_perfect   = build_A(N_POS, N_OBJ, p_contact=p_c, p_bg=p_bg,
                          with_proprio=True, proprio_accuracy=1.0)
    A_noisy     = build_A(N_POS, N_OBJ, p_contact=p_c, p_bg=p_bg,
                          with_proprio=True, proprio_accuracy=ACC_NOISY)

    pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                          tactile_noise_floor=0.1, contact_triggered=True)
    pm_base = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1)

    kw = dict(n_episodes=args.episodes, seed_offset=args.seed, max_steps=args.max_steps)

    print(f"Baseline Comparison  (n={args.episodes}, noisy_acc={ACC_NOISY})\n")

    results = []

    # AIF conditions
    print("── AIF conditions ──")
    results.append(_report("AIF-noproprio (C)",
        run_condition_aif(A_noproprio, pm_base, 0.0, **kw), args.max_steps))
    results.append(_report(f"AIF-perfect (E, acc=1.0)",
        run_condition_aif(A_perfect, pm, 1.0, **kw), args.max_steps))
    results.append(_report(f"AIF-noisy (acc={ACC_NOISY})",
        run_condition_aif(A_noisy, pm, ACC_NOISY, **kw), args.max_steps))

    # Systematic baseline conditions
    print("\n── Systematic scan baseline ──")
    results.append(_report("SYS-perfect (acc=1.0)",
        run_condition_baseline(lambda rng: SystematicPolicy(),
                               1.0, **kw), args.max_steps))
    results.append(_report(f"SYS-noisy (acc={ACC_NOISY})",
        run_condition_baseline(lambda rng: SystematicPolicy(),
                               ACC_NOISY, **kw), args.max_steps))

    # Random lower bound
    print("\n── Random baseline ──")
    results.append(_report("RANDOM",
        run_condition_baseline(lambda rng: RandomPolicy(rng),
                               0.0, **kw), args.max_steps))

    # Key comparisons
    print("\n── Key comparisons ──")
    comparisons = {}
    comparisons["aif_perfect_vs_sys_perfect"] = _compare(
        "AIF-perfect", results[1]["steps"],
        "SYS-perfect", results[3]["steps"])
    comparisons["aif_noisy_vs_sys_noisy"] = _compare(
        f"AIF-noisy(acc={ACC_NOISY})", results[2]["steps"],
        f"SYS-noisy(acc={ACC_NOISY})", results[4]["steps"])
    comparisons["aif_noisy_vs_aif_perfect"] = _compare(
        f"AIF-noisy(acc={ACC_NOISY})", results[2]["steps"],
        "AIF-perfect",                 results[1]["steps"])

    print("\n── Interpretation ──")
    d_key = comparisons["aif_noisy_vs_sys_noisy"]["cohens_d"]
    p_key = comparisons["aif_noisy_vs_sys_noisy"]["p"]
    if p_key < 0.05 and d_key > 0.5:
        print("  → AIF outperforms systematic scan under noisy proprio:")
        print("    Benefit is NOT just information quantity — AIF uncertainty handling matters.")
    elif p_key >= 0.05:
        print("  → No significant difference between AIF and systematic scan under noisy proprio:")
        print("    Cannot rule out information-quantity explanation.")
    else:
        print(f"  → Marginal difference (d={d_key:.2f}, p={p_key:.4f}). Borderline case.")

    if not args.no_plot:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _plot(results, comparisons, args.out)


if __name__ == "__main__":
    main()
