"""
CT-switching の貢献分離実験。

仮説: CT-switching は tactile 偽陽性が多い環境で効果を発揮する。
     現在の物理タッチセンサ環境では偽陽性がほぼゼロのため
     always-on と同等になる。

設計:
  2 要因:
    - Proprio (ON/OFF) × CT-switching (ON/OFF)
  操作変数:
    - 偽陽性率 fp_rate ∈ {0.00, 0.05, 0.10, 0.20, 0.30}
      エージェントへの tactile 観測に確率的に 1 を注入
      （実環境は変えない = 本物の接触は正しく検出）

予測:
  fp_rate が低い → E ≈ F, G ≈ C (CT-sw 無関係)
  fp_rate が高い → E > F, G > C (CT-sw が偽陽性を抑制)
  Proprio は fp_rate によらず常に有効

Usage
-----
    uv run python experiments/run_ctswitch_decomposition.py
    uv run python experiments/run_ctswitch_decomposition.py --episodes 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_calib_robustness.core.generative_model.model_builder import build_A, build_B, build_C, build_D
from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager
from aif_calib_robustness.simulation.so101_env import SO101OcclusionEnv

N_POS     = 5
N_OBJ     = 3
N_VIS     = N_OBJ + 1
N_TAC     = 2
N_PROPRIO = N_POS


def run_episode(
    env: SO101OcclusionEnv,
    A, B, C, D,
    pm: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    fp_rate: float = 0.0,
    with_proprio: bool = False,
    max_steps: int = 20,
) -> int:
    """
    fp_rate: エージェントへの tactile 偽陽性率（実環境は不変）。
    """
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(
        A, B, C, D,
        precision_manager=pm,
        policy_len=2, inference_horizon=2,
    )
    agent.reset()
    rng = np.random.default_rng(seed + 99999)

    for t in range(max_steps):
        env_obs = env._get_obs()

        # 偽陽性注入（エージェントの観測のみ、物理は不変）
        tac_obs = env_obs.tactile_obs_idx
        if tac_obs == 0 and rng.random() < fp_rate:
            tac_obs = 1

        obs = [env_obs.visual_obs_idx, tac_obs]
        if with_proprio:
            obs.append(env_obs.arm_pos_idx)

        result = agent.step(obs, c_visual=env.c_visual)
        action = int(result.action[0])
        step_result = env.step(action)

        # 本物の接触のみ成功とする
        if step_result.obs.tactile_obs_idx > 0:
            return t + 1

    return max_steps


def run_condition(
    A, pm: PrecisionManager,
    fp_rate: float,
    with_proprio: bool,
    n_episodes: int,
    seed_offset: int,
    max_steps: int = 20,
) -> list[int]:
    n_proprio = N_PROPRIO if with_proprio else 0
    env = SO101OcclusionEnv(
        occlusion_mode="full",
        n_arm_positions=N_POS,
        max_steps=max_steps + 5,
    )
    B = build_B(N_POS, N_OBJ)
    C_pref = build_C(N_VIS, N_TAC, n_proprio=n_proprio)
    D = build_D(N_POS, N_OBJ)

    return [
        run_episode(
            env, A, B, C_pref, D, pm,
            obj_loc_idx=ep % N_OBJ,
            seed=seed_offset + ep,
            fp_rate=fp_rate,
            with_proprio=with_proprio,
            max_steps=max_steps,
        )
        for ep in range(n_episodes)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes",   type=int,  default=60)
    parser.add_argument("--seed",       type=int,  default=0)
    parser.add_argument("--max-steps",  type=int,  default=20)
    parser.add_argument("--out",        type=Path, default=Path("results/ctswitch_decomposition.png"))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    fp_rates = np.array([0.00, 0.05, 0.10, 0.20, 0.30])

    A_no_proprio = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05)
    A_proprio    = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                           with_proprio=True, proprio_accuracy=1.0)

    # CT-sw OFF: tactile_noise_floor=0.0（ノイズなし、常にシャープ）
    # CT-sw ON:  tactile_noise_floor=0.1（ノイズあり、接触時のみシャープ）
    pm_no_sw = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                                tactile_noise_floor=0.0)
    pm_ct_sw = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                                tactile_noise_floor=0.1, contact_triggered=True)

    conditions = [
        ("C: proprio=OFF, CT-sw=OFF", A_no_proprio, pm_no_sw, False, "C0519A", "-"),
        ("G: proprio=OFF, CT-sw=ON",  A_no_proprio, pm_ct_sw, False, "FF9800", ":"),
        ("F: proprio=ON,  CT-sw=OFF", A_proprio,    pm_no_sw, True,  "2196F3", "--"),
        ("E: proprio=ON,  CT-sw=ON",  A_proprio,    pm_ct_sw, True,  "4CAF50", "-"),
    ]

    print(f"CT-switching 貢献分離実験 (n={args.episodes}, seed={args.seed})")
    print(f"偽陽性率 = {fp_rates}")
    print()

    results = {}

    for label, A, pm, proprio, color, ls in conditions:
        contacts_per_fp = []
        print(f"  {label}:")
        for fp in fp_rates:
            steps = run_condition(
                A=A, pm=pm,
                fp_rate=float(fp),
                with_proprio=proprio,
                n_episodes=args.episodes,
                seed_offset=args.seed,
                max_steps=args.max_steps,
            )
            n_contact = sum(s < args.max_steps for s in steps)
            contacts_per_fp.append(n_contact)
            print(f"    fp={fp:.2f}: contact={n_contact}/{args.episodes}  "
                  f"mean={np.mean(steps):.2f}")
        results[label] = (contacts_per_fp, color, ls)
        print()

    # 貢献の分離分析
    print("  === 貢献の分離 ===")
    print(f"  {'fp_rate':>8}  "
          f"{'proprio寄与':>12}  "
          f"{'CT-sw寄与(no-p)':>16}  "
          f"{'CT-sw寄与(p)':>14}")
    print(f"  {'-'*58}")

    c_contacts = results["C: proprio=OFF, CT-sw=OFF"][0]
    g_contacts = results["G: proprio=OFF, CT-sw=ON"][0]
    f_contacts = results["F: proprio=ON,  CT-sw=OFF"][0]
    e_contacts = results["E: proprio=ON,  CT-sw=ON"][0]

    for i, fp in enumerate(fp_rates):
        proprio_effect   = f_contacts[i] - c_contacts[i]   # F - C
        ctswitch_no_p    = g_contacts[i] - c_contacts[i]   # G - C
        ctswitch_with_p  = e_contacts[i] - f_contacts[i]   # E - F
        print(f"  {fp:>8.2f}  "
              f"{proprio_effect:>+12}  "
              f"{ctswitch_no_p:>+16}  "
              f"{ctswitch_with_p:>+14}")

    # ── 図の作成 ──────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "CT-switching Contribution Decomposition\n"
        f"(occlusion=full, n={args.episodes}, seed={args.seed})",
        fontsize=12,
    )

    # Panel 1: 接触成功率 vs 偽陽性率
    ax = axes[0]
    for label, A, pm, proprio, color, ls in conditions:
        contacts, clr, linestyle = results[label]
        rates = [c / args.episodes * 100 for c in contacts]
        short = label.split(":")[0]
        ax.plot(fp_rates * 100, rates,
                marker="o", linestyle=linestyle,
                color=f"#{clr}", label=short, linewidth=2)

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Tactile false positive rate [%]", fontsize=11)
    ax.set_ylabel("Contact success rate [%]", fontsize=11)
    ax.set_title("Success Rate vs False Positive Rate", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(fp_rates * 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: CT-switching の寄与（成功数の差）
    ax = axes[1]
    ct_effect_no_p   = [g_contacts[i] - c_contacts[i] for i in range(len(fp_rates))]
    ct_effect_with_p = [e_contacts[i] - f_contacts[i] for i in range(len(fp_rates))]

    ax.plot(fp_rates * 100, ct_effect_no_p,
            marker="o", color="#FF9800", linewidth=2, label="CT-sw effect (no proprio)")
    ax.plot(fp_rates * 100, ct_effect_with_p,
            marker="s", color="#4CAF50", linewidth=2, label="CT-sw effect (with proprio)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Tactile false positive rate [%]", fontsize=11)
    ax.set_ylabel("CT-sw contribution [contacts gained]", fontsize=11)
    ax.set_title("Marginal Contribution of CT-switching", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"\n  図を保存: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
