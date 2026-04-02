"""
キャリブレーション系統誤差スイープ実験。

仮説：CT-switchingはランダムノイズには効かないが、
     系統的キャリブレーション誤差に対して効果を持つ可能性がある。

設定：
  - calib_offset: 実際のarm角度に加算してエージェントに報告する系統的オフセット
  - エージェントのA行列は変えない（「自分のキャリブレーションは正しい」と信じている）
  - 物理環境（タッチセンサ、実際の動作）はオフセットの影響を受けない

Usage
-----
    uv run python experiments/run_calib_offset_sweep.py
    uv run python experiments/run_calib_offset_sweep.py --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_occlusion.core.generative_model.model_builder import build_A, build_B, build_C, build_D
from aif_occlusion.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_occlusion.core.precision.precision_manager import PrecisionManager
from aif_occlusion.simulation.so101_env import SO101OcclusionEnv

N_POS    = 5
N_OBJ    = 3
N_VIS    = N_OBJ + 1
N_TAC    = 2
N_PROPRIO = N_POS


def run_episode(
    env: SO101OcclusionEnv,
    A, B, C, D,
    pm: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    max_steps: int = 20,
    with_proprio: bool = False,
) -> int:
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(
        A, B, C, D,
        precision_manager=pm,
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
    calib_offset: float,
    occlusion_mode: str,
    A, pm: PrecisionManager,
    n_episodes: int,
    seed_offset: int = 0,
    max_steps: int = 20,
    with_proprio: bool = False,
) -> list[int]:
    n_proprio = N_PROPRIO if with_proprio else 0
    env = SO101OcclusionEnv(
        occlusion_mode=occlusion_mode,
        n_arm_positions=N_POS,
        max_steps=max_steps + 5,
        calib_offset=calib_offset,
    )
    return [
        run_episode(
            env, A,
            build_B(N_POS, N_OBJ),
            build_C(N_VIS, N_TAC, n_proprio=n_proprio),
            build_D(N_POS, N_OBJ),
            pm,
            obj_loc_idx=ep % N_OBJ,
            seed=seed_offset + ep,
            max_steps=max_steps,
            with_proprio=with_proprio,
        )
        for ep in range(n_episodes)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes",   type=int,   default=60)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--max-steps",  type=int,   default=20)
    parser.add_argument("--out",        type=Path,  default=Path("results/calib_offset_sweep.png"))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    # キャリブレーションオフセット範囲
    # bin幅 = 0.16 rad。1bin = 0.16 rad
    BIN_WIDTH = 0.16
    offsets = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20])
    offset_labels = [f"{o:.2f}\n({o/BIN_WIDTH:.1f}bin)" for o in offsets]

    A_soft        = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05)
    A_soft_proprio = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                             with_proprio=True, proprio_accuracy=1.0)

    pm_no = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                             tactile_noise_floor=0.0)
    pm_sw = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                             tactile_noise_floor=0.1)
    pm_ct = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                             tactile_noise_floor=0.1, contact_triggered=True)

    # 比較条件
    conditions = [
        ("C: baseline\n(no proprio, no sw)",   A_soft,         pm_no, False, "C0519A", "-"),
        ("E: CT-sw + proprio",                  A_soft_proprio, pm_ct, True,  "E05B4A", "-"),
        ("F: always-on + proprio",              A_soft_proprio, pm_sw, True,  "2196F3", "--"),
        ("G: CT-sw only\n(no proprio)",         A_soft,         pm_ct, False, "FF9800", ":"),
    ]

    print(f"キャリブレーション系統誤差スイープ (n={args.episodes}, seed={args.seed})")
    print(f"bin幅 = {BIN_WIDTH:.2f} rad")
    print()

    results = {}  # cond_label -> list of mean_steps per offset

    for label, A, pm, proprio, color, ls in conditions:
        means = []
        contacts = []
        print(f"  {label.replace(chr(10), ' ')}:")
        for offset in offsets:
            steps = run_condition(
                calib_offset=offset,
                occlusion_mode="full",
                A=A, pm=pm,
                n_episodes=args.episodes,
                seed_offset=args.seed,
                max_steps=args.max_steps,
                with_proprio=proprio,
            )
            m = float(np.mean(steps))
            c = sum(s < args.max_steps for s in steps)
            means.append(m)
            contacts.append(c)
            print(f"    offset={offset:.2f}rad ({offset/BIN_WIDTH:.1f}bin): "
                  f"mean={m:.2f}  contact={c}/{args.episodes}")
        results[label] = (means, contacts)
        print()

    # ── 統計検定：offset=0 vs offset=0.16 (1bin) ──────────────────────────
    print("  C vs E (offset=0.16rad = 1bin):")
    offset_idx_1bin = list(offsets).index(0.16) if 0.16 in offsets else -1

    # ── 図の作成 ──────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("キャリブレーション系統誤差 vs AIF条件\n"
                 "(遮蔽=full, 系統的オフセット: モデルと環境のずれ)",
                 fontsize=12)

    # Panel 1: 平均ステップ数
    ax = axes[0]
    for label, A, pm, proprio, color, ls in conditions:
        means, _ = results[label]
        short = label.split("\n")[0]
        ax.plot(offsets / BIN_WIDTH, means,
                marker="o", linestyle=ls,
                color=f"#{color}", label=short, linewidth=2)

    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="0.5bin")
    ax.axvline(x=1.0, color="red",  linestyle=":", alpha=0.5, label="1.0bin (1bin誤認)")
    ax.set_xlabel("キャリブレーション誤差 [bin幅単位]", fontsize=11)
    ax.set_ylabel("平均ステップ数（低いほど良い）", fontsize=11)
    ax.set_title("平均ステップ数 vs キャリブレーション誤差", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, args.max_steps + 1)
    ax.set_xticks(offsets / BIN_WIDTH)
    ax.set_xticklabels([f"{o/BIN_WIDTH:.2f}" for o in offsets])

    # Panel 2: 接触成功率
    ax = axes[1]
    for label, A, pm, proprio, color, ls in conditions:
        _, contacts = results[label]
        rates = [c / args.episodes * 100 for c in contacts]
        short = label.split("\n")[0]
        ax.plot(offsets / BIN_WIDTH, rates,
                marker="s", linestyle=ls,
                color=f"#{color}", label=short, linewidth=2)

    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=1.0, color="red",  linestyle=":", alpha=0.5)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3)
    ax.set_xlabel("キャリブレーション誤差 [bin幅単位]", fontsize=11)
    ax.set_ylabel("接触成功率 [%]", fontsize=11)
    ax.set_title("接触成功率 vs キャリブレーション誤差", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xticks(offsets / BIN_WIDTH)
    ax.set_xticklabels([f"{o/BIN_WIDTH:.2f}" for o in offsets])

    plt.tight_layout()
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"\n  図を保存: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
