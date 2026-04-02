"""
ステップ3: AIF による自己キャリブレーション補正。

アルゴリズム:
  フェーズ1（推定フェーズ）:
    N_calib エピソード実行し，接触成功時の (reported_arm_bin, obj_loc) を収集。
    期待接触ビンとの差分でキャリブレーションオフセットをビン単位で推定。

  フェーズ2（タスクフェーズ）:
    推定オフセットで A 行列の obj_arm マッピングを補正し N_test エピソード実行。
    未補正条件との成功率・ステップ数を比較。

Usage
-----
    uv run python experiments/run_step3_self_calib.py
    uv run python experiments/run_step3_self_calib.py --n-calib 30 --n-test 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_occlusion.core.generative_model.model_builder import (
    build_A, build_B, build_C, build_D, DEFAULT_OBJ_ARM,
)
from aif_occlusion.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_occlusion.core.precision.precision_manager import PrecisionManager
from aif_occlusion.simulation.so101_env import SO101OcclusionEnv

N_POS    = 5
N_OBJ    = 3
N_VIS    = N_OBJ + 1
N_TAC    = 2
N_PROPRIO = N_POS
BIN_WIDTH = 0.80 / N_POS  # 0.16 rad


# ---------------------------------------------------------------------------
# Episode runner — returns steps + contact info
# ---------------------------------------------------------------------------

def run_episode(
    env: SO101OcclusionEnv,
    A, B, C, D,
    pm: PrecisionManager,
    obj_loc_idx: int,
    seed: int,
    max_steps: int = 20,
) -> tuple[int, Optional[int]]:
    """
    Returns
    -------
    steps : int
        Steps taken (max_steps if no contact).
    arm_bin_at_contact : int or None
        Reported arm bin when contact was first detected; None on failure.
    """
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(
        A, B, C, D,
        precision_manager=pm,
        policy_len=2, inference_horizon=2,
    )
    agent.reset()

    for t in range(max_steps):
        env_obs = env._get_obs()
        obs = [env_obs.visual_obs_idx, env_obs.tactile_obs_idx, env_obs.arm_pos_idx]

        result = agent.step(obs, c_visual=env.c_visual)
        action = int(result.action[0])
        step_result = env.step(action)
        if step_result.obs.tactile_obs_idx > 0:
            return t + 1, step_result.obs.arm_pos_idx

    return max_steps, None


# ---------------------------------------------------------------------------
# Calibration estimation
# ---------------------------------------------------------------------------

def estimate_shift_bins(
    calib_offset: float,
    pm: PrecisionManager,
    A_naive: np.ndarray,
    n_calib: int,
    seed_offset: int,
    max_steps: int = 20,
) -> tuple[int, list]:
    """
    Run n_calib episodes and estimate calibration shift in bins.

    Returns
    -------
    shift_bins : int
        Estimated shift, rounded to nearest integer.
    raw_offsets : list of int
        Per-contact raw bin offsets (for diagnostics).
    """
    env = SO101OcclusionEnv(
        occlusion_mode="full",
        n_arm_positions=N_POS,
        max_steps=max_steps + 5,
        calib_offset=calib_offset,
    )
    B = build_B(N_POS, N_OBJ)
    C = build_C(N_VIS, N_TAC, n_proprio=N_PROPRIO)
    D = build_D(N_POS, N_OBJ)

    raw_offsets = []
    for ep in range(n_calib):
        obj_loc = ep % N_OBJ
        _, arm_bin = run_episode(
            env, A_naive, B, C, D, pm,
            obj_loc_idx=obj_loc,
            seed=seed_offset + ep,
            max_steps=max_steps,
        )
        if arm_bin is not None:
            expected_bin = DEFAULT_OBJ_ARM[obj_loc]
            raw_offsets.append(arm_bin - expected_bin)

    if not raw_offsets:
        return 0, []
    shift = int(round(float(np.mean(raw_offsets))))
    return shift, raw_offsets


# ---------------------------------------------------------------------------
# Corrected A builder
# ---------------------------------------------------------------------------

def build_corrected_A(shift_bins: int) -> np.ndarray:
    corrected_obj_arm = {
        j: int(np.clip(DEFAULT_OBJ_ARM[j] + shift_bins, 0, N_POS - 1))
        for j in range(N_OBJ)
    }
    return build_A(
        N_POS, N_OBJ,
        p_contact=0.9, p_bg=0.05,
        obj_arm=corrected_obj_arm,
        with_proprio=True,
        proprio_accuracy=1.0,
    )


# ---------------------------------------------------------------------------
# Test phase
# ---------------------------------------------------------------------------

def run_test(
    calib_offset: float,
    A: np.ndarray,
    pm: PrecisionManager,
    n_test: int,
    seed_offset: int,
    max_steps: int = 20,
) -> list[int]:
    env = SO101OcclusionEnv(
        occlusion_mode="full",
        n_arm_positions=N_POS,
        max_steps=max_steps + 5,
        calib_offset=calib_offset,
    )
    B = build_B(N_POS, N_OBJ)
    C = build_C(N_VIS, N_TAC, n_proprio=N_PROPRIO)
    D = build_D(N_POS, N_OBJ)

    results = []
    for ep in range(n_test):
        steps, _ = run_episode(
            env, A, B, C, D, pm,
            obj_loc_idx=ep % N_OBJ,
            seed=seed_offset + ep,
            max_steps=max_steps,
        )
        results.append(steps)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-calib",   type=int,  default=20)
    parser.add_argument("--n-test",    type=int,  default=60)
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--max-steps", type=int,  default=20)
    parser.add_argument("--out",       type=Path, default=Path("results/step3_self_calib.png"))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    offsets = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])

    # CT-switching + proprio (条件 E)
    A_naive = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                      with_proprio=True, proprio_accuracy=1.0)
    pm = PrecisionManager(
        theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
        tactile_noise_floor=0.1, contact_triggered=True,
    )

    print(f"ステップ3: 自己キャリブレーション補正実験")
    print(f"  推定フェーズ: {args.n_calib} エピソード")
    print(f"  テストフェーズ: {args.n_test} エピソード")
    print(f"  bin幅 = {BIN_WIDTH:.2f} rad\n")

    naive_contacts   = []
    corrected_contacts = []
    shift_estimates  = []
    raw_offset_lists = []

    for offset in offsets:
        # フェーズ1: キャリブレーション推定
        shift, raw = estimate_shift_bins(
            calib_offset=offset,
            pm=pm,
            A_naive=A_naive,
            n_calib=args.n_calib,
            seed_offset=args.seed,
            max_steps=args.max_steps,
        )
        shift_estimates.append(shift)
        raw_offset_lists.append(raw)

        # フェーズ2a: 未補正
        naive_steps = run_test(
            calib_offset=offset,
            A=A_naive,
            pm=pm,
            n_test=args.n_test,
            seed_offset=args.seed + 1000,
            max_steps=args.max_steps,
        )
        n_naive = sum(s < args.max_steps for s in naive_steps)
        naive_contacts.append(n_naive)

        # フェーズ2b: 補正済み
        A_corrected = build_corrected_A(shift)
        corrected_steps = run_test(
            calib_offset=offset,
            A=A_corrected,
            pm=pm,
            n_test=args.n_test,
            seed_offset=args.seed + 1000,
            max_steps=args.max_steps,
        )
        n_corrected = sum(s < args.max_steps for s in corrected_steps)
        corrected_contacts.append(n_corrected)

        raw_str = f"[{', '.join(map(str, raw))}]" if raw else "[]"
        print(
            f"  offset={offset:.2f}rad ({offset/BIN_WIDTH:.2f}bin): "
            f"推定shift={shift:+d}bin  "
            f"未補正={n_naive}/{args.n_test}  "
            f"補正済み={n_corrected}/{args.n_test}  "
            f"raw={raw_str}"
        )

    print()
    print("  まとめ:")
    print(f"  {'offset':>8}  {'shift':>6}  {'未補正':>8}  {'補正済み':>8}  {'改善':>6}")
    print(f"  {'-'*48}")
    for i, offset in enumerate(offsets):
        improved = corrected_contacts[i] - naive_contacts[i]
        sign = "+" if improved > 0 else ""
        print(
            f"  {offset:+8.2f}  {shift_estimates[i]:+6d}  "
            f"{naive_contacts[i]:>4}/{args.n_test}  "
            f"{corrected_contacts[i]:>4}/{args.n_test}  "
            f"{sign}{improved:>4}"
        )

    # ── 図の作成 ──────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Step 3: Self-Calibration Correction\n"
        "(occlusion=full, CT-sw + proprio, "
        f"calib_phase={args.n_calib}ep, test={args.n_test}ep)",
        fontsize=12,
    )

    x = offsets / BIN_WIDTH
    naive_rates     = [c / args.n_test * 100 for c in naive_contacts]
    corrected_rates = [c / args.n_test * 100 for c in corrected_contacts]

    # Panel 1: 接触成功率
    ax = axes[0]
    ax.plot(x, naive_rates,     marker="o", color="#C0519A", linewidth=2, label="Naive (no correction)")
    ax.plot(x, corrected_rates, marker="s", color="#2196F3", linewidth=2, label="Self-calibrated")
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3)
    ax.axvline(x=0.75, color="gray",  linestyle=":", alpha=0.5, label="threshold 0.75bin")
    ax.set_xlabel("Calibration error [bin units]", fontsize=11)
    ax.set_ylabel("Contact success rate [%]", fontsize=11)
    ax.set_title("Contact Success Rate", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o/BIN_WIDTH:.2f}" for o in offsets])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: 推定シフト量
    ax = axes[1]
    ax.bar(x, shift_estimates, width=0.15, color="#FF9800", alpha=0.8, label="Estimated shift [bins]")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Calibration error [bin units]", fontsize=11)
    ax.set_ylabel("Estimated shift [bins]", fontsize=11)
    ax.set_title("Calibration Shift Estimate", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o/BIN_WIDTH:.2f}" for o in offsets])
    ax.set_yticks(range(-2, 3))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"\n  図を保存: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
