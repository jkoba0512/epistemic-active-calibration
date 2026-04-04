"""
マルチシード統計実験。

3つの実験を seed=0..4 の5シードで再実行し、
Wilson 信頼区間・Fisher 正確検定・Bonferroni 補正を計算する。

Usage
-----
    uv run python experiments/run_multiseed_stats.py
    uv run python experiments/run_multiseed_stats.py --episodes 60 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_calib_robustness.core.generative_model.model_builder import (
    build_A, build_B, build_C, build_D, DEFAULT_OBJ_ARM,
)
from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager
from aif_calib_robustness.simulation.so101_env import SO101OcclusionEnv

N_POS     = 5
N_OBJ     = 3
N_VIS     = N_OBJ + 1
N_TAC     = 2
N_PROPRIO = N_POS
BIN_WIDTH = 0.16

OFFSETS  = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
FP_RATES = [0, 5, 10, 20, 30]


# ── 共通エピソードランナー ───────────────────────────────────────────────────

def run_episode_sweep(env, A, B, C, D, pm, obj_loc_idx, seed,
                      max_steps=20, with_proprio=False):
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(A, B, C, D, precision_manager=pm,
                               policy_len=2, inference_horizon=2)
    agent.reset()
    for t in range(max_steps):
        obs_raw = env._get_obs()
        obs = [obs_raw.visual_obs_idx, obs_raw.tactile_obs_idx]
        if with_proprio:
            obs.append(obs_raw.arm_pos_idx)
        result = agent.step(obs, c_visual=env.c_visual)
        sr = env.step(int(result.action[0]))
        if sr.obs.tactile_obs_idx > 0:
            return t + 1
    return max_steps


def run_episode_ctswitch(env, A, B, C, D, pm, obj_loc_idx, seed,
                         fp_rate=0.0, with_proprio=False, max_steps=20):
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(A, B, C, D, precision_manager=pm,
                               policy_len=2, inference_horizon=2)
    agent.reset()
    rng = np.random.default_rng(seed + 99999)
    for t in range(max_steps):
        obs_raw = env._get_obs()
        tac = obs_raw.tactile_obs_idx
        if tac == 0 and rng.random() < fp_rate / 100.0:
            tac = 1
        obs = [obs_raw.visual_obs_idx, tac]
        if with_proprio:
            obs.append(obs_raw.arm_pos_idx)
        result = agent.step(obs, c_visual=env.c_visual)
        sr = env.step(int(result.action[0]))
        if sr.obs.tactile_obs_idx > 0:
            return t + 1
    return max_steps


def run_episode_calib(env, A, B, C, D, pm, obj_loc_idx, seed, max_steps=20):
    env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(A, B, C, D, precision_manager=pm,
                               policy_len=2, inference_horizon=2)
    agent.reset()
    for t in range(max_steps):
        obs_raw = env._get_obs()
        obs = [obs_raw.visual_obs_idx, obs_raw.tactile_obs_idx,
               obs_raw.arm_pos_idx]
        result = agent.step(obs, c_visual=env.c_visual)
        sr = env.step(int(result.action[0]))
        if sr.obs.tactile_obs_idx > 0:
            return t + 1, sr.obs.arm_pos_idx
    return max_steps, None


# ── 実験1: キャリブレーションオフセットスイープ ─────────────────────────────

def exp_sweep(n_episodes, seeds, max_steps=20):
    """条件 C/E/F/G × オフセット × シード"""
    A_no = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05)
    A_pr = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                   with_proprio=True, proprio_accuracy=1.0)
    pm_no = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.0)
    pm_ct = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.1,
                             contact_triggered=True)
    pm_sw = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.1)

    conditions = {
        "C": (A_no, pm_no, False),
        "E": (A_pr, pm_ct, True),
        "F": (A_pr, pm_sw, True),
        "G": (A_no, pm_ct, False),
    }

    # data[cond][offset_idx][seed_idx] = (n_contact, mean_steps)
    data = {c: [[None]*len(seeds) for _ in OFFSETS] for c in conditions}

    for ci, (cond, (A, pm, proprio)) in enumerate(conditions.items()):
        print(f"  Sweep condition {cond}...")
        n_proprio = N_PROPRIO if proprio else 0
        for oi, offset in enumerate(OFFSETS):
            env = SO101OcclusionEnv(occlusion_mode="full",
                                   n_arm_positions=N_POS,
                                   max_steps=max_steps + 5,
                                   calib_offset=offset)
            B = build_B(N_POS, N_OBJ)
            C_pref = build_C(N_VIS, N_TAC, n_proprio=n_proprio)
            D = build_D(N_POS, N_OBJ)
            for si, seed in enumerate(seeds):
                steps = [
                    run_episode_sweep(env, A, B, C_pref, D, pm,
                                      ep % N_OBJ, seed * 1000 + ep,
                                      max_steps, proprio)
                    for ep in range(n_episodes)
                ]
                n_c = sum(s < max_steps for s in steps)
                data[cond][oi][si] = (n_c, float(np.mean(steps)))
    return data


# ── 実験2: CT-switching 分解 ─────────────────────────────────────────────────

def exp_ctswitch(n_episodes, seeds, max_steps=20):
    """条件 C/E/F/G × fp_rate × シード"""
    A_no = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05)
    A_pr = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                   with_proprio=True, proprio_accuracy=1.0)
    pm_no = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.0)
    pm_ct = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.1,
                             contact_triggered=True)
    pm_sw = PrecisionManager(theta=0.4, pi_tactile_max=5.0,
                             pi_visual_min=0.1, tactile_noise_floor=0.1)

    conditions = {
        "C": (A_no, pm_no, False),
        "E": (A_pr, pm_ct, True),
        "F": (A_pr, pm_sw, True),
        "G": (A_no, pm_ct, False),
    }

    data = {c: [[None]*len(seeds) for _ in FP_RATES] for c in conditions}

    for cond, (A, pm, proprio) in conditions.items():
        print(f"  CT-switch condition {cond}...")
        n_proprio = N_PROPRIO if proprio else 0
        for fi, fp in enumerate(FP_RATES):
            env = SO101OcclusionEnv(occlusion_mode="full",
                                   n_arm_positions=N_POS,
                                   max_steps=max_steps + 5)
            B = build_B(N_POS, N_OBJ)
            C_pref = build_C(N_VIS, N_TAC, n_proprio=n_proprio)
            D = build_D(N_POS, N_OBJ)
            for si, seed in enumerate(seeds):
                steps = [
                    run_episode_ctswitch(env, A, B, C_pref, D, pm,
                                        ep % N_OBJ, seed * 1000 + ep,
                                        fp, proprio, max_steps)
                    for ep in range(n_episodes)
                ]
                n_c = sum(s < max_steps for s in steps)
                data[cond][fi][si] = (n_c, float(np.mean(steps)))
    return data


# ── 実験3: 自己キャリブレーション ───────────────────────────────────────────

def exp_selfcalib(n_calib, n_test, seeds, max_steps=20):
    """naive vs corrected × オフセット × シード"""
    A_naive = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                      with_proprio=True, proprio_accuracy=1.0)
    pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                          tactile_noise_floor=0.1, contact_triggered=True)

    data = {"naive": [[None]*len(seeds) for _ in OFFSETS],
            "corrected": [[None]*len(seeds) for _ in OFFSETS],
            "shift": [[None]*len(seeds) for _ in OFFSETS]}

    print("  Self-calib...")
    for oi, offset in enumerate(OFFSETS):
        for si, seed in enumerate(seeds):
            # Phase 1: estimate shift
            env_c = SO101OcclusionEnv(occlusion_mode="full",
                                      n_arm_positions=N_POS,
                                      max_steps=max_steps + 5,
                                      calib_offset=offset)
            B = build_B(N_POS, N_OBJ)
            C_pref = build_C(N_VIS, N_TAC, n_proprio=N_PROPRIO)
            D = build_D(N_POS, N_OBJ)
            raw_offsets = []
            for ep in range(n_calib):
                obj_loc = ep % N_OBJ
                _, arm_bin = run_episode_calib(
                    env_c, A_naive, B, C_pref, D, pm,
                    obj_loc, seed * 1000 + ep, max_steps)
                if arm_bin is not None:
                    raw_offsets.append(arm_bin - DEFAULT_OBJ_ARM[obj_loc])
            shift = int(round(float(np.mean(raw_offsets)))) if raw_offsets else 0

            # Phase 2a: naive test
            env_t = SO101OcclusionEnv(occlusion_mode="full",
                                      n_arm_positions=N_POS,
                                      max_steps=max_steps + 5,
                                      calib_offset=offset)
            naive_steps = [
                run_episode_calib(env_t, A_naive, B, C_pref, D, pm,
                                  ep % N_OBJ, seed * 1000 + 1000 + ep,
                                  max_steps)[0]
                for ep in range(n_test)
            ]
            n_naive = sum(s < max_steps for s in naive_steps)

            # Phase 2b: corrected test
            corr_obj_arm = {
                j: int(np.clip(DEFAULT_OBJ_ARM[j] + shift, 0, N_POS - 1))
                for j in range(N_OBJ)
            }
            A_corr = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                             obj_arm=corr_obj_arm,
                             with_proprio=True, proprio_accuracy=1.0)
            corr_steps = [
                run_episode_calib(env_t, A_corr, B, C_pref, D, pm,
                                  ep % N_OBJ, seed * 1000 + 1000 + ep,
                                  max_steps)[0]
                for ep in range(n_test)
            ]
            n_corr = sum(s < max_steps for s in corr_steps)

            data["naive"][oi][si]     = (n_naive, float(np.mean(naive_steps)))
            data["corrected"][oi][si] = (n_corr,  float(np.mean(corr_steps)))
            data["shift"][oi][si]     = shift
    return data


# ── 統計計算 ─────────────────────────────────────────────────────────────────

def wilson_ci(successes_list, n):
    """各シードの成功数リストから Wilson 95% CI を計算"""
    total_s = sum(successes_list)
    total_n = n * len(successes_list)
    lo, hi = proportion_confint(total_s, total_n, alpha=0.05, method="wilson")
    return total_s / total_n, lo, hi


def fisher_test_pair(s1_list, s2_list, n):
    """2条件の Fisher 正確検定（合計成功数 vs 合計失敗数）"""
    s1 = sum(s1_list)
    s2 = sum(s2_list)
    total_n = n * len(s1_list)
    table = [[s1, total_n - s1],
             [s2, total_n - s2]]
    _, p = fisher_exact(table)
    return float(p)


def compute_stats_sweep(data, n, seeds):
    """スイープ実験の統計"""
    stats = {}
    for cond in data:
        stats[cond] = []
        for oi in range(len(OFFSETS)):
            s_list = [data[cond][oi][si][0] for si in range(len(seeds))]
            rate, lo, hi = wilson_ci(s_list, n)
            mean_steps = np.mean([data[cond][oi][si][1] for si in range(len(seeds))])
            stats[cond].append({
                "offset": OFFSETS[oi],
                "successes_per_seed": s_list,
                "rate": rate, "ci_lo": lo, "ci_hi": hi,
                "mean_steps": float(mean_steps),
            })
    return stats


def compute_stats_ctswitch(data, n, seeds):
    """CT-switch 実験の統計"""
    stats = {}
    for cond in data:
        stats[cond] = []
        for fi in range(len(FP_RATES)):
            s_list = [data[cond][fi][si][0] for si in range(len(seeds))]
            rate, lo, hi = wilson_ci(s_list, n)
            stats[cond].append({
                "fp_rate": FP_RATES[fi],
                "successes_per_seed": s_list,
                "rate": rate, "ci_lo": lo, "ci_hi": hi,
            })
    return stats


def compute_stats_selfcalib(data, n_test, seeds):
    """自己キャリブレーション実験の統計"""
    stats = {"naive": [], "corrected": [], "shift": []}
    for oi in range(len(OFFSETS)):
        for key in ("naive", "corrected"):
            s_list = [data[key][oi][si][0] for si in range(len(seeds))]
            rate, lo, hi = wilson_ci(s_list, n_test)
            stats[key].append({
                "offset": OFFSETS[oi],
                "successes_per_seed": s_list,
                "rate": rate, "ci_lo": lo, "ci_hi": hi,
            })
        shifts = [data["shift"][oi][si] for si in range(len(seeds))]
        stats["shift"].append({"offset": OFFSETS[oi], "shifts": shifts,
                               "mean": float(np.mean(shifts))})
    return stats


def run_fisher_tests(sweep_stats, ct_stats, n, seeds):
    """主要な条件ペアの Fisher 検定"""
    n_total = n * len(seeds)
    tests = []

    # --- Sweep: E vs C at each offset ---
    print("\n  Fisher tests (sweep E vs C):")
    for oi, offset in enumerate(OFFSETS):
        s_e = sweep_stats["E"][oi]["successes_per_seed"]
        s_c = sweep_stats["C"][oi]["successes_per_seed"]
        p = fisher_test_pair(s_e, s_c, n)
        tests.append({"comparison": f"sweep E vs C, offset={offset:.2f}",
                      "p_raw": p})
        print(f"    offset={offset:.2f}: p={p:.4f}")

    # --- Sweep: F vs C at delta>=0.12 (proprioception effect) ---
    print("  Fisher tests (sweep F vs C, offset>=0.12):")
    for oi, offset in enumerate(OFFSETS):
        if offset < 0.12:
            continue
        s_f = sweep_stats["F"][oi]["successes_per_seed"]
        s_c = sweep_stats["C"][oi]["successes_per_seed"]
        p = fisher_test_pair(s_f, s_c, n)
        tests.append({"comparison": f"sweep F vs C, offset={offset:.2f}",
                      "p_raw": p})
        print(f"    offset={offset:.2f}: p={p:.4f}")

    # --- CT-switch: G vs C at each fp_rate ---
    print("  Fisher tests (CT-sw G vs C):")
    for fi, fp in enumerate(FP_RATES):
        s_g = ct_stats["G"][fi]["successes_per_seed"]
        s_c = ct_stats["C"][fi]["successes_per_seed"]
        p = fisher_test_pair(s_g, s_c, n)
        tests.append({"comparison": f"ct G vs C, fp={fp}%", "p_raw": p})
        print(f"    fp={fp}%: p={p:.4f}")

    # Bonferroni correction
    n_tests = len(tests)
    for t in tests:
        t["p_bonferroni"] = float(min(t["p_raw"] * n_tests, 1.0))
        t["significant"] = bool(t["p_bonferroni"] < 0.05)

    return tests


# ── メイン ───────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seeds",    type=int, default=5)
    parser.add_argument("--n-calib",  type=int, default=20)
    parser.add_argument("--max-steps",type=int, default=20)
    parser.add_argument("--out",      type=Path,
                        default=Path("results/multiseed_stats.json"))
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    print(f"マルチシード統計実験")
    print(f"  seeds={seeds}, episodes={args.episodes}")
    print()

    print("=== 実験1: キャリブレーションスイープ ===")
    sweep_raw  = exp_sweep(args.episodes, seeds, args.max_steps)
    sweep_stats = compute_stats_sweep(sweep_raw, args.episodes, seeds)

    print("\n=== 実験2: CT-switching 分解 ===")
    ct_raw   = exp_ctswitch(args.episodes, seeds, args.max_steps)
    ct_stats = compute_stats_ctswitch(ct_raw, args.episodes, seeds)

    print("\n=== 実験3: 自己キャリブレーション ===")
    calib_raw   = exp_selfcalib(args.n_calib, args.episodes, seeds,
                                args.max_steps)
    calib_stats = compute_stats_selfcalib(calib_raw, args.episodes, seeds)

    print("\n=== Fisher 正確検定 ===")
    fisher_results = run_fisher_tests(sweep_stats, ct_stats,
                                      args.episodes, seeds)

    # --- 結果のサマリー表示 ---
    print("\n=== スイープ結果 (rate [95% CI]) ===")
    for cond in ["C", "E", "F", "G"]:
        print(f"  {cond}:", end="")
        for s in sweep_stats[cond]:
            print(f"  δ={s['offset']:.2f}: "
                  f"{s['rate']*100:.1f}% [{s['ci_lo']*100:.1f},{s['ci_hi']*100:.1f}]",
                  end="")
        print()

    print("\n=== CT-switch 結果 (rate [95% CI]) ===")
    for cond in ["C", "E", "F", "G"]:
        print(f"  {cond}:", end="")
        for s in ct_stats[cond]:
            print(f"  fp={s['fp_rate']}%: "
                  f"{s['rate']*100:.1f}% [{s['ci_lo']*100:.1f},{s['ci_hi']*100:.1f}]",
                  end="")
        print()

    print("\n=== 自己キャリブレーション結果 ===")
    for key in ["naive", "corrected"]:
        print(f"  {key}:", end="")
        for s in calib_stats[key]:
            print(f"  δ={s['offset']:.2f}: "
                  f"{s['rate']*100:.1f}% [{s['ci_lo']*100:.1f},{s['ci_hi']*100:.1f}]",
                  end="")
        print()

    print("\n=== Bonferroni 補正後 有意な比較 ===")
    sig = [t for t in fisher_results if t["significant"]]
    if sig:
        for t in sig:
            print(f"  {t['comparison']}: p_raw={t['p_raw']:.4f}, "
                  f"p_bonf={t['p_bonferroni']:.4f} *")
    else:
        print("  なし（全て有意でない）")

    # JSON 保存
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {"seeds": seeds, "episodes": args.episodes,
                   "n_calib": args.n_calib},
        "sweep":  sweep_stats,
        "ctswitch": ct_stats,
        "selfcalib": calib_stats,
        "fisher": fisher_results,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n結果を保存: {args.out}")


if __name__ == "__main__":
    main()
