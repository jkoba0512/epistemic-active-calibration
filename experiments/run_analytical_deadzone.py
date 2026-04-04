"""
デッドゾーン原因切り分け実験。

MuJoCo 物理エンジンを使わず、解析的な接触判定（arm の離散ビンが
オブジェクトの接触ビンと一致した瞬間に接触）を用いた純粋な離散シミュレーション
で自己キャリブレーションを実行し、δ=0.12 のデッドゾーンが消えることを確認する。

結論:
  - MuJoCo: グリッパーの物理的な厚みにより、接触が公称位置 (-0.200 rad) より
    左で発生することがある (-0.237 rad)。このため報告ビンが 1 か 2 かが
    エピソードごとにばらつき、推定シフトの平均が 0.5 → round(0.5) = 0 となる。
  - 解析的: 接触は常に公称ビン位置 (-0.20 rad → bin1 → reported bin2 with δ=0.12)
    で発生するため推定シフト = +1 で一致。デッドゾーン消失。

Usage
-----
    uv run python experiments/run_analytical_deadzone.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_calib_robustness.core.generative_model.model_builder import (
    build_A, build_B, build_C, build_D, DEFAULT_OBJ_ARM,
)
from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager
from aif_calib_robustness.utils.discretizer import JointDiscretizer

# ── 定数（SO101OcclusionEnv と同一） ──────────────────────────────────────────
N_POS       = 5
N_OBJ       = 3
N_VIS       = N_OBJ + 1   # ambiguous + 3 locations
N_TAC       = 2
N_PROPRIO   = N_POS
ANGLE_MIN   = -0.40
ANGLE_MAX   =  0.40
STEP        =  0.20        # decode_delta の刻み幅 (range/(n_bins-1) = 0.80/4)
OFFSETS     = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
N_CALIB     = 20
N_TEST      = 60
SEEDS       = [0, 1, 2, 3, 4]

_joint_disc = JointDiscretizer(n_bins=N_POS,
                                angle_min=ANGLE_MIN,
                                angle_max=ANGLE_MAX)

VIS_AMBIGUOUS = N_OBJ   # 視覚が全閉塞のときの index


# ── 解析的環境 ────────────────────────────────────────────────────────────────

@dataclass
class AnalyticalObs:
    arm_pos_idx:     int
    tactile_obs_idx: int
    visual_obs_idx:  int


class AnalyticalEnv:
    """
    MuJoCo を使わない純粋な離散環境。
    接触判定: encode(arm_angle) == DEFAULT_OBJ_ARM[obj_loc_idx]
    固有感覚: encode(clip(arm_angle + calib_offset))
    """

    def __init__(self, calib_offset: float = 0.0):
        self.calib_offset = calib_offset
        self._arm_angle:    float = ANGLE_MIN
        self._obj_loc_idx:  int   = 0
        self._step_count:   int   = 0

    def reset(self, obj_loc_idx: int = 0, seed: Optional[int] = None) -> AnalyticalObs:
        self._arm_angle   = ANGLE_MIN
        self._obj_loc_idx = obj_loc_idx
        self._step_count  = 0
        return self._get_obs()

    def step(self, action: int):
        delta = {0: -STEP, 1: 0.0, 2: +STEP}[action]
        self._arm_angle = float(np.clip(
            self._arm_angle + delta, ANGLE_MIN, ANGLE_MAX))
        self._step_count += 1
        return self._get_obs()

    def _get_obs(self) -> AnalyticalObs:
        # 接触判定: 実際のビンがオブジェクトの接触ビンと一致
        actual_bin  = _joint_disc.encode(self._arm_angle)
        contact_bin = DEFAULT_OBJ_ARM[self._obj_loc_idx]
        tactile     = 1 if actual_bin == contact_bin else 0

        # 固有感覚: キャリブレーションオフセット込みの報告値
        reported_angle = float(np.clip(
            self._arm_angle + self.calib_offset, ANGLE_MIN, ANGLE_MAX))
        arm_pos_idx = _joint_disc.encode(reported_angle)

        return AnalyticalObs(
            arm_pos_idx=arm_pos_idx,
            tactile_obs_idx=tactile,
            visual_obs_idx=VIS_AMBIGUOUS,
        )

    @property
    def c_visual(self) -> float:
        return 0.0   # 全閉塞


# ── エピソードランナー ────────────────────────────────────────────────────────

def run_episode(env, A, B, C_pref, D, pm,
                obj_loc_idx: int, seed: int,
                max_steps: int = 20):
    """
    1エピソードを実行し (steps, reported_contact_arm_bin) を返す。
    接触なし終了の場合は (max_steps, None)。
    """
    obs = env.reset(obj_loc_idx=obj_loc_idx, seed=seed)
    agent = MultiModalAIFAgent(A, B, C_pref, D, precision_manager=pm,
                               policy_len=2, inference_horizon=2)
    agent.reset()

    for t in range(max_steps):
        obs_list = [obs.visual_obs_idx, obs.tactile_obs_idx, obs.arm_pos_idx]
        result   = agent.step(obs_list, c_visual=env.c_visual)
        obs      = env.step(int(result.action[0]))
        if obs.tactile_obs_idx > 0:
            return t + 1, obs.arm_pos_idx

    return max_steps, None


# ── 実験: キャリブレーション ─────────────────────────────────────────────────

def run_selfcalib_at_offset(offset: float, n_calib: int, n_test: int,
                             seed: int) -> dict:
    """
    1シード × 1オフセットで自己キャリブレーション実験を実行。
    Returns dict with naive_success, corrected_success, shift, raw_offsets.
    """
    A_naive = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                      with_proprio=True, proprio_accuracy=1.0)
    pm = PrecisionManager(theta=0.4, pi_tactile_max=5.0, pi_visual_min=0.1,
                          tactile_noise_floor=0.1, contact_triggered=True)

    # ── Phase 1: シフト推定 ────────────────────────────────────────────────
    env_c = AnalyticalEnv(calib_offset=offset)
    B      = build_B(N_POS, N_OBJ)
    C_pref = build_C(N_VIS, N_TAC, n_proprio=N_PROPRIO)
    D      = build_D(N_POS, N_OBJ)

    raw_offsets = []
    for ep in range(n_calib):
        obj_loc = ep % N_OBJ
        _, arm_bin = run_episode(env_c, A_naive, B, C_pref, D, pm,
                                  obj_loc, seed * 1000 + ep)
        if arm_bin is not None:
            raw_offsets.append(arm_bin - DEFAULT_OBJ_ARM[obj_loc])

    shift = int(round(float(np.mean(raw_offsets)))) if raw_offsets else 0

    # ── Phase 2a: naive test ──────────────────────────────────────────────
    env_t = AnalyticalEnv(calib_offset=offset)
    naive_steps = [
        run_episode(env_t, A_naive, B, C_pref, D, pm,
                    ep % N_OBJ, seed * 1000 + 1000 + ep)[0]
        for ep in range(n_test)
    ]
    n_naive = sum(s < 20 for s in naive_steps)

    # ── Phase 2b: corrected test ──────────────────────────────────────────
    corr_obj_arm = {
        j: int(np.clip(DEFAULT_OBJ_ARM[j] + shift, 0, N_POS - 1))
        for j in range(N_OBJ)
    }
    A_corr = build_A(N_POS, N_OBJ, p_contact=0.9, p_bg=0.05,
                     obj_arm=corr_obj_arm,
                     with_proprio=True, proprio_accuracy=1.0)
    corr_steps = [
        run_episode(env_t, A_corr, B, C_pref, D, pm,
                    ep % N_OBJ, seed * 1000 + 1000 + ep)[0]
        for ep in range(n_test)
    ]
    n_corr = sum(s < 20 for s in corr_steps)

    return {
        "naive_success":     n_naive,
        "corrected_success": n_corr,
        "shift":             shift,
        "raw_offsets":       raw_offsets,
    }


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("解析的環境によるデッドゾーン原因切り分け実験")
    print(f"  seeds={SEEDS}, n_calib={N_CALIB}, n_test={N_TEST}")
    print()

    results = {}
    for offset in OFFSETS:
        seed_results = [
            run_selfcalib_at_offset(offset, N_CALIB, N_TEST, seed)
            for seed in SEEDS
        ]
        shifts        = [r["shift"]             for r in seed_results]
        naive_succs   = [r["naive_success"]      for r in seed_results]
        corr_succs    = [r["corrected_success"]  for r in seed_results]
        naive_rate    = sum(naive_succs) / (N_TEST * len(SEEDS)) * 100
        corr_rate     = sum(corr_succs)  / (N_TEST * len(SEEDS)) * 100

        results[f"{offset:.2f}"] = {
            "shifts":       shifts,
            "naive_rate":   naive_rate,
            "corr_rate":    corr_rate,
        }

        print(f"δ={offset:.2f}  shifts={shifts}"
              f"  naive={naive_rate:.1f}%  corrected={corr_rate:.1f}%")

    print()
    print("── まとめ ──")
    print("δ=0.12 のシフト推定:")
    r12 = results["0.12"]
    print(f"  解析的環境: {r12['shifts']}  (一致: {len(set(r12['shifts']))==1})")
    print(f"  → naive={r12['naive_rate']:.1f}%  corrected={r12['corr_rate']:.1f}%")
    print()
    print("解析的環境でデッドゾーンが消えれば、原因は MuJoCo の物理精度（グリッパー厚み）。")
    print("消えなければ、アルゴリズム自体の数学的欠陥。")


if __name__ == "__main__":
    main()
