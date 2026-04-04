"""
2-DoF キャリブレーションロバスト性格子実験。

目的
----
独立関節定理を実験的に検証する:

  定理1（独立関節）:
    各関節の遷移行列がテンソル積構造 B = B^(1) ⊗ B^(2) を持つとき、
    ロバスト性領域は直積構造になる:
      R = { (δ1, δ2) : δ1 < δ_c^(1)  かつ  δ2 < δ_c^(2) }
    ここで δ_c^(k) = range_k * (n_bins_k - 2) / (n_bins_k * (n_bins_k - 1))

  Skip Deadlock 仮説:
    δ1 > δ_c^(1) かつ δ2 > δ_c^(2) のとき、
    2-DoF 軌跡が接触タプル (c1*, c2*) を「同時に」訪問しない可能性がある。

実装
----
  実験1: 解析的ビン被覆 (fast)
    各 (δ1, δ2) で訪問ビン集合を計算し、テンソル積予測を出力。

  実験2: AIF エージェント (pymdp, combined-pose 状態因子)
    実際のエピソード成功率を計測してテンソル積予測と比較。
    combined-pose: pose = j1_bin * N_POS + j2_bin
    5 actions: j1_left(0), j1_right(1), j2_left(2), j2_right(3), stay(4)

接触条件
--------
  オブジェクト k は pose (j1_bin = k+1, j2_bin = k+1) で接触。
  両関節が正しい位置に同時にある場合のみ接触 → 2-DoF skip deadlock を再現。

Usage
-----
    uv run python experiments/run_2dof_calibration_grid.py
    uv run python experiments/run_2dof_calibration_grid.py --analytical-only
    uv run python experiments/run_2dof_calibration_grid.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_calib_robustness.core.generative_model.multimodal_agent import MultiModalAIFAgent
from aif_calib_robustness.core.precision.precision_manager import PrecisionManager
from aif_calib_robustness.utils.discretizer import JointDiscretizer
from pymdp.legacy import utils

# ── 定数 ──────────────────────────────────────────────────────────────────────

N_POS     = 5        # 各関節の離散ビン数
N_OBJ     = 3        # オブジェクト数
ANGLE_MIN = -0.40    # rad
ANGLE_MAX =  0.40    # rad
STEP      = (ANGLE_MAX - ANGLE_MIN) / (N_POS - 1)   # = 0.20 rad

# 1-DoF 理論閾値 (各関節に同一パラメータを使用)
# δ_c = range*(n_bins-2) / (n_bins*(n_bins-1)) = 0.80*3/20 = 0.12 rad
DELTA_C = (ANGLE_MAX - ANGLE_MIN) * (N_POS - 2) / (N_POS * (N_POS - 1))

# 2-DoF 接触条件: obj k → (j1_bin=k+1, j2_bin=k+1)
# 対角配置: 各オブジェクトが j1 と j2 の同じビンを要求する
CONTACT_BINS: dict[int, tuple[int, int]] = {0: (1, 1), 1: (2, 2), 2: (3, 3)}

# combined-pose: pose = j1_bin * N_POS + j2_bin
N_POSES = N_POS * N_POS   # = 25

# N_PROPRIO も N_POS (各関節の固有感覚ビン数)
N_VIS = N_OBJ + 1   # = 4 (0..N_OBJ-1 は位置, N_OBJ は ambiguous)

# 実験グリッド
DELTA_VALUES = np.linspace(-0.20, 0.20, 11)   # 11 × 11 = 121 点
N_SEEDS      = 3
N_EPISODES   = 30    # per seed per (δ1, δ2) point
MAX_STEPS    = 30    # per episode

_joint_disc = JointDiscretizer(n_bins=N_POS, angle_min=ANGLE_MIN, angle_max=ANGLE_MAX)


# ── ユーティリティ ──────────────────────────────────────────────────────────

def pose_to_bins(pose: int) -> tuple[int, int]:
    """combined-pose インデックス → (j1_bin, j2_bin)"""
    return (pose // N_POS, pose % N_POS)


def bins_to_pose(j1: int, j2: int) -> int:
    """(j1_bin, j2_bin) → combined-pose インデックス"""
    return j1 * N_POS + j2


# ── 実験1: 解析的ビン被覆 ──────────────────────────────────────────────────

def visited_bins_1dof(delta: float) -> set[int]:
    """
    1-DoF スウィープ (左→右) で δ オフセット付きのとき訪問されるビン集合。
    スウィープは N_POS ステップ (ビン中心を巡回)。
    """
    visited = set()
    for i in range(N_POS):
        angle = ANGLE_MIN + i * STEP
        reported = float(np.clip(angle + delta, ANGLE_MIN, ANGLE_MAX))
        visited.add(_joint_disc.encode(reported))
    return visited


def analytical_contact_map(delta1: float, delta2: float) -> dict[int, bool]:
    """
    テンソル積予測: 各 obj で接触タプルが訪問されるか否かを返す。

    独立関節定理 (Theorem 1):
      B = B^(1) ⊗ B^(2) ならば
      (c1*, c2*) が訪問される ⟺ c1* ∈ visited1(δ1) AND c2* ∈ visited2(δ2)
    """
    v1 = visited_bins_1dof(delta1)
    v2 = visited_bins_1dof(delta2)
    return {obj: (c1 in v1) and (c2 in v2) for obj, (c1, c2) in CONTACT_BINS.items()}


def analytical_success_rate(delta1: float, delta2: float) -> float:
    """テンソル積予測の期待成功率 (全 obj 均等)。"""
    return np.mean(list(analytical_contact_map(delta1, delta2).values()))


# ── 実験2: 2-DoF AIF エージェント ────────────────────────────────────────────

def build_A_2dof(p_contact: float = 0.9, p_bg: float = 0.05) -> np.ndarray:
    """
    4-modality A matrices for 2-DoF combined-pose model.

    A[0]: visual    (N_VIS, N_POSES, N_OBJ)   -- 全閉塞 → ambiguous
    A[1]: tactile   (2,     N_POSES, N_OBJ)   -- 接触条件
    A[2]: proprio_j1 (N_POS, N_POSES, N_OBJ)  -- j1 成分の固有感覚 (キャリブ誤差なしモデル)
    A[3]: proprio_j2 (N_POS, N_POSES, N_OBJ)  -- j2 成分の固有感覚
    """
    # Visual: 完全な可視性を仮定 (PrecisionManager が全閉塞時に uniform 化)
    A_visual = np.zeros((N_VIS, N_POSES, N_OBJ))
    for obj in range(N_OBJ):
        A_visual[obj, :, obj] = 1.0   # 観測 = obj_loc なら p=1

    # Tactile
    A_tactile = np.zeros((2, N_POSES, N_OBJ))
    for pose in range(N_POSES):
        j1, j2 = pose_to_bins(pose)
        for obj in range(N_OBJ):
            c1, c2 = CONTACT_BINS[obj]
            if j1 == c1 and j2 == c2:
                A_tactile[1, pose, obj] = p_contact
                A_tactile[0, pose, obj] = 1.0 - p_contact
            else:
                A_tactile[1, pose, obj] = p_bg
                A_tactile[0, pose, obj] = 1.0 - p_bg

    # Proprio j1: A_j1[obs_j1, pose, obj] = 1 iff obs_j1 == j1-component(pose)
    A_proprio_j1 = np.zeros((N_POS, N_POSES, N_OBJ))
    for pose in range(N_POSES):
        j1, _ = pose_to_bins(pose)
        A_proprio_j1[j1, pose, :] = 1.0

    # Proprio j2: A_j2[obs_j2, pose, obj] = 1 iff obs_j2 == j2-component(pose)
    A_proprio_j2 = np.zeros((N_POS, N_POSES, N_OBJ))
    for pose in range(N_POSES):
        _, j2 = pose_to_bins(pose)
        A_proprio_j2[j2, pose, :] = 1.0

    A = utils.obj_array(4)
    A[0] = A_visual
    A[1] = A_tactile
    A[2] = A_proprio_j1
    A[3] = A_proprio_j2
    return A


def build_B_2dof() -> np.ndarray:
    """
    B matrices for 2-DoF combined-pose model (5 actions, テンソル積構造).

    Actions:
        0: j1 left   (j1-1, j2)
        1: j1 right  (j1+1, j2)
        2: j2 left   (j1, j2-1)
        3: j2 right  (j1, j2+1)
        4: stay

    B^(pose) が B^(j1) ⊗ B^(j2) と同一であることは、各アクションが
    一方の関節のみを動かすことから直接確認できる (独立関節定理の前提)。
    """
    n_actions = 5
    B_pose = np.zeros((N_POSES, N_POSES, n_actions))

    for j1 in range(N_POS):
        for j2 in range(N_POS):
            p = bins_to_pose(j1, j2)
            B_pose[bins_to_pose(max(0, j1 - 1), j2),        p, 0] = 1.0  # j1 left
            B_pose[bins_to_pose(min(N_POS-1, j1 + 1), j2),  p, 1] = 1.0  # j1 right
            B_pose[bins_to_pose(j1, max(0, j2 - 1)),        p, 2] = 1.0  # j2 left
            B_pose[bins_to_pose(j1, min(N_POS-1, j2 + 1)),  p, 3] = 1.0  # j2 right
            B_pose[p,                                         p, 4] = 1.0  # stay

    B_obj = np.zeros((N_OBJ, N_OBJ, n_actions))
    for a in range(n_actions):
        np.fill_diagonal(B_obj[:, :, a], 1.0)

    B = utils.obj_array(2)
    B[0] = B_pose
    B[1] = B_obj
    return B


def build_C_2dof() -> np.ndarray:
    C = utils.obj_array(4)
    C[0] = np.zeros(N_VIS)
    C[1] = np.array([-3.0, 3.0])
    C[2] = np.zeros(N_POS)
    C[3] = np.zeros(N_POS)
    return C


def build_D_2dof() -> np.ndarray:
    D = utils.obj_array(2)
    D[0] = np.ones(N_POSES) / N_POSES
    D[1] = np.ones(N_OBJ) / N_OBJ
    return D


# ── 2-DoF 解析的環境 ──────────────────────────────────────────────────────────

@dataclass
class Obs2DoF:
    """2-DoF エピソードの1ステップ観測。"""
    j1_actual:       int   # 実際の j1 ビン (環境内部)
    j2_actual:       int   # 実際の j2 ビン
    tactile_obs:     int   # 0=no-contact, 1=contact
    visual_obs:      int   # 常に N_OBJ (全閉塞)
    proprio_j1_obs:  int   # キャリブ誤差 δ1 込みの報告値
    proprio_j2_obs:  int   # キャリブ誤差 δ2 込みの報告値


class AnalyticalEnv2DoF:
    """
    2-DoF 純離散環境 (MuJoCo なし)。

    接触判定: 実際の (j1_bin, j2_bin) == CONTACT_BINS[obj_loc] のとき接触。
    固有感覚: encode(clip(j_angle + delta_k)) でバイアスされた観測を返す。
    """

    def __init__(self, delta1: float = 0.0, delta2: float = 0.0):
        self.delta1 = delta1
        self.delta2 = delta2
        self._j1:      int = 0
        self._j2:      int = 0
        self._obj_loc: int = 0
        self._steps:   int = 0

    def reset(self, obj_loc: int = 0) -> Obs2DoF:
        self._j1      = 0
        self._j2      = 0
        self._obj_loc = obj_loc
        self._steps   = 0
        return self._get_obs()

    def step(self, action: int) -> Obs2DoF:
        """action: 0=j1-, 1=j1+, 2=j2-, 3=j2+, 4=stay"""
        if action == 0:
            self._j1 = max(0, self._j1 - 1)
        elif action == 1:
            self._j1 = min(N_POS - 1, self._j1 + 1)
        elif action == 2:
            self._j2 = max(0, self._j2 - 1)
        elif action == 3:
            self._j2 = min(N_POS - 1, self._j2 + 1)
        # action == 4: stay
        self._steps += 1
        return self._get_obs()

    def _get_obs(self) -> Obs2DoF:
        c1, c2 = CONTACT_BINS[self._obj_loc]
        tactile = 1 if (self._j1 == c1 and self._j2 == c2) else 0

        j1_angle = ANGLE_MIN + self._j1 * STEP
        j2_angle = ANGLE_MIN + self._j2 * STEP
        rep_j1 = float(np.clip(j1_angle + self.delta1, ANGLE_MIN, ANGLE_MAX))
        rep_j2 = float(np.clip(j2_angle + self.delta2, ANGLE_MIN, ANGLE_MAX))

        return Obs2DoF(
            j1_actual=self._j1,
            j2_actual=self._j2,
            tactile_obs=tactile,
            visual_obs=N_OBJ,   # 全閉塞: 常に ambiguous
            proprio_j1_obs=_joint_disc.encode(rep_j1),
            proprio_j2_obs=_joint_disc.encode(rep_j2),
        )

    @property
    def obj_loc(self) -> int:
        return self._obj_loc


# ── エピソードランナー ────────────────────────────────────────────────────────

def run_episode_2dof(
    env: AnalyticalEnv2DoF,
    A, B, C, D,
    obj_loc: int,
    max_steps: int = MAX_STEPS,
) -> bool:
    """1エピソード実行。接触成功なら True。"""
    obs = env.reset(obj_loc=obj_loc)

    pm = PrecisionManager(
        theta=0.4,
        pi_tactile_max=5.0,
        pi_visual_min=0.1,
        visual_modality_idx=0,
        tactile_modality_idx=1,
        tactile_noise_floor=0.0,
        contact_triggered=False,
    )
    agent = MultiModalAIFAgent(A, B, C, D, precision_manager=pm,
                               policy_len=1, inference_horizon=1)
    agent.reset()

    for _ in range(max_steps):
        obs_list = [
            obs.visual_obs,
            obs.tactile_obs,
            obs.proprio_j1_obs,
            obs.proprio_j2_obs,
        ]
        result = agent.step(obs_list, c_visual=0.0)   # 全閉塞
        obs    = env.step(int(result.action[0]))
        if obs.tactile_obs > 0:
            return True

    return False


def run_grid_point(delta1: float, delta2: float,
                   A, B, C, D,
                   n_seeds: int, n_episodes: int,
                   rng: np.random.Generator) -> dict:
    """
    (δ1, δ2) の1格子点で複数シード・エピソードを実行。

    Returns
    -------
    dict with keys:
        aif_success_rate : float  実際の成功率
        tp_success_rate  : float  テンソル積予測の成功率
        aif_successes    : list[int]  シードごとの成功数
    """
    env = AnalyticalEnv2DoF(delta1=delta1, delta2=delta2)
    tp_rate = analytical_success_rate(delta1, delta2)

    aif_successes = []
    for seed in range(n_seeds):
        successes = 0
        for ep in range(n_episodes):
            obj_loc = ep % N_OBJ
            success = run_episode_2dof(env, A, B, C, D, obj_loc,
                                        max_steps=MAX_STEPS)
            if success:
                successes += 1
        aif_successes.append(successes)

    total = n_seeds * n_episodes
    aif_rate = sum(aif_successes) / total

    return {
        "aif_success_rate": aif_rate,
        "tp_success_rate":  tp_rate,
        "aif_successes":    aif_successes,
    }


# ── メイン ────────────────────────────────────────────────────────────────────

def main(analytical_only: bool = False, no_plot: bool = False) -> None:
    print("2-DoF キャリブレーションロバスト性格子実験")
    print(f"  N_POS={N_POS}, N_OBJ={N_OBJ}, δ_c(理論値)={DELTA_C:.4f} rad")
    print(f"  接触条件: {CONTACT_BINS}")
    print()

    # ── 実験1: 解析的ビン被覆 ─────────────────────────────────────────────

    print("=== 実験1: 解析的テンソル積予測 ===")
    print(f"{'δ1':>7} {'δ2':>7} | {'obj0':>6} {'obj1':>6} {'obj2':>6} | {'success':>8}")
    print("-" * 55)
    for d1 in DELTA_VALUES[::2]:   # 間引きして表示
        for d2 in DELTA_VALUES[::2]:
            cmap = analytical_contact_map(d1, d2)
            rate = np.mean(list(cmap.values()))
            print(f"{d1:+.3f} {d2:+.3f} | "
                  f"{'Y' if cmap[0] else 'N':>6} "
                  f"{'Y' if cmap[1] else 'N':>6} "
                  f"{'Y' if cmap[2] else 'N':>6} | "
                  f"{rate:.3f}")

    print()
    print(f"  → δ_c={DELTA_C:.4f} rad: 独立閾値境界を破線で可視化予定")
    print()

    if analytical_only:
        print("[--analytical-only] AIF エージェント実験をスキップ。")
        return

    # ── 実験2: AIF エージェント格子実験 ──────────────────────────────────

    print("=== 実験2: AIF エージェント格子実験 ===")
    print(f"  グリッド: {len(DELTA_VALUES)}×{len(DELTA_VALUES)}={len(DELTA_VALUES)**2} 点")
    print(f"  シード×エピソード: {N_SEEDS}×{N_EPISODES}={N_SEEDS*N_EPISODES} per 格子点")
    total_pts = len(DELTA_VALUES) ** 2
    print(f"  総エピソード: {total_pts * N_SEEDS * N_EPISODES}")
    print()

    A = build_A_2dof()
    B = build_B_2dof()
    C = build_C_2dof()
    D = build_D_2dof()
    rng = np.random.default_rng(42)

    grid_results = {}
    completed = 0

    for i, d1 in enumerate(DELTA_VALUES):
        for j, d2 in enumerate(DELTA_VALUES):
            result = run_grid_point(d1, d2, A, B, C, D,
                                     N_SEEDS, N_EPISODES, rng)
            key = f"{d1:.3f}_{d2:.3f}"
            grid_results[key] = {
                "delta1":           float(d1),
                "delta2":           float(d2),
                "aif_success_rate": result["aif_success_rate"],
                "tp_success_rate":  result["tp_success_rate"],
                "aif_successes":    result["aif_successes"],
            }
            completed += 1
            if completed % 11 == 0 or completed == 1:
                print(f"  [{completed:3d}/{total_pts}] δ1={d1:+.2f} δ2={d2:+.2f} "
                      f"AIF={result['aif_success_rate']:.2f} "
                      f"TP={result['tp_success_rate']:.2f}")

    print()

    # ── 結果保存 ─────────────────────────────────────────────────────────

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "2dof_calibration_grid.json"

    summary = {
        "parameters": {
            "N_POS":       N_POS,
            "N_OBJ":       N_OBJ,
            "ANGLE_MIN":   ANGLE_MIN,
            "ANGLE_MAX":   ANGLE_MAX,
            "DELTA_C":     float(DELTA_C),
            "CONTACT_BINS": {str(k): list(v) for k, v in CONTACT_BINS.items()},
            "N_SEEDS":     N_SEEDS,
            "N_EPISODES":  N_EPISODES,
            "MAX_STEPS":   MAX_STEPS,
        },
        "grid": grid_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"結果を保存: {out_path}")

    # ── 一致率の計算 ─────────────────────────────────────────────────────

    n_match = 0
    n_total = 0
    for v in grid_results.values():
        tp_success = v["tp_success_rate"] > 0.5
        aif_success = v["aif_success_rate"] > 0.5
        if tp_success == aif_success:
            n_match += 1
        n_total += 1
    print(f"テンソル積予測 vs AIF 一致率: {n_match}/{n_total} = {n_match/n_total:.1%}")
    print()

    # ── 可視化 ───────────────────────────────────────────────────────────

    if no_plot:
        print("[--no-plot] 図の生成をスキップ。")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        D1, D2 = np.meshgrid(DELTA_VALUES, DELTA_VALUES, indexing="ij")
        AIF_RATE = np.zeros_like(D1)
        TP_RATE  = np.zeros_like(D1)

        for i, d1 in enumerate(DELTA_VALUES):
            for j, d2 in enumerate(DELTA_VALUES):
                key = f"{d1:.3f}_{d2:.3f}"
                AIF_RATE[i, j] = grid_results[key]["aif_success_rate"]
                TP_RATE[i, j]  = grid_results[key]["tp_success_rate"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 左: AIF 成功率ヒートマップ
        ax = axes[0]
        im = ax.contourf(D1, D2, AIF_RATE, levels=np.linspace(0, 1, 21), cmap="RdYlGn")
        ax.axvline(x=DELTA_C,  color="blue",   lw=2, ls="--", label=f"δ_c={DELTA_C:.3f}")
        ax.axhline(y=DELTA_C,  color="blue",   lw=2, ls="--")
        ax.axvline(x=-DELTA_C, color="gray",   lw=1, ls=":")
        ax.axhline(y=-DELTA_C, color="gray",   lw=1, ls=":")
        ax.axvline(x=0,        color="black",  lw=0.5, ls="-")
        ax.axhline(y=0,        color="black",  lw=0.5, ls="-")
        fig.colorbar(im, ax=ax, label="success rate")
        ax.set_xlabel("δ1 (rad)")
        ax.set_ylabel("δ2 (rad)")
        ax.set_title("AIF Agent Success Rate")
        ax.legend(fontsize=8)

        # 中: テンソル積予測
        ax = axes[1]
        im2 = ax.contourf(D1, D2, TP_RATE, levels=np.linspace(0, 1, 21), cmap="RdYlGn")
        ax.axvline(x=DELTA_C,  color="blue",   lw=2, ls="--")
        ax.axhline(y=DELTA_C,  color="blue",   lw=2, ls="--")
        ax.axvline(x=0,        color="black",  lw=0.5, ls="-")
        ax.axhline(y=0,        color="black",  lw=0.5, ls="-")
        fig.colorbar(im2, ax=ax, label="TP predicted rate")
        ax.set_xlabel("δ1 (rad)")
        ax.set_ylabel("δ2 (rad)")
        ax.set_title("Tensor Product Prediction")

        # 右: 差分 (AIF - TP)
        ax = axes[2]
        diff = AIF_RATE - TP_RATE
        lim = max(abs(diff.min()), abs(diff.max()), 0.1)
        im3 = ax.contourf(D1, D2, diff, levels=np.linspace(-lim, lim, 21), cmap="bwr")
        ax.axvline(x=DELTA_C, color="blue", lw=2, ls="--", label=f"δ_c={DELTA_C:.3f}")
        ax.axhline(y=DELTA_C, color="blue", lw=2, ls="--")
        ax.axvline(x=0,       color="black", lw=0.5, ls="-")
        ax.axhline(y=0,       color="black", lw=0.5, ls="-")
        fig.colorbar(im3, ax=ax, label="AIF − TP")
        ax.set_xlabel("δ1 (rad)")
        ax.set_ylabel("δ2 (rad)")
        ax.set_title("Discrepancy (AIF − Tensor Product)")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"2-DoF Calibration Robustness Grid\n"
            f"N_bins={N_POS}, δ_c={DELTA_C:.4f} rad "
            f"({N_SEEDS} seeds × {N_EPISODES} eps/seed)",
            fontsize=11,
        )
        fig.tight_layout()

        fig_path = out_dir / "2dof_calibration_grid.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"図を保存: {fig_path}")

    except Exception as e:
        print(f"[警告] 図の生成に失敗: {e}")

    # ── まとめ ──────────────────────────────────────────────────────────

    print()
    print("── まとめ ──")
    print(f"  1-DoF 理論閾値: δ_c = {DELTA_C:.4f} rad")
    print(f"  独立関節定理の予測: ロバスト領域 = {{(δ1,δ2): δ1<δ_c AND δ2<δ_c}}")
    print()

    # 4象限の平均成功率
    q_labels = [
        ("δ1<0, δ2<0 (両方負)", lambda d1, d2: d1 < -1e-6 and d2 < -1e-6),
        ("δ1<δ_c, δ2<δ_c (両方ロバスト)", lambda d1, d2: 0 <= d1 < DELTA_C and 0 <= d2 < DELTA_C),
        ("δ1>δ_c, δ2<δ_c (j1のみ超過)", lambda d1, d2: d1 > DELTA_C and 0 <= d2 < DELTA_C),
        ("δ1<δ_c, δ2>δ_c (j2のみ超過)", lambda d1, d2: 0 <= d1 < DELTA_C and d2 > DELTA_C),
        ("δ1>δ_c, δ2>δ_c (両方超過 = Deadlock?)", lambda d1, d2: d1 > DELTA_C and d2 > DELTA_C),
    ]
    for label, cond in q_labels:
        vals = [v["aif_success_rate"] for v in grid_results.values()
                if cond(v["delta1"], v["delta2"])]
        if vals:
            print(f"  {label}: AIF={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    print()
    print(f"テンソル積予測との一致率: {n_match/n_total:.1%}")
    print("  (一致率が高い → 独立関節定理が成立)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytical-only", action="store_true",
                        help="解析的予測のみ実行 (pymdpなし)")
    parser.add_argument("--no-plot", action="store_true",
                        help="matplotlib 図の生成をスキップ")
    args = parser.parse_args()
    main(analytical_only=args.analytical_only, no_plot=args.no_plot)
