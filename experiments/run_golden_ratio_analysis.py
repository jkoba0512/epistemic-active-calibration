"""
黄金比ヒューリスティック検証実験スクリプト。

「ρ=φ が bin-skip パターンの多様性を最大化する」という
informal heuristic を数値的・解析的に検証する。

Theory
------
三距離定理（Steinhaus three-distance theorem）:
  {0, ρ, 2ρ, ..., (N-1)ρ} mod 1 のギャップ列は高々 3 種類。
  ρ=φ=(1+√5)/2 のとき最大ギャップが最小化され、
  bin-skip パターンが δ に対して最も均一に分布する。

連結:
  各 δ 値は異なる bin-skip パターン（どのビンが報告されるか）を生む。
  ρ=φ のとき、ギャップが均一 → どの δ でも少なくとも 1 つの
  contact bin がスキップされる確率が最大化される（仮説生存率の最大化）。

Experiments
-----------
  実験 1: P_robust(ρ) vs ρ  [固定対象物配置 {1,2,3}]
    - average-case と worst-case δ の両方で計算
  実験 2: 三距離定理のギャップ構造 vs ρ
    - 最大ギャップと distinct visited パターン数の ρ 依存性
  実験 3: 対象物配置のランダム化
    - 全配置に対する平均・最悪 P_robust で汎化性を確認
  実験 4: effective δ_c(ρ)
    - 各 ρ に対する robustness threshold の数値的計算

Usage
-----
    uv run python experiments/run_golden_ratio_analysis.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 定数 ─────────────────────────────────────────────────────────────
THETA_MIN  = -0.40   # rad
THETA_MAX  = +0.40   # rad
ANGLE_RANGE = 0.80   # rad
N_POS      = 5       # bins（論文と同じ設定）

W = ANGLE_RANGE / N_POS  # bin 幅 = 0.16 rad

PHI         = (1 + np.sqrt(5)) / 2  # 黄金比 ≈ 1.6180
RHO_CURRENT = 1.25                   # 論文現設定

CONTACT_BINS_DEFAULT = [1, 2, 3]     # 論文の対象物配置（bin 1, 2, 3）

OUTPUT_PATH = Path(__file__).parent.parent / "results" / "golden_ratio_analysis.json"
FIGURE_PATH = Path(__file__).parent.parent / "results" / "golden_ratio_figure.png"


# ── コア解析関数 ──────────────────────────────────────────────────────

def compute_visited_bins(rho: float,
                          delta: float,
                          n_pos: int = N_POS) -> frozenset[int]:
    """
    オフセット delta のもとで step ratio ρ の左→右スウィープが
    報告するビン集合を返す。

    ステップ数は θ_min から θ_max に到達するまで（到達後は停止）。
    step size Δ = ρ * w
    reported bin at step i: encode(clip(θ_min + i*Δ + δ, θ_min, θ_max))
    """
    step_size = rho * (ANGLE_RANGE / n_pos)
    w_local   = ANGLE_RANGE / n_pos

    visited: set[int] = set()
    i = 0
    while True:
        pos = THETA_MIN + i * step_size
        if pos > THETA_MAX + 1e-9:
            break
        reported = float(np.clip(pos + delta, THETA_MIN, THETA_MAX))
        b = int(np.floor((reported - THETA_MIN) / w_local))
        b = max(0, min(b, n_pos - 1))
        visited.add(b)
        i += 1
        if i > 10 * n_pos:  # 安全ガード（無限ループ防止）
            break

    return frozenset(visited)


def hypothesis_survives(contact_bin: int, visited: frozenset[int]) -> bool:
    """contact_bin がスキップされた（仮説が生存した）かどうかを返す。"""
    return contact_bin not in visited


def p_robust_metrics(rho: float,
                      contact_bins: list[int],
                      n_delta: int = 2000) -> dict:
    """
    密な δ グリッドでロバスト性指標を計算する。

    Returns
    -------
    p_any_survive  : ∃j: c_j ∉ visited となる δ の割合（average-case）
    p_all_survive  : ∀j: c_j ∉ visited となる δ の割合
    e_skip_count   : スキップされた contact bins の期待数
    worst_skip     : min_δ |skip_set|
    worst_any      : min_δ [|skip_set| > 0]  (0/1)
    """
    # δ の範囲: [0, ANGLE_RANGE/2]（論文でテストした正のオフセット全域）
    delta_grid = np.linspace(0.0, ANGLE_RANGE / 2, n_delta, endpoint=True)

    any_survive  = 0
    all_survive  = 0
    total_skip   = 0
    min_skip     = len(contact_bins)

    for delta in delta_grid:
        visited = compute_visited_bins(rho, delta)
        n_skip = sum(1 for c in contact_bins if c not in visited)

        if n_skip > 0:
            any_survive += 1
        if n_skip == len(contact_bins):
            all_survive += 1
        total_skip += n_skip
        if n_skip < min_skip:
            min_skip = n_skip

    n = len(delta_grid)
    return {
        "p_any_survive" : any_survive  / n,
        "p_all_survive" : all_survive  / n,
        "e_skip_count"  : total_skip   / n,
        "worst_skip"    : min_skip,
        "worst_any"     : int(min_skip > 0),
    }


def three_distance_max_gap(rho: float, n: int = N_POS) -> float:
    """
    {0, ρ, 2ρ, ..., (n-1)ρ} mod 1 の最大ギャップを返す。
    三距離定理の数値的検証: ρ=φ のとき最大ギャップが最小化される。
    """
    points = np.array([(i * rho) % 1.0 for i in range(n)])
    points = np.sort(points)
    gaps   = np.diff(points)
    wrap   = 1.0 - points[-1] + points[0]  # 循環ギャップ
    return float(np.append(gaps, wrap).max())


def distinct_visited_patterns(rho: float,
                               n_delta: int = 1000) -> int:
    """
    δ を変化させたときの distinct visited bin パターン数を返す。
    これが「bin-skip 多様性」の直接指標。
    """
    delta_grid = np.linspace(0.0, ANGLE_RANGE / 2, n_delta, endpoint=True)
    patterns: set[frozenset[int]] = set()
    for delta in delta_grid:
        patterns.add(compute_visited_bins(rho, delta))
    return len(patterns)


def effective_delta_c(rho: float,
                       contact_bins: list[int],
                       n_delta: int = 4000) -> float:
    """
    各 ρ に対する実効的 robustness threshold δ_c を数値的に計算する。
    「全 contact bins が visited されるようになる最初の δ」を返す。
    """
    delta_grid = np.linspace(0.0, ANGLE_RANGE / 2, n_delta, endpoint=True)
    for delta in delta_grid:
        visited = compute_visited_bins(rho, delta)
        if all(c in visited for c in contact_bins):
            return float(delta)
    return float(ANGLE_RANGE / 2)  # δ_c が範囲内に存在しない場合


# ── 実験関数 ─────────────────────────────────────────────────────────

def run_experiment_1(rho_grid: np.ndarray,
                      contact_bins: list[int]) -> list[dict]:
    """実験 1: P_robust(ρ) vs ρ（固定対象物配置）"""
    print("実験 1: P_robust(ρ) vs ρ を計算中 ...")
    results = []
    for k, rho in enumerate(rho_grid):
        if k % 50 == 0:
            print(f"  {k}/{len(rho_grid)} (ρ={rho:.3f})")
        m = p_robust_metrics(rho, contact_bins)
        m["rho"] = float(rho)
        results.append(m)
    return results


def run_experiment_2(rho_grid: np.ndarray) -> list[dict]:
    """実験 2: 三距離定理ギャップ構造 vs ρ"""
    print("実験 2: 三距離定理ギャップ + パターン数を計算中 ...")
    results = []
    for k, rho in enumerate(rho_grid):
        if k % 50 == 0:
            print(f"  {k}/{len(rho_grid)} (ρ={rho:.3f})")
        results.append({
            "rho"              : float(rho),
            "max_gap"          : three_distance_max_gap(rho, N_POS),
            "n_distinct_patterns": distinct_visited_patterns(rho),
        })
    return results


def run_experiment_3(rho_grid: np.ndarray) -> list[dict]:
    """実験 3: 全対象物配置に対する汎化性検証"""
    print("実験 3: 対象物配置ランダム化を計算中 ...")
    # 内部ビン（境界 0, N_POS-1 以外）から選ぶ全組み合わせ
    inner_bins   = list(range(1, N_POS - 1))   # [1, 2, 3] for N_POS=5
    n_obj        = min(3, len(inner_bins))
    all_placements = list(combinations(inner_bins, n_obj))

    results = []
    for k, rho in enumerate(rho_grid):
        if k % 50 == 0:
            print(f"  {k}/{len(rho_grid)} (ρ={rho:.3f})")
        p_any_list   = []
        worst_list   = []
        for placement in all_placements:
            m = p_robust_metrics(rho, list(placement), n_delta=500)
            p_any_list.append(m["p_any_survive"])
            worst_list.append(m["worst_any"])
        results.append({
            "rho"        : float(rho),
            "p_any_mean" : float(np.mean(p_any_list)),
            "p_any_min"  : float(np.min(p_any_list)),
            "worst_mean" : float(np.mean(worst_list)),
            "worst_min"  : int(np.min(worst_list)),
        })
    return results


def run_experiment_4(rho_grid: np.ndarray,
                      contact_bins: list[int]) -> list[dict]:
    """実験 4: effective δ_c(ρ) vs ρ"""
    print("実験 4: effective δ_c(ρ) を計算中 ...")
    results = []
    for k, rho in enumerate(rho_grid):
        if k % 50 == 0:
            print(f"  {k}/{len(rho_grid)} (ρ={rho:.3f})")
        dc = effective_delta_c(rho, contact_bins)
        results.append({
            "rho"     : float(rho),
            "delta_c" : dc,
        })
    return results


# ── 図生成 ───────────────────────────────────────────────────────────

def make_figure(rho_grid: np.ndarray,
                exp1: list[dict],
                exp2: list[dict],
                exp3: list[dict],
                exp4: list[dict]) -> None:
    """4 パネル検証図を生成・保存する。"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Golden Ratio Heuristic Validation  (N_pos={N_POS}, contact bins {{1,2,3}})\n"
        f"Does ρ=φ≈{PHI:.4f} maximise bin-skip diversity?",
        fontsize=13,
    )

    rho_arr = np.array([r["rho"] for r in exp1])

    def vlines(ax):
        ax.axvline(PHI,         color="red",   lw=2.0, ls="-",
                   label=f"ρ=φ≈{PHI:.3f} (golden ratio)")
        ax.axvline(RHO_CURRENT, color="green", lw=2.0, ls="--",
                   label=f"ρ={RHO_CURRENT} (current setting)")
        for ri in [1, 2]:
            ax.axvline(ri, color="gray", lw=0.6, ls=":", alpha=0.5)

    # ── パネル (A): P_robust(ρ)  average & all-survive ────────────
    ax = axes[0, 0]
    p_any = np.array([r["p_any_survive"] for r in exp1])
    p_all = np.array([r["p_all_survive"] for r in exp1])
    e_skip = np.array([r["e_skip_count"] for r in exp1])

    ax.plot(rho_arr, p_any,  color="tab:orange", lw=2,
            label="P(any hypothesis survives)")
    ax.plot(rho_arr, p_all,  color="tab:blue",   lw=2, ls="--",
            label="P(all hypotheses survive)")
    ax.plot(rho_arr, e_skip / len(CONTACT_BINS_DEFAULT),
            color="tab:purple", lw=1.5, ls=":",
            label="E[fraction of bins skipped]")

    vlines(ax)
    ax.set_xlabel("Step/bin ratio ρ")
    ax.set_ylabel("Probability / fraction")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.set_title("(A) Hypothesis Survival Probability vs ρ\n"
                 "(δ ∈ [0, 0.40] rad; contact bins {1,2,3})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ── パネル (B): 三距離定理 ─────────────────────────────────────
    ax = axes[0, 1]
    rho2  = np.array([r["rho"] for r in exp2])
    mgap  = np.array([r["max_gap"] for r in exp2])
    npat  = np.array([r["n_distinct_patterns"] for r in exp2])

    ax2 = ax.twinx()
    l1, = ax.plot(rho2, mgap, color="tab:purple", lw=2,
                  label="Max gap (three-distance theorem)")
    l2, = ax2.plot(rho2, npat, color="tab:cyan",  lw=1.5, ls="--",
                   label="# distinct visit patterns")

    ax.axvline(PHI,         color="red",   lw=2.0, ls="-")
    ax.axvline(RHO_CURRENT, color="green", lw=2.0, ls="--")
    for ri in [1, 2]:
        ax.axvline(ri, color="gray", lw=0.6, ls=":", alpha=0.5)

    ax.set_xlabel("Step/bin ratio ρ")
    ax.set_ylabel("Max gap on [0,1) circle", color="tab:purple")
    ax2.set_ylabel("# distinct visited-bin patterns", color="tab:cyan")
    ax.set_title("(B) Three-Distance Theorem\n"
                 "(ρ=φ minimises max gap → uniform bin-skip coverage)")
    ax.legend([l1, l2,
               plt.Line2D([0],[0], color="red",   lw=2, ls="-"),
               plt.Line2D([0],[0], color="green", lw=2, ls="--")],
              [l1.get_label(), l2.get_label(),
               f"ρ=φ≈{PHI:.3f}", f"ρ={RHO_CURRENT}"],
              fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── パネル (C): 配置ランダム化 ────────────────────────────────
    ax = axes[1, 0]
    rho3   = np.array([r["rho"] for r in exp3])
    p_mean = np.array([r["p_any_mean"] for r in exp3])
    p_min  = np.array([r["p_any_min"]  for r in exp3])

    ax.fill_between(rho3, p_min, p_mean,
                    alpha=0.25, color="tab:orange",
                    label="Range (min to mean over placements)")
    ax.plot(rho3, p_mean, color="tab:orange", lw=2,
            label="Mean over all contact-bin placements")
    ax.plot(rho3, p_min,  color="tab:red",    lw=1.5, ls="--",
            label="Worst-case placement")

    vlines(ax)
    ax.set_xlabel("Step/bin ratio ρ")
    ax.set_ylabel("P(any hypothesis survives)")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.set_title("(C) Generalisation: All Contact-Bin Placements\n"
                 f"(all C({N_POS-2},3) combinations of inner bins)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ── パネル (D): effective δ_c(ρ) ──────────────────────────────
    ax = axes[1, 1]
    rho4 = np.array([r["rho"] for r in exp4])
    dc4  = np.array([r["delta_c"] for r in exp4])

    # 論文の解析的公式（ρ=1.25 で導出）
    dc_formula = ANGLE_RANGE * (N_POS - 2) / (N_POS * (N_POS - 1))

    ax.plot(rho4, dc4, color="tab:brown", lw=2,
            label="Effective δ_c(ρ) [numerical]")
    ax.axhline(dc_formula, color="navy", lw=1.5, ls="--",
               label=f"Analytical formula δ_c={dc_formula:.3f} rad  (ρ=1.25 derivation)")

    vlines(ax)
    ax.set_xlabel("Step/bin ratio ρ")
    ax.set_ylabel("Robustness threshold δ_c (rad)")
    ax.set_title("(D) Effective Robustness Threshold δ_c vs ρ\n"
                 "(first δ > 0 where all contact bins are visited)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n図を保存: {FIGURE_PATH}")


# ── サマリー表示 ─────────────────────────────────────────────────────

def print_comparison_table(exp1: list[dict],
                            exp2: list[dict],
                            exp3: list[dict],
                            exp4: list[dict]) -> None:
    """ρ=1.25 vs ρ=φ vs ρ=integer近傍 の比較表を表示する。"""

    def nearest(data: list[dict], rho_target: float, key: str) -> float:
        return min(data, key=lambda r: abs(r["rho"] - rho_target))[key]

    rho_targets = [
        ("ρ=1.0 (integer)",     1.001),
        ("ρ=1.25 (current)",    1.25),
        ("ρ=φ≈1.618 (golden)",  PHI),
        ("ρ=2.0 (integer)",     2.001),
    ]

    header = f"{'ρ 設定':<28} | {'P_any':>6} | {'P_all':>6} | {'E_skip':>6} | {'W_any':>5} | {'MaxGap':>7} | {'Patterns':>8} | {'δ_c (rad)':>10}"
    print("\n" + "=" * len(header))
    print("ρ 値別ロバスト性比較")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, rho in rho_targets:
        p_any  = nearest(exp1, rho, "p_any_survive")
        p_all  = nearest(exp1, rho, "p_all_survive")
        e_skip = nearest(exp1, rho, "e_skip_count")
        w_any  = nearest(exp1, rho, "worst_any")
        mgap   = nearest(exp2, rho, "max_gap")
        npat   = nearest(exp2, rho, "n_distinct_patterns")
        dc     = nearest(exp4, rho, "delta_c")
        print(f"{label:<28} | {p_any:6.3f} | {p_all:6.3f} | {e_skip:6.3f} | {w_any:5d} | {mgap:7.4f} | {npat:8d} | {dc:10.4f}")

    print("=" * len(header))
    print("  P_any  = P(少なくとも 1 仮説が生存) over δ ∈ [0, 0.40] rad")
    print("  P_all  = P(全仮説が生存) over δ")
    print("  E_skip = E[スキップされた contact bin 数]")
    print("  W_any  = worst-case で少なくとも 1 仮説が生存するか (0/1)")
    print("  MaxGap = 三距離定理の最大ギャップ（小さいほど均一）")
    print("  δ_c    = 全 contact bins が初めて visited される δ")


# ── メイン ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ρ グリッド: 整数値（完全 coverage で bin-skip なし）を避けて
    # 1.01 から 2.50 まで 300 点
    rho_grid      = np.linspace(1.01, 2.50, 300)
    contact_bins  = CONTACT_BINS_DEFAULT

    print(f"黄金比 φ = {PHI:.6f}")
    print(f"現設定  ρ = {RHO_CURRENT}")
    print(f"ρ グリッド: {rho_grid[0]:.3f} ~ {rho_grid[-1]:.3f}  ({len(rho_grid)} 点)")
    print()

    exp1 = run_experiment_1(rho_grid, contact_bins)
    exp2 = run_experiment_2(rho_grid)
    exp3 = run_experiment_3(rho_grid)
    exp4 = run_experiment_4(rho_grid, contact_bins)

    print_comparison_table(exp1, exp2, exp3, exp4)
    make_figure(rho_grid, exp1, exp2, exp3, exp4)

    # ── JSON 保存 ────────────────────────────────────────────────────
    results = {
        "parameters": {
            "N_POS"          : N_POS,
            "ANGLE_RANGE"    : ANGLE_RANGE,
            "W"              : W,
            "PHI"            : PHI,
            "RHO_CURRENT"    : RHO_CURRENT,
            "contact_bins"   : contact_bins,
            "n_rho_points"   : len(rho_grid),
            "rho_min"        : float(rho_grid[0]),
            "rho_max"        : float(rho_grid[-1]),
            "delta_range"    : [0.0, ANGLE_RANGE / 2],
        },
        "experiment_1_p_robust"       : exp1,
        "experiment_2_three_distance" : exp2,
        "experiment_3_random_placement": exp3,
        "experiment_4_delta_c"        : exp4,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n結果を保存: {OUTPUT_PATH}")
