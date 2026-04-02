"""
Q1: キャリブレーション閾値の理論的導出（修正版）。

正しい失敗機構:
  エージェントはスイープ中に「モデル期待ビン B_k を訪問したが接触なし」を
  観測すると obj_loc=k を消去する。
  全 obj_loc が消去されると，エージェントは bin 4（モデル外）への
  移動を選ばず停止 → 物理接触に到達できず失敗。

  重要: あるステップ幅が bin を「スキップ」すると，
        そのビンの obj 仮説は消去されず生き残る
        → エージェントはその仮説を追って移動を続ける → 成功。

Usage
-----
    uv run python experiments/analyze_calib_threshold.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aif_occlusion.utils.discretizer import JointDiscretizer
from aif_occlusion.simulation.so101_env import _CONTACT_ANGLES, _SHOULDER_MIN, _SHOULDER_MAX

N_BINS    = 5
STEP_SIZE = (_SHOULDER_MAX - _SHOULDER_MIN) / (N_BINS - 1)  # 0.20 rad
BIN_WIDTH = (_SHOULDER_MAX - _SHOULDER_MIN) / N_BINS         # 0.16 rad

disc = JointDiscretizer(n_bins=N_BINS, angle_min=_SHOULDER_MIN, angle_max=_SHOULDER_MAX)

# 腕の到達可能位置
ARM_POSITIONS = np.array([_SHOULDER_MIN + i * STEP_SIZE for i in range(N_BINS)])
# → [-0.40, -0.20, 0.00, +0.20, +0.40]

# A_tactile で定義された接触ビン
MODEL_CONTACT_BINS = {0: 1, 1: 2, 2: 3}

print("=" * 68)
print("Q1: キャリブレーション閾値の理論的導出")
print("=" * 68)
print(f"\n基本パラメータ:")
print(f"  bin 幅         = {BIN_WIDTH:.4f} rad")
print(f"  ステップ幅     = {STEP_SIZE:.4f} rad")
print(f"  ステップ/bin   = {STEP_SIZE/BIN_WIDTH:.4f}  ← 非整数：「bin スキップ」が発生しうる")
print(f"\nbin 境界: {disc._edges}")
print(f"腕の到達可能位置: {ARM_POSITIONS}")

# ── Step 1: スイープ中に訪問する報告ビン列 ─────────────────────────────────
print("\n" + "=" * 68)
print("Step 1: offset ごとの「報告ビン訪問列」（左端 -0.40 からのスイープ）")
print("=" * 68)

offsets = np.round(np.arange(0.0, 0.22, 0.04), 4)

print(f"\n{'offset':>8}  {'bin幅比':>7}  訪問ビン列  スキップ")
print("-" * 60)

visited_bins_by_offset = {}
for offset in offsets:
    visited = []
    for a in ARM_POSITIONS:
        r = float(np.clip(a + offset, _SHOULDER_MIN, _SHOULDER_MAX))
        b = disc.encode(r)
        visited.append(b)
    skipped = sorted(
        set(range(min(visited), max(visited)+1)) - set(visited)
    )
    visited_bins_by_offset[offset] = visited
    skip_str = f"bin {skipped}" if skipped else "なし"
    print(f"  {offset:+.4f}  {offset/BIN_WIDTH:>6.2f}   {visited}   {skip_str}")

# ── Step 2: 失敗機構の詳細分析 ─────────────────────────────────────────────
print("\n" + "=" * 68)
print("Step 2: 失敗機構の詳細分析")
print()
print("  失敗条件 =")
print("    (A) 物理接触ビン B'_j(δ) ∉ モデル接触ビン集合 {1,2,3}")
print("    かつ")
print("    (B) スイープ中に {1,2,3} を全て「接触なしで」訪問する")
print("        ← どれかがスキップされると，その obj 仮説が生き残り，")
print("           エージェントは動き続けて物理接触に到達できる")
print("=" * 68)

model_bins_set = set(MODEL_CONTACT_BINS.values())  # {1, 2, 3}

for obj_loc in range(3):
    c = _CONTACT_ANGLES[obj_loc]
    model_bin_j = MODEL_CONTACT_BINS[obj_loc]
    print(f"\n  obj_loc={obj_loc}  接触角={c:+.3f}rad  モデルbin={model_bin_j}")
    print(f"  {'offset':>8}  {'物理接触bin':>11}  "
          f"{'モデルbinスキップ':>16}  {'全仮説消去?':>11}  {'判定':>8}")
    print(f"  {'-'*65}")

    for offset in offsets:
        visited = visited_bins_by_offset[offset]
        phys_bin = disc.encode(float(np.clip(c + offset, _SHOULDER_MIN, _SHOULDER_MAX)))

        # (A): 物理接触がモデル範囲外か
        cond_A = phys_bin not in model_bins_set

        # (B): モデルビン集合 {1,2,3} が全て訪問されるか？
        #      ただし，物理接触ビン到達「前」の訪問のみカウント
        #      = 物理接触ビン(phys_bin)が初めて現れる index 以前の訪問
        try:
            phys_reach_idx = visited.index(phys_bin)
        except ValueError:
            phys_reach_idx = len(visited)  # 到達しない
        visited_before_contact = set(visited[:phys_reach_idx])
        skipped_model_bins = model_bins_set - visited_before_contact

        cond_B = len(skipped_model_bins) == 0  # 全モデルbin訪問済み

        # 残存する仮説（スキップされたモデルビンの obj_loc）
        surviving_hyps = [
            j for j, b in MODEL_CONTACT_BINS.items() if b in skipped_model_bins
        ]

        if cond_A and cond_B:
            verdict = "✗ 失敗"
        elif cond_A and not cond_B:
            surviving_str = f"obj{surviving_hyps} 生存"
            verdict = f"✓ 成功({surviving_str})"
        else:
            verdict = "✓ 成功(正位置)"

        skip_str = f"bin{sorted(skipped_model_bins)}" if skipped_model_bins else "なし"
        print(f"  {offset:+8.4f}  {'bin'+str(phys_bin):>11}  "
              f"{skip_str:>16}  {'Yes' if cond_B else 'No':>11}  {verdict}")

# ── Step 3: 全 obj_loc × offset の予測マトリクス ───────────────────────────
print("\n" + "=" * 68)
print("Step 3: 予測成功率 vs 実験値")
print("=" * 68)

exp_E_contacts = {0.00: 60, 0.04: 60, 0.08: 60, 0.12: 40, 0.16: 40, 0.20: 40}

print(f"\n  {'offset':>8}  {'bin幅比':>7}  obj0  obj1  obj2  予測    実験")
print(f"  {'-'*55}")

for offset in offsets:
    visited = visited_bins_by_offset[offset]
    preds = []
    for obj_loc in range(3):
        c = _CONTACT_ANGLES[obj_loc]
        phys_bin = disc.encode(float(np.clip(c + offset, _SHOULDER_MIN, _SHOULDER_MAX)))
        cond_A = phys_bin not in model_bins_set
        try:
            phys_idx = visited.index(phys_bin)
        except ValueError:
            phys_idx = len(visited)
        visited_before = set(visited[:phys_idx])
        cond_B = model_bins_set.issubset(visited_before)
        preds.append("✗" if (cond_A and cond_B) else "✓")

    n_ok = preds.count("✓")
    pred_n = n_ok / 3 * 60
    exp_n = exp_E_contacts.get(round(float(offset), 2), "?")
    match = "✓" if abs(pred_n - exp_n) < 1 else "✗"
    print(f"  {offset:+8.4f}  {offset/BIN_WIDTH:>6.2f}   "
          f"{'  '.join(preds)}   {pred_n:.0f}/60   {exp_n}/60  {match}")

# ── Step 4: 閾値の幾何学的説明 ─────────────────────────────────────────────
print("\n" + "=" * 68)
print("Step 4: 閾値 δ_c = 0.12 rad の幾何学的説明")
print("=" * 68)
print("""
  スキップが消えるのは，連続する arm 位置 [a, a+0.20] の「報告ビン遷移」が
  隣接ビンになる（飛ばしがなくなる）瞬間。

  着目ステップ: actual = -0.20 → 0.00 (スイープの第2ステップ)
    報告 = (-0.20+δ) → (0.00+δ)

  bin2 = [-0.08, +0.08) がスキップされる条件:
    encode(-0.20+δ) ≤ 1  かつ  encode(0.00+δ) ≥ 3
    ↔ (-0.20+δ) < -0.08  かつ  (0.00+δ) ≥ +0.08
    ↔ δ < 0.12           かつ  δ ≥ 0.08
    ↔ 0.08 ≤ δ < 0.12
""")

c0 = _CONTACT_ANGLES[0]  # -0.20
right_edge_bin1 = disc._edges[2]  # -0.08
threshold = right_edge_bin1 - c0
print(f"  δ_c = (bin1の右端) - (obj_loc=0の接触角)")
print(f"      = {right_edge_bin1:.4f} - ({c0:.4f})")
print(f"      = {threshold:.4f} rad = {threshold/BIN_WIDTH:.4f} bins")
print()
print(f"  解釈: obj_loc=0 の接触角 c_0 = {c0:.2f} rad は")
print(f"        bin 1 の右端 {right_edge_bin1:.2f} rad まで {threshold:.2f} rad (= 0.75 bin) の余裕がある。")
print(f"        offset がこの余裕を超えると，c_0 が bin 2 に踏み込み，")
print(f"        「bin 2 を接触なしで訪問」が起き，obj=1 仮説が消去される。")
print(f"        この時点で全仮説が消去 → エージェント停止。")

# ── Step 5: 一般化 ──────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Step 5: 一般化 — 閾値の決定因子")
print("=" * 68)
print("""
  本実験設定での閾値を決めた要素:

  1. 接触角とbin境界の距離
       c_0 = -0.20 rad → bin1 右端 -0.08 rad まで 0.12 rad
       これが「bin スキップが消える」閾値になる

  2. ステップサイズ vs bin幅の非整合（0.20 / 0.16 = 1.25）
       この非整合が「skipped bin」を生む。
       ステップ = bin幅 なら常にスキップなし → 閾値は最小化

  3. 「生き残り仮説」機構
       スキップされた bin の obj 仮説が消去されないことで
       エージェントの探索継続が保証される

  一般式（n_pos, step, contact_angles が変わった場合）:
    δ_c = min_{j} (右端_bin(c_j) - c_j)   ただしスキップが消える条件のもとで
                                            (step と bin幅の関係に依存)
""")
print("  本実験での δ_c 確認:")
for obj_loc in range(3):
    c = _CONTACT_ANGLES[obj_loc]
    b = MODEL_CONTACT_BINS[obj_loc]
    right_edge = disc._edges[b + 1]
    dist_to_right = right_edge - c
    left_edge = disc._edges[b]
    dist_to_left = c - left_edge
    print(f"    obj_loc={obj_loc}: c={c:+.3f}, bin{b}=[{left_edge:.3f},{right_edge:.3f})")
    print(f"           左端まで {dist_to_left:.4f} rad ({dist_to_left/BIN_WIDTH:.2f}bin),  "
          f"右端まで {dist_to_right:.4f} rad ({dist_to_right/BIN_WIDTH:.2f}bin)")

print(f"\n  → 実験で最初に崩壊するのは「右端への距離が最小の obj_loc」")
print(f"    = obj_loc=0 の右端距離 0.12 rad = 0.75 bin  ← これが δ_c  ✓")
