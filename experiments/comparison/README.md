# DEM 比較実験: MATLAB (SPM) vs Python (JAX)

## 概要

このディレクトリは、Friston の DEM (Dynamic Expectation Maximization) の
MATLAB 実装 (SPM) と Python/JAX 実装を数値的に比較検証するための
サンプルプログラムを含む。

### 対象システム: 減衰線形システム (Damped Linear System)

```
真のシステム:
  dx/dt = a * x + v    (真の減衰係数 a = -1.0, 外部入力 v = 0)
  y = x + ε            (観測ノイズ ε ~ N(0, σ_y²))

初期状態: x(0) = 1.0
時定数:   τ = 1/|a| = 1.0 秒
解析解:   x(t) = exp(-t)
```

このシステムを選んだ理由:
1. **解析解が既知**: x(t) = exp(-t) と比較できる
2. **D-step + E-step の両方を使う**: 状態追跡とパラメータ推定の両方を検証できる
3. **SPM の既存デモ `DEM_demo_GF.m` と類似**: 対応が容易
4. **シンプルな設定**: MATLAB/Python で同一パラメータを設定しやすい

---

## ディレクトリ構造

```
experiments/comparison/
├── README.md                    # このファイル
├── matlab/
│   └── dem_linear_system.m     # MATLAB + SPM 実装
├── python/
│   └── dem_linear_system.py    # Python + JAX 実装
└── compare_results.py          # 結果比較スクリプト

results/
├── comparison_matlab.csv       # MATLAB 実行結果
├── comparison_python.csv       # Python 実行結果
└── comparison_result.png       # 比較プロット
```

---

## 共通パラメータ設定

以下のパラメータは MATLAB/Python 実装で完全に一致させる:

| パラメータ | 記号 | 値 | SPM フィールド | Python フィールド |
|-----------|------|-----|---------------|-----------------|
| 真の減衰係数 | a_true | -1.0 | (データ生成) | (データ生成) |
| 初期パラメータ推定 | a_init | 0.0 | `pE.a = 0` | `params[0] = 0.0` |
| パラメータ事前精度 | pi_a | exp(-4) ≈ 0.018 | `pC.a = exp(-4)` | `params_prior_pi` |
| 観測精度 | pi_y | 16.0 | `M(1).V = 16` | `PI_Y = 16.0` |
| 状態精度 | pi_x | 1.0 | `M(1).W = 1` | `PI_X = 1.0` |
| 埋め込み次数 | p | 4 | `M(1).n = 4` | `n_order = 4` |
| 平滑化パラメータ | s | 1.0 | `M(1).E.s = 1.0` | `s_y = s_x = 1.0` |
| 初期状態 | x₀ | 1.0 | `M(1).x = [1]` | mu_x0[0] = 1.0 |
| シミュレーション時間 | T | 4.0 秒 | `U.dt * N` | dt * N_steps |
| 観測時間刻み | dt | 0.1 秒 | `U.dt = 0.1` | dt = 0.1 |
| 観測ノイズ標準偏差 | σ_y | exp(-2) ≈ 0.135 | (データ生成) | noise_std = exp(-2) |

**注意: s=0.5 (SPM のデフォルト) と pi_y=exp(4) の組み合わせは、Python 版の固定ステップ幅 Euler 積分では R 行列スケール (s^{-k}) が大きすぎて数値発散する。両実装で s=1.0, pi_y=16 を採用する。**

---

## セットアップ

### MATLAB + SPM のセットアップ

1. SPM12 をダウンロード・インストール: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
2. MATLAB を起動し、SPM のパスを追加:

```matlab
addpath('/path/to/spm12');          % SPM のパスを設定
addpath('/path/to/spm12/toolbox');  % ツールボックスのパスも追加
spm('Defaults', 'FMRI');            % SPM の初期化
```

3. プロジェクトルートに移動:

```matlab
cd('/home/jkoba/SynologyDrive_private/SynologyDrive/aif_occlusion_manipulator');
```

### Python + JAX のセットアップ

```bash
cd /home/jkoba/SynologyDrive_private/SynologyDrive/aif_occlusion_manipulator
uv sync  # または pip install -e .
```

---

## 実行手順

### Step 1: MATLAB 版を実行

```matlab
% MATLAB コンソールで実行
run('experiments/comparison/matlab/dem_linear_system.m')
% → results/comparison_matlab.csv が生成される
```

### Step 2: Python 版を実行

```bash
cd /home/jkoba/SynologyDrive_private/SynologyDrive/aif_occlusion_manipulator
python experiments/comparison/python/dem_linear_system.py
# → results/comparison_python.csv が生成される
```

### Step 3: 結果を比較

```bash
python experiments/comparison/compare_results.py
# → results/comparison_result.png が生成される
# → コンソールに比較指標が表示される
```

---

## 期待される結果

### 状態推定 (D-step)
- 両実装とも x(t) = exp(-t) の解析解に収束
- 観測ノイズを除いた真の軌跡を精度良く追跡

### パラメータ推定 (E-step)
- 初期値 a₀ = 0.0 から真値 a = -1.0 方向へ収束
- 観測時系列が長いほど収束精度が向上

### VFE 軌跡
- 時間とともに単調減少（または概ね減少）
- 両実装で類似したスケールと形状を示す

---

## 合格基準

| 指標 | 基準 |
|------|------|
| 状態推定 RMSE (Python vs MATLAB) | < 0.05 |
| パラメータ収束先の差異 \|a_py - a_matlab\| | < 0.1 |
| VFE 最終値の相対誤差 | < 10% |

---

## 注意事項

- Python 版の E-step は現在実装中であり、現バージョンでは D-step（状態推定）のみ動作する
- E-step が完成次第、`python/dem_linear_system.py` の E-step 部分を有効化する
- MATLAB 版は SPM12 の `spm_DEM.m` または `spm_ADEM.m` を使用する
- SPM のバージョンによって関数名や構造体フィールドが異なる場合がある（SPM12 を推奨）

## 参考文献

- Friston, K. (2008). Hierarchical models in the brain. PLoS Computational Biology, 4(11).
- Friston, K. et al. (2010). Generalised filtering. Mathematical Problems in Engineering.
- SPM DEM デモ: `spm12/toolbox/DEM/DEM_demo_GF.m`
