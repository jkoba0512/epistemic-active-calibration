"""Build the KUKA iiwa terminal-controllability case-study figure (RA-L task #1).

Reads the sweep / AUC result JSONs and produces a 3-panel figure:
  (a) failTask vs Phase-2 horizon H (Wilson CI) -> terminal finite-horizon
      controllability: good calibration but short-horizon task failure that
      recovers as H grows.
  (b) Predictor AUC for failTask@H=50 -> parameter error and static
      singularity do not predict failure; short-horizon rollout risk does
      (incl. oracle parameters).
  (c) Per-seed scatter rmse@change vs rollout risk, coloured by task failure ->
      failure separates along rollout risk, not along parameter error.

Mirrors the planar 4-DoF main result on a commercial 7-DoF arm.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
results = root / "results"

sweep = json.loads((results / "iiwa_horizon_sweep_varied.json").read_text())
auc = json.loads((results / "iiwa_rollout_risk_auc.json").read_text())

H_eval = auc["H_eval"]
horizons = sweep["horizons"]
res = sweep["results"]

COND_STYLE = {
    "probe->plain":   ("tab:red",   "o-", "probe -> plain"),
    "probe->posture": ("tab:orange","s--", "probe -> posture"),
    "plain->plain":   ("tab:blue",  "^-", "plain -> plain"),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

# --- (a) failTask vs H -----------------------------------------------------
ax = axes[0]
for cond, (color, style, label) in COND_STYLE.items():
    hd = res[cond]
    ft = [hd[str(H)]["fail_task"] for H in horizons]
    lo = [hd[str(H)]["fail_task_ci"][0] for H in horizons]
    hi = [hd[str(H)]["fail_task_ci"][1] for H in horizons]
    ax.plot(horizons, ft, style, color=color, label=label, markersize=5)
    ax.fill_between(horizons, lo, hi, color=color, alpha=0.12)
ax.set_xlabel("Phase-2 horizon $H$ (steps)")
ax.set_ylabel("Task failure rate")
ax.set_ylim(-0.03, 1.05)
ax.set_title("(a) Terminal finite-horizon controllability")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

# --- (b) predictor AUC -----------------------------------------------------
ax = axes[1]
auc_keys = ["rmse_at_change", "inv_sigma_min", "rollout_risk_hat", "rollout_risk_oracle"]
auc_labels = ["param\nRMSE", "$1/\\sigma_{min}$\n(static)", "rollout\nrisk", "rollout\nrisk\n(oracle)"]
vals = [auc["auc"][k] for k in auc_keys]
colors = ["0.6", "0.6", "tab:green", "tab:green"]
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.6)
ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(auc_labels, fontsize=8)
ax.set_ylabel(f"AUC for failTask at $H$={H_eval}")
ax.set_ylim(0, 1.05)
ax.set_title("(b) What predicts task failure?")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8)
ax.grid(alpha=0.25, axis="y")

# --- (c) per-seed scatter: rmse vs rollout risk ----------------------------
ax = axes[2]
per = auc["per_seed"]
rmse = np.array([p["rmse_at_change"] for p in per])
rr = np.array([p["r_rollout_hat"] for p in per])
fail = np.array([p["fail_task"] for p in per], bool)
ax.scatter(rmse[~fail], rr[~fail], c="tab:blue", marker="o", s=30,
           label="task success", edgecolor="k", linewidth=0.4)
ax.scatter(rmse[fail], rr[fail], c="tab:red", marker="X", s=55,
           label="task failure", edgecolor="k", linewidth=0.4)
ax.set_xlabel("Parameter RMSE at Phase-1 end")
ax.set_ylabel("Short-horizon rollout risk")
ax.set_title("(c) Failure separates by rollout risk, not RMSE")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()

for ext in ("png", "pdf"):
    out = results / f"iiwa_case_study.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
