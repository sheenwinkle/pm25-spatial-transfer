"""Plot corrected post-hoc audit-resolution estimates for the manuscript."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research" / "manuscript_revision_20261004" / "review_evidence" / "diagnostics" / "corrected_resolution_20261005" / "corrected_resolution_by_config.csv"
OUTPUT = ROOT / "research" / "manuscript_revision_20261004" / "review_evidence" / "diagnostics" / "corrected_resolution_20261005" / "audit_resolution_corrected.png"

data = pd.read_csv(SOURCE)
groups = [
    ("CHN_to_AUS", "temporal_audit", "China to Australia · time"),
    ("CHN_to_AUS", "new_city_audit", "China to Australia · new city"),
    ("AUS_to_CHN", "temporal_audit", "Australia to China · time"),
    ("AUS_to_CHN", "new_city_audit", "Australia to China · new city"),
]
colors = {"CHN_to_AUS": "#d56b3c", "AUS_to_CHN": "#287d75"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=170)
rng = np.random.default_rng(412)
for i, (direction, stage, label) in enumerate(groups):
    values = data.loc[(data.direction == direction) & (data.stage == stage), "conservative_mde80_pct"].dropna().to_numpy()
    color = colors[direction]
    ax.scatter(values, i + rng.uniform(-0.16, 0.16, len(values)), s=16, alpha=0.68,
               color=color, edgecolor="white", linewidth=0.35)
    median = float(np.median(values))
    ax.plot([median, median], [i - 0.28, i + 0.28], color="#202727", linewidth=2)
    ax.text(max(values.max(), median) * 1.08, i, f"median {median:.0f}% (n={len(values)})",
            va="center", fontsize=8, color="#555d5b")
ax.axvline(5, color="#303737", linewidth=1, linestyle="--")
ax.text(0.015, 0.975, "Dashed line: 5% point margin", transform=ax.transAxes,
        fontsize=8, va="top", color="#303737")
ax.set_xscale("log")
ax.set_xlim(5, 650)
ax.set_xticks([5, 10, 20, 50, 100, 200, 500])
ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500"])
ax.set_yticks(range(4))
ax.set_yticklabels([group[2] for group in groups])
ax.set_ylim(3.45, -0.55)
ax.set_xlabel("Approximate 80% minimum detectable RMSE improvement (%) · log scale")
ax.set_title("Post-hoc audit resolution under the corrected replay", loc="left", fontsize=10.5)
ax.grid(axis="x", color="#e2e6e3", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(OUTPUT)
