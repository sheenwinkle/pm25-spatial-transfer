"""Figures for the round-6 manuscript: label-budget ablations (k and h)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_tables import load, with_best
from paths import GENERATED

G = GENERATED
G.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
SHOW = [("GBR (target-only)", "Target-only GBR (benchmark)", "#7f7f7f", "o", "-"),
        ("NNW[LDF]", "NNW[LDF] (benchmark best)", "#c0392b", "s", "-"),
        ("Best local", "Per-split post-hoc best (8 target-only)", "#27ae60", "D", ":"),
        ("Source-affine (ours)", "Source-affine (this report)", "#1f4e9c", "o", "-"),
        ("Source-affine[LDF] (ours)", "Source-affine[LDF] (this report)", "#5dade2", "v", "--")]
NAME = {"yrd": "T1: YRD target", "aus": "T2: Australian target"}


def ci(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def k_figure(d):
    panels = [(t, p) for t in ("yrd", "aus") for p in ("B", "S") if ((d.task == t) & (d.protocol == p)).any()]
    if not panels:
        return
    nr = 2 if len(panels) > 2 else 1; nc = int(np.ceil(len(panels) / nr))
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 2.9 * nr + 0.6), squeeze=False)
    for ax, (t, p) in zip(axes.ravel(), panels):
        g = d[(d.task == t) & (d.protocol == p)]
        ks = sorted(g.k.unique())
        for key, lab, col, mk, ls in SHOW:
            h = g[g.model == key].groupby("k").rmse.mean()
            if len(h):
                ax.plot(h.index, h.values, color=col, marker=mk, ls=ls, ms=4, lw=1.4, label=lab)
        ax.set_xticks(ks)
        ax.set_xlabel("labelled target FUAs, $k$")
        ax.set_title(f"{NAME[t]}\n{'same-period (benchmark) protocol' if p == 'B' else 'later-period protocol'}", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean test RMSE (µg/m³)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(G / "fig_k_ablation.png", dpi=220); plt.close(fig)


def h_figure(d):
    mm = d[d.protocol.isin(["M3", "M6", "M12", "S"])].copy()
    if not len(mm):
        return
    mm["h"] = mm.protocol.map({"M3": 3, "M6": 6, "M12": 12, "S": 72})
    panels = [(t, k) for t, ks in (("yrd", (5, 11)), ("aus", (3, 5))) for k in ks
              if ((mm.task == t) & (mm.k == k) & mm.protocol.str.startswith("M")).any()]
    nr = 2 if len(panels) > 2 else 1; nc = int(np.ceil(len(panels) / nr))
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 2.9 * nr + 0.6), squeeze=False)
    for ax, (t, k) in zip(axes.ravel(), panels):
        g = mm[(mm.task == t) & (mm.k == k)]
        for key, lab, col, mk, ls in SHOW:
            h = g[g.model == key].groupby("h").rmse.mean()
            if len(h):
                ax.plot(h.index, h.values, color=col, marker=mk, ls=ls, ms=4, lw=1.4, label=lab)
        ax.set_xscale("log"); ax.set_xticks([3, 6, 12, 72]); ax.set_xticklabels(["3", "6", "12", "72"])
        ax.minorticks_off()
        ax.set_xlabel("months of target labels, $h$")
        ax.set_title(f"{NAME[t]}, $k={k}$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean RMSE, 2021–2022 (µg/m³)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(G / "fig_history_ablation.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    d = with_best(load())
    k_figure(d); h_figure(d)
    print("figures written")
