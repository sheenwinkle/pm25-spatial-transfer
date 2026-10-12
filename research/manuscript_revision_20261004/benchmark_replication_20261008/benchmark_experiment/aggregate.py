"""Benchmark-style aggregation of results/*/k*_split*.csv (see PROTOCOL.md).

Outputs (results/summary/):
  benchmark_table_{task}_{protocol}.csv   mean R2 / RMSE over splits per model and k (benchmark Table 1 format)
  paired.csv                               paired per-split comparisons for the declared pairs
  summary.md                               human-readable digest
  fig_k_ablation.png, fig_m_ablation.png   label-budget ablations
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from paths import RESULTS

R = RESULTS
OUT = R / "summary"; OUT.mkdir(exist_ok=True)
OURS = "Source-affine (ours)"
PAIRS = [  # (candidate, reference, label)  -- declared in PROTOCOL.md
    (OURS, "NNW[LDF]", "primary: affine vs NNW[LDF]"),
    (OURS, "GBR (target-only)", "primary: affine vs target-only GBR"),
    ("NNW[LDF]", "GBR (target-only)", "benchmark claim: NNW[LDF] vs GBR"),
    ("Source-affine[LDF] (ours)", "NNW[LDF]", "secondary: affine[LDF] vs NNW[LDF]"),
    (OURS, "Shuffled-source affine (control)", "control: affine vs shuffled-source affine"),
    ("NNW[LDF]", "NNW", "LDF ablation: NNW[LDF] vs NNW"),
    ("GBR[LDF] (target-only)", "GBR (target-only)", "LDF ablation: GBR[LDF] vs GBR"),
    ("KLIEP[LDF]", "KLIEP", "LDF ablation: KLIEP[LDF] vs KLIEP"),
    ("KMM[LDF]", "KMM", "LDF ablation: KMM[LDF] vs KMM"),
    ("NNW (sample-weight)", "NNW", "weighting: sample-weight vs released (NNW)"),
    ("NNW[LDF] (sample-weight)", "NNW[LDF]", "weighting: sample-weight vs released (NNW[LDF])"),
]
LOCAL = ["Local mean", "Local ridge", "Local RBF", "Local RF", "Local Extra Trees", "Local LightGBM"]
BENCH = ["RF (target-only)", "GBR (target-only)", "GBR[LDF] (target-only)", "NNW", "NNW[LDF]", "KLIEP",
         "KLIEP[LDF]", "KMM", "KMM[LDF]", "NNW (sample-weight)", "NNW[LDF] (sample-weight)"]


def load():
    frames = []
    for f in sorted(R.glob("*_*/k*_split*.csv")):
        if f.name.startswith("singlepass"):
            continue
        d = pd.read_csv(f)
        if "SKIPPED" in set(d.model):
            continue
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["protocol"] = d.protocol.astype(str)
    return d


def add_best(d):
    """Per-split oracle best of the six local learners and of the benchmark methods (post-hoc, descriptive)."""
    extra = []
    for key, g in d.groupby(["task", "protocol", "k", "split"]):
        for name, pool in (("Best local (per split, post-hoc)", LOCAL), ("Best benchmark (per split, post-hoc)", BENCH)):
            h = g[g.model.isin(pool)]
            if len(h):
                r = h.loc[h.rmse.idxmin()].copy(); r["model"] = name; extra.append(r)
    return pd.concat([d, pd.DataFrame(extra)], ignore_index=True)


def paired(d):
    rows = []
    for (task, prot, k), g in d.groupby(["task", "protocol", "k"]):
        w = g.pivot_table(index="split", columns="model", values="rmse")
        r2 = g.pivot_table(index="split", columns="model", values="r2")
        pairs = PAIRS + [(OURS, "Best local (per split, post-hoc)", "post-hoc: affine vs best local"),
                         (OURS, "Best benchmark (per split, post-hoc)", "post-hoc: affine vs best benchmark")]
        for c, ref, lab in pairs:
            if c not in w or ref not in w:
                continue
            ok = w[[c, ref]].dropna()
            if not len(ok):
                continue
            rel = 1 - ok[c] / ok[ref]
            dr2 = (r2.loc[ok.index, c] - r2.loc[ok.index, ref])
            rows.append({"task": task, "protocol": prot, "k": k, "comparison": lab, "candidate": c, "reference": ref,
                         "splits": len(ok), "rmse_candidate_mean": ok[c].mean(), "rmse_reference_mean": ok[ref].mean(),
                         "rel_rmse_reduction_mean_pct": 100 * rel.mean(), "rel_rmse_reduction_median_pct": 100 * rel.median(),
                         "ratio_of_means_pct": 100 * (1 - ok[c].mean() / ok[ref].mean()),
                         "wins": int((ok[c] < ok[ref]).sum()),
                         "r2_candidate_mean": r2.loc[ok.index, c].mean(), "r2_reference_mean": r2.loc[ok.index, ref].mean(),
                         "r2_diff_points_mean": 100 * dr2.mean()})
    return pd.DataFrame(rows)


def bench_tables(d):
    out = {}
    for (task, prot), g in d.groupby(["task", "protocol"]):
        t = g.groupby(["model", "k"]).agg(r2=("r2", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"),
                                          splits=("split", "nunique")).reset_index()
        wide = t.pivot(index="model", columns="k", values=["r2", "rmse", "splits"])
        out[(task, prot)] = wide
        wide.to_csv(OUT / f"benchmark_table_{task}_{prot}.csv")
    return out


def figures(d):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("GBR (target-only)", "#7f7f7f", "o"), ("NNW[LDF]", "#d62728", "s"), ("Best benchmark (per split, post-hoc)", "#ff9896", "^"),
            ("Local Extra Trees", "#2ca02c", "D"), (OURS, "#1f77b4", "o"), ("Source-affine[LDF] (ours)", "#17becf", "v")]
    # k ablation: P-B and P-S per task
    combos = [(t, p) for t in ("yrd", "aus") for p in ("B", "S") if ((d.task == t) & (d.protocol == p)).any()]
    if combos:
        fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 3.6), squeeze=False)
        for ax, (t, p) in zip(axes[0], combos):
            g = d[(d.task == t) & (d.protocol == p)]
            for m, col, mk in show:
                h = g[g.model == m].groupby("k").rmse.mean()
                if len(h):
                    ax.plot(h.index, h.values, marker=mk, color=col, label=m.replace(" (per split, post-hoc)", " (post-hoc)"))
            ax.set_title({"yrd": "T1 YRD", "aus": "T2 Australia"}[t] + {"B": ", benchmark protocol", "S": ", later-period protocol"}[p], fontsize=10)
            ax.set_xlabel("labelled target FUAs (k)"); ax.set_ylabel("mean RMSE over splits")
            ax.set_xticks(sorted(g.k.unique()))
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(OUT / "fig_k_ablation.png", dpi=200); plt.close(fig)
    # m ablation: months of target labels, P-S as m = 72
    mm = d[d.protocol.str.match(r"^M\d+$") | (d.protocol == "S")].copy()
    if len(mm):
        mm["m"] = np.where(mm.protocol == "S", 72, mm.protocol.str.extract(r"M(\d+)")[0].astype(float))
        tasks = [(t, k) for t in ("yrd", "aus") for k in sorted(mm[mm.task == t].k.unique())
                 if ((mm.task == t) & (mm.k == k) & (mm.protocol != "S")).any()]
        if tasks:
            fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 3.6), squeeze=False)
            for ax, (t, k) in zip(axes[0], tasks):
                g = mm[(mm.task == t) & (mm.k == k)]
                for m, col, mk in show:
                    h = g[g.model == m].groupby("m").rmse.mean()
                    if len(h):
                        ax.plot(h.index, h.values, marker=mk, color=col, label=m.replace(" (per split, post-hoc)", " (post-hoc)"))
                ax.set_xscale("log"); ax.set_xticks([3, 6, 12, 72]); ax.set_xticklabels(["3", "6", "12", "72"])
                ax.set_title(f"{ {'yrd': 'T1 YRD', 'aus': 'T2 Australia'}[t] }, k = {k}", fontsize=10)
                ax.set_xlabel("months of target labels (m)"); ax.set_ylabel("mean RMSE, 2021-2022")
            axes[0][0].legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(OUT / "fig_m_ablation.png", dpi=200); plt.close(fig)


def main():
    d = add_best(load())
    d.to_csv(OUT / "all_split_metrics.csv", index=False)
    tabs = bench_tables(d)
    pr = paired(d); pr.to_csv(OUT / "paired.csv", index=False)
    figures(d)
    lines = ["# Benchmark-aligned results (auto-generated)", ""]
    cnt = d[~d.model.str.contains("per split")].groupby(["task", "protocol", "k"]).split.nunique()
    lines += ["## Completed splits", "", cnt.to_frame("splits").to_markdown(), ""]
    for (task, prot), wide in tabs.items():
        lines += [f"## {task} / {prot}: mean R2 and RMSE over splits", ""]
        show = wide[["r2", "rmse"]].round(3)
        show.columns = [f"{a}_k{b}" for a, b in show.columns]
        lines += [show.to_markdown(), ""]
    lines += ["## Paired comparisons", "", pr.round(3).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
