"""Appendix tables for the exploratory checks (explore/README.md).

Usage: python explore/make_explore_report.py <explore_dir>
Writes tab_explore_ablation.tex, tab_explore_hybrid.tex and tab_explore_windows.tex into <explore_dir>. The hybrid
caption reads the Australian label counts from <explore_dir>/aus_hybrid_predictions.csv.gz; no model is refitted.
"""
import json
import sys
from pathlib import Path

import pandas as pd

E = Path(sys.argv[1])
abl = json.loads((E / "ablation_summary.json").read_text())
mech = json.loads((E / "mechanism.json").read_text())
hyb = pd.read_csv(E / "hybrid_gains.csv")


def f1(v):
    return f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}"


def f2(v):
    return f"$-${abs(v):.2f}" if v < 0 else f"{v:.2f}"


def ci(g, lo, hi):
    return f"{f1(g)} [{f1(lo)}, {f1(hi)}]"


# ---- condition-group ablation
NAMES = {"no meteorology": "Meteorology (temperature, diurnal range, humidity, wind, precipitation)",
         "no fire": "Fire (log fire count within 100 km)", "no season": "Season (local-season sine and cosine)",
         "no conditions": "All conditions (level and scale only)"}
lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
         r"\caption{Where to adapt by condition group (exploratory). The ECA-S prior is re-learnt without one group "
         r"of conditions and compared with the full prior on the 216 within-China pseudo-transfers, in the three-fold "
         r"region cross-validation of Section~\ref{eca-method} (learnt on six development regions and evaluated on the "
         r"other three). Entries are the mean paired "
         r"RMSE reduction (\%) of the full prior over the reduced prior on the held-out tasks, with a 95\% interval "
         r"from 1,000 bootstrap draws of the nine regions, and the number of regions in which the full prior is "
         r"more accurate; positive values mean that the removed group carries useful correction.}",
         r"\label{tab:explore-ablation}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}p{0.52\linewidth}rr@{}}", r"\toprule",
         r"Removed group & Full prior better by (\%) & Regions \\", r"\midrule"]
for key, label in NAMES.items():
    v = abl["variants"][key]
    lines.append(f"{label} & {ci(v['full_vs_variant_gain'], v['lo'], v['hi'])} & "
                 f"{v['regions_full_better']}/{v['regions']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(E / "tab_explore_ablation.tex").write_text("\n".join(lines) + "\n")

# ---- hybrid strategy against the most recent months
# Some Australian FUAs miss part of late 2020: the last-months design (S0) then falls short of its budget, while the
# hybrid fills its coverage part, so it can hold one label more. The caption states this for the Australian rows.
_p = pd.read_csv(E / "aus_hybrid_predictions.csv.gz").groupby(["m", "k", "split"]).first()
_d = (_p["H recent + target-aware|0|n_rows"] - _p["S0 last consecutive|0|n_rows"]).xs(3, level="k")
assert set(_d.unique()) <= {0, 1}
_n = {m: int((_d.xs(m, level="m") > 0).sum()) for m in (3, 6)}
assert _n[3] == _n[6]
AUS_NOTE = (f"In Australia, the hybrid received one label more than the last-months design in {_n[3]} of the 20 "
            f"splits shown for each $m$; see the text for a matched-budget check. ")
TASKS = [("yrd", 5, "T1 YRD, $k=5$"), ("locked", 5, "T3 north-west China, $k=5$"), ("aus", 3, "T2 Australia, $k=3$")]
lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
         r"\caption{Which months to label: the hybrid strategy (exploratory). Each labelled FUA is assigned its "
         r"$\lfloor m/2\rfloor$ most recent months of 2020 and the remaining months by target-aware coverage. "
         + AUS_NOTE + r"Entries "
         r"are mean paired RMSE reductions (\%) of the hybrid over the last $m$ months (S0) and over random months "
         r"(S1), with 95\% crossed bootstrap intervals; positive values favour the hybrid.}",
         r"\label{tab:explore-hybrid}", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}llllll@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Hybrid vs last months} & \multicolumn{2}{c}{Hybrid vs random months} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Target & Estimator & $m=3$ & $m=6$ & $m=3$ & $m=6$ \\", r"\midrule"]
for task, k, label in TASKS:
    first = True
    for method in ("Source-affine", "ECA-R", "ECA-S"):
        cells = []
        for comp in ("H vs S0", "H vs S1"):
            for m in (3, 6):
                r = hyb[(hyb.task == task) & (hyb.k == k) & (hyb.m == m) & (hyb.method == method)
                        & (hyb.comparison == comp)].iloc[0]
                cells.append(ci(r.gain, r.lo, r.hi))
        lines.append(f"{label if first else ''} & {method} & " + " & ".join(cells) + r" \\")
        first = False
    if task != "aus":
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
(E / "tab_explore_hybrid.tex").write_text("\n".join(lines) + "\n")

# ---- three-month label windows across 2019-2020
TT = [("yrd", "T1 YRD"), ("locked", "T3 north-west China"), ("aus", "T2 Australia")]
lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
         r"\caption{Position of a three-month label window (exploratory). For the 22 windows ending between March "
         r"2019 and December 2020 ($k=5$, 20 splits): the range of mean 2021--2022 RMSE ($\mu$g/m$^3$), the rank of "
         r"the most recent window (October--December 2020) and the rank correlation between window position and "
         r"RMSE (negative: later windows are more accurate).}",
         r"\label{tab:explore-windows}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Target & Estimator & Best & Worst & Rank of latest & Rank correlation \\", r"\midrule"]
for task, label in TT:
    first = True
    for method in ("Source-affine", "ECA-R", "ECA-S"):
        w = mech["window_scan"][task][method]
        lines.append(f"{label if first else ''} & {method} & {w['min']:.2f} & {w['max']:.2f} & "
                     f"{w['rank_last']}/{w['windows']} & {f2(w['spearman_rho'])} \\\\")
        first = False
    if task != "aus":
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(E / "tab_explore_windows.tex").write_text("\n".join(lines) + "\n")
print("wrote", sorted(p.name for p in E.glob("tab_explore_*.tex")))
