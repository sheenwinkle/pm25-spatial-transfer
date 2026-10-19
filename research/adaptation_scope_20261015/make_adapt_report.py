"""Tables, figure and numbers for the adaptation-scope and label-coverage chapter.

Usage: python make_adapt_report.py <results_dir> <summary_dir> <out_dir>
Writes tab_adapt_rmse.tex, tab_adapt_confirm.tex, tab_adapt_ablation.tex, tab_adapt_labels.tex,
fig_adapt_budget.png and adapt_numbers.json. The label-selection caption reads the Australian label counts from
<results_dir>/aus_q2_selected.csv; no model is refitted.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

RES, SUM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
LOCAL = ["Local mean", "Local ridge", "Local RBF", "Local RF", "Local Extra Trees", "Local LightGBM"]
means = pd.read_csv(SUM / "q1_means.csv")
gains = pd.read_csv(SUM / "q1_gains.csv")
q2g = pd.read_csv(SUM / "q2_gains.csv")
q2m = pd.read_csv(SUM / "q2_means.csv")
NAME = {"ECA2": "ECA-S", "ECA-fixed": "ECA-R", "Best local six": "Best local"}
num = {}


def mean_rmse(task, design, cell, k, model):
    s = means[(means.task == task) & (means.design == design) & (means.cell == cell) & (means.k == k)
              & (means.model == model)].mean_rmse
    assert len(s) == 1, (task, design, cell, k, model)
    return float(s.iloc[0])


def gain(task, design, cell, k, comparison):
    s = gains[(gains.task == task) & (gains.design == design) & (gains.cell == cell) & (gains.k == k)
              & (gains.comparison == comparison)]
    assert len(s) == 1, (task, design, cell, k, comparison)
    return s.iloc[0]


def fmt_gain(r):
    def neg(v):
        s = f"{abs(v):.2f}" if 0 < abs(v) < 0.05 else f"{abs(v):.1f}"    # keep the sign of bounds near zero
        return f"$-${s}" if v < 0 else s
    return f"{neg(r.gain)} [{neg(r.lo)}, {neg(r.hi)}] ({int(r.wins)})"


# ------------------------------------------------------------------ Table A: mean RMSE
BLOCKS = [("yrd", "history", "T1 YRD, current source", (5, 11)),
          ("yrd", "historyN", "T1 YRD, source up to 2020", (5, 11)),
          ("aus", "history", "T2 Australia, current source", (3, 5)),
          ("locked", "history", "T3 north-west China, current source", (5, 11)),
          ("locked", "historyN", "T3 north-west China, source up to 2020", (5, 11))]
COLS = ["Direct", "Source-affine", "Best local six", "ECA-fixed", "ECA2"]
lines = []
for task, design, label, ks in BLOCKS:
    lines.append(r"\multicolumn{8}{@{}l}{\emph{" + label + "}} \\\\")
    for k in ks:
        for m in (3, 6, 12):
            vals = [mean_rmse(task, design, f"M{m}", k, c) for c in COLS]
            best = min(vals)
            cells = [(r"\textbf{%.2f}" % v) if abs(v - best) < 5e-3 else f"{v:.2f}" for v in vals]
            lines.append(f"\\quad & {k} & {m} & " + " & ".join(cells) + r" \\")
            for c, v in zip(COLS, vals):
                num[f"rmse|{task}|{design}|M{m}|k{k}|{NAME.get(c, c)}"] = round(v, 2)
    lines.append(r"\addlinespace")
lines = lines[:-1]
tab = r"""\begin{table}[H]
\centering
\footnotesize
\caption{How much to adapt: mean 2021--2022 RMSE ($\mu$g/m$^3$) over the 20 benchmark splits, with $k$ labelled target FUAs observed in the last $m$ months of 2020. Direct transfer uses no target labels. Best local is the per-split post-hoc minimum of the six local learners. ECA-R recalibrates globally and then adds the evidence-controlled condition correction; ECA-S shrinks every adjustment toward the unadapted source model. ``Source up to 2020'' fits the source model without 2021--2022 source labels. T1 and T2 are development tasks; T3 is the locked confirmation region, run once. Bold marks the lowest value in each row.}
\label{tab:adapt-rmse}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Task and source model & $k$ & $m$ & Direct & Source-affine & Best local & ECA-R & ECA-S \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_adapt_rmse.tex").write_text(tab)

# ------------------------------------------------------------------ Table B: confirmatory comparisons on T3
CONF = [("history", "ECA2 vs Direct", "ECA-S vs direct transfer"),
        ("history", "ECA2 vs Source-affine", "ECA-S vs source-affine"),
        ("history", "Source-affine vs Direct", "Source-affine vs direct transfer"),
        ("historyN", "ECA-fixed vs Source-affine", "ECA-R vs source-affine"),
        ("historyN", "ECA-fixed vs Direct", "ECA-R vs direct transfer"),
        ("historyN", "ECA2 vs Direct", "ECA-S vs direct transfer")]
lines = []
prev = None
for design, comp, label in CONF:
    if design != prev:
        head = "Current source model" if design == "history" else "Source model fitted up to 2020"
        lines.append(r"\multicolumn{4}{@{}l}{\emph{" + head + "}} \\\\")
        prev = design
    for m in (3, 6, 12):
        cells = []
        for k in (5, 11):
            r = gain("locked", design, f"M{m}", k, comp)
            cells.append(fmt_gain(r))
            num[f"gain|locked|{design}|M{m}|k{k}|{comp}"] = [round(r.gain, 1), round(r.lo, 1), round(r.hi, 1),
                                                           int(r.wins)]
        lines.append(f"\\quad {label if m == 3 else ''} & {m} & " + " & ".join(cells) + r" \\")
tab = r"""\begin{table}[H]
\centering
\footnotesize
\caption{Pre-declared confirmatory comparisons on the locked region T3. Entries are mean paired RMSE reductions (\%) of the first-named estimator over the 20 splits, with 95\% crossed FUA-by-quarter bootstrap intervals and the number of splits won; positive values favour the first-named estimator.}
\label{tab:adapt-confirm}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrll@{}}
\toprule
Comparison & $m$ & $k=5$ & $k=11$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_adapt_confirm.tex").write_text(tab)

# ------------------------------------------------------------------ Table C: ablations (T3 and T1)
ABL = [("history", "ECA2 vs ECA2-affine", "ECA-S vs no condition term"),
       ("history", "ECA2 vs ECA2-isotropic", "ECA-S vs one length-scale"),
       ("history", "ECA2 vs ECA2-no-meta", "ECA-S vs no meta-learning"),
       ("history", "ECA2 vs ECA2-unbounded", "ECA-S vs prior/noise ratio $\\times 10$"),
       ("history", "ECA2 vs ECA2-MAP", "ECA-S vs amount set on target labels"),
       ("historyN", "ECA-fixed vs ECA-no-meta", "ECA-R vs no meta-learning"),
       ("historyN", "ECA-fixed vs ECA-unbounded", "ECA-R vs prior/noise ratio $\\times 100$"),
       ("historyN", "ECA-fixed vs ECA", "ECA-R vs amount set on target labels")]
lines = []
prev = None
for design, comp, label in ABL:
    if design != prev:
        head = "Current source model" if design == "history" else "Source model fitted up to 2020"
        lines.append(r"\multicolumn{5}{@{}l}{\emph{" + head + "}} \\\\")
        prev = design
    for m in (3, 6):
        cells = []
        for k in (5, 11):
            r = gain("locked", design, f"M{m}", k, comp)
            cells.append(fmt_gain(r))
            num[f"gain|locked|{design}|M{m}|k{k}|{comp}"] = [round(r.gain, 1), round(r.lo, 1), round(r.hi, 1),
                                                           int(r.wins)]
        r1 = gain("yrd", design, f"M{m}", 5, comp)
        num[f"gain|yrd|{design}|M{m}|k5|{comp}"] = [round(r1.gain, 1), round(r1.lo, 1), round(r1.hi, 1), int(r1.wins)]
        lines.append(f"\\quad {label if m == 3 else ''} & {m} & " + " & ".join(cells) + f" & {fmt_gain(r1)}" + r" \\")
tab = r"""\begin{table}[H]
\centering
\footnotesize
\caption{Where and how much to adapt: ablations. Entries as in Table~\ref{tab:adapt-confirm}; positive values favour the full estimator. The weak-shrinkage rows multiply the ratio of the prior to the noise standard deviation by 10 for ECA-S and by 100 for ECA-R (amplitude $\times 10$, noise SD $\div 10$). T3 columns are confirmation results; the last column gives the development result on T1 at $k=5$.}
\label{tab:adapt-ablation}
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{@{}lrlll@{}}
\toprule
 & & \multicolumn{2}{c}{T3 (confirmation)} & T1 (development) \\
\cmidrule(lr){3-4}\cmidrule(lr){5-5}
Comparison & $m$ & $k=5$ & $k=11$ & $k=5$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
(OUT / "tab_adapt_ablation.tex").write_text(tab)

# ------------------------------------------------------------------ Table D: label selection
# Some Australian FUAs miss part of late 2020, so the last-months design (S0) can receive fewer labels than its
# budget; every other strategy is full. The caption reports this for the Australian cell shown in the table.
sel = pd.read_csv(RES / "aus_q2_selected.csv")
s0 = sel[(sel.strategy == "S0 last consecutive") & (sel.m == 3) & (sel.k == 3)]
assert len(s0) == 20
others = sel[sel.strategy != "S0 last consecutive"]
assert bool((others.n == others.m * others.k).all())
n_short, n_mean = int((s0.n < 9).sum()), float(s0.n.mean())
num["q2labels|aus|m3|k3|S0"] = [n_short, round(n_mean, 2)]
AUS_NOTE = (f"In Australia, the last-months design received fewer labels than its budget in {n_short} of the 20 "
            f"splits shown ({n_mean:.2f} of 9 on average); see the text for a matched-budget check. ")
STRAT = ["S0 last consecutive", "S1 random", "S2 season-spread", "S3 condition coverage", "S4 target-aware"]
SLAB = {"S0 last consecutive": "Last $m$ months of 2020", "S1 random": "Random months",
        "S2 season-spread": "Spread over seasons", "S3 condition coverage": "Condition coverage",
        "S4 target-aware": "Target-aware coverage"}
METH = ["Source-affine", "ECA-R", "ECA-S", "Best local six"]
TASKS = [("yrd", 3, 5, "T1 YRD (development), $m=3$, $k=5$"),
         ("aus", 3, 3, "T2 Australia (development), $m=3$, $k=3$"),
         ("locked", 3, 5, "T3 north-west China (confirmation), $m=3$, $k=5$"),
         ("locked", 6, 5, "T3 north-west China (confirmation), $m=6$, $k=5$")]
lines = []
for task, m, k, label in TASKS:
    d = q2m[(q2m.task == task) & (q2m.m == m) & (q2m.k == k)]
    direct = float(d[d.method == "Direct"].mean_rmse.iloc[0])
    lines.append(r"\multicolumn{5}{@{}l}{\emph{" + label + f"; direct transfer {direct:.2f}" + "}} \\\\")
    num[f"q2rmse|{task}|m{m}|k{k}|Direct"] = round(direct, 2)
    for st in STRAT:
        vals = []
        for me in METH:
            v = float(d[(d.method == me) & (d.strategy == st)].mean_rmse.iloc[0])
            vals.append(v)
            num[f"q2rmse|{task}|m{m}|k{k}|{NAME.get(me, me)}|{st[:2]}"] = round(v, 2)
        lines.append(f"\\quad {SLAB[st]} & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    lines.append(r"\addlinespace")
lines = lines[:-1]
tab = r"""\begin{table}[H]
\centering
\footnotesize
\caption{Which months to label: mean 2021--2022 RMSE ($\mu$g/m$^3$) when each labelled FUA is assigned $m$ labelled months chosen from 2019--2020 by each strategy. """ + AUS_NOTE + r"""Random and season-spread values average five draws. Condition coverage maximises the spread of each FUA's labelled conditions; target-aware coverage chooses the months that most reduce the correction's uncertainty over the conditions to be predicted.}
\label{tab:adapt-labels}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Strategy & Source-affine & ECA-R & ECA-S & Best local \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_adapt_labels.tex").write_text(tab)
for task in ("yrd", "aus", "locked"):
    for _, r in q2g[q2g.task == task].iterrows():
        num[f"q2gain|{task}|m{r.m}|k{r.k}|{NAME.get(r.method, r.method)}|{r.comparison}"] = \
            [round(r.gain, 1), round(r.lo, 1), round(r.hi, 1), int(r.wins)]

# ------------------------------------------------------------------ all other gains used in the text
for _, r in gains.iterrows():
    key = f"gain|{r.task}|{r.design}|{r.cell}|k{r.k}|{r.comparison}"
    num.setdefault(key, [round(r.gain, 1), round(r.lo, 1), round(r.hi, 1), int(r.wins)])
arch = pd.read_csv(SUM / "q1_archived.csv")
for _, r in arch.iterrows():
    num[f"archived|{r.task}|{r.cell}|k{r.k}|{r.comparison}"] = [round(r.gain, 1), int(r.wins), round(r.mean_a, 2),
                                                                round(r.mean_b, 2)]
eff = pd.read_csv(SUM / "q1_label_efficiency.csv")
for _, r in eff.iterrows():
    num[f"eff|{r.task}|{r.design}|k{r.k}|{r.comparison}"] = [round(r.gain, 1), round(r.lo, 1), round(r.hi, 1),
                                                              int(r.wins)]
for _, r in means.iterrows():
    num.setdefault(f"rmse|{r.task}|{r.design}|{r.cell}|k{r.k}|{NAME.get(r.model, r.model)}", round(r.mean_rmse, 2))
(OUT / "adapt_numbers.json").write_text(json.dumps(num, indent=0, sort_keys=True))

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SHOW = [("Source-affine", "Source-affine (global recalibration)", "#1f4e9c", "o", "-"),
        ("ECA-fixed", "ECA-R (recalibrate, then correct by condition)", "#d35400", "s", "-"),
        ("ECA2", "ECA-S (anchored to the source model)", "#7d3c98", "^", "-"),
        ("Best local six", "Per-split post-hoc best of six local learners", "#27ae60", "D", ":")]
PANELS = [("yrd", "history", 5, "T1 YRD (development)\ncurrent source model, $k=5$"),
          ("aus", "history", 3, "T2 Australia (development)\ncurrent source model, $k=3$"),
          ("locked", "history", 5, "T3 north-west China (confirmation)\ncurrent source model, $k=5$"),
          ("locked", "historyN", 5, "T3 north-west China (confirmation)\nsource model fitted up to 2020, $k=5$")]
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
for ax, (task, design, k, title) in zip(axes.ravel(), PANELS):
    xs = [3, 6, 12]
    for model, lab, col, mk, ls in SHOW:
        ys = [mean_rmse(task, design, f"M{m}", k, model) for m in xs]
        ax.plot(xs, ys, color=col, marker=mk, ls=ls, ms=5, lw=1.6, label=lab)
    d = mean_rmse(task, design, "M3", k, "Direct")
    ax.axhline(d, color="#333333", ls="-.", lw=1.2, label="Direct transfer (no target labels)")
    ax.set_xticks(xs); ax.set_xlabel("labelled months per FUA, $m$")
    if ax in axes[:, 0]:
        ax.set_ylabel("mean 2021–2022 RMSE (µg/m³)")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    if task == "aus":
        ax.set_yscale("log"); ax.set_yticks([1.5, 2, 3, 5, 10, 15]); ax.set_yticklabels(["1.5", "2", "3", "5", "10", "15"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("mean RMSE, log scale (µg/m³)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT / "fig_adapt_budget.png", dpi=220); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.iterdir()))
