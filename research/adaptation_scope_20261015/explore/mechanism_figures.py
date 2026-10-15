"""Exploratory, added after the T3 run (explore/README.md, item 3): two mechanism figures.

(a) Monthly mean bias of the current source model (prediction minus observation) over the eligible target FUAs,
    January 2019 to December 2022, for T1 (YRD), T3 (north-west China) and T2 (Australia); with the distance between
    each candidate label month's bias (2019-2020) and the mean test-period bias (2021-2022).
(b) The ECA-R condition correction against log fire count in the Australian October-December 2019 window: for each of
    the 20 splits at k = 5, the correction is evaluated at the median test-row conditions with the fire term varied;
    curve = mean over splits, band = 10th-90th percentile over splits.
Panels (a)-(c) of the figure plot explore/window_scan.csv (explore/README.md, item 4); (a) above is reported in
mechanism.json only.
Usage: python explore/mechanism_figures.py <theta.json> <out_dir>
Writes mechanism.json and fig_adapt_mechanism.png.
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_lib as al
from adapt_lib import X, Y, core

theta_path, OUT = Path(sys.argv[1]), Path(sys.argv[2])
th = json.loads(theta_path.read_text())["ARD"]
THETA = dict(ls=np.array(th["ls"]), af=th["af"], an=th["an"])
ECA_R = "#d35400"
FIRE = al.COND.index("log_fire_count_100km")
NAMES = {"yrd": "T1 YRD", "locked": "T3 north-west China", "aus": "T2 Australia"}
res = {"note": "exploratory, after the T3 run", "bias": {}, "fire": {}}

# ---- (a) monthly bias of the current source model
for task in ("yrd", "locked", "aus"):
    src, tgt, eligible = al.load(task)
    f = al.source_model(src)
    tgt["raw"] = core.pos(f.predict(tgt[X]))
    tgt["err"] = tgt.raw - tgt[Y]
    win = tgt[tgt.year.between(2019, 2022)]
    monthly = win.groupby(["year", "month"]).err.mean()
    test_bias = float(tgt[tgt.year.ge(2021)].err.mean())
    cand = monthly.loc[[(y, m) for y in (2019, 2020) for m in range(1, 13)]]
    dist = (cand - test_bias).abs()
    last3 = float(tgt[tgt.year.eq(2020) & tgt.month.ge(10)].err.mean())
    res["bias"][task] = dict(test_mean_bias=round(test_bias, 4), oct_dec_2020_bias=round(last3, 4),
                             gap_last3=round(abs(last3 - test_bias), 2),
                             median_gap_candidate_months=round(float(dist.median()), 2),
                             rank_of_dec_2020_gap=int((dist <= dist.loc[(2020, 12)]).sum()),
                             mean_bias_2019=round(float(tgt[tgt.year.eq(2019)].err.mean()), 4),
                             mean_bias_2020=round(float(tgt[tgt.year.eq(2020)].err.mean()), 4))
    mon = cand.to_numpy()                       # the 24 candidate months in calendar order
    for m in (3, 6):
        last = abs(mon[-m:].mean() - test_bias)
        gaps = np.array([abs(mon[list(c)].mean() - test_bias) for c in itertools.combinations(range(24), m)])
        res["bias"][task][f"m{m}_last_gap"] = round(float(last), 2)
        res["bias"][task][f"m{m}_share_random_subsets_with_larger_gap"] = round(float((gaps > last).mean()), 3)
    print(task, res["bias"][task], flush=True)

# ---- (b) ECA-R correction against fire count, Australia, labels October-December 2019
src, tgt, eligible = al.load("aus")
f = al.source_model(src)
tgt["raw"] = core.pos(f.predict(tgt[X]))
Z = al.zmat(tgt)
t_of = tgt.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
ta, tb = int(t_of[(2019, 10)]), int(t_of[(2019, 12)])
mu, sd = float(al.Z_MU.iloc[FIRE]), float(al.Z_SD.iloc[FIRE])
grid = np.linspace(0, float(tgt.log_fire_count_100km.max()), 60)
curves, lab_fire, test_fire, g_test, g_lab = [], [], [], [], []
for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, 5, 20)):
    trn = (tgt.eFUA_ID.isin(train_ids) & tgt.t.between(ta, tb)).to_numpy()
    tst = (tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy()
    a, b = tgt[trn], tgt[tst]
    y, raw, eraw = a[Y].to_numpy(), a.raw.to_numpy(), b.raw.to_numpy()
    aff_l, aff_t = al.affine(raw, y, eraw)
    zg = np.repeat(np.median(Z[tst], 0)[None, :], len(grid), 0)
    zg[:, FIRE] = (grid - mu) / sd
    g, _ = al.eca_predict("ECA-fixed", THETA, Z[trn], y - aff_l, np.vstack([zg, Z[tst], Z[trn]]), y.mean())
    curves.append(g[:len(grid)])
    g_test.append(g[len(grid):len(grid) + tst.sum()]); g_lab.append(g[len(grid) + tst.sum():])
    lab_fire.append(a.log_fire_count_100km.to_numpy()); test_fire.append(b.log_fire_count_100km.to_numpy())
curves = np.array(curves)
lab_fire, test_fire = np.concatenate(lab_fire), np.concatenate(test_fire)
res["fire"] = dict(splits=len(curves), k=5,
                   median_log_fire_label=round(float(np.median(lab_fire)), 2),
                   median_log_fire_test=round(float(np.median(test_fire)), 2),
                   share_test_rows_below_label_median=round(float(np.mean(test_fire < np.median(lab_fire))), 3),
                   correction_at_test_median=round(float(np.interp(np.median(test_fire), grid, curves.mean(0))), 2),
                   correction_at_label_median=round(float(np.interp(np.median(lab_fire), grid, curves.mean(0))), 2),
                   mean_correction_test_rows=round(float(np.mean(np.concatenate(g_test))), 4),
                   mean_correction_label_rows=round(float(np.mean(np.concatenate(g_lab))), 2))
print("fire", res["fire"], flush=True)
(OUT / "mechanism.json").write_text(json.dumps(res, indent=1))

# ---- window scan summary (explore/window_scan.py must have been run)
from scipy.stats import spearmanr
ws = pd.read_csv(OUT / "window_scan.csv")
wmean = ws.groupby(["task", "model", "end"]).rmse.mean()
res["window_scan"] = {}
for task in ("yrd", "locked", "aus"):
    res["window_scan"][task] = {}
    for model in ("Source-affine", "ECA-R", "ECA-S"):
        s = wmean.loc[(task, model)]
        rho, pv = spearmanr(np.arange(len(s)), s.to_numpy())
        res["window_scan"][task][model] = dict(windows=len(s), last=round(float(s.iloc[-1]), 2),
                                               rank_last=int((s < s.iloc[-1]).sum()) + 1,
                                               min=round(float(s.min()), 2), min_end=str(s.idxmin()),
                                               max=round(float(s.max()), 2), max_end=str(s.idxmax()),
                                               median=round(float(s.median()), 2),
                                               spearman_rho=round(float(rho), 2), spearman_p=round(float(pv), 3))
    res["window_scan"][task]["Direct"] = round(float(wmean.loc[(task, "Direct")].iloc[0]), 2)
(OUT / "mechanism.json").write_text(json.dumps(res, indent=1))

# ---- figure, in the style of fig_adapt_budget.png (make_adapt_report.py)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SHOW = [("Source-affine", "Source-affine (global recalibration)", "#1f4e9c", "o"),
        ("ECA-R", "ECA-R (recalibrate, then correct by condition)", "#d35400", "s"),
        ("ECA-S", "ECA-S (anchored to the source model)", "#7d3c98", "^")]
PANELS = [("yrd", "(a) T1 YRD"), ("locked", "(b) T3 north-west China"), ("aus", "(c) T2 Australia")]
ends = sorted(ws.end.unique())
xi = np.arange(len(ends))
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
for ax, (task, title) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], PANELS):
    ax.axvspan(len(ends) - 1.5, len(ends) - 0.5, color="#f2f2f2", lw=0, zorder=0)
    if task == "aus":
        bs = [i for i, e in enumerate(ends) if "2019-12" <= e <= "2020-03"]
        ax.axvspan(bs[0] - 0.5, bs[-1] + 0.5, color="#fbe5d6", lw=0, zorder=0)
        ax.text((bs[0] + bs[-1]) / 2, 0.03, "windows with\nBlack Summer\nmonths", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#5a5a5a")
    for model, lab, col, mk in SHOW:
        ys = [wmean.loc[(task, model, e)] for e in ends]
        ax.plot(xi, ys, color=col, marker=mk, ms=3, lw=1.3, label=lab)
    ax.axhline(wmean.loc[(task, "Direct")].iloc[0], color="#333333", ls="-.", lw=1.2,
               label="Direct transfer (no target labels)")
    ticks = [ends.index(e) for e in ("2019-06", "2019-12", "2020-06", "2020-12")]
    ax.set_xticks(ticks, ["Jun 2019", "Dec 2019", "Jun 2020", "Dec 2020"])
    ax.set_xlim(-0.5, len(ends) - 0.5)
    ax.set_xlabel("last of the three labelled months")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_ylabel("mean 2021–2022 RMSE (µg/m³)")
    if task == "aus":
        ax.set_yscale("log"); ax.set_yticks([1.5, 2, 3, 5, 10, 15]); ax.set_yticklabels(["1.5", "2", "3", "5", "10", "15"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("mean RMSE, log scale (µg/m³)")
ax = axes[1, 1]
lo, hi = np.percentile(curves, [10, 90], 0)
ax.fill_between(grid, lo, hi, color=ECA_R, alpha=0.18, lw=0)
ax.plot(grid, curves.mean(0), color=ECA_R, lw=1.6)
ax.axhline(0, color="#9a9a9a", lw=0.6)
ymin = min(lo.min(), 0); ymax = hi.max(); span = ymax - ymin
r1, r2 = ymin - 0.12 * span, ymin - 0.27 * span
ax.plot(test_fire, np.full(len(test_fire), r1), "|", color="#7a7a7a", ms=5, alpha=0.3, mew=0.5)
ax.plot(lab_fire, np.full(len(lab_fire), r2), "|", color=ECA_R, ms=5, alpha=0.6, mew=0.6)
ax.text(0.0, r1 + 0.03 * span, "2021–2022 test months", ha="left", va="bottom", fontsize=7, color="#4a4a4a")
ax.text(0.0, r2 + 0.03 * span, "Oct–Dec 2019 label months", ha="left", va="bottom", fontsize=7, color="#4a4a4a")
ax.set_ylim(r2 - 0.08 * span, ymax + 0.05 * span)
ax.set_xlabel("log fire count within 100 km")
ax.set_ylabel("ECA-R condition correction (µg/m³)")
ax.set_title("(d) T2 Australia: correction learnt from\nOct–Dec 2019 labels, $k=5$", fontsize=9)
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT / "fig_adapt_mechanism.png", dpi=220); plt.close(fig)
print("saved figure")
