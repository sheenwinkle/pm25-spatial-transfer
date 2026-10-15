"""Check every number of the adaptation study quoted in main.tex against the saved summaries.

Usage: python check_adapt_claims.py <adapt_dir> <main.tex>
Each check recomputes a value from summary/*.csv, meta/*.json, aoa/di_summary.json or results/*.csv.gz, formats
it as the text does and requires that string (with the stated context) to occur in main.tex.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, TEX = Path(sys.argv[1]), Path(sys.argv[2])
text = re.sub(r"\s+", " ", TEX.read_text(encoding="utf-8"))
g = pd.read_csv(A / "summary/q1_gains.csv"); m = pd.read_csv(A / "summary/q1_means.csv")
q2 = pd.read_csv(A / "summary/q2_gains.csv"); q2m = pd.read_csv(A / "summary/q2_means.csv")
arch = pd.read_csv(A / "summary/q1_archived.csv")
th = json.loads((A / "meta/theta.json").read_text()); th2 = json.loads((A / "meta/theta2.json").read_text())
di = json.loads((A / "aoa/di_summary.json").read_text())
results = []


def f1(v):
    return f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}"


def G(task, design, cell, k, comp):
    r = g[(g.task == task) & (g.design == design) & (g.cell == cell) & (g.k == k) & (g.comparison == comp)]
    assert len(r) == 1, (task, design, cell, k, comp)
    return r.iloc[0]


def M(task, design, cell, k, model):
    r = m[(m.task == task) & (m.design == design) & (m.cell == cell) & (m.k == k) & (m.model == model)]
    assert len(r) == 1, (task, design, cell, k, model)
    return float(r.mean_rmse.iloc[0])


def Q2(task, mm, k, method, comp):
    r = q2[(q2.task == task) & (q2.m == mm) & (q2.k == k) & (q2.method == method) & (q2.comparison == comp)]
    assert len(r) == 1, (task, mm, k, method, comp)
    return r.iloc[0]


def check(name, needle):
    ok = re.sub(r"\s+", " ", needle) in text
    results.append((name, ok, needle))


def gi(r):            # "10.5\% [2.0, 23.1]"
    return f"{f1(r.gain)}\\% [{f1(r.lo)}, {f1(r.hi)}]"


def rng(vals):
    lo, hi = min(vals), max(vals)
    return f"{f1(lo)}--{f1(hi)}" if lo >= 0 else f"{f1(lo)} to {f1(hi)}"


L = "locked"
# ---- confirmation, current source model
r = G(L, "history", "M3", 5, "ECA2 vs Direct"); check("T3 ECA-S vs direct m3 k5", "by " + gi(r) + " with three months")
r = G(L, "history", "M3", 11, "ECA2 vs Direct"); check("T3 ECA-S vs direct m3 k11", gi(r))
r = G(L, "history", "M6", 5, "ECA2 vs Direct"); check("T3 ECA-S vs direct m6 k5", gi(r))
r = G(L, "history", "M6", 11, "ECA2 vs Direct"); check("T3 ECA-S vs direct m6 k11", gi(r))
r = G(L, "history", "M3", 5, "ECA2 vs Source-affine"); check("T3 ECA-S vs affine m3 k5 (abstract)", f"its {f1(r.gain)}\\% gain over recalibration")
v = [G(L, "history", c, k, "Source-affine vs Direct").gain for c in ("M3", "M6", "M12") for k in (5, 11)]
check("T3 affine vs direct range", f"by {rng(v)}\\%, with every interval")
assert all(G(L, "history", c, k, "Source-affine vs Direct").lo < 0 for c in ("M3", "M6", "M12") for k in (5, 11))
v = [G(L, "history", c, k, "ECA2 vs Source-affine").gain for c in ("M3", "M6") for k in (5, 11)]
check("T3 ECA-S vs affine range", f"by {rng(v)}\\% with three or six months")
assert all(G(L, "history", c, k, "ECA2 vs Source-affine").lo < 0 for c in ("M3", "M6") for k in (5, 11))
v = [M(L, "history", "M12", k, x) for k in (5, 11) for x in ("Source-affine", "ECA2", "ECA-fixed")]
spread = max(max(M(L, "history", "M12", k, x) for x in ("Source-affine", "ECA2", "ECA-fixed")) /
             min(M(L, "history", "M12", k, x) for x in ("Source-affine", "ECA2", "ECA-fixed")) - 1 for k in (5, 11))
results.append(("T3 m12 adapted within 3%", spread < 0.03, f"{spread:.4f}"))
r = G(L, "history", "M3", 5, "ECA2 vs Best local six"); check("T3 ECA-S vs best local", gi(r))
# ---- confirmation, source up to 2020
check("T3 direct stale", f"rises to {M(L, 'historyN', 'M3', 5, 'Direct'):.2f} at five FUAs in T3 and {M('yrd', 'historyN', 'M3', 5, 'Direct'):.2f} in T1")
r = G(L, "historyN", "M3", 5, "ECA-fixed vs Source-affine"); check("T3 ECA-R vs affine m3 k5", gi(r) + " with three months at five FUAs")
r = G(L, "historyN", "M3", 11, "ECA-fixed vs Source-affine"); check("T3 ECA-R vs affine m3 k11", gi(r) + " at eleven")
v = [G(L, "historyN", c, k, "ECA-fixed vs Source-affine").gain for c in ("M6", "M12") for k in (5, 11)]
check("T3 ECA-R vs affine m6/12", f"({rng(v)}\\%)")
v = [G(L, "historyN", c, k, "ECA-fixed vs Direct") for c in ("M3", "M6", "M12") for k in (5, 11)]
check("T3 ECA-R vs direct range", f"by {rng([x.gain for x in v])}\\% in every T3 cell")
assert all(x.lo > 0 for x in v)
v = [G(L, "historyN", c, k, "ECA2 vs Direct").gain for c in ("M3", "M6", "M12") for k in (5, 11)]
check("T3 ECA-S vs direct stale range", f"by {rng(v)}\\%, but has")
r = G(L, "historyN", "M3", 5, "ECA2 vs Source-affine")
check("T3 ECA-S vs affine stale", f"has {abs(r.gain):.1f}\\% higher RMSE than source-affine with three months at five FUAs (interval {f1(r.lo)} to {f1(r.hi)})")
# ---- YRD development
check("T1 direct", f"({M('yrd', 'history', 'M3', 5, 'Direct'):.2f} at \\(k=5\\))")
a = arch[(arch.task == "yrd") & (arch.cell == "M3") & (arch.comparison == "Direct vs NNW[LDF] (archived)")]
a5, a11 = a[a.k == 5].iloc[0], a[a.k == 11].iloc[0]
check("T1 direct vs NNW k5", f"by {a5.gain:.1f}\\% (20 of 20 splits)"); assert a5.wins == 20
check("T1 direct vs NNW k11", f"{a11.gain:.1f}\\% ({int(a11.wins)}/20)")
check("abstract direct vs NNW", f"relative to NNW[LDF] by {a5.gain:.1f}\\% (20 of 20 splits)")
r = G("yrd", "history", "M3", 5, "Source-affine vs Direct")
check("T1 affine vs direct", f"has {abs(r.gain):.1f}\\% higher RMSE than direct transfer at \\(k=5\\) (interval {f1(r.lo)} to {f1(r.hi)})")
check("T1 ECA-S m3", f"transfer ({M('yrd', 'history', 'M3', 5, 'ECA2'):.2f}) and is lower with twelve months ({M('yrd', 'history', 'M12', 5, 'ECA2'):.2f})")
check("T1 2019 window", f"source-affine has RMSE {M('yrd', 'window', '2019Q4', 5, 'Source-affine'):.2f}, ECA-S has {M('yrd', 'window', '2019Q4', 5, 'ECA2'):.2f}, a reduction of " + gi(G("yrd", "window", "2019Q4", 5, "ECA2 vs Source-affine")).replace("\\%", "\\%"))
# ---- Australia
check("T2 ECA-S m3", f"{M('aus', 'history', 'M3', 5, 'ECA2'):.2f}--{M('aus', 'history', 'M3', 3, 'ECA2'):.2f} with three months, against {M('aus', 'history', 'M3', 5, 'Source-affine'):.2f}--{M('aus', 'history', 'M3', 3, 'Source-affine'):.2f}")
v = [abs(G("aus", "history", c, k, "ECA-fixed vs Source-affine").gain) for c in ("M3", "M6", "M12") for k in (3, 5)]
check("T2 ECA-R within", f"within {max(v):.1f}\\% of source-affine")
assert all(G("aus", "history", c, k, x).gain < 0 for c in ("M3", "M6") for k in (3, 5)
           for x in ("ECA-fixed vs Best local six", "Source-affine vs Best local six"))
r3, r5 = G("aus", "window", "2019Q4", 3, "ECA-fixed vs Source-affine"), G("aus", "window", "2019Q4", 5, "ECA-fixed vs Source-affine")
check("T2 2019 window", f"by {gi(r3)} at three FUAs and {gi(r5)} at five")
b3, b5 = G("aus", "window", "2019Q4", 3, "ECA-fixed vs Best local six"), G("aus", "window", "2019Q4", 5, "ECA-fixed vs Best local six")
check("T2 2019 vs local", f"by {b3.gain:.1f}\\% and {b5.gain:.1f}\\%")
v = [-G("aus", "window", c, k, "ECA-fixed vs Source-affine").gain for c in ("2020Q2", "2020Q3", "2020Q4") for k in (3, 5)]
check("T2 other windows", f"it is {min(v):.1f}--{max(v):.1f}\\% worse")
# ---- ablations
r5, r11 = G(L, "history", "M3", 5, "ECA2 vs ECA2-affine"), G(L, "history", "M3", 11, "ECA2 vs ECA2-affine")
check("abl condition m3", f"by {gi(r5)} and {gi(r11)} with three months")
r5, r11 = G(L, "history", "M6", 5, "ECA2 vs ECA2-affine"), G(L, "history", "M6", 11, "ECA2 vs ECA2-affine")
check("abl condition m6", f"by {r5.gain:.1f}\\% and {r11.gain:.1f}\\% with six"); assert r5.lo > 0 and r11.lo > 0
check("abl meta", f"by {rng([G(L, 'history', 'M3', k, 'ECA2 vs ECA2-no-meta').gain for k in (5, 11)])}\\% with three months")
check("abl bound", f"cost {rng([G(L, 'history', 'M3', k, 'ECA2 vs ECA2-unbounded').gain for k in (5, 11)])}\\% with three")
v = [abs(G(L, "history", c, k, "ECA2 vs ECA2-isotropic").gain) for c in ("M3", "M6") for k in (5, 11)]
check("abl isotropic", f"(within {max(v):.1f}\\%)")
v = [G(L, "history", "M3", k, "ECA2 vs ECA2-MAP").gain for k in (5, 11)]
check("abl MAP", f"({rng(v)}\\%)")
check("abl ECA-R meta", f"by {rng([G(L, 'historyN', 'M3', k, 'ECA-fixed vs ECA-no-meta').gain for k in (5, 11)])}\\% with three")
v = [G(L, "historyN", "M3", k, "ECA-fixed vs ECA-unbounded").gain for k in (5, 11)]
results.append(("abl ECA-R bound doubles", all(1.8 < 1 / (1 - x / 100) < 2.3 for x in v), str(v)))
check("abl T1", f"({f1(G('yrd', 'history', 'M3', 5, 'ECA2 vs ECA2-affine').gain)}\\%), while meta-learning and the bound do ({G('yrd', 'history', 'M3', 5, 'ECA2 vs ECA2-no-meta').gain:.1f}\\% and {G('yrd', 'history', 'M3', 5, 'ECA2 vs ECA2-unbounded').gain:.1f}\\%)")
# ---- label selection
d = q2m[(q2m.task == "aus") & (q2m.m == 3) & (q2m.k == 3) & (q2m.method == "Source-affine")].set_index("strategy").mean_rmse
others = [d[s] for s in d.index if not s.startswith("S0")]
check("labels AUS", f"has RMSE {d['S0 last consecutive']:.2f} at three FUAs, against {min(others):.2f}--{max(others):.2f}")
r5, r11 = Q2(L, 3, 5, "ECA-R", "S4 target-aware vs S0 last consecutive"), Q2(L, 3, 11, "ECA-R", "S4 target-aware vs S0 last consecutive")
check("labels T3 ECA-R", f"by {-r5.gain:.1f}\\% [{-r5.hi:.1f}, {-r5.lo:.1f}] and {-r11.gain:.1f}\\% [{-r11.hi:.1f}, {-r11.lo:.1f}]")
s5, s11 = Q2(L, 3, 5, "ECA-S", "S4 target-aware vs S0 last consecutive"), Q2(L, 3, 11, "ECA-S", "S4 target-aware vs S0 last consecutive")
check("labels T3 ECA-S", f"ECA-S's by {-s5.gain:.1f}\\% and {-s11.gain:.1f}\\%"); assert s5.hi > 0 and s11.hi > 0
for task in ("locked",):
    for mm in (3, 6):
        for k in (5, 11):
            for meth in ("Source-affine", "ECA-R", "ECA-S"):
                d = q2m[(q2m.task == task) & (q2m.m == mm) & (q2m.k == k) & (q2m.method == meth)].set_index("strategy").mean_rmse
                results.append((f"T3 S0 best {meth} m{mm} k{k}", d.idxmin().startswith("S0"), d.idxmin()))
for mm in (3, 6):
    for k in (3, 5):
        for meth in ("Source-affine", "ECA-R"):
            d = q2m[(q2m.task == "aus") & (q2m.m == mm) & (q2m.k == k) & (q2m.method == meth)].set_index("strategy").mean_rmse
            results.append((f"T2 S0 best {meth} m{mm} k{k}", d.idxmin().startswith("S0"), d.idxmin()))
d = q2m[(q2m.task == "yrd") & (q2m.m == 3) & (q2m.k == 5) & (q2m.method == "ECA-S")].set_index("strategy").mean_rmse
r = Q2("yrd", 3, 5, "ECA-S", "S4 target-aware vs S0 last consecutive")
check("labels T1", f"has RMSE {d['S4 target-aware']:.2f} at five FUAs, below direct transfer (6.25) and {r.gain:.1f}\\% below the last-months design (interval {f1(r.lo)} to {f1(r.hi)})")
v = [abs(Q2(t, mm, k, "ECA-S", "S4 target-aware vs S1 random").gain) for t in ("yrd", "locked") for mm in (3, 6) for k in (5, 11)]
check("labels S4 vs random", f"within {max(v):.1f}\\% of random")
for mm in (3, 6):
    for k in (5, 11):
        d = q2m[(q2m.task == "yrd") & (q2m.m == mm) & (q2m.k == k) & (q2m.method == "Source-affine")].set_index("strategy").mean_rmse
        results.append((f"T1 affine S0 best m{mm} k{k}", d.idxmin().startswith("S0"), d.idxmin()))
# ---- meta-learning and applicability
cv2 = [100 * (1 - np.exp(c["held_out_obj"])) for c in th2["region_cv"]]
check("meta CV ECA-S", f"by {cv2[0]:.1f}\\%, {cv2[1]:.1f}\\% and {cv2[2]:.1f}\\%")
check("meta affine", f"increases MSE by {100 * (np.exp(th2['source_affine_objective']) - 1):.1f}\\%")
cv1 = [100 * (1 - np.exp(c["held_out_obj"])) for c in th["region_cv"]]
check("meta CV ECA-R", f"by {cv1[0]:.1f}\\%, {cv1[1]:.1f}\\% and {cv1[2]:.1f}\\%")
an = np.exp(th2["ARD"]["an"])
check("ratios", f"(signal-to-noise ratio {np.exp(th2['ARD']['aa']) / an:.4f}), some to a scale change ({np.exp(th2['ARD']['ab']) / an:.2f}) and most to the condition term ({np.exp(th2['ARD']['ag']) / an:.2f})")
ls = dict(zip(th2["conditions"], np.exp(th2["ARD"]["ls"])))
check("ls ECA-S", f"temperature ({ls['monthly_power_temperature_c']:.1f}), fire count ({ls['log_fire_count_100km']:.1f}), wind speed ({ls['monthly_power_wind_mps']:.1f}) and the seasonal sine ({ls['local_season_sin']:.1f}); precipitation ({ls['monthly_power_precip_mm_day']:.1f}) and diurnal range ({ls['temp_range']:.1f})")
ls1 = dict(zip(th["conditions"], np.exp(th["ARD"]["ls"])))
check("ls ECA-R", f"temperature ({ls1['monthly_power_temperature_c']:.1f}), the seasonal cosine ({ls1['local_season_cos']:.1f}) and fire count ({ls1['log_fire_count_100km']:.1f})")
check("AOA", f"{100 * di['T1 rest of China -> YRD']['target_outside']:.1f}\\% of YRD FUA-months, {100 * di['T3 China without region 8 -> region 8']['target_outside']:.1f}\\% of north-west Chinese FUA-months and {100 * di['T2 China -> Australia']['target_outside']:.1f}\\% of Australian")
lev = {}
for task in ("yrd", "locked", "aus"):
    P = pd.read_csv(A / f"results/{task}_history_predictions.csv.gz", usecols=["eFUA_ID", "t", "observed", "Direct"])
    P = P.drop_duplicates(["eFUA_ID", "t"])
    lev[task] = (P.Direct.mean() - P.observed.mean(), P.observed.mean())
check("levels", f"by {lev['yrd'][0]:.1f}~\\(\\mu\\)g/m\\(^3\\) in the YRD, {lev['locked'][0]:.1f} in north-west China and {lev['aus'][0]:.1f} in Australia (observed means {lev['yrd'][1]:.1f}, {lev['locked'][1]:.1f} and {lev['aus'][1]:.1f})")
# ---- exploratory distance check over the nine development regions (aoa/distance_check.json, added after T3)
dist = json.loads((A / "aoa/distance_check.json").read_text())
dr = {x["region"]: x for x in dist["regions"]}
xj, yt = dr[1], dr[6]
rest = [x for k_, x in dr.items() if k_ not in (1, 6)]
results.append(("dist Xinjiang only region wholly outside",
                xj["outside_pct"] == 100 and all(x["outside_pct"] < 100 for k_, x in dr.items() if k_ != 1), ""))
results.append(("dist other seven lose", all(x["gain_pct"] < 0 for x in rest) and xj["gain_pct"] > 0 and yt["gain_pct"] > 0,
                str([x["gain_pct"] for x in rest])))
check("dist Xinjiang", f"regions' area of applicability ({xj['gain_pct']:.1f}\\%, {xj['wins']} of {xj['tasks']} pseudo-transfers)")
check("dist Yunnan-Tibet", f"in Yunnan and Tibet ({yt['gain_pct']:.1f}\\%, {yt['wins']}/{yt['tasks']}, with {yt['outside_pct']:.1f}\\% outside)")
check("dist other seven", f"with {min(x['outside_pct'] for x in rest):.1f}--{max(x['outside_pct'] for x in rest):.1f}\\% outside, its mean gain ranged from {f1(min(x['gain_pct'] for x in rest))}\\% to {f1(max(x['gain_pct'] for x in rest))}\\%")
check("dist rank correlation", f"is {dist['spearman_rho']:.2f} (\\(p={dist['spearman_p']:.2f}\\))")
for x in dist["regions"]:
    check(f"dist table {x['region']}", f"{x['label']} & {x['eligible_fuas']} & {x['outside_pct']:.1f} & {x['median_di']:.2f} & {f1(x['gain_pct'])} & {x['wins']}/{x['tasks']} \\\\")
# ---- exploratory checks after T3 (explore/)
E = A / "explore"
abl = json.loads((E / "ablation_summary.json").read_text())["variants"]
mech = json.loads((E / "mechanism.json").read_text())
ovl = json.loads((E / "condition_overlap.json").read_text())
hyb = pd.read_csv(E / "hybrid_gains.csv")
groups = [abl[v]["full_vs_variant_gain"] for v in ("no meteorology", "no fire", "no season")]
results.append(("explore groups within 0.5", max(abs(round(x, 1)) for x in groups) <= 0.5, str(groups)))
results.append(("explore all intervals include zero",
                all(abl[v]["lo"] < 0 < abl[v]["hi"] for v in abl if v != "full"), ""))
nc = abl["no conditions"]
check("explore no conditions", f"condition term by {f1(nc['full_vs_variant_gain'])}\\% [{f1(nc['lo'])}, {f1(nc['hi'])}]")
W = pd.read_csv(A / "results/aus_window_predictions.csv.gz")
W = W[(W.cell == "2019Q4") & (W.k == 5)]
check("explore fire excess", f"by {(W['Source-affine'] - W.observed).mean():.1f}~\\(\\mu\\)g/m\\(^3\\) on average; the "
      f"correction lowers this excess to {(W['ECA-fixed'] - W.observed).mean():.1f}")
ws = mech["window_scan"]
check("explore window ranges", f"from {ws['yrd']['Source-affine']['min']:.2f} to {ws['yrd']['Source-affine']['max']:.2f}"
      f"~\\(\\mu\\)g/m\\(^3\\) in the YRD, from {ws['locked']['Source-affine']['min']:.2f} to "
      f"{ws['locked']['Source-affine']['max']:.2f} in north-west China and from {ws['aus']['Source-affine']['min']:.2f}"
      f" to {ws['aus']['Source-affine']['max']:.2f} in Australia")
rk = [ws[t][mm]["rank_last"] for t in ("yrd", "locked", "aus") for mm in ("Source-affine", "ECA-R")]
results.append(("explore latest window among four best", max(rk) <= 4, str(rk)))
rhos = [ws[t][mm]["spearman_rho"] for t in ("yrd", "locked", "aus") for mm in ("Source-affine", "ECA-R")]
results.append(("explore rank correlations negative", max(rhos) < 0, str(rhos)))
check("explore rank correlations", f"rank correlations $-${abs(max(rhos)):.2f} to $-${abs(min(rhos)):.2f}")
check("explore ECA-S window ranges", f"({ws['yrd']['ECA-S']['min']:.2f}--{ws['yrd']['ECA-S']['max']:.2f} in the YRD, "
      f"{ws['locked']['ECA-S']['min']:.2f}--{ws['locked']['ECA-S']['max']:.2f} in north-west China)")
h0 = hyb[hyb.comparison == "H vs S0"]; h1 = hyb[hyb.comparison == "H vs S1"]
n_pos0, n_neg0 = int((h0.lo > 0).sum()), int((h0.hi < 0).sum())
n_pos1, n_neg1 = int((h1.lo > 0).sum()), int((h1.hi < 0).sum())
check("explore hybrid counts text", f"improved on random months in {n_pos1} of {len(h1)} comparisons, but on the latest "
      f"months only in {n_pos0}")
check("explore hybrid counts appendix", f"improves on the latest months with an interval above zero in {n_pos0} and "
      f"loses in {n_neg0}, and improves on random months in {n_pos1} and loses in {n_neg1}")
pos = h0[h0.lo > 0]
results.append(("explore hybrid positive cells", set(map(tuple, pos[["task", "m"]].drop_duplicates().to_numpy()))
                <= {("yrd", 6), ("aus", 3), ("aus", 6)} and set(pos[pos.task == "aus"].method) == {"ECA-S"}
                and set(pos[pos.task == "yrd"].method) <= {"Source-affine", "ECA-R"}, pos.to_string()))
yp = pos[pos.task == "yrd"].gain
check("explore hybrid YRD six months", f"with six months, by {yp.min():.1f}--{yp.max():.1f}\\% for source-affine and ECA-R")
check("explore hybrid YRD three months", f"It lost up to {abs(h0[(h0.task == 'yrd') & (h0.m == 3)].gain.min()):.1f}\\% in "
      f"the YRD with three months")
aus_sa = h0[(h0.task == "aus") & (h0.method == "Source-affine")]
results.append(("explore hybrid AUS source-affine losses resolved", bool((aus_sa.hi < 0).all()), ""))
check("explore hybrid AUS", f"{abs(aus_sa.gain.max()):.1f}--{abs(aus_sa.gain.min()):.1f}\\% for source-affine in "
      f"Australia")
b = mech["bias"]
check("explore bias by year", "".join([
    f"was {f1(b['yrd']['mean_bias_2019'])}, {f1(b['yrd']['mean_bias_2020'])} and {f1(b['yrd']['test_mean_bias'])}",
    f"~\\(\\mu\\)g/m\\(^3\\) in 2019, 2020 and 2021--2022 in the YRD, {f1(b['locked']['mean_bias_2019'])}, ",
    f"{f1(b['locked']['mean_bias_2020'])} and {f1(b['locked']['test_mean_bias'])} in north-west China, and ",
    f"{f1(b['aus']['mean_bias_2019'])}, {f1(b['aus']['mean_bias_2020'])} and {f1(b['aus']['test_mean_bias'])} in "
    f"Australia"]))
check("explore bias subsets", f"than that of {100 * b['yrd']['m3_share_random_subsets_with_larger_gap']:.0f}\\%, "
      f"{100 * b['locked']['m3_share_random_subsets_with_larger_gap']:.0f}\\% and "
      f"{100 * b['aus']['m3_share_random_subsets_with_larger_gap']:.0f}\\%")
fr = mech["fire"]
check("explore fire medians", f"median log fire count is {fr['median_log_fire_label']:.2f} in the label months and "
      f"{fr['median_log_fire_test']:.2f} in the test months, and {100 * fr['share_test_rows_below_label_median']:.1f}\\% "
      f"of test rows")
check("explore overlap", f"\\(R^2\\) of {ovl['monthly_power_temperature_c']:.2f} and {ovl['local_season_cos']:.2f}")
for name in ("ablation", "windows", "hybrid"):
    for line in (E / f"tab_explore_{name}.tex").read_text(encoding="utf-8").splitlines():
        if "&" in line:
            check(f"explore table {name}", line)
# ---- appendix counts
conf =[("history", "ECA2 vs Direct"), ("history", "ECA2 vs Source-affine"), ("history", "Source-affine vs Direct"),
        ("historyN", "ECA-fixed vs Source-affine"), ("historyN", "ECA-fixed vs Direct"), ("historyN", "ECA2 vs Direct")]
rows = [G(L, d_, c, k, comp) for d_, comp in conf for c in ("M3", "M6", "M12") for k in (5, 11)]
imp = sum(r.lo > 0 for r in rows); loss = sum(r.hi < 0 for r in rows)
check("appendix counts", f"Of the 36 estimator comparisons in Table~\\ref{{tab:adapt-confirm}}, {imp} show an improvement")
results.append(("appendix loss count", loss == 1, str(loss)))
lab = [Q2(L, mm, k, meth, comp) for mm in (3, 6) for k in (5, 11) for meth in ("ECA-S", "ECA-R")
       for comp in ("S4 target-aware vs S0 last consecutive", "S4 target-aware vs S1 random")]
results.append(("appendix label counts", sum(r.lo > 0 for r in lab) == 0 and sum(r.hi < 0 for r in lab) == 2,
                f"{sum(r.lo > 0 for r in lab)} / {sum(r.hi < 0 for r in lab)}"))
fails = [r for r in results if not r[1]]
for name, ok, needle in results:
    print("PASS" if ok else "FAIL", name, "" if ok else "| " + needle)
print(f"{len(results) - len(fails)}/{len(results)} checks pass")
sys.exit(1 if fails else 0)
