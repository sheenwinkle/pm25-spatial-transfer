"""Exploratory check, added after the T3 run: does distance from the source data predict whether recalibration
pays among the nine development regions?

Distance: the covariate-only dissimilarity index of di_probe.py (same predictors, standardisation, threshold rule
and random seed), with the other eight development regions as the source, as in the pseudo-transfers.
Gain: mean paired RMSE reduction (%) of source-affine relative to direct transfer over the region's 24
pseudo-transfers in meta/meta2_tasks.csv (five labelled FUAs; m = 3, 6, 12; consecutive or scattered; four draws).
Usage: python distance_check.py <panel> <regions.json> <meta2_tasks.csv> <out.json> <out_table.tex>
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PANEL, REG, TASKS, OUT, TAB = sys.argv[1:6]
ENV = ['monthly_power_temperature_c', 'monthly_power_temperature_max_c', 'monthly_power_temperature_min_c',
       'monthly_power_humidity_pct', 'monthly_power_wind_mps', 'monthly_power_precip_mm_day',
       'log_population_density', 'distance_to_coast_km', 'urban_centre_area_share', 'log_fire_count_100km']
LABEL = {0: 'Central China and Fujian (Wuhan, Changsha, Xiamen)',
         1: 'Xinjiang (Urumqi, Karamay)',
         2: 'Sichuan Basin and Guizhou (Chengdu, Chongqing)',
         3: 'Liaoning and north Hebei (Shenyang, Tangshan)',
         4: 'South China (Guangzhou, Nanning, Haikou)',
         5: "Shanxi, Henan and Shaanxi (Taiyuan, Xi'an)",
         6: 'Yunnan and Tibet (Kunming, Lhasa)',
         7: 'North-east (Harbin, Changchun)',
         9: 'North China Plain (Beijing, Jinan, Qingdao)'}

p = pd.read_csv(PANEL, usecols=['country_iso', 'eFUA_ID', 't'] + ENV)
regions = json.load(open(REG))
reg = {int(k): v for k, v in regions["fua_region"].items()}
rng = np.random.default_rng(5993)


def sqdist(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def probe(src, tgt):
    """As in di_probe.py: share of target rows outside the source's area of applicability, and median DI."""
    mu, sd = src[ENV].mean(), src[ENV].std()
    S = ((src[ENV] - mu) / sd).to_numpy(float); T = ((tgt[ENV] - mu) / sd).to_numpy(float)
    idx = rng.choice(len(S), size=min(4000, len(S)), replace=False)
    pair = np.sqrt(sqdist(S[idx], S[idx])); dbar = pair[np.triu_indices(len(idx), 1)].mean()
    fua = src.eFUA_ID.to_numpy(); train = np.empty(len(S))
    for i in range(0, len(S), 1000):
        d = sqdist(S[i:i + 1000], S); d[fua[i:i + 1000][:, None] == fua[None, :]] = np.inf
        train[i:i + 1000] = np.sqrt(d.min(1)) / dbar
    q1, q3 = np.percentile(train, [25, 75]); thr = train[train <= q3 + 1.5 * (q3 - q1)].max()
    di = np.concatenate([np.sqrt(sqdist(T[i:i + 1000], S).min(1)) / dbar for i in range(0, len(T), 1000)])
    return float(np.mean(di > thr)), float(np.median(di))


chn = p[p.country_iso.eq('CHN')]
r = chn.eFUA_ID.map(reg)
DEV = sorted(k for k in set(reg.values()) if k != regions["locked_region"])
n_months = p.t.nunique()
tasks = pd.read_csv(TASKS)
tasks['gain'] = 100 * (1 - tasks.rmse_affine / tasks.rmse_direct)

rows = []
for c in DEV:
    outside, med = probe(chn[r.isin([k for k in DEV if k != c])], chn[r.eq(c)])
    cnt = chn[r.eq(c)].groupby('eFUA_ID').size()
    g = tasks[tasks.region.eq(c)]
    rows.append(dict(region=c, label=LABEL[c], eligible_fuas=int((cnt >= np.ceil(0.8 * n_months)).sum()),
                     outside_pct=round(100 * outside, 1), median_di=round(med, 2),
                     gain_pct=round(float(g.gain.mean()), 1), wins=int((g.gain > 0).sum()), tasks=len(g)))
rows.sort(key=lambda x: x['outside_pct'])
rho, pval = spearmanr([x['outside_pct'] for x in rows], [x['gain_pct'] for x in rows])
out = dict(note="exploratory, computed after the T3 run", regions=rows,
           spearman_rho=round(float(rho), 2), spearman_p=round(float(pval), 2))
json.dump(out, open(OUT, 'w'), indent=1)


def num(v):
    return f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}"


lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
         r"\caption{Distance from the source data and the gain from recalibration in the nine development regions "
         r"(exploratory, computed after the confirmation run). Outside: share of the region's FUA-months outside the "
         r"area of applicability of the other eight development regions; DI: median dissimilarity index. Gain: mean "
         r"paired RMSE reduction (\%) of source-affine relative to direct transfer over the region's 24 "
         r"pseudo-transfers (five labelled FUAs, three to twelve months); negative values favour direct transfer. "
         r"Wins: pseudo-transfers in which source-affine is more accurate.}",
         r"\label{tab:adapt-distance}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Region & FUAs & Outside (\%) & DI & Gain (\%) & Wins \\", r"\midrule"]
for x in rows:
    lines.append(f"{x['label']} & {x['eligible_fuas']} & {x['outside_pct']:.1f} & {x['median_di']:.2f} & "
                 f"{num(x['gain_pct'])} & {x['wins']}/{x['tasks']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(TAB, 'w').write("\n".join(lines) + "\n")
print(json.dumps(out, indent=1))
