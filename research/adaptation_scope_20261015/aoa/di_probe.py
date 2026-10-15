"""Covariate-only dissimilarity of target FUA-months from the source environment (no PM2.5 is read).

Dissimilarity index after Meyer & Pebesma (2021), simplified: ten standardised environmental predictors with equal
weights; DI = distance to the nearest source row / mean pairwise source distance; the applicability threshold is
the upper whisker (Q3 + 1.5 IQR, capped at the data) of leave-FUA-out DI within the source.
Usage: python di_probe.py <panel> <membership> <regions.json> <out.json>
"""
import json, sys
import numpy as np, pandas as pd

PANEL, MEMBER, REG, OUT = sys.argv[1:5]
ENV = ['monthly_power_temperature_c', 'monthly_power_temperature_max_c', 'monthly_power_temperature_min_c',
       'monthly_power_humidity_pct', 'monthly_power_wind_mps', 'monthly_power_precip_mm_day',
       'log_population_density', 'distance_to_coast_km', 'urban_centre_area_share', 'log_fire_count_100km']
p = pd.read_csv(PANEL, usecols=['country_iso', 'eFUA_ID', 'year', 'month'] + ENV)
yrd = set(pd.read_csv(MEMBER, usecols=['country_iso', 'eFUA_ID']).query("country_iso == 'CHN'").eFUA_ID)
reg = {int(k): v for k, v in json.load(open(REG))["fua_region"].items()}
rng = np.random.default_rng(5993)


def sqdist(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def probe(src, tgt):
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
    return dict(threshold=round(float(thr), 3), source_outside=round(float(np.mean(train > thr)), 4),
                target_median_di=round(float(np.median(di)), 3), target_outside=round(float(np.mean(di > thr)), 4),
                target_rows=len(T), target_fuas=int(tgt.eFUA_ID.nunique()))


chn, aus = p[p.country_iso.eq('CHN')], p[p.country_iso.eq('AUS')]
r = chn.eFUA_ID.map(reg)
out = {"T1 rest of China -> YRD": probe(chn[~chn.eFUA_ID.isin(yrd)], chn[chn.eFUA_ID.isin(yrd)]),
       "T2 China -> Australia": probe(chn, aus),
       "T3 China without region 8 -> region 8": probe(chn[~r.eq(8)], chn[r.eq(8)])}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
