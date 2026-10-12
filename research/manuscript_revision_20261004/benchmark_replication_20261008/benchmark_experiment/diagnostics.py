"""Post-hoc descriptive diagnostic: share of target FUA-month variance explained by calendar-month
(time-index) means and by additive month + FUA effects. Written to generated/diagnostics.json."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from paths import GENERATED, PANEL, RESULTS, require


def shares(d):
    y = d.observed_pm25.to_numpy(); tot = ((y - y.mean()) ** 2).sum()
    mt = d.groupby("t").observed_pm25.transform("mean").to_numpy()
    a = np.zeros(len(d)); b = np.zeros(len(d))
    for _ in range(100):
        a = pd.Series(y - b, index=d.index).groupby(d.t).transform("mean").to_numpy()
        b = pd.Series(y - a, index=d.index).groupby(d.eFUA_ID).transform("mean").to_numpy()
    return {"month_share": float(1 - ((y - mt) ** 2).sum() / tot), "month_fua_share": float(1 - ((y - a - b) ** 2).sum() / tot),
            "fuas": int(d.eFUA_ID.nunique()), "rows": int(len(d))}


p = pd.read_csv(require(PANEL, "BENCH_PANEL"))
el = json.loads((RESULTS / "yrd_B" / "eligible.json").read_text())["eligible"]
y = p[p.eFUA_ID.isin(el)]
a = p[p.country_iso.eq("AUS")]; cnt = a.groupby("eFUA_ID").size(); a = a[a.eFUA_ID.isin(cnt[cnt >= 48].index)]
out = {"T1_all": shares(y), "T1_2021_2022": shares(y[y.year >= 2021]), "T2_all": shares(a), "T2_2021_2022": shares(a[a.year >= 2021]),
       "T1_mean_by_year": y.groupby("year").observed_pm25.mean().round(2).to_dict(),
       "T2_mean_by_year": a.groupby("year").observed_pm25.mean().round(2).to_dict(),
       "T2_2020_mean_by_month": a[a.year.eq(2020)].groupby("month").observed_pm25.mean().round(2).to_dict(),
       "T2_2021_2022_sd": float(a[a.year >= 2021].observed_pm25.std())}
GENERATED.mkdir(parents=True, exist_ok=True)
(GENERATED / "diagnostics.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
