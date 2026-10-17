"""Paired comparisons for the hybrid strategy (explore/README.md, item 2).

H is compared with S0, S1 (mean over its five draws), S4 and direct transfer for source-affine, ECA-R and ECA-S,
using the crossed FUA-by-quarter bootstrap of summarise.py (functions copied from it; same seed, so the draws match
those behind summary/q2_gains.csv).
Usage: python explore/hybrid_summary.py <study_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEED, DRAWS = 20261015, 1000
H = "H recent + target-aware"
KEY = ["m", "k", "split", "eFUA_ID", "t"]


def draws_for(frame):
    fuas = np.array(sorted(frame.eFUA_ID.unique())); quarters = np.array(sorted(frame.quarter_id.unique()))
    rng = np.random.default_rng(SEED)
    wf = rng.multinomial(len(fuas), np.full(len(fuas), 1 / len(fuas)), size=DRAWS).astype(float)
    wq = rng.multinomial(len(quarters), np.full(len(quarters), 1 / len(quarters)), size=DRAWS).astype(float)
    return fuas, quarters, wf, wq


def split_rmse(df, cols, fuas, quarters, wf, wq):
    w = wf[:, np.searchsorted(fuas, df.eFUA_ID.to_numpy())] * wq[:, np.searchsorted(quarters, df.quarter_id.to_numpy())]
    tot = w.sum(1)
    tot = np.where(tot > 0, tot, np.nan)
    y = df.observed.to_numpy()
    out = {}
    for name, c in cols.items():
        e = (df[c].to_numpy() - y) ** 2
        out[name] = (np.sqrt(e.mean()), np.sqrt(w @ e / tot))
    return out


def gains(rows, a, b):
    pa = np.array([r[a][0] for r in rows]); pb = np.array([r[b][0] for r in rows])
    ba = np.vstack([r[a][1] for r in rows]); bb = np.vstack([r[b][1] for r in rows])
    g = 100 * np.mean(1 - pa / pb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 1 - ba / bb
    ok = np.isfinite(rel)
    d = 100 * np.where(ok.any(0), np.nansum(np.where(ok, rel, 0), 0) / np.maximum(ok.sum(0), 1), np.nan)
    d = d[np.isfinite(d)]
    return dict(gain=g, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), wins=int((pa < pb).sum()),
                n=len(rows), mean_a=pa.mean(), mean_b=pb.mean())


rows_out, means_out = [], []
for task in ("yrd", "aus", "locked"):
    Q = pd.read_csv(A / f"results/{task}_q2_predictions.csv.gz")
    Hd = pd.read_csv(A / f"explore/{task}_hybrid_predictions.csv.gz")
    hcols = [c for c in Hd.columns if c.startswith(H + "|") and not c.endswith("n_rows")]
    P = Q.merge(Hd[KEY + hcols], on=KEY, how="left", validate="one_to_one")
    assert P[hcols].notna().all().all()
    fuas, quarters, wf, wq = draws_for(P)
    for (m, k), g in P.groupby(["m", "k"]):
        for method in ("Source-affine", "ECA-R", "ECA-S"):
            per = []
            for _, s in g.groupby("split"):
                cols = {"Direct": "Direct", "S0": f"S0 last consecutive|0|{method}", "S4": f"S4 target-aware|0|{method}",
                        "H": f"{H}|0|{method}"}
                cols.update({f"S1|{r}": f"S1 random|{r}|{method}" for r in range(5)})
                r = split_rmse(s, cols, fuas, quarters, wf, wq)
                s1 = [r[f"S1|{i}"] for i in range(5)]
                r["S1"] = (np.mean([x[0] for x in s1]), np.mean([x[1] for x in s1], 0))
                per.append(r)
            for name in ("Direct", "S0", "S1", "S4", "H"):
                means_out.append(dict(task=task, m=m, k=k, method=method, strategy=name,
                                      mean_rmse=np.mean([r[name][0] for r in per])))
            for b in ("S0", "S1", "S4", "Direct"):
                rows_out.append(dict(task=task, m=m, k=k, method=method, comparison=f"H vs {b}", **gains(per, "H", b)))
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows_out).to_csv(OUT / "hybrid_gains.csv", index=False, float_format="%.4f")
pd.DataFrame(means_out).to_csv(OUT / "hybrid_means.csv", index=False, float_format="%.4f")
print(pd.DataFrame(rows_out).round(1).to_string(index=False))
