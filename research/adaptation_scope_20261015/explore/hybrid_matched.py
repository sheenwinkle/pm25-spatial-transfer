"""Matched-budget check for the hybrid strategy (post hoc, added 19 October 2026 after the sixth draft).

In Australia some labelled FUAs miss part of late 2020. The last-months design (S0) then receives fewer labels than
its budget, while the hybrid (H) fills its coverage part in full, so H received one label more than S0 in 50 of
the 80 Australian splits. This script repeats the hybrid selection of hybrid_labels.py exactly (same functions and
seeds) and then cuts each FUA's hybrid labels to the number S0 received for that FUA, keeping the recent months
first and then the coverage months in their selection order. The estimators are refitted on the cut labels.

Usage: python explore/hybrid_matched.py <task: yrd|aus|locked> <theta.json> <theta2.json> <out_dir>
Writes <out_dir>/<task>_hybridmatched_predictions.csv.gz in the format of <task>_hybrid_predictions.csv.gz.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path(__file__).with_name("hybrid_labels.py").read_text()
exec(_src.split('STRATS = {"S0 last consecutive": s0')[0])   # data, source model, S0 and H exactly as in hybrid_labels.py

FUA = tgt.eFUA_ID.to_numpy()
H = "H recent + target-aware"


def cut(rows, quota):
    keep, used = [], {}
    for r in rows:
        u = FUA[r]
        if used.get(u, 0) < quota.get(u, 0):
            keep.append(r); used[u] = used.get(u, 0) + 1
    return np.array(keep, dtype=int)


preds_all = []
for m in (3, 6):
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, k, 20)):
            tst = np.flatnonzero((tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy())
            b = tgt.iloc[tst]; vy = b[Y].to_numpy(); eraw = b.raw.to_numpy()
            frame = b[["eFUA_ID", "t", "year", "month", "quarter_id"]].copy()
            frame.insert(0, "split", sid); frame.insert(0, "k", k); frame.insert(0, "m", m)
            frame["observed"] = vy
            frame["Direct"] = eraw
            s0_rows = s0(train_ids, m, None)
            quota = pd.Series(FUA[s0_rows]).value_counts().to_dict()
            rng = np.random.default_rng([al.SEED, m, k, sid, 0])
            full = hybrid(train_ids, m, rng, tst)
            rows = cut(full, quota)
            assert len(rows) == len(s0_rows)
            a = tgt.iloc[rows]; y = a[Y].to_numpy(); raw = a.raw.to_numpy()
            aff_l, aff_t = al.affine(raw, y, eraw)
            frame[f"{H}|0|Source-affine"] = core.pos(aff_t)
            g, _ = al.eca_predict("ECA-fixed", THETA, Z[rows], y - aff_l, Z[tst], y.mean())
            frame[f"{H}|0|ECA-R"] = core.pos(aff_t + g)
            s_l, s_t = al.scaled_source(raw, eraw, SRC_SD)
            d2, _ = al.eca2_predict("ECA2", TH2, Z[rows], s_l, y - raw, Z[tst], s_t)
            frame[f"{H}|0|ECA-S"] = core.pos(eraw + d2)
            frame[f"{H}|0|n_rows"] = len(rows)
            frame[f"{H}|0|n_rows_full"] = len(full)
            preds_all.append(frame)
        print(f"{task} hybrid matched m={m} k={k} done", flush=True)

pd.concat(preds_all).to_csv(out / f"{task}_hybridmatched_predictions.csv.gz", index=False, float_format="%.6g")
