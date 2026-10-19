"""Q2 matched-budget sensitivity (post hoc, added after the fifth draft).

In Australia, some labelled FUAs have fewer than m observed months in the last m months of 2020, so S0 receives
fewer labels than S1-S4. This script re-runs S1-S4 with each FUA's labels cut to the number S0 actually received
(keeping each strategy's own selection order), so every strategy uses the same labels per FUA and split.

Usage: python run_q2_matched.py <task> meta/theta.json meta/theta2.json <out_dir>
Writes <out_dir>/<task>_q2matched_metrics.csv and _selected.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src_code = Path(__file__).with_name("run_q2.py").read_text()
exec(src_code.split("metrics, preds_all, selected = [], [], []")[0])   # loads data, models and S0-S4 unchanged

out.mkdir(parents=True, exist_ok=True)
fua_of = tgt.eFUA_ID.to_numpy()


def truncate(rows, quota):
    keep, used = [], {}
    for r in rows:
        u = fua_of[r]
        if used.get(u, 0) < quota.get(u, 0):
            keep.append(r); used[u] = used.get(u, 0) + 1
    return np.array(keep, dtype=int)


metrics, selected = [], []
for m in (3, 6):
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, k, 20)):
            tst = np.flatnonzero((tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy())
            b = tgt.iloc[tst]; vy = b[Y].to_numpy(); eraw = b.raw.to_numpy()
            s0_rows = s0(train_ids, m, None)
            quota = pd.Series(fua_of[s0_rows]).value_counts().to_dict()
            for sname, (fn, reps) in STRATS.items():
                for rep in range(reps):
                    rng = np.random.default_rng([al.SEED, m, k, sid, rep])
                    rows = fn(train_ids, m, rng, tst) if fn is s4 else fn(train_ids, m, rng)
                    rows = truncate(rows, quota)
                    if len(rows) < 5:
                        continue
                    a = tgt.iloc[rows]; y = a[Y].to_numpy(); raw = a.raw.to_numpy()
                    aff_l, aff_t = al.affine(raw, y, eraw)
                    p = {"Source-affine": core.pos(aff_t)}
                    g, _ = al.eca_predict("ECA-fixed", THETA, Z[rows], y - aff_l, Z[tst], y.mean())
                    p["ECA-R"] = core.pos(aff_t + g)
                    s_l, s_t = al.scaled_source(raw, eraw, SRC_SD)
                    d2, _ = al.eca2_predict("ECA2", TH2, Z[rows], s_l, y - raw, Z[tst], s_t)
                    p["ECA-S"] = core.pos(eraw + d2)
                    p.update(al.local_six(a[X], y, b[X]))
                    for mname, pr in p.items():
                        metrics.append(dict(task=task, m=m, k=k, split=sid, strategy=sname, rep=rep, model=mname,
                                            rmse=al.rmse(vy, pr), n_train_rows=len(rows)))
                    selected.append(dict(task=task, m=m, k=k, split=sid, strategy=sname, rep=rep, n=len(rows),
                                         n_s0=len(s0_rows)))
        print(f"{task} q2matched m={m} k={k} done", flush=True)

stem = out / f"{task}_q2matched"
pd.DataFrame(metrics).to_csv(f"{stem}_metrics.csv", index=False)
pd.DataFrame(selected).to_csv(f"{stem}_selected.csv", index=False)
