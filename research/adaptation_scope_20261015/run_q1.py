"""Q1: how much and where to adapt (see PROTOCOL.md).

Usage: python run_q1.py <task: yrd|aus|locked> <design: history|historyN|window> <theta.json> <theta2.json> <out_dir>
historyN: as history, but the source model is fitted on source years up to 2020 only (no contemporaneous source
labels).
Writes <out_dir>/<task>_<design>_metrics.csv, _predictions.csv.gz and _hyper.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import adapt_lib as al
from adapt_lib import X, Y, core

task, design, theta_path, theta2_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), \
    Path(sys.argv[5])
TH2 = json.loads(theta2_path.read_text())
out.mkdir(parents=True, exist_ok=True)
th = json.loads(theta_path.read_text())
THETA = {k: dict(ls=np.array(v["ls"]), af=v["af"], an=v["an"]) for k, v in th.items() if k in ("ARD", "isotropic")}
ks = [3, 5] if task == "aus" else [5, 11]

src, tgt, eligible = al.load(task)
if design == "historyN":
    src = src[src.year.le(2020)].reset_index(drop=True)
f = al.source_model(src)
tgt["raw"] = core.pos(f.predict(tgt[X]))
SRC_SD = float(np.std(core.pos(f.predict(src[X]))))
Z = al.zmat(tgt)
t_of = tgt.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
if design in ("history", "historyN"):
    t_dec = int(t_of[(2020, 12)])
    cells = {f"M{m}": (t_dec - m + 1, t_dec) for m in (3, 6, 12)}
else:
    cells = {"2020Q4": (int(t_of[(2020, 10)]), int(t_of[(2020, 12)])),
             "2020Q2": (int(t_of[(2020, 4)]), int(t_of[(2020, 6)])),
             "2020Q3": (int(t_of[(2020, 7)]), int(t_of[(2020, 9)])),
             "2019Q4": (int(t_of[(2019, 10)]), int(t_of[(2019, 12)]))}
VARIANTS = ["ECA", "ECA-fixed", "ECA-no-meta", "ECA-unbounded", "ECA-isotropic"]
VARIANTS2 = ["ECA2", "ECA2-MAP", "ECA2-affine", "ECA2-isotropic", "ECA2-no-meta", "ECA2-unbounded"]
metrics, preds_all, hyper = [], [], []
t0 = time.time()
for cell, (ta, tb) in cells.items():
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, k, 20)):
            trn = (tgt.eFUA_ID.isin(train_ids) & tgt.t.between(ta, tb)).to_numpy()
            tst = (tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy()
            if trn.sum() < 5:
                continue
            a, b = tgt[trn], tgt[tst]
            y, raw, eraw = a[Y].to_numpy(), a.raw.to_numpy(), b.raw.to_numpy()
            p = {"Direct": eraw}
            aff_l, aff_t = al.affine(raw, y, eraw)
            p["Source-affine"] = core.pos(aff_t)
            p["Source-residual"] = core.pos(eraw + core.tree().fit(a[X], y - raw).predict(b[X]))
            p.update(al.local_six(a[X], y, b[X]))
            for v in VARIANTS:
                g, (ls, af, an) = al.eca_predict(v, THETA["isotropic" if v == "ECA-isotropic" else "ARD"],
                                                 Z[trn], y - aff_l, Z[tst], y.mean())
                p[v] = core.pos(aff_t + g)
                hyper.append(dict(task=task, design=design, cell=cell, k=k, split=sid, variant=v, a_f=af, a_n=an,
                                  **{f"ls_{c}": l for c, l in zip(al.COND, np.broadcast_to(ls, (len(al.COND),)))}))
            s_l, s_t = al.scaled_source(raw, eraw, SRC_SD)
            for v in VARIANTS2:
                d2, (ls, ra, rb, rg) = al.eca2_predict(v, TH2, Z[trn], s_l, y - raw, Z[tst], s_t)
                p[v] = core.pos(eraw + d2)
                hyper.append(dict(task=task, design=design, cell=cell, k=k, split=sid, variant=v, rho_a=ra, rho_b=rb,
                                  rho_g=rg,
                                  **{f"ls_{c}": l for c, l in zip(al.COND, np.broadcast_to(ls, (len(al.COND),)))}))
            vy = b[Y].to_numpy()
            for mname, pr in p.items():
                metrics.append(dict(task=task, design=design, cell=cell, k=k, split=sid, model=mname,
                                    rmse=al.rmse(vy, pr), n_train_rows=int(trn.sum()), n_test=int(tst.sum())))
            frame = b[["eFUA_ID", "t", "year", "month", "quarter_id"]].copy()
            frame.insert(0, "split", sid); frame.insert(0, "k", k); frame.insert(0, "cell", cell)
            frame["observed"] = vy
            for mname, pr in p.items():
                frame[mname] = pr
            preds_all.append(frame)
        print(f"{task} {design} {cell} k={k} done ({time.time() - t0:.0f}s)", flush=True)

stem = out / f"{task}_{design}"
pd.DataFrame(metrics).to_csv(f"{stem}_metrics.csv", index=False)
pd.concat(preds_all).to_csv(f"{stem}_predictions.csv.gz", index=False, float_format="%.6g")
pd.DataFrame(hyper).to_csv(f"{stem}_hyper.csv", index=False, float_format="%.5g")

if (design == "history" and task in ("yrd", "aus")) or (design == "historyN" and task == "yrd"):
    # reproduction check against the archived results (history: M protocol; historyN: N protocol)
    m = pd.DataFrame(metrics)
    diffs = []
    for cell in cells:
        for k in ks:
            for sid in range(20):
                folder = cell if design == "history" else cell.replace("M", "N")
                fpath = al.BENCH / "results" / f"{task}_{folder}" / f"k{k}_split{sid:02d}.csv"
                arch = pd.read_csv(fpath)
                if "Source-affine (ours)" not in set(arch.model):
                    continue
                ours = m[(m.cell == cell) & (m.k == k) & (m.split == sid) & (m.model == "Source-affine")].rmse
                diffs.append(abs(float(ours.iloc[0]) - float(arch[arch.model == "Source-affine (ours)"].rmse.iloc[0])))
    print(f"reproduction: {len(diffs)} source-affine RMSEs, max |difference| {max(diffs):.2e}")
