"""Exploratory, added after the T3 run (see explore/README.md, item 2): hybrid label selection.

Strategy H gives each labelled FUA its floor(m/2) most recent months of 2020 and chooses the remaining months by the
target-aware criterion of S4 after conditioning on the recent months. Tasks, splits, seeds and the source-affine,
ECA-R and ECA-S estimators are those of run_q2.py (functions copied from it); S0 is recomputed as a check against
results/<task>_q2_predictions.csv.gz.

Usage: python explore/hybrid_labels.py <task: yrd|aus|locked> <theta.json> <theta2.json> <out_dir>
Writes <out_dir>/<task>_hybrid_predictions.csv.gz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_lib as al
from adapt_lib import X, Y, core

task, theta_path, theta2_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
th = json.loads(theta_path.read_text())["ARD"]
THETA = dict(ls=np.array(th["ls"]), af=th["af"], an=th["an"])
LS = np.exp(THETA["ls"]); RHO = (np.exp(THETA["an"]) / np.exp(THETA["af"])) ** 2
TH2 = json.loads(theta2_path.read_text())
ks = [3, 5] if task == "aus" else [5, 11]

src, tgt, eligible = al.load(task)
f = al.source_model(src)
tgt["raw"] = core.pos(f.predict(tgt[X]))
SRC_SD = float(np.std(core.pos(f.predict(src[X]))))
Z = al.zmat(tgt)
t_of = tgt.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
T_LO, T_HI, T_DEC = int(t_of[(2019, 1)]), int(t_of[(2020, 12)]), int(t_of[(2020, 12)])


def pool_rows(fua):
    return np.flatnonzero((tgt.eFUA_ID.eq(fua) & tgt.t.between(T_LO, T_HI)).to_numpy())


def s0(lab, m, rng):
    return np.flatnonzero((tgt.eFUA_ID.isin(lab) & tgt.t.between(T_DEC - m + 1, T_DEC)).to_numpy())


def hybrid(lab, m, rng, test_rows):
    r = m // 2
    recent = s0(lab, r, rng)
    rest = {u: np.setdiff1d(pool_rows(u), recent) for u in lab}
    cand = np.concatenate([rest[u] for u in lab])
    quota = {u: min(m - r, len(rest[u])) for u in lab}
    fua = tgt.eFUA_ID.to_numpy()[cand]
    u_rows = test_rows if len(test_rows) <= 400 else rng.choice(test_rows, 400, replace=False)
    allz = np.vstack([Z[u_rows], Z[recent], Z[cand]])
    C = al.kern(allz, allz, LS)
    nu, nr = len(u_rows), len(recent)
    for j in range(nr):                      # condition on the recent months
        col = C[:, nu + j].copy()
        C -= np.outer(col, col) / (C[nu + j, nu + j] + RHO)
    off = nu + nr
    chosen, used = [], {u: 0 for u in lab}
    while len(chosen) < sum(quota.values()):
        cx = C[off:, off:].diagonal()
        gain = (C[:nu, off:] ** 2).sum(0) / (cx + RHO)
        for j, c in enumerate(cand):
            if used[fua[j]] >= quota[fua[j]] or c in chosen:
                gain[j] = -np.inf
        j = int(np.argmax(gain)); chosen.append(cand[j]); used[fua[j]] += 1
        col = C[:, off + j].copy()
        C -= np.outer(col, col) / (C[off + j, off + j] + RHO)
    return np.concatenate([recent, np.array(chosen, dtype=int)])


STRATS = {"S0 last consecutive": s0, "H recent + target-aware": hybrid}
preds_all = []
t0 = time.time()
for m in (3, 6):
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, k, 20)):
            tst = np.flatnonzero((tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy())
            b = tgt.iloc[tst]; vy = b[Y].to_numpy(); eraw = b.raw.to_numpy()
            frame = b[["eFUA_ID", "t", "year", "month", "quarter_id"]].copy()
            frame.insert(0, "split", sid); frame.insert(0, "k", k); frame.insert(0, "m", m)
            frame["observed"] = vy
            frame["Direct"] = eraw
            for sname, fn in STRATS.items():
                rng = np.random.default_rng([al.SEED, m, k, sid, 0])
                rows = fn(train_ids, m, rng, tst) if fn is hybrid else fn(train_ids, m, rng)
                a = tgt.iloc[rows]; y = a[Y].to_numpy(); raw = a.raw.to_numpy()
                aff_l, aff_t = al.affine(raw, y, eraw)
                frame[f"{sname}|0|Source-affine"] = core.pos(aff_t)
                g, _ = al.eca_predict("ECA-fixed", THETA, Z[rows], y - aff_l, Z[tst], y.mean())
                frame[f"{sname}|0|ECA-R"] = core.pos(aff_t + g)
                s_l, s_t = al.scaled_source(raw, eraw, SRC_SD)
                d2, _ = al.eca2_predict("ECA2", TH2, Z[rows], s_l, y - raw, Z[tst], s_t)
                frame[f"{sname}|0|ECA-S"] = core.pos(eraw + d2)
                frame[f"{sname}|0|n_rows"] = len(rows)
            preds_all.append(frame)
        print(f"{task} hybrid m={m} k={k} done ({time.time() - t0:.0f}s)", flush=True)

P = pd.concat(preds_all)
P.to_csv(out / f"{task}_hybrid_predictions.csv.gz", index=False, float_format="%.6g")
ref = pd.read_csv(Path(__file__).resolve().parents[1] / f"results/{task}_q2_predictions.csv.gz")
key = ["m", "k", "split", "eFUA_ID", "t"]
mrg = P[key + [f"S0 last consecutive|0|{x}" for x in ("Source-affine", "ECA-R", "ECA-S")]].merge(
    ref[key + [f"S0 last consecutive|0|{x}" for x in ("Source-affine", "ECA-R", "ECA-S")]], on=key, suffixes=("", "_ref"))
dev = max(float(np.max(np.abs(mrg[c] - mrg[c + "_ref"]) / np.maximum(1, np.abs(mrg[c + "_ref"]))))
          for c in [f"S0 last consecutive|0|{x}" for x in ("Source-affine", "ECA-R", "ECA-S")])
print(f"S0 reproduces run_q2: rows {len(mrg)}/{len(P)}, max relative deviation {dev:.2e}", flush=True)
