"""Q2: which conditions the few labels should cover (see PROTOCOL.md, Amendment 2).

Usage: python run_q2.py <task: yrd|aus|locked> <theta.json> <theta2.json> <out_dir>
S3 and S4 use the condition kernel of ECA-R (theta.json), as pre-specified. Methods: direct transfer, source-affine,
ECA-R (ECA-fixed), ECA-S (ECA2) and the six local learners.
Writes <out_dir>/<task>_q2_metrics.csv, _predictions.csv.gz and _selected.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import adapt_lib as al
from adapt_lib import X, Y, core

task, theta_path, theta2_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
th = json.loads(theta_path.read_text())["ARD"]
THETA = dict(ls=np.array(th["ls"]), af=th["af"], an=th["an"])
LS = np.exp(THETA["ls"]); RHO = (np.exp(THETA["an"]) / np.exp(THETA["af"])) ** 2   # noise-to-signal variance ratio
TH2 = json.loads(theta2_path.read_text())
ks = [3, 5] if task == "aus" else [5, 11]

src, tgt, eligible = al.load(task)
f = al.source_model(src)
tgt["raw"] = core.pos(f.predict(tgt[X]))
SRC_SD = float(np.std(core.pos(f.predict(src[X]))))
Z = al.zmat(tgt)
RAW = tgt.raw.to_numpy()
t_of = tgt.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
T_LO, T_HI, T_DEC = int(t_of[(2019, 1)]), int(t_of[(2020, 12)]), int(t_of[(2020, 12)])
# four seasons from the local-season month index (3 consecutive local months per season)
local_month = (np.round(np.arctan2(tgt.local_season_sin, tgt.local_season_cos) / (2 * np.pi) * 12) % 12).astype(int)
tgt["season4"] = (local_month // 3).to_numpy()


def pool_rows(fua):
    return np.flatnonzero((tgt.eFUA_ID.eq(fua) & tgt.t.between(T_LO, T_HI)).to_numpy())


def s0(lab, m, rng):
    return np.flatnonzero((tgt.eFUA_ID.isin(lab) & tgt.t.between(T_DEC - m + 1, T_DEC)).to_numpy())


def s1(lab, m, rng):
    return np.concatenate([rng.choice(pool_rows(u), min(m, len(pool_rows(u))), replace=False) for u in lab])


def s2(lab, m, rng):
    sel = []
    for u in lab:
        rows = pool_rows(u)
        order = rng.permutation(4)
        by = {s: list(rng.permutation(rows[tgt.season4.to_numpy()[rows] == s])) for s in range(4)}
        pick, i = [], 0
        while len(pick) < min(m, len(rows)):
            s = order[i % 4]; i += 1
            if by[s]:
                pick.append(by[s].pop())
        sel.extend(pick)
    return np.array(sel)


def s3(lab, m, rng):
    """Condition coverage: greedy max-min distance in the scaled conditions, within each labelled FUA."""
    sel = []
    for u in lab:
        rows = pool_rows(u); z = Z[rows] / LS
        first = int(np.argmin(((z - z.mean(0)) ** 2).sum(1)))
        pick = [first]
        d = ((z - z[first]) ** 2).sum(1)
        while len(pick) < min(m, len(rows)):
            j = int(np.argmax(d)); pick.append(j); d = np.minimum(d, ((z - z[j]) ** 2).sum(1))
        sel.extend(rows[pick])
    return np.array(sel)


def s4(lab, m, rng, test_rows):
    """Target-aware coverage: greedy reduction of the posterior variance of the condition correction over the
    conditions of the prediction rows (covariates only), with quota m per labelled FUA."""
    cand = np.concatenate([pool_rows(u) for u in lab])
    quota = {u: min(m, len(pool_rows(u))) for u in lab}
    fua = tgt.eFUA_ID.to_numpy()[cand]
    u_rows = test_rows if len(test_rows) <= 400 else rng.choice(test_rows, 400, replace=False)
    allz = np.vstack([Z[u_rows], Z[cand]])
    C = al.kern(allz, allz, LS)                    # prior covariance in units of the signal variance
    rho = RHO
    nu = len(u_rows)
    chosen, used = [], {u: 0 for u in lab}
    while len(chosen) < sum(quota.values()):
        cx = C[nu:, nu:].diagonal()
        gain = (C[:nu, nu:] ** 2).sum(0) / (cx + rho)
        for j, c in enumerate(cand):
            if used[fua[j]] >= quota[fua[j]] or c in chosen:
                gain[j] = -np.inf
        j = int(np.argmax(gain)); chosen.append(cand[j]); used[fua[j]] += 1
        col = C[:, nu + j].copy()
        C -= np.outer(col, col) / (C[nu + j, nu + j] + rho)
    return np.array(chosen)


STRATS = {"S0 last consecutive": (s0, 1), "S1 random": (s1, 5), "S2 season-spread": (s2, 5),
          "S3 condition coverage": (s3, 1), "S4 target-aware": (s4, 1)}
metrics, preds_all, selected = [], [], []
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
            for sname, (fn, reps) in STRATS.items():
                for rep in range(reps):
                    rng = np.random.default_rng([al.SEED, m, k, sid, rep])
                    rows = fn(train_ids, m, rng, tst) if fn is s4 else fn(train_ids, m, rng)
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
                    tag = f"{sname}|{rep}"
                    for mname, pr in p.items():
                        metrics.append(dict(task=task, m=m, k=k, split=sid, strategy=sname, rep=rep, model=mname,
                                            rmse=al.rmse(vy, pr), n_train_rows=len(rows)))
                        frame[f"{tag}|{mname}"] = pr
                    selected.append(dict(task=task, m=m, k=k, split=sid, strategy=sname, rep=rep, n=len(rows),
                                         **{c: float(a[c].mean()) for c in al.COND},
                                         sd_temp=float(a.monthly_power_temperature_c.std()),
                                         sd_fire=float(a.log_fire_count_100km.std()),
                                         n_seasons=int(a.season4.nunique()),
                                         months=" ".join(sorted(a.month.astype(str)))))
            preds_all.append(frame)
        print(f"{task} q2 m={m} k={k} done ({time.time() - t0:.0f}s)", flush=True)

stem = out / f"{task}_q2"
pd.DataFrame(metrics).to_csv(f"{stem}_metrics.csv", index=False)
pd.concat(preds_all).to_csv(f"{stem}_predictions.csv.gz", index=False, float_format="%.6g")
pd.DataFrame(selected).to_csv(f"{stem}_selected.csv", index=False, float_format="%.5g")
