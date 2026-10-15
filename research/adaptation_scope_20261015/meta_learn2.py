"""Amendment 2: meta-learn the priors of ECA2 (shrink-toward-source adaptation) on source-side pseudo-transfers.

Same 216 pseudo-transfer tasks as meta_learn.py (same regions, random draws and seed). Objective: mean over tasks
of log(MSE of ECA2 / MSE of direct transfer) on the pseudo-target's test rows. Three families are fitted: ARD
(8 length-scales + level, scale, condition and noise SDs), isotropic (one length-scale) and affine (level and scale
only, no condition component).

Usage: python meta_learn2.py <out_dir>
Writes theta2.json, meta2_tasks.csv and meta2_log.txt.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import adapt_lib as al
from adapt_lib import X, Y, core

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "meta2_log.txt", "w")


def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); log.write(s + "\n"); log.flush()


P = al._P
reg = P.eFUA_ID.map(al.FUA_REGION)
DEV = sorted(r for r in set(al.FUA_REGION.values()) if r != al.LOCKED)
rng = np.random.default_rng(al.SEED)
t_of = P.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
T_LO, T_HI = int(t_of[(2019, 1)]), int(t_of[(2020, 12)])
T_END_LO = int(t_of[(2019, 6)])

tasks, rows = [], []
for c in DEV:
    src = P[P.country_iso.eq("CHN") & reg.isin([r for r in DEV if r != c])]
    tgt = P[P.country_iso.eq("CHN") & reg.eq(c)]
    elig = al.eligible_fuas(tgt, .8)
    tgt = tgt[tgt.eFUA_ID.isin(elig)].copy()
    f = al.source_model(src)
    src_sd = float(np.std(core.pos(f.predict(src[X]))))
    tgt["raw"] = core.pos(f.predict(tgt[X]))
    z = al.zmat(tgt)
    for m in (3, 6, 12):
        for design in ("consecutive", "scattered"):
            for rep in range(4):
                lab = rng.choice(elig, 5, replace=False)
                is_lab = tgt.eFUA_ID.isin(lab).to_numpy()
                if design == "consecutive":
                    t_end = int(rng.integers(T_END_LO, T_HI + 1))
                    sel = is_lab & tgt.t.between(t_end - m + 1, t_end).to_numpy()
                else:
                    sel = np.zeros(len(tgt), bool)
                    for u in lab:
                        idx = np.flatnonzero((tgt.eFUA_ID.eq(u) & tgt.t.between(T_LO, T_HI)).to_numpy())
                        sel[rng.choice(idx, min(m, len(idx)), replace=False)] = True
                test = (~is_lab) & tgt.year.ge(2021).to_numpy()
                if sel.sum() < 5 or test.sum() < 20:
                    continue
                y_l, y_t = tgt[Y].to_numpy()[sel], tgt[Y].to_numpy()[test]
                raw_l, raw_t = tgt.raw.to_numpy()[sel], tgt.raw.to_numpy()[test]
                s_l, s_t = al.scaled_source(raw_l, raw_t, src_sd)
                aff_l, aff_t = al.affine(raw_l, y_l, raw_t)
                mse_dir = float(np.mean((y_t - raw_t) ** 2))
                tasks.append(dict(region=c, m=m, design=design, zl=z[sel], sl=s_l, r=y_l - raw_l, zt=z[test], st=s_t,
                                  y_t=y_t, raw_t=raw_t, ybar=y_l.mean(), mse_dir=mse_dir))
                rows.append(dict(region=c, m=m, design=design, rep=rep, n_lab=int(sel.sum()), n_test=int(test.sum()),
                                 rmse_direct=np.sqrt(mse_dir),
                                 rmse_affine=np.sqrt(np.mean((y_t - core.pos(aff_t)) ** 2))))
    say(f"region {c}: {len(elig)} eligible FUAs, source rows {len(src)}, tasks so far {len(tasks)}")

D = len(al.COND)
FAMILIES = {"ARD": D + 4, "isotropic": 5, "affine": 3}


def unpack(family, x):
    if family == "ARD":
        return np.exp(x[:D]), *np.exp(x[D:])
    if family == "isotropic":
        return np.full(D, np.exp(x[0])), *np.exp(x[1:])
    aa, ab, an = np.exp(x)
    return np.ones(D), aa, ab, 0.0, an


def task_mse(t, hp):
    delta = al.eca2_fit_predict(t["zl"], t["sl"], t["r"], t["zt"], t["st"], t["ybar"], *hp)
    return float(np.mean((t["y_t"] - core.pos(t["raw_t"] + delta)) ** 2))


def objective(x, family, idx):
    hp = unpack(family, x)
    return float(np.mean([np.log(task_mse(tasks[i], hp) / tasks[i]["mse_dir"]) for i in idx]))


def starts(family):
    base = [(0.0, .2, .2, .1, .1), (np.log(3.0), .05, .05, .05, .2), (np.log(.5), .5, .5, .2, .05)]
    out = []
    for l, aa, ab, ag, an in base:
        if family == "ARD":
            out.append(np.r_[np.full(D, l), np.log([aa, ab, ag, an])])
        elif family == "isotropic":
            out.append(np.r_[l, np.log([aa, ab, ag, an])])
        else:
            out.append(np.log([aa, ab, an]))
    return out


def fit(family, idx):
    best = None
    for s in starts(family):
        res = minimize(objective, s, args=(family, idx), method="L-BFGS-B",
                       bounds=[(-6, 5)] * FAMILIES[family], options=dict(maxiter=80))
        if best is None or res.fun < best.fun:
            best = res
    return best


all_idx = np.arange(len(tasks))
t0 = time.time()
say(f"{len(tasks)} pseudo-transfer tasks; direct transfer has objective 0 by definition")
affine_obj = float(np.mean([np.log(r["rmse_affine"] ** 2 / r["rmse_direct"] ** 2) for r in rows]))
say(f"source-affine (thesis) objective: {affine_obj:.4f}")
folds = [DEV[i::3] for i in range(3)]
cv = []
for fold in folds:
    tr = [i for i in all_idx if tasks[i]["region"] not in fold]
    te = [i for i in all_idx if tasks[i]["region"] in fold]
    res = fit("ARD", tr)
    cv.append(dict(held_out=fold, train_obj=float(res.fun), held_out_obj=objective(res.x, "ARD", te)))
    say(f"held-out regions {fold}: train {res.fun:.4f}, held-out {cv[-1]['held_out_obj']:.4f}")
theta2 = {"conditions": al.COND, "region_cv": cv, "n_tasks": len(tasks), "source_affine_objective": affine_obj}
for family in FAMILIES:
    res = fit(family, all_idx)
    x = res.x
    if family == "ARD":
        theta2[family] = dict(ls=x[:D].tolist(), aa=float(x[D]), ab=float(x[D + 1]), ag=float(x[D + 2]),
                              an=float(x[D + 3]), objective=float(res.fun))
    elif family == "isotropic":
        theta2[family] = dict(ls=[float(x[0])] * D, aa=float(x[1]), ab=float(x[2]), ag=float(x[3]), an=float(x[4]),
                              objective=float(res.fun))
    else:
        theta2[family] = dict(ls=None, aa=float(x[0]), ab=float(x[1]), ag=None, an=float(x[2]),
                              objective=float(res.fun))
    hp = unpack(family, x)
    say(f"{family}: objective {res.fun:.4f} (mean MSE ratio to direct {np.exp(res.fun):.3f}); "
        f"aa {hp[1]:.3f} ab {hp[2]:.3f} ag {hp[3]:.3f} an {hp[4]:.3f}"
        + (f"; length-scales {dict(zip(al.COND, np.round(hp[0], 2)))}" if family != "affine" else ""))
    for i, t in enumerate(tasks):
        rows[i][f"rmse_eca2_{family}"] = np.sqrt(task_mse(t, hp))
theta2["seconds"] = round(time.time() - t0)
(OUT / "theta2.json").write_text(json.dumps(theta2, indent=1))
pd.DataFrame(rows).to_csv(OUT / "meta2_tasks.csv", index=False)
