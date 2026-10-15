"""Meta-learn where to adapt (ARD length-scales) and default adjustment size on source-side pseudo-transfers.

Usage: python meta_learn.py <out_dir>
Writes theta.json (ARD and isotropic), meta_tasks.csv (one row per pseudo-transfer task) and meta_log.txt.
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
log = open(OUT / "meta_log.txt", "w")


def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); log.write(s + "\n"); log.flush()


P = al._P
reg = P.eFUA_ID.map(al.FUA_REGION)
DEV = sorted(r for r in set(al.FUA_REGION.values()) if r != al.LOCKED)
rng = np.random.default_rng(al.SEED)
t_of = P.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
T_LO, T_HI = int(t_of[(2019, 1)]), int(t_of[(2020, 12)])
T_END_LO = int(t_of[(2019, 6)])

tasks, meta_rows = [], []
for c in DEV:
    src = P[P.country_iso.eq("CHN") & reg.isin([r for r in DEV if r != c])]
    tgt = P[P.country_iso.eq("CHN") & reg.eq(c)]
    elig = al.eligible_fuas(tgt, .8)
    tgt = tgt[tgt.eFUA_ID.isin(elig)].copy()
    f = al.source_model(src)
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
                aff_l, aff_t = al.affine(raw_l, y_l, raw_t)
                mse_aff = float(np.mean((y_t - core.pos(aff_t)) ** 2))
                tasks.append(dict(region=c, m=m, design=design, zl=z[sel], r=y_l - aff_l, zt=z[test], y_t=y_t,
                                  aff_t=aff_t, ybar=y_l.mean(), mse_aff=mse_aff))
                meta_rows.append(dict(region=c, m=m, design=design, rep=rep, n_lab=int(sel.sum()),
                                      n_test=int(test.sum()), rmse_affine=np.sqrt(mse_aff)))
    say(f"region {c}: {len(elig)} eligible FUAs, source rows {len(src)}, tasks so far {len(tasks)}")


def objective(theta, idx, iso=False):
    d = len(al.COND)
    ls = np.full(d, np.exp(theta[0])) if iso else np.exp(theta[:d])
    af, an = np.exp(theta[-2]), np.exp(theta[-1])
    vals = []
    for i in idx:
        t = tasks[i]
        g = al.gp_mean(t["zl"], t["r"], t["zt"], ls, af * t["ybar"], an * t["ybar"])
        vals.append(np.log(np.mean((t["y_t"] - core.pos(t["aff_t"] + g)) ** 2) / t["mse_aff"]))
    return float(np.mean(vals))


def fit(idx, iso=False):
    d = 1 if iso else len(al.COND)
    starts = [np.r_[np.full(d, 0.0), np.log(.1), np.log(.1)],
              np.r_[np.full(d, np.log(3.0)), np.log(.05), np.log(.2)],
              np.r_[np.full(d, np.log(.5)), np.log(.2), np.log(.05)]]
    best = None
    for s in starts:
        res = minimize(objective, s, args=(idx, iso), method="L-BFGS-B", bounds=[(-4, 5)] * (d + 2),
                       options=dict(maxiter=80))
        if best is None or res.fun < best.fun:
            best = res
    return best


all_idx = np.arange(len(tasks))
t0 = time.time()
say(f"{len(tasks)} pseudo-transfer tasks; affine-only objective is 0 by definition")
# region-held-out check of the meta-learning (diagnostic): three folds of three regions
folds = [DEV[i::3] for i in range(3)]
cv = []
for fold in folds:
    tr = [i for i in all_idx if tasks[i]["region"] not in fold]
    te = [i for i in all_idx if tasks[i]["region"] in fold]
    res = fit(tr)
    cv.append(dict(held_out=fold, train_obj=res.fun, held_out_obj=objective(res.x, te)))
    say(f"held-out regions {fold}: train objective {res.fun:.4f}, held-out objective {cv[-1]['held_out_obj']:.4f}")
final = fit(all_idx)
iso = fit(all_idx, iso=True)
d = len(al.COND)
theta = {"ARD": {"ls": final.x[:d].tolist(), "af": float(final.x[d]), "an": float(final.x[d + 1]),
                 "objective": float(final.fun)},
         "isotropic": {"ls": [float(iso.x[0])] * d, "af": float(iso.x[1]), "an": float(iso.x[2]),
                       "objective": float(iso.fun)},
         "conditions": al.COND, "region_cv": cv, "n_tasks": len(tasks), "seconds": round(time.time() - t0)}
(OUT / "theta.json").write_text(json.dumps(theta, indent=1))
pd.DataFrame(meta_rows).to_csv(OUT / "meta_tasks.csv", index=False)
say("ARD length-scales:", dict(zip(al.COND, np.round(np.exp(final.x[:d]), 2))))
say(f"a_f {np.exp(final.x[d]):.3f}, a_n {np.exp(final.x[d + 1]):.3f}, objective {final.fun:.4f} "
    f"(mean MSE ratio {np.exp(final.fun):.3f}); isotropic objective {iso.fun:.4f}, ls {np.exp(iso.x[0]):.2f}")
