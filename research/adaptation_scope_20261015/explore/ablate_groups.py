"""Exploratory, added after the T3 run: where to adapt, by condition group.

The ECA-S prior is re-learnt with one group of conditions removed from z and compared with the full prior on the
same 216 within-China pseudo-transfers, using the region-held-out cross-validation of meta_learn2.py (learn on six
development regions, evaluate on the other three). Groups: meteorology (temperature, diurnal range, humidity, wind,
precipitation), fire (log fire count within 100 km) and season (local-season sine and cosine); "none" removes the
condition term altogether (the affine family). Tasks, objective, starting points and optimiser are those of
meta_learn2.py; the task-building loop is copied from it so that the 216 tasks are identical (checked against
meta/meta2_tasks.csv).

Usage: python explore/ablate_groups.py <out_dir>   (run from the study folder, with the env.sh paths set)
Writes ablation_tasks.csv and ablation_summary.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_lib as al
from adapt_lib import X, Y, core

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {"meteorology": [0, 1, 2, 3, 4], "fire": [5], "season": [6, 7]}
VARIANTS = {"full": list(range(8))}
VARIANTS.update({f"no {g}": [j for j in range(8) if j not in idx] for g, idx in GROUPS.items()})
VARIANTS["no conditions"] = []
log = open(OUT / "ablation_log.txt", "w")


def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); log.write(s + "\n"); log.flush()


# ---- the 216 pseudo-transfer tasks, built exactly as in meta_learn2.py
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
                tasks.append(dict(region=c, zl=z[sel], sl=s_l, r=y_l - raw_l, zt=z[test], st=s_t,
                                  y_t=y_t, raw_t=raw_t, ybar=y_l.mean(), mse_dir=mse_dir))
                rows.append(dict(region=c, m=m, design=design, rep=rep, rmse_direct=np.sqrt(mse_dir),
                                 rmse_affine=np.sqrt(np.mean((y_t - core.pos(aff_t)) ** 2))))
    say(f"region {c}: tasks so far {len(tasks)}")

ref = pd.read_csv(Path(__file__).resolve().parents[1] / "meta/meta2_tasks.csv")
same = np.allclose(ref.rmse_direct, [r["rmse_direct"] for r in rows]) and \
    np.allclose(ref.rmse_affine, [r["rmse_affine"] for r in rows])
say(f"tasks identical to meta/meta2_tasks.csv: {same}")
assert same


# ---- objective and fitting, as in meta_learn2.py, for a subset of conditions
def unpack(keep, x):
    D = len(keep)
    if D == 0:
        aa, ab, an = np.exp(x)
        return np.ones(1), aa, ab, 0.0, an
    return np.exp(x[:D]), *np.exp(x[D:])


def task_mse(t, keep, hp):
    zl = t["zl"][:, keep] if keep else np.zeros((len(t["zl"]), 1))
    zt = t["zt"][:, keep] if keep else np.zeros((len(t["zt"]), 1))
    delta = al.eca2_fit_predict(zl, t["sl"], t["r"], zt, t["st"], t["ybar"], *hp)
    return float(np.mean((t["y_t"] - core.pos(t["raw_t"] + delta)) ** 2))


def objective(x, keep, idx):
    hp = unpack(keep, x)
    return float(np.mean([np.log(task_mse(tasks[i], keep, hp) / tasks[i]["mse_dir"]) for i in idx]))


def starts(keep):
    base = [(0.0, .2, .2, .1, .1), (np.log(3.0), .05, .05, .05, .2), (np.log(.5), .5, .5, .2, .05)]
    if not keep:
        return [np.log([aa, ab, an]) for _, aa, ab, _, an in base]
    return [np.r_[np.full(len(keep), l), np.log([aa, ab, ag, an])] for l, aa, ab, ag, an in base]


def fit(keep, idx):
    n = len(keep) + 4 if keep else 3
    best = None
    for s in starts(keep):
        res = minimize(objective, s, args=(keep, idx), method="L-BFGS-B", bounds=[(-6, 5)] * n,
                       options=dict(maxiter=80))
        if best is None or res.fun < best.fun:
            best = res
    return best


all_idx = np.arange(len(tasks))
folds = [DEV[i::3] for i in range(3)]
held = {v: np.full(len(tasks), np.nan) for v in VARIANTS}
fold_obj = {v: [] for v in VARIANTS}
t0 = time.time()
for v, keep in VARIANTS.items():
    for fold in folds:
        tr = [i for i in all_idx if tasks[i]["region"] not in fold]
        te = [i for i in all_idx if tasks[i]["region"] in fold]
        res = fit(keep, tr)
        hp = unpack(keep, res.x)
        for i in te:
            held[v][i] = task_mse(tasks[i], keep, hp)
        fold_obj[v].append(float(np.mean([np.log(held[v][i] / tasks[i]["mse_dir"]) for i in te])))
        say(f"{v:14s} held-out {fold}: {fold_obj[v][-1]:.4f}  ({time.time() - t0:.0f}s)")

tab = pd.DataFrame(rows)
for v in VARIANTS:
    tab[f"rmse_{v.replace(' ', '_')}"] = np.sqrt(held[v])
tab.to_csv(OUT / "ablation_tasks.csv", index=False)

boot = np.random.default_rng(20261015)
regions = np.array(DEV)
summary = {"note": "exploratory, after the T3 run; region-held-out cross-validation over the 216 pseudo-transfers",
           "folds": [list(map(int, f)) for f in folds],
           "full_cv_matches_theta2": None, "variants": {}}
th2 = json.loads((Path(__file__).resolve().parents[1] / "meta/theta2.json").read_text())
summary["full_cv_matches_theta2"] = bool(np.allclose(fold_obj["full"], [c["held_out_obj"] for c in th2["region_cv"]],
                                                     atol=1e-6))
r_full = tab["rmse_full"].to_numpy()
for v in VARIANTS:
    rv = tab[f"rmse_{v.replace(' ', '_')}"].to_numpy()
    gain_vs_direct = 100 * (1 - rv / tab.rmse_direct.to_numpy())
    entry = dict(held_out_objective_by_fold=[round(x, 4) for x in fold_obj[v]],
                 mean_gain_vs_direct=round(float(gain_vs_direct.mean()), 2))
    if v != "full":
        g = 100 * (1 - r_full / rv)            # positive: removing the group raises the error
        by_reg = pd.Series(g).groupby(tab.region.to_numpy())
        draws = []
        for _ in range(1000):
            pick = boot.choice(regions, len(regions), replace=True)
            draws.append(np.mean(np.concatenate([g[tab.region.to_numpy() == r] for r in pick])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        entry.update(full_vs_variant_gain=round(float(g.mean()), 2), lo=round(float(lo), 2), hi=round(float(hi), 2),
                     tasks_full_better=int((g > 0).sum()), tasks=len(g),
                     regions_full_better=int((by_reg.mean() > 0).sum()), regions=len(regions),
                     by_region={int(k): round(float(x), 2) for k, x in by_reg.mean().items()})
    summary["variants"][v] = entry
    say(v, json.dumps(entry))
summary["seconds"] = round(time.time() - t0)
(OUT / "ablation_summary.json").write_text(json.dumps(summary, indent=1))
say("full CV reproduces theta2 region_cv:", summary["full_cv_matches_theta2"])
