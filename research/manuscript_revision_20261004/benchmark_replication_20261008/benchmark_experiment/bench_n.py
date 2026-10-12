"""Addendum B (PROTOCOL.md): P-N(h), short target history with NO contemporaneous source labels.

As bench_m.py (training-target rows = k training FUAs in the last h months of 2020; test rows = test
FUAs in 2021-2022) but every method sees source rows from years <= 2020 only. LDF-based models are
not run (no same-month labelled neighbours exist for 2021-2022 test months).

Usage: python3 bench_n.py --task yrd --m 3 [--ks 5,11] [--splits 20]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import bench_harness as bh
from bench_harness import core, X, Y, LDF_TAGS
from paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--ks", default=None); ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    assert a.m in (3, 6, 12)
    ks = [int(v) for v in a.ks.split(",")] if a.ks else ([5, 11] if a.task == "yrd" else [3, 5])
    out = RESULTS / f"{a.task}_N{a.m}"
    out.mkdir(parents=True, exist_ok=True)
    src, tgt, eligible = bh.load_task(a.task)
    src = src[src.year.le(2020)].reset_index(drop=True)          # no contemporaneous source labels
    base = core.tree(True).fit(src[X], src[Y])
    shuf = core.tree(True).fit(src[X], np.random.default_rng(5993).permutation(src[Y].to_numpy()))
    tgt["src_pred"] = core.pos(base.predict(tgt[X])); tgt["shuf_pred"] = core.pos(shuf.predict(tgt[X]))
    emb = core.embed(base, tgt[X])
    t_dec2020 = int(tgt.loc[tgt.year.eq(2020) & tgt.month.eq(12), "t"].iloc[0])
    window = (t_dec2020 - a.m + 1, t_dec2020)
    (out / "design.json").write_text(json.dumps({"m": a.m, "train_t_window": window, "test_years": [2021, 2022],
                                                 "source_years": "<=2020", "source_rows": len(src),
                                                 "eligible": len(eligible)}, indent=1))
    feat = LDF_TAGS[:-1]
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(bh.benchmark_splits(eligible, k, a.splits)):
            f = out / f"k{k}_split{sid:02d}.csv"
            if f.exists():
                continue
            t0 = time.time()
            trn = tgt[tgt.eFUA_ID.isin(train_ids) & tgt.t.between(*window)]
            tst = tgt[tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)]
            if len(trn) < 5:
                pd.DataFrame([{"task": a.task, "protocol": f"N{a.m}", "k": k, "split": sid, "model": "SKIPPED",
                               "n_train_rows": len(trn)}]).to_csv(f, index=False)
                continue
            # benchmark centre features: (x - mean) / sd over source + training-target rows (ddof 0), pm25 placeholder 0
            pool = pd.concat([src[feat], trn[feat]]).to_numpy(float)
            mu, sd = pool.mean(axis=0), pool.std(axis=0); sd[sd == 0] = 1
            z = lambda d: np.c_[(d[feat].to_numpy(float) - mu) / sd, np.zeros(len(d))]
            cs, ct, cv = z(src), z(trn), z(tst)
            sy, ty, vy = src[Y].to_numpy(float), trn[Y].to_numpy(float), tst[Y].to_numpy(float)
            preds = {}
            preds["RF (target-only)"] = bh.rf().fit(ct, ty).predict(cv)
            preds["GBR (target-only)"] = bh.gbr().fit(ct, ty).predict(cv)
            w = bh.itl_weights("NNW", cs, ct)
            preds["NNW"] = bh.itl_fit_predict(cs, sy, ct, ty, cv, w)
            preds["NNW (sample-weight)"] = bh.itl_fit_predict(cs, sy, ct, ty, cv, w, released=False)
            x, ex, y = trn[X], tst[X], ty
            preds["Local mean"] = np.full(len(ex), y.mean())
            zz0 = core.norm(); xx = zz0.fit_transform(x); ee = zz0.transform(ex)
            preds["Local ridge"] = core.pos(Ridge(alpha=10).fit(xx, y).predict(ee))
            preds["Local RBF"] = core.kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y)
            preds["Local RF"] = core.pos(RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee))
            preds["Local Extra Trees"] = core.pos(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee))
            preds["Local LightGBM"] = core.pos(core.tree().fit(x, y).predict(ex))
            raw, eraw = trn.src_pred.to_numpy(), tst.src_pred.to_numpy()
            s1 = StandardScaler(); rx = s1.fit_transform(raw[:, None]); re = s1.transform(eraw[:, None])
            preds["Source-affine (ours)"] = core.pos(Ridge(alpha=10).fit(rx, y).predict(re))
            preds["Source-residual (ours)"] = core.pos(eraw + core.tree().fit(x, y - raw).predict(ex))
            ia = tgt.index.get_indexer(trn.index); ib = tgt.index.get_indexer(tst.index)
            ea, eb = emb[ia], emb[ib]
            preds["Source-kernel (ours)"] = core.kernel((ea @ ea.T).toarray(), (eb @ ea.T).toarray(), y)
            sr, se = trn.shuf_pred.to_numpy(), tst.shuf_pred.to_numpy()
            s2 = StandardScaler(); preds["Shuffled-source affine (control)"] = core.pos(Ridge(alpha=10).fit(s2.fit_transform(sr[:, None]), y).predict(s2.transform(se[:, None])))
            rows = [{"task": a.task, "protocol": f"N{a.m}", "k": k, "split": sid, "model": m, **bh.score(vy, p),
                     "n_train_rows": len(ty), "n_test_fuas": len(test_ids), "seconds": round(time.time() - t0, 1)}
                    for m, p in preds.items()]
            pd.DataFrame(rows).to_csv(f, index=False)
            r = {row["model"]: row["rmse"] for row in rows}
            print(f"{a.task}-N{a.m} k={k} split={sid} {time.time()-t0:.0f}s rows={len(ty)} affine={r['Source-affine (ours)']:.3f} "
                  f"NNW={r['NNW']:.3f} GBR={r['GBR (target-only)']:.3f}", flush=True)


if __name__ == "__main__":
    main()
