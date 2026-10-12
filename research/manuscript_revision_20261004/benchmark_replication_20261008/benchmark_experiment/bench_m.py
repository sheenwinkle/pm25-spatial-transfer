"""Addendum A (PROTOCOL.md): temporal label budget P-M(m), m in {3, 6, 12}.

Training-target rows = k training FUAs in the last m months of 2020; test rows = test FUAs in
2021-2022; source rows = all source rows. Same splits, LDF pipeline and estimators as
bench_harness.py; KLIEP and KMM are not run under P-M (fixed before fitting).

Usage: python3 bench_m.py --task yrd|aus --m 3 [--ks 5,11] [--splits 20]
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
from bench_harness import core, X, Y, CENTER, MEASURE
from paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--ks", default=None); ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    assert a.m in (3, 6, 12)
    ks = [int(v) for v in a.ks.split(",")] if a.ks else ([5, 11] if a.task == "yrd" else [3, 5])
    out = RESULTS / f"{a.task}_M{a.m}"
    out.mkdir(parents=True, exist_ok=True)
    src, tgt, eligible = bh.load_task(a.task)
    base = core.tree(True).fit(src[X], src[Y])
    shuf = core.tree(True).fit(src[X], np.random.default_rng(5993).permutation(src[Y].to_numpy()))
    tgt["src_pred"] = core.pos(base.predict(tgt[X])); tgt["shuf_pred"] = core.pos(shuf.predict(tgt[X]))
    emb = core.embed(base, tgt[X])
    t_dec2020 = int(tgt.loc[tgt.year.eq(2020) & tgt.month.eq(12), "t"].iloc[0])
    window = (t_dec2020 - a.m + 1, t_dec2020)
    (out / "design.json").write_text(json.dumps({"m": a.m, "train_t_window": window, "test_years": [2021, 2022],
                                                 "eligible": len(eligible)}, indent=1))
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(bh.benchmark_splits(eligible, k, a.splits)):
            f = out / f"k{k}_split{sid:02d}.csv"
            if f.exists():
                continue
            t0 = time.time()
            trn_mask = tgt.eFUA_ID.isin(train_ids) & tgt.t.between(*window)
            tst_mask = tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)
            trn, tst = tgt[trn_mask], tgt[tst_mask]
            if len(trn) < 5:
                pd.DataFrame([{"task": a.task, "protocol": f"M{a.m}", "k": k, "split": sid, "model": "SKIPPED",
                               "n_train_rows": len(trn)}]).to_csv(f, index=False)
                continue
            tr_all = pd.concat([src, trn])
            mu, sd = tr_all[MEASURE].mean(), tr_all[MEASURE].std()
            clouds, labels, order = bh.compose_clouds(src, trn, tst, mu, sd)
            norm = bh.normalise(clouds)
            ldf = bh.train_ldf(norm, labels, seed=sid)
            ctr = {kk: v[:, :, CENTER].copy() for kk, v in norm.items()}
            ctr_ldf = {kk: v.copy() for kk, v in ctr.items()}
            for kk in ctr_ldf:
                ctr_ldf[kk][:, -1] = ldf[kk]
            sy, ty, vy = labels["source"], labels["train"], labels["test"]
            preds = {}
            preds["RF (target-only)"] = bh.rf().fit(ctr["train"], ty).predict(ctr["test"])
            preds["GBR (target-only)"] = bh.gbr().fit(ctr["train"], ty).predict(ctr["test"])
            preds["GBR[LDF] (target-only)"] = bh.gbr().fit(ctr_ldf["train"], ty).predict(ctr_ldf["test"])
            w_nf = bh.itl_weights("NNW", ctr["source"], ctr["train"])
            w_l = bh.itl_weights("NNW", ctr_ldf["source"], ctr_ldf["train"])
            preds["NNW"] = bh.itl_fit_predict(ctr["source"], sy, ctr["train"], ty, ctr["test"], w_nf)
            preds["NNW[LDF]"] = bh.itl_fit_predict(ctr_ldf["source"], sy, ctr_ldf["train"], ty, ctr_ldf["test"], w_l)
            preds["NNW (sample-weight)"] = bh.itl_fit_predict(ctr["source"], sy, ctr["train"], ty, ctr["test"], w_nf, released=False)
            preds["NNW[LDF] (sample-weight)"] = bh.itl_fit_predict(ctr_ldf["source"], sy, ctr_ldf["train"], ty, ctr_ldf["test"], w_l, released=False)
            tst_o = tgt.loc[order["test"]]; trn_o = tgt.loc[order["train"]]
            assert np.allclose(tst_o[Y].to_numpy(), vy) and np.allclose(trn_o[Y].to_numpy(), ty)
            x, ex, y = trn_o[X], tst_o[X], trn_o[Y].to_numpy()
            preds["Local mean"] = np.full(len(ex), y.mean())
            z = core.norm(); xx = z.fit_transform(x); ee = z.transform(ex)
            preds["Local ridge"] = core.pos(Ridge(alpha=10).fit(xx, y).predict(ee))
            preds["Local RBF"] = core.kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y)
            preds["Local RF"] = core.pos(RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee))
            preds["Local Extra Trees"] = core.pos(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee))
            preds["Local LightGBM"] = core.pos(core.tree().fit(x, y).predict(ex))
            raw, eraw = trn_o.src_pred.to_numpy(), tst_o.src_pred.to_numpy()
            zz = StandardScaler(); rx = zz.fit_transform(raw[:, None]); re = zz.transform(eraw[:, None])
            preds["Source-affine (ours)"] = core.pos(Ridge(alpha=10).fit(rx, y).predict(re))
            preds["Source-residual (ours)"] = core.pos(eraw + core.tree().fit(x, y - raw).predict(ex))
            ia = np.searchsorted(tgt.index.to_numpy(), order["train"]); ib = np.searchsorted(tgt.index.to_numpy(), order["test"])
            ea, eb = emb[ia], emb[ib]
            preds["Source-kernel (ours)"] = core.kernel((ea @ ea.T).toarray(), (eb @ ea.T).toarray(), y)
            sr, se = trn_o.shuf_pred.to_numpy(), tst_o.shuf_pred.to_numpy()
            z2 = StandardScaler(); preds["Shuffled-source affine (control)"] = core.pos(Ridge(alpha=10).fit(z2.fit_transform(sr[:, None]), y).predict(z2.transform(se[:, None])))
            z3 = StandardScaler(); A = z3.fit_transform(np.c_[raw, ldf["train"]]); B = z3.transform(np.c_[eraw, ldf["test"]])
            preds["Source-affine[LDF] (ours)"] = core.pos(Ridge(alpha=10).fit(A, y).predict(B))
            rows = [{"task": a.task, "protocol": f"M{a.m}", "k": k, "split": sid, "model": mname, **bh.score(vy, p),
                     "n_train_rows": len(ty), "n_test_fuas": len(test_ids), "seconds": round(time.time() - t0, 1)}
                    for mname, p in preds.items()]
            pd.DataFrame(rows).to_csv(f, index=False)
            r = {row["model"]: row["rmse"] for row in rows}
            print(f"{a.task}-M{a.m} k={k} split={sid} {time.time()-t0:.0f}s rows={len(ty)} "
                  f"affine={r['Source-affine (ours)']:.3f} NNW[LDF]={r['NNW[LDF]']:.3f} GBR={r['GBR (target-only)']:.3f}", flush=True)


if __name__ == "__main__":
    main()
