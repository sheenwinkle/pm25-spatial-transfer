"""Post-hoc fixed-window replay with retained row predictions; see round-8 protocol.

Weighting check (review, 2026-10-11): also fits NNW[LDF] with the NNW weights passed as sample weights,
using the same LDF codes and weights, so Table 7 can report the stronger of the two released/standard
weightings. Run with a new --out directory; all other columns should reproduce window_results exactly.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

import bench_harness as bh
from paths import PANEL, RESULTS
from nnw_cpu_weights import nnw_weights

WINDOWS = {"2020Q4": (2020, 10, 12), "2020Q2": (2020, 4, 6),
           "2020Q3": (2020, 7, 9), "2019Q4": (2019, 10, 12)}
AFF = "Source-affine (ours)"
NNW = "NNW[LDF]"
NNW_SW = "NNW[LDF] (sample-weight)"  # same weights passed to the loss instead of multiplying the features
HYBRID = "Source-affine[LDF] (ours)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="2020Q4,2020Q2,2020Q3,2019Q4")
    ap.add_argument("--ks", default="5,11")
    ap.add_argument("--split-start", type=int, default=0)
    ap.add_argument("--split-stop", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "adapt", "tensorflow")}
    versions.update(python=platform.python_version(), system=platform.platform())
    (args.out / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    input_hash = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    print("Loading corrected panel", input_hash, flush=True)
    src, tgt, eligible = bh.load_task("yrd")
    base = bh.core.tree(True).fit(src[bh.X], src[bh.Y])
    tgt["src_pred"] = bh.core.pos(base.predict(tgt[bh.X]))
    for window_name in args.windows.split(","):
        year, lo, hi = WINDOWS[window_name]
        for k in map(int, args.ks.split(",")):
            for sid, (train_ids, test_ids) in enumerate(bh.benchmark_splits(eligible, k, 20)):
                if not args.split_start <= sid < args.split_stop:
                    continue
                folder = args.out / window_name / f"k{k}_split{sid:02d}"
                offsets = [0] if args.replay_only or window_name != "2020Q4" else [0, 10000]
                if all((folder / f"seed{offset}_metrics.json").exists() for offset in offsets):
                    continue
                start = time.monotonic()
                folder.mkdir(parents=True, exist_ok=True)
                trn = tgt[tgt.eFUA_ID.isin(train_ids) & tgt.year.eq(year) & tgt.month.between(lo, hi)]
                tst = tgt[tgt.eFUA_ID.isin(test_ids) & tgt.year.between(2021, 2022)]
                assert set(trn.eFUA_ID).isdisjoint(set(tst.eFUA_ID))
                assert len(trn) >= 5 and trn.year.max() < tst.year.min()
                train_all = pd.concat([src, trn])
                clouds, labels, order = bh.compose_clouds(
                    src, trn, tst, train_all[bh.MEASURE].mean(), train_all[bh.MEASURE].std())
                norm = bh.normalise(clouds)
                del clouds, train_all
                ctr = {key: value[:, :, bh.CENTER].copy() for key, value in norm.items()}
                sy, ty, vy = labels["source"], labels["train"], labels["test"]
                trn_o, tst_o = tgt.loc[order["train"]], tgt.loc[order["test"]]
                assert np.array_equal(ty, trn_o[bh.Y].to_numpy())
                assert np.array_equal(vy, tst_o[bh.Y].to_numpy())
                x, ex, y = trn_o[bh.X], tst_o[bh.X], ty
                scaler = bh.core.norm()
                xx, ee = scaler.fit_transform(x), None
                ee = scaler.transform(ex)
                raw, eraw = trn_o.src_pred.to_numpy(), tst_o.src_pred.to_numpy()
                z = StandardScaler()
                fixed = {
                    "RF (target-only)": bh.rf().fit(ctr["train"], ty).predict(ctr["test"]),
                    "GBR (target-only)": bh.gbr().fit(ctr["train"], ty).predict(ctr["test"]),
                    "Local mean": np.full(len(ex), y.mean()),
                    "Local ridge": bh.core.pos(Ridge(alpha=10).fit(xx, y).predict(ee)),
                    "Local RBF": bh.core.kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y),
                    "Local RF": bh.core.pos(RandomForestRegressor(n_estimators=80, min_samples_leaf=3,
                        max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee)),
                    "Local Extra Trees": bh.core.pos(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3,
                        max_features=1.0, n_jobs=1, random_state=5993).fit(xx, y).predict(ee)),
                    "Local LightGBM": bh.core.pos(bh.core.tree().fit(x, y).predict(ex)),
                    AFF: bh.core.pos(Ridge(alpha=10).fit(z.fit_transform(raw[:, None]), y)
                                     .predict(z.transform(eraw[:, None]))),
                }
                for offset in offsets:
                    done = folder / f"seed{offset}_metrics.json"
                    if done.exists():
                        continue
                    print(f"FIT {window_name} k={k} split={sid} LDF_seed={sid + offset}", flush=True)
                    ldf = bh.train_ldf(norm, labels, seed=sid + offset)
                    with_ldf = {key: value.copy() for key, value in ctr.items()}
                    for key in with_ldf:
                        with_ldf[key][:, -1] = ldf[key]
                    weights = nnw_weights(with_ldf["source"], with_ldf["train"])
                    preds = dict(fixed)
                    preds[NNW] = bh.itl_fit_predict(with_ldf["source"], sy, with_ldf["train"], ty,
                                                  with_ldf["test"], weights)
                    preds[NNW_SW] = bh.itl_fit_predict(with_ldf["source"], sy, with_ldf["train"], ty,
                                                     with_ldf["test"], weights, released=False)
                    zh = StandardScaler()
                    aa = zh.fit_transform(np.c_[raw, ldf["train"]])
                    bb = zh.transform(np.c_[eraw, ldf["test"]])
                    preds[HYBRID] = bh.core.pos(Ridge(alpha=10).fit(aa, y).predict(bb))
                    scores = {name: bh.score(vy, p) for name, p in preds.items()}
                    diffs = {}
                    if window_name == "2020Q4" and offset == 0:
                        archived = pd.read_csv(RESULTS / "yrd_M3" / f"k{k}_split{sid:02d}.csv").set_index("model")
                        for name, score in scores.items():
                            diffs[name] = {metric: float(score[metric] - archived.loc[name, metric])
                                           for metric in ("rmse", "mae", "r2")}
                    prediction = tst_o[["eFUA_ID", "year", "month", "t", "quarter_id", bh.Y]].copy()
                    prediction["split"] = sid
                    prediction["k"] = k
                    prediction["ldf_seed"] = sid + offset
                    for name, value in preds.items():
                        prediction[name] = value
                    prediction.to_csv(folder / f"seed{offset}_predictions.csv.gz", index=False,
                                      float_format="%.17g")
                    meta = {"window": window_name, "k": k, "split": sid, "ldf_seed": sid + offset,
                            "source_panel_sha256": input_hash, "train_fuas": train_ids.tolist(),
                            "test_fuas": test_ids.tolist(), "train_rows": len(trn_o),
                            "train_keys": trn_o[["eFUA_ID", "year", "month"]].values.tolist(),
                            "scores": scores, "replay_differences": diffs,
                            "elapsed_seconds": time.monotonic() - start}
                    done.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    worst = max((abs(v) for row in diffs.values() for v in row.values()), default=0)
                    print(f"DONE {window_name} k={k} split={sid} offset={offset} "
                          f"affine={scores[AFF]['rmse']:.8f} NNW={scores[NNW]['rmse']:.8f} "
                          f"replay_max={worst:.3g} elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
