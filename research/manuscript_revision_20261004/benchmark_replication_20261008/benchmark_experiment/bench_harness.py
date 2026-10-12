"""Benchmark-aligned comparison (see PROTOCOL.md). One process = one task x protocol.

Usage: python3 bench_harness.py --task yrd|aus --protocol B|S [--ks 5,7,9,11] [--splits 20]

Benchmark reproduction follows the released code of Gupta et al. (2024)
(github.com/YongbeeIngkle/spatial-transfer-learning): LDF station-cloud composition
(data_process/allocate.py), normalisation (data_process/compose.py), autoencoder
(model/autoencoder.py) and ITL + GBR training (model/adapt.py). This report's
estimators are imported unchanged from thesis_completion_20260912/run_monitor_models.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.spatial import distance_matrix
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from paths import MEMBER, PANEL, RESULTS, THESIS, require

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
require(THESIS / "run_monitor_models.py", "THESIS_COMPLETION_DIR")
sys.path.insert(0, str(THESIS))
import run_monitor_models as core  # thesis estimators: tree, embed, kernel, pos, norm
# torch's lazily imported optimiser/dynamo modules segfault if first loaded after TensorFlow
# (pulled in by adapt), so initialise them before importing adapt.
import torch._dynamo  # noqa: F401
torch.optim.NAdam([torch.zeros(1, requires_grad=True)], lr=0.001)

FEATURES12 = json.loads((THESIS / "audit/build_summary.json").read_text())["features"]
X = FEATURES12 + ["month", "t", "fua_centroid_lon", "fua_centroid_lat"]
MEASURE = ["fua_centroid_lon", "fua_centroid_lat", "monthly_power_temperature_c",
           "monthly_power_temperature_max_c", "monthly_power_temperature_min_c",
           "monthly_power_humidity_pct", "monthly_power_wind_mps", "monthly_power_precip_mm_day",
           "log_population_density", "distance_to_coast_km", "urban_centre_area_share",
           "log_fire_count_100km"]
Y = "observed_pm25"
LDF_TAGS = X + ["pm25_value"]          # benchmark: input tags end with pm25_value
NEAR = 12                              # benchmark default (near_station_number = 12)
CENTER = (NEAR + 1) // 2               # benchmark: center_id = (n+1)//2


# ----------------------------------------------------------------------------- data
def load_task(task):
    p = pd.read_csv(require(PANEL, "BENCH_PANEL"))
    if task == "yrd":
        m = pd.read_csv(require(MEMBER, "BENCH_MEMBERSHIP"))
        yrd = set(m.loc[m.country_iso.eq("CHN"), "eFUA_ID"])
        assert len(yrd) == 43
        src = p[p.country_iso.eq("CHN") & ~p.eFUA_ID.isin(yrd)].copy()
        tgt = p[p.country_iso.eq("CHN") & p.eFUA_ID.isin(yrd)].copy()
        share = 0.80
    else:
        src = p[p.country_iso.eq("CHN")].copy()
        tgt = p[p.country_iso.eq("AUS")].copy()
        share = 0.50
    nmonths = p.t.nunique()
    cnt = tgt.groupby("eFUA_ID").size()
    eligible = np.array(sorted(cnt[cnt >= np.ceil(share * nmonths)].index))
    tgt = tgt[tgt.eFUA_ID.isin(eligible)].copy()
    for d in (src, tgt):
        d["pm25_value"] = d[Y]
        d.reset_index(drop=True, inplace=True)
    return src, tgt, eligible


def benchmark_splits(eligible, k, n):
    """Replicates CaliforniaSourcetargetId._split_data_coord (seed 1000, per k)."""
    np.random.seed(1000)
    out = []
    for _ in range(n):
        train = np.random.choice(eligible, k, replace=False)
        test = np.array([i for i in eligible if i not in train])
        out.append((train, test))
    return out


# -------------------------------------------------------------- LDF (benchmark code)
def _spatially_weight(cloud, tags, measure):
    idx = [tags.index(m) for m in measure]
    comp = cloud[:, idx, :]
    center = comp[:, :, CENTER][:, :, None]
    dist = np.linalg.norm(comp - center, axis=1)          # (n, 13); raw measure tags
    off = 1 / np.delete(dist, CENTER, 1)
    sumw = off.sum(axis=1)
    out = cloud.copy()
    for p in range(cloud.shape[2]):
        if p == CENTER:
            continue
        w = (1 / dist[:, p]) / sumw
        out[:, :, p] = cloud[:, :, p] * w[:, None]
    return out


def compose_clouds(src, trn, tst, tr_mean, tr_std):
    """Per month: neighbours = labelled rows (source + training target) of the same month."""
    blocks = {"source": [], "train": [], "test": []}
    labels = {"source": [], "train": [], "test": []}
    order = {"source": [], "train": [], "test": []}
    for month in sorted(set(src.t) | set(trn.t) | set(tst.t)):
        s, a, v = src[src.t.eq(month)], trn[trn.t.eq(month)], tst[tst.t.eq(month)]
        pool = pd.concat([s, a])
        if len(s) < 1:          # benchmark: skip dates without source rows
            continue
        pool_m = ((pool[MEASURE] - tr_mean) / tr_std).to_numpy()
        pool_v = pool[LDF_TAGS].to_numpy(float)
        for name, c in (("source", s), ("train", a), ("test", v)):
            if not len(c):
                continue
            cm = ((c[MEASURE] - tr_mean) / tr_std).to_numpy()
            d = distance_matrix(cm, pool_m)
            d[d == 0] = np.inf                       # benchmark: self/identical -> inf
            nn_idx = np.argsort(d, axis=1)[:, :NEAR]
            neigh = pool_v[nn_idx]                   # (n, 12, tags)
            centre = c[LDF_TAGS].to_numpy(float).copy()
            centre[:, -1] = 0                        # benchmark: centre pm25 set to 0
            stack = np.concatenate([neigh[:, :CENTER], centre[:, None, :], neigh[:, CENTER:]], axis=1)
            stack = np.transpose(stack, (0, 2, 1))   # (n, tags, 13)
            stack = _spatially_weight(stack, LDF_TAGS, MEASURE)
            blocks[name].append(stack)
            labels[name].append(c[Y].to_numpy(float))
            order[name].append(c.index.to_numpy())
    cat = lambda z: np.concatenate(z) if z else np.empty((0,))
    return ({k: cat(v) for k, v in blocks.items()}, {k: cat(v) for k, v in labels.items()},
            {k: cat(v) for k, v in order.items()})


def normalise(clouds):
    tr = np.vstack([clouds["source"], clouds["train"]])
    mean, std = tr.mean(axis=0), tr.std(axis=0)
    mean[-1, :] = 0; std[-1, :] = 1                 # benchmark: pm25_value mean 0, std 1
    std[std == 0] = 1
    return {k: (v - mean) / std for k, v in clouds.items()}


class AutoencoderFeature(nn.Module):                 # copied from model/autoencoder.py
    def __init__(self, latent_dim, target_num, input_dim, input_len):
        super().__init__()
        self.latent_dim, self.target_num, self.input_dim, self.input_len = latent_dim, target_num, input_dim, input_len
        self.enocde_layer1 = nn.Conv1d(input_dim, 16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(16, 8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(8, 4, kernel_size=3, padding="same")
        self.encode_layer4 = nn.Linear(4 * input_len, latent_dim)
        self.deocde_layer1 = nn.Linear(latent_dim, input_len)
        self.decode_layer2 = nn.Conv1d(1, 8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(8, 16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(16, input_dim, kernel_size=1)
        self.estimator_layer = nn.Linear(latent_dim, target_num)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x)); x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x)); x = x.view(-1, 4 * self.input_len)
        return self.encode_layer4(x)

    def decode(self, x):
        x = torch.relu(self.deocde_layer1(x)); x = x.view(-1, 1, self.input_len)
        x = torch.relu(self.decode_layer2(x)); x = torch.relu(self.decode_layer3(x))
        return self.decode_layer4(x)

    def estimate(self, x):
        return self.estimator_layer(self.encode(x))

    def forward(self, x):
        return self.decode(self.encode(x))


class _Set(Dataset):
    def __init__(self, a, b):
        self.a = torch.tensor(a, dtype=torch.float32); self.b = torch.tensor(b, dtype=torch.float32)
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return self.a[i], self.b[i]


def train_ldf(norm, labels, seed):
    torch.manual_seed(seed)
    model = AutoencoderFeature(1, 1, norm["source"].shape[1], norm["source"].shape[2])
    opt = torch.optim.NAdam(model.parameters(), lr=0.001)
    lossf = nn.MSELoss()
    x = np.vstack([norm["source"], norm["train"]]); y = np.hstack([labels["source"], labels["train"]])
    loader = DataLoader(_Set(x, y), batch_size=128, shuffle=False)   # benchmark: shuffle=False
    for _ in range(30):                                                # benchmark: 30 epochs
        for bx, by in loader:
            by = by.reshape(-1, 1)
            recon = model(bx); pred = model.estimate(bx)
            l1 = lossf(recon, bx); l2 = lossf(pred, by)
            opt.zero_grad(); l1.backward(); l2.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        enc = {k: model.encode(torch.tensor(v, dtype=torch.float32)).numpy().ravel() for k, v in norm.items()}
    return enc


# ------------------------------------------------------- ITL + GBR (benchmark code)
def gbr():
    return GradientBoostingRegressor(n_estimators=400, max_leaf_nodes=4, max_depth=None, random_state=2,
                                     min_samples_split=5, learning_rate=0.1, subsample=0.5)


def rf():
    return RandomForestRegressor(n_estimators=400, max_leaf_nodes=4, max_depth=None, random_state=2,
                                 min_samples_split=5)


def itl_weights(name, sx, tx):
    from adapt.instance_based import KMM, KLIEP, NearestNeighborsWeighting

    if name == "KMM":
        w = KMM(DecisionTreeRegressor(max_depth=6), Xt=tx, kernel="rbf", gamma=0.1, verbose=0, random_state=0)
    elif name == "KLIEP":
        w = KLIEP(DecisionTreeRegressor(max_depth=6), Xt=tx, kernel="poly", gamma=0.1, verbose=0, random_state=0)
    else:
        w = NearestNeighborsWeighting(DecisionTreeRegressor(max_depth=6), Xt=tx, n_neighbors=6, random_state=0)
    return np.asarray(w.fit_weights(sx, tx)).ravel()


def itl_fit_predict(sx, sy, tx, ty, vx, weights, released=True):
    m = gbr()
    if released:   # model/adapt.py: source_adapt_x = source_x * weights
        m.fit(np.vstack([sx * weights[:, None], tx]), np.hstack([sy, ty]))
    else:          # standard importance weighting
        m.fit(np.vstack([sx, tx]), np.hstack([sy, ty]), sample_weight=np.hstack([weights, np.ones(len(ty))]))
    return m.predict(vx)


# ------------------------------------------------------------------------- metrics
def score(y, p):
    p = np.asarray(p, float)
    return {"r2": r2_score(y, p), "r2_pearson": pearsonr(y, p)[0] ** 2 if np.std(p) > 0 else np.nan,
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)), "n": len(y)}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--ks", default=None); ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--pass", dest="stage", default="core", choices=["core", "kmm"],
                    help="core: all methods except KMM; kmm: add KMM and KMM[LDF] to existing split files")
    a = ap.parse_args()
    ks = [int(v) for v in a.ks.split(",")] if a.ks else ([5, 7, 9, 11] if a.task == "yrd" else [3, 5])
    if a.stage == "kmm" and not (a.task == "yrd" and a.protocol == "B"):
        print(f"KMM pass skipped for {a.task}-{a.protocol}: KMM is run only under the benchmark protocol "
              "(T1, P-B); see PROTOCOL.md execution notes.", flush=True)
        return
    out = RESULTS / f"{a.task}_{a.protocol}"
    out.mkdir(parents=True, exist_ok=True)
    src, tgt, eligible = load_task(a.task)
    (out / "eligible.json").write_text(json.dumps({"eligible": [float(e) for e in eligible], "n": len(eligible),
                                                     "source_rows": len(src), "source_fuas": int(src.eFUA_ID.nunique()),
                                                     "target_rows": len(tgt)}, indent=1))
    # this report's source models are fitted once per task (source data do not depend on split)
    base = core.tree(True).fit(src[X], src[Y])
    shuf = core.tree(True).fit(src[X], np.random.default_rng(5993).permutation(src[Y].to_numpy()))
    tgt["src_pred"] = core.pos(base.predict(tgt[X])); tgt["shuf_pred"] = core.pos(shuf.predict(tgt[X]))
    emb = core.embed(base, tgt[X])
    for k in ks:
        for sid, (train_ids, test_ids) in enumerate(benchmark_splits(eligible, k, a.splits)):
            if a.only >= 0 and sid != a.only:
                continue
            f = out / f"k{k}_split{sid:02d}.csv"
            cache = out / f"k{k}_split{sid:02d}_ldf.npz"
            if a.stage == "core" and f.exists():
                continue
            if a.stage == "kmm":
                if not f.exists() or not cache.exists():
                    continue
                if pd.read_csv(f).model.eq("KMM").any():
                    continue
            t0 = time.time()
            trn_mask = tgt.eFUA_ID.isin(train_ids); tst_mask = tgt.eFUA_ID.isin(test_ids)
            if a.protocol == "S":
                trn_mask &= tgt.year.le(2020); tst_mask &= tgt.year.ge(2021)
            trn, tst = tgt[trn_mask], tgt[tst_mask]
            # ---- benchmark pipeline
            tr_all = pd.concat([src, trn])
            mu, sd = tr_all[MEASURE].mean(), tr_all[MEASURE].std()
            clouds, labels, order = compose_clouds(src, trn, tst, mu, sd)
            norm = normalise(clouds)
            t1 = time.time()
            if a.stage == "core":
                ldf = train_ldf(norm, labels, seed=sid)
                np.savez_compressed(cache, **ldf)
            else:   # reuse the exact LDF encodings of the core pass
                z_ = np.load(cache); ldf = {kk: z_[kk] for kk in ("source", "train", "test")}
            stage = {"pass": a.stage, "clouds_s": round(t1 - t0, 1), "ldf_s": round(time.time() - t1, 1)}
            ctr = {kk: v[:, :, CENTER].copy() for kk, v in norm.items()}       # NF input (pm25 col = 0)
            ctr_ldf = {kk: v.copy() for kk, v in ctr.items()}
            for kk in ctr_ldf:
                ctr_ldf[kk][:, -1] = ldf[kk]
            sy, ty, vy = labels["source"], labels["train"], labels["test"]
            preds = {}
            if a.stage == "kmm":
                t2 = time.time()
                w_nf = itl_weights("KMM", ctr["source"], ctr["train"])
                preds["KMM"] = itl_fit_predict(ctr["source"], sy, ctr["train"], ty, ctr["test"], w_nf)
                w_l = itl_weights("KMM", ctr_ldf["source"], ctr_ldf["train"])
                preds["KMM[LDF]"] = itl_fit_predict(ctr_ldf["source"], sy, ctr_ldf["train"], ty, ctr_ldf["test"], w_l)
                stage["KMM_s"] = round(time.time() - t2, 1)
                old_rows = pd.read_csv(f)
                new_rows = pd.DataFrame([{"task": a.task, "protocol": a.protocol, "k": k, "split": sid, "model": m,
                                          **score(vy, pp), "n_train_rows": len(ty), "n_test_fuas": len(test_ids),
                                          "seconds": round(time.time() - t0, 1)} for m, pp in preds.items()])
                pd.concat([old_rows, new_rows], ignore_index=True).to_csv(f, index=False)
                with open(out / "timing.jsonl", "a") as fh:
                    fh.write(json.dumps({"k": k, "split": sid, **stage, "total_s": round(time.time() - t0, 1)}) + "\n")
                print(f"{a.task}-{a.protocol} k={k} split={sid} KMM pass {time.time()-t0:.0f}s", flush=True)
                continue
            preds["RF (target-only)"] = rf().fit(ctr["train"], ty).predict(ctr["test"])
            preds["GBR (target-only)"] = gbr().fit(ctr["train"], ty).predict(ctr["test"])
            preds["GBR[LDF] (target-only)"] = gbr().fit(ctr_ldf["train"], ty).predict(ctr_ldf["test"])
            itl = ["NNW", "KLIEP"] if a.protocol == "B" else ["NNW"]   # KMM: separate pass; KLIEP: P-B only (cost)
            for name in itl:
                t2 = time.time()
                w_nf = itl_weights(name, ctr["source"], ctr["train"])
                preds[name] = itl_fit_predict(ctr["source"], sy, ctr["train"], ty, ctr["test"], w_nf)
                w_l = itl_weights(name, ctr_ldf["source"], ctr_ldf["train"])
                preds[f"{name}[LDF]"] = itl_fit_predict(ctr_ldf["source"], sy, ctr_ldf["train"], ty, ctr_ldf["test"], w_l)
                if name == "NNW":
                    preds["NNW (sample-weight)"] = itl_fit_predict(ctr["source"], sy, ctr["train"], ty, ctr["test"], w_nf, released=False)
                    preds["NNW[LDF] (sample-weight)"] = itl_fit_predict(ctr_ldf["source"], sy, ctr_ldf["train"], ty, ctr_ldf["test"], w_l, released=False)
                stage[f"{name}_s"] = round(time.time() - t2, 1)
                if name == "NNW":
                    stage["nnw_zero_weight_share"] = float(np.mean(w_nf == 0))
            # align this report's estimators to the benchmark test-row order
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
            rows = [{"task": a.task, "protocol": a.protocol, "k": k, "split": sid, "model": m, **score(vy, p),
                     "n_train_rows": len(ty), "n_test_fuas": len(test_ids), "seconds": round(time.time() - t0, 1)}
                    for m, p in preds.items()]
            pd.DataFrame(rows).to_csv(f, index=False)
            with open(out / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"k": k, "split": sid, **stage, "total_s": round(time.time() - t0, 1)}) + "\n")
            print(f"{a.task}-{a.protocol} k={k} split={sid} {time.time()-t0:.0f}s "
                  f"affine={rows[[r['model'] for r in rows].index('Source-affine (ours)')]['rmse']:.3f} "
                  f"NNW[LDF]={rows[[r['model'] for r in rows].index('NNW[LDF]')]['rmse']:.3f}", flush=True)


if __name__ == "__main__":
    main()
