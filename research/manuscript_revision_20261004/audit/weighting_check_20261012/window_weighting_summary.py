"""Weighting check for the fixed-window analysis (thesis Section 5.6, Table 5.6).

Reads the re-run window_results_sw/<window>/k<k>_split<ss>/seed<offset>_predictions.csv.gz, which
reproduce every column of audit/round8_20261009/window_results exactly and add
"NNW[LDF] (sample-weight)": NNW[LDF] fitted with the same LDF codes and NNW weights, the weights
passed to the loss instead of multiplying the source features.

For every window x k x LDF-seed cell it reports mean paired relative RMSE reductions
  100 * mean_s (1 - RMSE_A,s / RMSE_B,s)  over the 20 splits,
with 95% percentile intervals from the crossed FUA-by-calendar-quarter bootstrap of Table 5.6:
same generator, seed (20261009), draw order and 1,000 draws, so the Table 5.6 columns are
reproduced exactly and the new comparisons use the same draws.

Usage: python3 window_weighting_summary.py <window_results_sw_dir> <out_prefix>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
DRAWS, RNG_SEED = 1000, 20261009
AFF, NNW, SW, HYB = ("Source-affine (ours)", "NNW[LDF]", "NNW[LDF] (sample-weight)",
                     "Source-affine[LDF] (ours)")
EIGHT = ["RF (target-only)", "GBR (target-only)", "Local mean", "Local ridge", "Local RBF",
         "Local RF", "Local Extra Trees", "Local LightGBM"]
MODELS = EIGHT + [AFF, NNW, SW, HYB]
Y = "observed_pm25"
CELLS = [("2020Q4", 0), ("2020Q2", 0), ("2020Q3", 0), ("2019Q4", 0), ("2020Q4", 10000)]


def files(window, k, seed):
    found = sorted(glob.glob(os.path.join(ROOT, window, f"k{k}_split*", f"seed{seed}_predictions.csv.gz")))
    assert len(found) == 20, (window, k, seed, len(found))
    return found


rows, means = [], []
for k in (5, 11):
    frames = {cell: [pd.read_csv(f) for f in files(cell[0], k, cell[1])] for cell in CELLS}
    fuas = np.array(sorted({u for dfs in frames.values() for df in dfs for u in df.eFUA_ID.unique()}))
    quarters = np.array(sorted({q for dfs in frames.values() for df in dfs for q in df.quarter_id.unique()}))
    assert len(fuas) == 30 and len(quarters) == 8, (len(fuas), len(quarters))
    rng = np.random.default_rng(RNG_SEED)
    wf = rng.multinomial(len(fuas), np.full(len(fuas), 1 / len(fuas)), size=DRAWS).astype(float)
    wq = rng.multinomial(len(quarters), np.full(len(quarters), 1 / len(quarters)), size=DRAWS).astype(float)
    for (window, seed), dfs in frames.items():
        boot = {m: np.empty((20, DRAWS)) for m in MODELS}
        point = {m: np.empty(20) for m in MODELS}
        for s, df in enumerate(sorted(dfs, key=lambda d: int(d.split.iloc[0]))):
            assert int(df.split.iloc[0]) == s and int(df.k.iloc[0]) == k and int(df.ldf_seed.iloc[0]) == s + seed
            weight = wf[:, np.searchsorted(fuas, df.eFUA_ID.to_numpy())] * \
                wq[:, np.searchsorted(quarters, df.quarter_id.to_numpy())]
            total = weight.sum(1)
            y = df[Y].to_numpy()
            for m in MODELS:
                error = df[m].to_numpy() - y
                boot[m][s] = np.sqrt(weight @ error ** 2 / total)
                point[m][s] = np.sqrt(np.mean(error ** 2))
        best_boot = np.min([boot[m] for m in EIGHT], axis=0)
        best_point = np.min([point[m] for m in EIGHT], axis=0)
        # stronger NNW[LDF] weighting in this cell: the one with the lower mean RMSE over the 20 splits
        strong = SW if point[SW].mean() < point[NNW].mean() else NNW
        comparisons = {
            "aff_vs_nnw": (AFF, point[NNW], boot[NNW]),
            "aff_vs_best8": (AFF, best_point, best_boot),
            "hyb_vs_nnw": (HYB, point[NNW], boot[NNW]),
            "aff_vs_sw": (AFF, point[SW], boot[SW]),
            "hyb_vs_sw": (HYB, point[SW], boot[SW]),
            "sw_vs_nnw": (SW, point[NNW], boot[NNW]),
            "hyb_vs_stronger": (HYB, point[strong], boot[strong]),
            "aff_vs_stronger": (AFF, point[strong], boot[strong]),
        }
        for name, (a, pb, bb) in comparisons.items():
            gain = 100 * np.mean(1 - point[a] / pb)
            draws = 100 * np.mean(1 - boot[a] / bb, axis=0)
            lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
            rows.append(dict(window=window, k=k, ldf_seed_offset=seed, comparison=name, gain=gain, lo=lo,
                             hi=hi, wins=int(np.sum(point[a] < pb)), splits=20,
                             stronger=strong if "stronger" in name else ""))
        ranking = sorted(MODELS, key=lambda m: point[m].mean())
        for m in MODELS:
            means.append(dict(window=window, k=k, ldf_seed_offset=seed, model=m, mean_rmse=point[m].mean(),
                              rank=ranking.index(m) + 1))

gains, model_means = pd.DataFrame(rows), pd.DataFrame(means)
gains.to_csv(OUT + "_gains.csv", index=False, float_format="%.4f")
model_means.to_csv(OUT + "_model_means.csv", index=False, float_format="%.4f")
gains["cell"] = gains.apply(lambda r: f"{r.gain:6.1f} [{r.lo:6.1f}, {r.hi:6.1f}] {r.wins:2d}/20", axis=1)
print(gains.pivot_table(index=["window", "ldf_seed_offset", "comparison"], columns="k", values="cell",
                        aggfunc="first").to_string())
top = model_means[model_means["rank"] <= 3].sort_values(["window", "ldf_seed_offset", "k", "rank"])
print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
