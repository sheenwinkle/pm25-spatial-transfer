"""Shared FUA x quarter bootstrap of fixed, overlapping benchmark predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_tables import AFF, AFFL, NNWL, TARGET_ONLY

WINDOWS = ("2020Q4", "2020Q2", "2020Q3", "2019Q4")
WINDOW_MONTHS = {"2020Q4": (2020, 10, 12), "2020Q2": (2020, 4, 6),
                 "2020Q3": (2020, 7, 9), "2019Q4": (2019, 10, 12)}


def cell_arrays(frames, candidate=AFF, reference=NNWL):
    cities = sorted(set().union(*(set(d.eFUA_ID) for d in frames)))
    quarters = sorted(set().union(*(set(d.quarter_id) for d in frames)))
    ci, qi = {c: i for i, c in enumerate(cities)}, {q: i for i, q in enumerate(quarters)}
    count = np.zeros((len(frames), len(cities), len(quarters)))
    cand, ref = count.copy(), count.copy()
    for sid, d in enumerate(frames):
        c = d.eFUA_ID.map(ci).to_numpy()
        q = d.quarter_id.map(qi).to_numpy()
        np.add.at(count[sid], (c, q), 1)
        np.add.at(cand[sid], (c, q), (d[candidate] - d.observed_pm25).to_numpy() ** 2)
        np.add.at(ref[sid], (c, q), (d[reference] - d.observed_pm25).to_numpy() ** 2)
    return count, cand, ref, cities, quarters


def weighted_gains(count, cand, ref, city_weights, quarter_weights):
    den = np.einsum("bc,scq,bq->bs", city_weights, count, quarter_weights, optimize=True)
    c = np.einsum("bc,scq,bq->bs", city_weights, cand, quarter_weights, optimize=True)
    r = np.einsum("bc,scq,bq->bs", city_weights, ref, quarter_weights, optimize=True)
    valid = np.all((den > 0) & (r > 0), axis=1)
    # The row-count denominator cancels in the paired ratio for each split.
    gains = 100 * (1 - np.sqrt(c[valid] / r[valid])).mean(axis=1)
    return gains, valid


def paired_bootstrap(frames, candidate=AFF, reference=NNWL, n=1000, seed=20261009):
    count, cand, ref, cities, quarters = cell_arrays(frames, candidate, reference)
    rng = np.random.default_rng(seed)
    cw = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), n)
    qw = rng.multinomial(len(quarters), np.full(len(quarters), 1 / len(quarters)), n)
    gains, valid = weighted_gains(count, cand, ref, cw, qw)
    split_gain = 100 * (1 - np.sqrt(cand.sum((1, 2)) / ref.sum((1, 2))))
    return {"gain_pct": float(split_gain.mean()), "wins": int((split_gain > 0).sum()),
            "ci_low_pct": float(np.quantile(gains, .025)), "ci_high_pct": float(np.quantile(gains, .975)),
            "bootstrap_replicates": n, "valid_replicates": int(valid.sum()),
            "fuas": len(cities), "quarters": len(quarters), "splits": len(frames)}


def crossed_draws(frames, n=1000, seed=20261009):
    """The same city and quarter multiplicities as paired_bootstrap (same seed and draw order)."""
    cities = sorted(set().union(*(set(d.eFUA_ID) for d in frames)))
    quarters = sorted(set().union(*(set(d.quarter_id) for d in frames)))
    rng = np.random.default_rng(seed)
    cw = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), n)
    qw = rng.multinomial(len(quarters), np.full(len(quarters), 1 / len(quarters)), n)
    return cities, quarters, cw, qw


def weighted_mse(frames, model, cities, quarters, cw, qw):
    """Draw-by-split weighted mean squared error of one model's saved predictions."""
    ci, qi = {c: i for i, c in enumerate(cities)}, {q: i for i, q in enumerate(quarters)}
    count = np.zeros((len(frames), len(cities), len(quarters)))
    sse = count.copy()
    for sid, d in enumerate(frames):
        c = d.eFUA_ID.map(ci).to_numpy()
        q = d.quarter_id.map(qi).to_numpy()
        np.add.at(count[sid], (c, q), 1)
        np.add.at(sse[sid], (c, q), (d[model] - d.observed_pm25).to_numpy() ** 2)
    den = np.einsum("bc,scq,bq->bs", cw, count, qw, optimize=True)
    num = np.einsum("bc,scq,bq->bs", cw, sse, qw, optimize=True)
    assert (den > 0).all()
    return num / den


def best_eight_bootstrap(frames, candidate=AFF, n=1000, seed=20261009):
    """Candidate against the per-split best target-only learner, re-selected within every draw."""
    cities, quarters, cw, qw = crossed_draws(frames, n, seed)
    cand = weighted_mse(frames, candidate, cities, quarters, cw, qw)
    best = np.min([weighted_mse(frames, m, cities, quarters, cw, qw) for m in TARGET_ONLY], axis=0)
    gains = 100 * (1 - np.sqrt(cand / best)).mean(axis=1)
    rmse = {m: np.array([np.sqrt(np.mean((d[m] - d.observed_pm25) ** 2)) for d in frames])
            for m in TARGET_ONLY + [candidate]}
    split_best = np.min([rmse[m] for m in TARGET_ONLY], axis=0)
    split_gain = 100 * (1 - rmse[candidate] / split_best)
    return {"gain_pct": float(split_gain.mean()), "wins": int((split_gain > 0).sum()),
            "ci_low_pct": float(np.quantile(gains, .025)), "ci_high_pct": float(np.quantile(gains, .975))}


def self_test():
    base = pd.DataFrame({"eFUA_ID": [1, 1, 2, 2, 3, 3], "quarter_id": [1, 2, 1, 3, 2, 3],
                         "observed_pm25": [1., 3., 2., 7., 4., 8.],
                         AFF: [2., 3., 4., 5., 4., 7.], NNWL: [3., 3., 6., 4., 6., 6.]})
    frames = [base.iloc[:4].copy(), base.iloc[2:].copy()]
    count, a, b, cities, quarters = cell_arrays(frames)
    rng = np.random.default_rng(412)
    cw = rng.multinomial(3, [1 / 3] * 3, 100)
    qw = rng.multinomial(3, [1 / 3] * 3, 100)
    got, valid = weighted_gains(count, a, b, cw, qw)
    direct = []
    for i in np.flatnonzero(valid):
        split = []
        for d in frames:
            weights = np.array([cw[i, cities.index(c)] * qw[i, quarters.index(q)]
                                for c, q in zip(d.eFUA_ID, d.quarter_id)])
            ce = np.sum(weights * (d[AFF] - d.observed_pm25) ** 2)
            re = np.sum(weights * (d[NNWL] - d.observed_pm25) ** 2)
            split.append(100 * (1 - np.sqrt(ce / re)))
        direct.append(np.mean(split))
    np.testing.assert_allclose(got, direct, atol=1e-12)
    print("PASS crossed bootstrap matches direct row weights on overlapping splits")


def latex_table(rows):
    lookup = {(r["window"], r["k"], r["seed_offset"]): r for r in rows}
    def number(x):
        return f"{x:.1f}".replace("-", "$-$")
    def cell(r, prefix):
        return f"{number(r[prefix + 'gain_pct'])} [{number(r[prefix + 'ci_low_pct'])}, {number(r[prefix + 'ci_high_pct'])}]"
    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
             r"\caption{Window and LDF-seed sensitivity for T1. Entries are mean paired RMSE reductions (\%) over 20 splits; positive values favour the first-named model. Brackets give 95\% crossed FUA-by-quarter bootstrap intervals from the same 1,000 draws for every column, shared across the 20 overlapping splits and conditional on the fitted models. All runs use the same 2021--2022 test rows and fully labelled source years. Best-eight is the per-split post-hoc best of the eight target-only learners, re-selected within each draw. The final rows change only the LDF seed from split ID to split ID plus 10000.}",
             r"\label{tab:window-sensitivity}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Source-affine vs} & Source-affine[LDF] vs \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}",
             r"Calibration window & $k$ & NNW[LDF] & best-eight & NNW[LDF] \\",
             r"\midrule"]
    names = {"2020Q4": "Oct--Dec 2020", "2020Q2": "Apr--Jun 2020",
             "2020Q3": "Jul--Sep 2020", "2019Q4": "Oct--Dec 2019"}
    for window, offset in [(w, 0) for w in WINDOWS] + [("2020Q4", 10000)]:
        if offset:
            lines.append(r"\addlinespace")
        for k in (5, 11):
            r = lookup[window, k, offset]
            best = {"gain_pct": r["gain_vs_posthoc_best_pct"], "ci_low_pct": r["best_ci_low_pct"],
                    "ci_high_pct": r["best_ci_high_pct"]}
            label = (names[window] + (" (new seed)" if offset else "")) if k == 5 else ""
            lines.append(f"{label} & {k} & {cell(r, '')} & {cell(best, '')} & {cell(r, 'hybrid_')} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return
    assert args.input and args.out
    summaries, drifts, hashes = [], [], set()
    checked_fits, checked_rmse = 0, 0
    for window in WINDOWS:
        for k in (5, 11):
            for offset in ([0, 10000] if window == "2020Q4" else [0]):
                frames, metadata = [], []
                for sid in range(20):
                    folder = args.input / window / f"k{k}_split{sid:02d}"
                    meta = json.loads((folder / f"seed{offset}_metrics.json").read_text())
                    d = pd.read_csv(folder / f"seed{offset}_predictions.csv.gz", float_precision="round_trip")
                    assert (meta["window"], meta["k"], meta["split"], meta["ldf_seed"]) == (window, k, sid, sid + offset)
                    hashes.add(meta["source_panel_sha256"])
                    yy, lo, hi = WINDOW_MONTHS[window]
                    assert len(meta["train_keys"]) == meta["train_rows"]
                    assert all(c in meta["train_fuas"] and y == yy and lo <= m <= hi
                               for c, y, m in meta["train_keys"])
                    assert len(set(map(tuple, meta["train_keys"]))) == meta["train_rows"]
                    assert not d.duplicated(["eFUA_ID", "year", "month"]).any()
                    assert d.split.eq(sid).all() and d.k.eq(k).all()
                    assert d.ldf_seed.eq(sid + offset).all()
                    assert set(d.eFUA_ID).isdisjoint(meta["train_fuas"])
                    assert set(d.eFUA_ID) == set(meta["test_fuas"])
                    assert d.year.isin([2021, 2022]).all()
                    assert np.isfinite(d[TARGET_ONLY + [AFF, NNWL, AFFL, "observed_pm25"]]).all().all()
                    for name in TARGET_ONLY + [AFF, NNWL, AFFL]:
                        rmse = float(np.sqrt(np.mean((d[name] - d.observed_pm25) ** 2)))
                        assert abs(rmse - meta["scores"][name]["rmse"]) < 1e-10
                        checked_rmse += 1
                    if window != "2020Q4" or offset:
                        baseline = pd.read_csv(args.input / "2020Q4" / f"k{k}_split{sid:02d}"
                                               / "seed0_predictions.csv.gz", float_precision="round_trip")
                        pd.testing.assert_frame_equal(d[["eFUA_ID", "year", "month", "observed_pm25"]],
                                                      baseline[["eFUA_ID", "year", "month", "observed_pm25"]])
                        if offset:
                            np.testing.assert_array_equal(d[TARGET_ONLY + [AFF]], baseline[TARGET_ONLY + [AFF]])
                    for name, diffs in meta["replay_differences"].items():
                        drifts.append({"k": k, "split": sid, "model": name, **diffs})
                    frames.append(d)
                    metadata.append(meta)
                    checked_fits += 1
                row = {"window": window, "k": k, "seed_offset": offset,
                       **paired_bootstrap(frames),
                       "affine_rmse": float(np.mean([m["scores"][AFF]["rmse"] for m in metadata])),
                       "nnw_rmse": float(np.mean([m["scores"][NNWL]["rmse"] for m in metadata]))}
                best = [min(m["scores"][name]["rmse"] for name in TARGET_ONLY) for m in metadata]
                row["gain_vs_posthoc_best_pct"] = float(100 * np.mean([
                    1 - m["scores"][AFF]["rmse"] / ref for m, ref in zip(metadata, best)]))
                best8 = best_eight_bootstrap(frames)
                assert abs(best8["gain_pct"] - row["gain_vs_posthoc_best_pct"]) < 1e-9
                row.update(best_wins=best8["wins"], best_ci_low_pct=best8["ci_low_pct"],
                           best_ci_high_pct=best8["ci_high_pct"])
                hybrid = paired_bootstrap(frames, candidate=AFFL, reference=NNWL)
                row.update(hybrid_gain_pct=hybrid["gain_pct"], hybrid_wins=hybrid["wins"],
                           hybrid_ci_low_pct=hybrid["ci_low_pct"], hybrid_ci_high_pct=hybrid["ci_high_pct"],
                           hybrid_rmse=float(np.mean([m["scores"][AFFL]["rmse"] for m in metadata])))
                row["train_rows_min"] = min(m["train_rows"] for m in metadata)
                row["train_rows_max"] = max(m["train_rows"] for m in metadata)
                assert row["valid_replicates"] == row["bootstrap_replicates"]
                summaries.append(row)
    assert checked_fits == 200 and checked_rmse == 2200 and len(hashes) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out / "window_sensitivity_summary.csv", index=False)
    pd.DataFrame(drifts).to_csv(args.out / "replay_metric_differences.csv", index=False)
    (args.out / "window_sensitivity_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.out / "tab_window_sensitivity.tex").write_text(latex_table(summaries), encoding="utf-8")
    checks = {"completed_fits": checked_fits, "recomputed_rmse": checked_rmse,
              "panel_sha256": next(iter(hashes)), "summary_cells": len(summaries),
              "valid_bootstrap_draws_per_cell": 1000,
              "aligned_test_rows": True, "unchanged_non_ldf_predictions_across_seeds": True}
    (args.out / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
