"""Check the balanced round-8 narrative against saved fits and independent review."""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
AUD = REV / "audit/round8_20261009"
REVIEW = REV / "window_sensitivity_review_20261011"
EXP = REV / "benchmark_replication_20261008/benchmark_experiment"
sys.path.insert(0, str(EXP))
from make_tables import AFF, AFFL, NNWL, TARGET_ONLY

checks = []


def check(name, passed, detail=None):
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def key(row):
    return row["window"], row["k"], row["seed_offset"]


rows = json.loads((AUD / "window_summary/window_sensitivity_summary.json").read_text())
lookup = {key(r): r for r in rows}
old = json.loads((HERE / "before/window_sensitivity_summary.json").read_text())
max_old_change = max(abs(float(value) - float(lookup[key(r)][field]))
                     for r in old for field, value in r.items() if field != "window")
check("all previously reported summary fields unchanged", max_old_change < 1e-10, max_old_change)

review = pd.read_csv(REVIEW / "review/review_gains.csv")
fields = {
    "aff_vs_nnw": ("gain_pct", "ci_low_pct", "ci_high_pct", "wins"),
    "aff_vs_best8": ("gain_vs_posthoc_best_pct", "best_ci_low_pct", "best_ci_high_pct", "best_wins"),
    "hyb_vs_nnw": ("hybrid_gain_pct", "hybrid_ci_low_pct", "hybrid_ci_high_pct", "hybrid_wins"),
}
errors = []
for r in review.itertuples():
    if r.comparison not in fields:
        continue
    ours = lookup[r.window, r.k, r.ldf_seed_offset]
    errors.extend(abs(ours[f] - v) for f, v in zip(fields[r.comparison], (r.gain, r.lo, r.hi, r.wins)))
check("30 comparison cells match independent four-decimal calculation", len(errors) == 120
      and max(errors) <= 5.1e-5, {"values": len(errors), "max_rounding_error": max(errors)})

alt = pd.read_csv(REVIEW / "review/indep_seed20261011_review_gains.csv")
joined = review.merge(alt, on=["window", "k", "ldf_seed_offset", "comparison"], suffixes=("_a", "_b"))
joined = joined[joined.comparison.isin(fields)]
endpoint_change = float(np.max(np.abs(joined[["lo_a", "hi_a"]].to_numpy()
                                    - joined[["lo_b", "hi_b"]].to_numpy())))
alt_best = alt[(alt.comparison == "aff_vs_best8") & (alt.ldf_seed_offset == 0)]
alt_hybrid_2020 = alt[(alt.comparison == "hyb_vs_nnw") & (alt.ldf_seed_offset == 0)
                      & alt.window.str.startswith("2020")]
alt_q4_primary = alt[(alt.comparison == "aff_vs_nnw") & (alt.window == "2020Q4")]
alt_hybrid_2019 = alt[(alt.comparison == "hyb_vs_nnw") & (alt.window == "2019Q4")]
check("alternate bootstrap seed preserves every interval-sign conclusion",
      (alt_best.lo > 0).sum() == 7
      and (alt_hybrid_2020.lo > 0).sum() == 6
      and ((alt_q4_primary.lo < 0) & (alt_q4_primary.hi > 0)).sum() == 3
      and ((alt_hybrid_2019.lo < 0) & (alt_hybrid_2019.hi > 0)).sum() == 2,
      {"maximum_endpoint_shift_pct": endpoint_change,
       "note": "The independent review's stated 1.5-point bound was not reproduced; the maximum is 2.54."})

original = [r for r in rows if r["seed_offset"] == 0]
q4 = [r for r in rows if r["window"] == "2020Q4"]
year2020 = [r for r in original if r["window"].startswith("2020")]
check("best-eight gains 13.7-32.7 percent and seven of eight positive intervals",
      [round(f(r["gain_vs_posthoc_best_pct"] for r in original), 1) for f in (min, max)] == [13.7, 32.7]
      and sum(r["best_ci_low_pct"] > 0 for r in original) == 7)
check("hybrid gains 20.8-39.4 percent and all six 2020 intervals positive",
      [round(f(r["hybrid_gain_pct"] for r in year2020), 1) for f in (min, max)] == [20.8, 39.4]
      and all(r["hybrid_ci_low_pct"] > 0 for r in year2020))
check("only original window has second seed and both hybrid intervals are positive",
      len([r for r in rows if r["seed_offset"]]) == 2
      and all(r["window"] == "2020Q4" and r["hybrid_ci_low_pct"] > 0 for r in rows if r["seed_offset"]))
check("three of four original-window primary intervals include zero",
      sum(r["ci_low_pct"] < 0 < r["ci_high_pct"] for r in q4) == 3)
check("2019 hybrid intervals do not resolve the difference",
      all(r["hybrid_ci_low_pct"] < 0 < r["hybrid_ci_high_pct"] for r in original if r["window"] == "2019Q4"))

models = TARGET_ONLY + [AFF, NNWL, AFFL]
scores, biases = {}, {}
for r in rows:
    frames = [pd.read_csv(AUD / "window_results" / r["window"] / f"k{r['k']}_split{s:02d}"
                         / f"seed{r['seed_offset']}_predictions.csv.gz", float_precision="round_trip")
              for s in range(20)]
    scores[key(r)] = {m: np.array([np.sqrt(np.mean((d[m] - d.observed_pm25) ** 2)) for d in frames])
                      for m in models}
    biases[key(r)] = {m: float(np.concatenate([(d[m] - d.observed_pm25).to_numpy() for d in frames]).mean())
                      for m in models}

seed_review = pd.read_csv(REVIEW / "review/review_ldf_seed.csv")
seed_stats, errors = [], []
for r in seed_review.itertuples():
    a, b = scores["2020Q4", r.k, 0][r.model], scores["2020Q4", r.k, 10000][r.model]
    values = [float(a.mean()), float(b.mean()), float(np.abs(a - b).max()), int((np.abs(a - b) > 1).sum())]
    expected = [r.mean_rmse_seed0, r.mean_rmse_seed10000, r.max_abs_split_change, r.splits_changed_over_1]
    errors.extend(abs(a - b) for a, b in zip(values, expected))
    seed_stats.append({"k": r.k, "model": r.model, "mean_seed0": values[0], "mean_seed10000": values[1],
                       "max_split_change": values[2], "changes_above_one": values[3]})
check("seed-sensitivity magnitudes independently reconstructed from row predictions",
      len(errors) == 16 and max(errors) <= 5.1e-5, seed_stats)

best_counts, gaps, nnw_gains = 0, [], []
for r in original:
    sc = scores[key(r)]
    means = {m: v.mean() for m, v in sc.items()}
    best_counts += min(means, key=means.get) == AFFL
    gaps.append(float(means[AFFL] - min(means.values())))
    nnw_gains.append(float(100 * np.mean(1 - sc[NNWL] / np.min([sc[m] for m in TARGET_ONLY], axis=0))))
check("hybrid has lowest mean RMSE in seven cells and misses eighth by 0.02",
      best_counts == 7 and round(max(gaps), 2) == .02, {"best_cells": best_counts, "largest_gap": max(gaps)})
check("NNW against best-eight ranges from 15.1 percent worse to 33.1 percent better",
      round(min(nnw_gains), 1) == -15.1 and round(max(nnw_gains), 1) == 33.1, nnw_gains)

mean_review = pd.read_csv(REVIEW / "review/review_model_means.csv")
mean_errors, bias_errors = [], []
for r in mean_review.itertuples():
    cell = (r.window, r.k, r.ldf_seed_offset)
    mean_errors.append(abs(scores[cell][r.model].mean() - r.mean_rmse))
    bias_errors.append(abs(biases[cell][r.model] - r.mean_bias))
check("all model means and descriptive biases match independent reconstruction",
      max(mean_errors + bias_errors) <= 5.1e-5,
      {"model_cells": len(mean_errors), "max_rmse_error": max(mean_errors), "max_bias_error": max(bias_errors)})

linux_path = REVIEW / "linux_2019Q4/linux_2019Q4_per_split.csv"
linux = pd.read_csv(linux_path)
linux_gains, non_ldf_errors = {}, []
for k in (5, 11):
    group = linux[linux.k == k].pivot(index="split", columns="model", values="rmse").sort_index()
    assert list(group.index) == list(range(20))
    for m in TARGET_ONLY + [AFF]:
        non_ldf_errors.extend(abs(group[m].to_numpy() - scores["2019Q4", k, 0][m]))
    linux_gains[k] = float(100 * np.mean(1 - group[AFF] / group[NNWL]))
check("review-supplied Linux replay non-LDF results match within 5e-7",
      max(non_ldf_errors) <= 5e-7, float(max(non_ldf_errors)))
check("review-supplied Linux replay confirms primary 2019 reversal",
      [round(linux_gains[k], 1) for k in (5, 11)] == [-22.1, -18.4], linux_gains)

text = (REV / "manuscript/main.tex").read_text(encoding="utf-8")
abstract = text.split(r"\section*{Abstract}", 1)[1].split(r"\section{Introduction}", 1)[0]
conclusion = text.split(r"\section{Conclusion}", 1)[1].split(r"\section*{Declaration", 1)[0]
check("abstract and conclusion both report positive target-only and hybrid comparisons",
      all("13.7--32.7" in section and "20.8--39.4" in section for section in (abstract, conclusion)))
check("AI use dates and independent recomputation disclosed",
      re.search(r"accessed 4--\d{1,2} (?:October|November) 2026", text) is not None
      and "window-sensitivity comparisons and replaying one window on a second" in text)
check("primary candidate and secondary comparison retained", "Source-affine remains the primary candidate" in text)

before = (HERE / "before/main.tex").read_text(encoding="utf-8")
(HERE / "main_balance.diff").write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile="before/main.tex", tofile="manuscript/main.tex")), encoding="utf-8")
report = {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks),
          "manuscript_sha256": hashlib.sha256(text.encode()).hexdigest(),
          "linux_replay_provenance": "Independent reviewer-supplied run; no new Linux fitting in this audit",
          "linux_csv_sha256": hashlib.sha256(linux_path.read_bytes()).hexdigest()}
(HERE / "balance_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
lines = [f"{'PASS' if c['pass'] else 'FAIL'} {c['name']}" for c in checks]
lines.append(f"{report['passed']}/{report['total']} checks pass")
(HERE / "balance_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if all(c["pass"] for c in checks) else 1)
