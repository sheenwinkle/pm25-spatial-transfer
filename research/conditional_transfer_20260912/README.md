# Conditional China-Australia PM2.5 transfer, 2026-09-12

This is a completed three-round retrospective exploratory branch plus a calibration-score range audit. It does not replace the earlier manuscript or assert confirmed conditional transfer. The v2 review package adds the third round and range audit without overwriting the original review ZIP.

## Read first

- `review/本轮总览.md`: start here for the updated overall findings.
- `review/条件泛化续研报告.md`: monthly-weather/anomaly models, sentinel reduction, range rejection, source-removal evidence, and manuscript-ready result paragraphs.
- `review/中文审阅报告.md`: first-two-round checkpoint; literature review, 16 concrete deficiencies, methods and results. Its future-work list is updated by the continuation report.
- `review/论文主线与关键表述修订.md`: reviewable changes to the central question and misleading claims.
- `review/all_frozen_rule_results.csv`: every selected rule across all evaluation stages.
- `review/configuration_pass_status.csv`: 60 configurations; none passed both later stages.
- `review/condition_coverage_audit.png` and `review/label_window_baselines.png`: scientific result figures.
- `review/verification.json`: 240 saved-rule stage checks and 240 predicate checks passed.
- `review/prediction_csv_schema.json`: original numeric dtypes required when applying exact thresholds to exported prediction CSVs.
- `review/anomaly_verification_summary.json`: 288 third-round stage metric and predicate checks.
- `review/range_support_verification.json`: 288 range-audit metric and interval checks, 72 unchanged-prediction checks and 5,184 rolling interval timing/size checks.

## Reproduce

Python 3.12.14 was used, with versions recorded in `requirements.txt`. The input data are present in the existing project snapshot specified below and referenced by content hash in the run manifests. This lightweight review package does not include those raw archives.

Run from this directory:

```powershell
python run_study.py --reference-root "C:\Users\USER\.codex\.chatgpt-projects\g-p-69aaacd3007481919301f520a145f3f4"
python run_sequential.py
python summarize.py
python verify_results.py
python run_anomaly.py
python summarize_anomaly.py
python run_range_support.py
python verify_range_support.py
```

`run_study.py` creates a validated FUA-month cache and the static results. `run_sequential.py` requires that cache and the static source predictions. It performs strictly earlier-month fitting using development cities only. `summarize.py` generates the review tables and figures. Subsequent runs overwrite generated results in this exploratory branch; preserve a copy if changing the protocols.

`run_anomaly.py` requires the FUA cache and original city manifests. It changes source targets and uses actual monthly weather, with 2/5/10 sentinel cities. `run_range_support.py` reuses its frozen choices and predictions, adding a past-calibration score interval. No target outcomes are removed from the evaluation denominator. Its saved predictions remain unchanged on accepted rows.

The protocols were written before the respective experiment, but after older project analyses had inspected the broad time period. Later rounds respond to earlier findings. None is a public preregistration or a new untouched test. The supplied scripts implement transparent baselines and simple threshold search, not the full published DDGroup, RSD, TransFusion, KMM-CP, or CAP algorithms.

The third-round and range-audit prediction exports use 17-digit precision. The older exports require the supplied dtype restoration schema at exact gate boundaries. Two third-round local log-Ridge stage metrics overflowed under extrapolation; these failed comparator records are retained in `review/anomaly_nonfinite_model_metrics.csv`.

## Key interpretation

The per-fit target-label budget is not the total study label budget. Source-only predictions with target-learned selection gates are not zero-target-label procedures. SatPM pseudo-observations are not regulatory ground truth. Confidence intervals are exploratory clustered comparisons, not universal conditional coverage guarantees.

The small review ZIP omits raw inputs, the generated FUA cache, fitted model binaries and row-level prediction archives. Those remain in the full local `research/conditional_transfer_20260912` branch. Running the scripts recreates them. The review ZIP includes the rule search records, frozen protocols, manifests, summary metrics and scripts.
