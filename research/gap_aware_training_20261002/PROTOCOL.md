# Missing-day diagnosis and bounded observed-history calibration

Locked 2026-10-02 Australia/Sydney before new model evaluation.

## Source diagnosis

Inspect the original 2018 CNEMC ZIP calendar and the August/December daily CSVs.
Distinguish absent dates, absent PM2.5 hour rows, invalid/duplicate hours,
station-level missing/invalid values and the >=20-hour daily screen. Independently
recompute those station-month counts and means against the cached aggregates.
Check a small deterministic set of deficient dates and complete controls against
the public secondary daily archive, preserving request provenance and conflicts.
Do not auto-overwrite original values, claim a secondary feed independent, or
relax the monthly >=ceil(.75*calendar days) evaluation-target threshold.

## Model change

Replace a fixed calendar window with a bounded window of valid observations:
for each fixed development city, select the most recent W eligible city-month
labels strictly before the prediction month, within at most 12 calendar months.
W=3 is primary and W=1 is a sensitivity setting. Selection depends only on
availability, city and date, never concentration, model error or audit score.
The number of labels is capped at W times the fixed development-city count.
Retain the old minimum of five total target labels. Report actual contributing
cities, oldest/latest label ages, labels outside the nominal calendar window,
label counts and skipped months. Do not synthesize missing target labels.

Use the same corrected panel, source cutoff, city splits, features, model
hyperparameters and all 27 estimators as the history-window branch. Source
models remain unchanged. Apply the exact same selected labels to all local and
calibrated transfer models. This is an availability-repair method, not ADWIN,
BRITS or GRU-D, and makes no claim to reproduce those algorithms.

Primary estimator remains regional_kernel, primary cohort membership_all,
both directions and all three existing splits. Compare fixed and observed-history
windows on identical rows where both predict, and report native coverage and
newly predicted rows separately. Do not claim performance gains by comparing
different evaluation samples. W=3 is not replaced post hoc by W=1.

Keep stages, comparators, >=5% incremental-transfer threshold, per-year support
and clustered uncertainty from the preceding protocol. No condition search,
new target trimming, independently confirmed performance, or promise of R2.

## Verification and output

Regression-test the unchanged fixed-selector path against the old predictions;
unit-test missing-month, no-future, held-out-city, 12-month age bound and label-cap
invariants. Audit the exact selected row set for every fit. Independently
recompute reported metrics. Preserve all original artefacts and provide a
Chinese before/after explanation, English manuscript amendment and evidence.

Imputation/reanalysis/IDW reconstructions, if pursued later, must remain separate
estimated products evaluated by artificial masking of observed data; they do
not become observed validation labels. Global label-cost savings and operational
publication delays are outside this experiment's claims.
