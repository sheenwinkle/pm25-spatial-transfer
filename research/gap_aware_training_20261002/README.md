# Bounded valid-observation history experiment

Protocol locked 2026-10-02; evaluation and reader delivery 2026-10-03
(Australia/Sydney). This is a research branch, not a replacement of frozen
baseline results or a complete updated thesis.

## Scope

- Replace the calendar mask in the existing fit function with per-city latest-W
  eligible observations, aged at most 12 calendar months; W=3 primary, W=1
  sensitivity; minimum five total target labels remains unchanged.
- 24 modified configurations and one unchanged-selector regression run.
- Preserve source models, source cutoff, features, hyperparameters and splits.
- Preserve observed targets; no imputation, quality relaxation or condition search.
- Compare against 24 existing fixed-window configurations on paired common rows.

`run_gap_models.py::select_history` implements the only training-selection change.
`fitter` extracts the original fit function using Python's AST, changes exactly
one mask assignment and one manifest field, and injects the selector. Receipts
record that the nominal width is a label count, not calendar duration in valid
mode. Existing code and results are not overwritten.

## Verified outcome

All 1,152 modified rolling fits completed. The one-observation configurations
recovered 36 formerly skipped configuration-months. This produced 469 prediction
rows across overlapping cohorts/splits, or 152 distinct country-city-month keys;
these are not newly collected measurements. Primary three-observation new-city
errors changed little, and the primary joint transfer criterion remained 0/6.
Across both widths and cohorts it remained 0/24.

The raw audit located five HTML 404 payloads inside nominal daily CSVs and
reproduced 2,960 cached station-month aggregates. Six secondary-source probes
recovered zero new station-hour values. This is not an exhaustive source search.

## Outputs

- `audit/VERIFICATION.json`: independent recomputation and label-set checks.
- `audit/primary.csv`, `paired_primary.csv`, `metrics.csv.gz`: performance.
- `audit/label_accounting.csv`, `availability.csv`: label and prediction coverage.
- `valid_*/results/`: prediction files, source bundles, fit audits and exact labels.
- `audit/raw_day_availability.csv`, `raw_month_diagnosis.csv`: archive causes.
- `evidence/backup_probes/`: HTTP bodies, URLs, timestamps and hashes.
- `audit/model_run_receipt.json`: frozen input, code and protocol hashes.
- Reader files in `output/research/gap_aware_training_20261002` in the workspace.

## Dependencies and reproduction

Use the existing project's scientific Python environment. The scripts depend on
the sibling `regional_station_scope_20260920`, `boundary_regression_20261001`,
`history_window_20261002`, `focused_conditions_20260919` and
`thesis_completion_20260912` branches, plus original CNEMC archives in the prior
project. Absolute input paths and hashes are recorded in the receipts. This
package alone is not a self-contained reproduction of all earlier experiments.

Order: `audit_raw.py`, `probe_backup.py`, `coverage_inventory.py`,
`run_gap_models.py`, `evaluate.py`, `verify.py`, `build_review.py`,
`qa_review.cjs`, `package_review.py`. Training and probes resume cached results.
Do not edit a locked training script or protocol and reuse old completion files
as though they came from the new code. Start a new branch for new hypotheses.

The English reader is a manuscript amendment, not a compiled full dissertation.
No new external confirmation, real-time forecasting performance, uniform NSW
hourly verification or target-label acquisition savings is claimed.
