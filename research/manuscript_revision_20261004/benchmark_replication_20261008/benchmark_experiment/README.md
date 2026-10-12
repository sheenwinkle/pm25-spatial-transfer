# Benchmark-aligned experiment

The saved per-split results are in `results_per_split.zip`. Extract that
archive beside the scripts so that the files appear under `results/`, then run:

```text
python check_claims.py
python make_tables.py
python make_figures.py
```

These three commands need only the saved result files. `check_claims.py`
exits with a non-zero status if any check fails or if the input files of a
check block are missing (round 7: missing inputs are no longer counted as
passes).

`make_tables.py` regenerates Tables 2–6 of the manuscript exactly. Round 7
changes three things in it:

- ties at the displayed precision are all set in bold (Table 3, k = 5);
- R-squared is typeset as `\(R^2\)`;
- in Table 5 (no contemporaneous source labels), source-affine is compared
  with the target-only fits of the matching short-history run, because those
  learners use no source data. A refit would differ only in row order and
  feature scaling, yet seeded subsampling would move a split's GBR RMSE by
  up to 4.9 µg/m³.

Refitting the models (`bench_harness.py`, `bench_m.py`, `bench_n.py`) and
`diagnostics.py` also need the corrected modelling panel. It is shipped as
`data/all_networks_model_panel_corrected.csv.gz`:

- 34,639 FUA-months;
- SHA-256 `9026749a73704d45e218a4aeb96d79704280edc98ad8e6dab3ed34bb627f2a03`;
- built by the round-2 replay, `review_evidence/replay/prepare_panels.py`.

Refitting additionally reads two existing project files, which `paths.py`
locates relative to the `Dissertation Program` folder:

- the earlier estimator module, `research/thesis_completion_20260912/run_monitor_models.py`;
- the FUA membership file, `research/boundary_regression_20261001/data/membership_all_panel.csv.gz`.

Round 7 fixes `paths.py`, which pointed one folder too high. Any path can be
overridden with these environment variables:

- `DISSERTATION_ROOT`
- `BENCH_PANEL`
- `BENCH_MEMBERSHIP`
- `THESIS_COMPLETION_DIR`
- `BENCH_RESULTS_DIR`
- `BENCH_GENERATED_DIR`

## Current results and historical drafting files

`claims_check.txt` contains the regenerated round-8 checker output (139/139
checks pass). It includes the corrected eight-model comparator and the
66% / 44% retained-gain fractions.

For the current Tables 2-6, regenerate `generated/tab_*.tex` and
`generated/numbers.json` with `make_tables.py` from the saved per-split
results. Table 5 uses the shared target-only fits described above. The
per-split post-hoc best reference is the minimum test RMSE among all eight
target-only learners: benchmark RF and GBR plus the six additional local
learners. The internal key `Best local` is retained for compatibility, but
now means this eight-model minimum. This is different from selecting one
learner by lowest mean RMSE across splits, as in the Australian summaries.
GBR[LDF] is not target-only because its LDF feature uses source information.
This correction applies to the benchmark analysis, not to the earlier
conditional audit's fixed comparator set.

`fill.py` and `generated/fill_ledger.csv` are retained as historical
round-6 drafting artefacts, when draft files contained number placeholders.
The current manuscript does not use them. The ledger's Table 5 values
predate the round-7 correction; do not use that ledger to validate or
populate the current manuscript. `fill.py` is not part of the current
regeneration workflow.

## Post-hoc window and seed extension (round 8)

The fixed specification is in `../../audit/round8_20261009/POSTHOC_PROTOCOL.md`.
The extension reuses the 20 spatial splits, two label budgets and 2021-2022
test rows, and fits all four specified three-month windows. Predictions and
fit metadata are retained separately from the archived result CSVs:

```text
python bench_window_sensitivity.py --out ../../audit/round8_20261009/window_results
python summarise_window_sensitivity.py --input ../../audit/round8_20261009/window_results --out ../../audit/round8_20261009/window_summary
python summarise_window_sensitivity.py --self-test
python test_round8_invariants.py
```

The runner resumes completed cells and can be divided by `--split-start`
and `--split-stop`. Each completed cell has a compressed row-prediction file
and JSON metrics, training keys, panel hash and model seeds. The original
window is repeated with LDF seed offset 10000; other fits and evaluation
rows remain fixed. The summary requires every specified cell, checks row
alignment and recomputes RMSE from saved predictions before calculating
1,000 crossed FUA-by-quarter bootstrap draws. Resampling weights are shared
across overlapping splits. Intervals are conditional on the fitted models,
splits and window, and do not adjust for post-hoc selection.

The native Windows replay does not exactly reproduce the archived LDF
metrics. An unmodified `bench_m.py` replay matches the new prediction-saving
runner on the checked split, while non-LDF metrics reproduce the archive.
Therefore replay estimates and intervals must be reported separately from
the archived 12.2% / 12.5% points; the numerical drift has not been fully
attributed to a specific platform or library operation.

`nnw_cpu_weights.py` implements the ADAPT 0.4.5 NNWeighting neighbour-count
operation without importing TensorFlow. We checked exact weight equality
against ADAPT on random, tied and one-dimensional fixtures, and exact
metric equality on a full real-data split. `bench_harness.py` imports ADAPT
lazily for its original weighting methods. No model settings or training
rows are changed by this dependency refactor. The extension records its
actual package versions in `window_results/environment.json`.

The completed extension contains 200 prediction files. Its summary checks
2,200 reconstructed model RMSEs and writes `window_summary/validation.json`,
`window_sensitivity_summary.csv`, `replay_metric_differences.csv` and
`tab_window_sensitivity.tex` (the additional manuscript Table 7). Every
window has exactly 15 training labels at k=5 or 33 at k=11. In the original
seed replay, the October-December 2020 gains are 12.9% / 10.0%, with both
clustered intervals including zero; the corresponding 2019 window reverses
the ranking (-25.1% / -22.8%). The 2020 second- and third-quarter windows
remain favourable to source-affine against NNW[LDF]. These are post-hoc
window comparisons, not a newly selected primary protocol.

The 11 October summary also reports the comparisons available from those
same 200 fits: source-affine versus the per-split post-hoc best of eight
target-only learners, and the pre-listed secondary estimator
source-affine[LDF] versus NNW[LDF]. All three comparisons share the same
bootstrap draws; the eight-model reference is re-selected within each
draw. The primary source-affine-versus-NNW[LDF] estimates are unchanged.
Across the four windows and two budgets, source-affine improves on the
eight-model reference by 13.7-32.7%, with seven of eight intervals above
zero. The hybrid improves on NNW[LDF] by 20.8-39.4% in the six 2020
window-budget cells, all with intervals above zero. Only the original
2020Q4 window has a second LDF seed; its hybrid intervals are also above
zero. The 2019 hybrid intervals contain zero, leaving that difference
unresolved rather than demonstrating equivalence. The original-window
plain-affine intervals contain zero in three of four seed-budget cells.

`../../audit/round8_balance_20261011/` records the narrative revision and
checks against the independent review calculations. No new fits, source
rows, calibration windows or primary-model choices were introduced.
