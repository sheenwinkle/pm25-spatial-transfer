# Post-hoc window and uncertainty extension

Recorded on 2026-10-09 before fitting this extension. Motivated by the user and
review of the already-observed three-month result; this is not preregistration
or an independent confirmation sample. Existing results remain immutable.

## Comparators and reporting

The per-split post-hoc best target-only comparator is the minimum test RMSE
among benchmark RF, benchmark GBR, local mean, ridge, RBF, random forest,
Extra Trees and LightGBM. GBR[LDF] is excluded because it uses source labels.
This oracle-per-split reference is distinct from a single target-only model
chosen by lowest mean RMSE over splits, as in the Australian summaries.

## Fixed fits

- Task: non-YRD China to the same 30 eligible YRD FUAs (T1).
- k = 5 and 11; the original 20 splits from seed 1000; same test FUA-months
  in 2021-2022; source labels remain fully available in 2015-2022.
- Three-month windows: 2020 October-December (replay), 2020 April-June,
  2020 July-September and 2019 October-December.
- Models: source-affine, NNW[LDF], source-affine[LDF], and the eight
  target-only learners. No new hyperparameter search or outcome filtering.
- Use the existing LDF architecture, 30 epochs and released feature-weight
  implementation of NNW. First run the original LDF seed equal to split ID.
- For October-December 2020 only, repeat with LDF seed = split ID + 10000;
  all splits, non-LDF fits, target rows and GBR seed stay fixed.
- Save city/month keys, truth, predictions, split IDs, fit-window membership,
  environment versions and elapsed run logs. Do not overwrite original CSVs.
- Verify replay against archived three-month metrics before attaching an
  interval to the archived 12.2% / 12.5% values. Report any numerical drift.

## Paired uncertainty

Use the same crossed FUA x calendar-quarter multinomial bootstrap principle
as the conditional audit, with 1,000 replicates and seed 20261009. Draw one
set of FUA and quarter multiplicities per replicate and share it across all
20 overlapping splits and compared estimators. Recompute each split's
weighted RMSE, then average the paired relative reductions across splits.
Percentile 2.5/97.5 bounds describe test-panel uncertainty conditional on the
fixed fitted models, fixed splits and chosen window. They do not include
post-hoc selection or uncertainty from resampling the training procedure.
Report model-seed sensitivity separately; do not treat splits or two seeds
as independent datasets. Validate the cell-aggregated implementation against
direct row weighting on a small synthetic fixture.

## Scope

Report every specified window and both budgets, regardless of direction.
Do not select a new preferred window from these tests. Do not add a claim
about computational speed or a new causal mechanism.

## Execution notes (continued 2026-10-10)

The first original-window replay reproduces all non-LDF metrics, but differs
from the archived NNW[LDF] RMSE by -0.0632014 on k=5, split 0. Running the
unchanged original short-history script in the same native Windows
environment gives the same result as the prediction-saving runner. The
archive was produced on Linux with a different Python version; the precise
cause of LDF numerical drift remains unresolved. Report archived and replay
estimates separately and attach bootstrap intervals only to the latter.

To reduce dependency memory, the extension uses the same sklearn nearest-
neighbour count calculation as ADAPT 0.4.5 NNWeighting, without importing
TensorFlow. Exact equality was checked against ADAPT on three fixtures and
on the full first replay split. This changes dependency loading, not the
weighting algorithm or fitted-model specification.
