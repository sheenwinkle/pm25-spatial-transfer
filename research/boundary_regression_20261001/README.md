# Fixed-condition replay, 2026-10-01

This branch isolates the NSW year-end acquisition correction. It replays the
previous monthly regional experiment; it does not search for new conditions.
The two original role/geographic cohorts, city splits, model hyperparameters,
48 locked model/condition settings and calibration rules remain fixed.

## Completed

- Full 2015-2022 NSW monthly reconstruction from original daily cache plus
  independently archived missing December 31 observations.
- 144,655 station-month rows; no previously absent station-month was added.
- Recomputed all-network and both regional targets with unchanged covariates.
- 12 configurations, 576 rolling fits, 48 frozen settings, 240 audit rows,
  12,960 model metric rows, 11,367 prediction rows across configurations.
- Four Australian national source/shuffled models reproduced exactly on the
  original data before refitting with corrected targets.
- All metric rows independently recomputed; time ordering, target-label budgets,
  rule/model identity and 298 frozen-file hashes checked.
- Chinese and English HTML/Markdown reports, paired tables and browser QA.

## Result

Joint temporal/new-city passes: 0/48 before and after.
Early passes: 19 before, 16 after.
Positive corrected new-city gain settings: 0/48.
These are overlapping retrospective settings, not 48 independent trials and
not fresh external confirmation. Neither positive transfer nor universal
impossibility is established.

## Scripts

`replay.py` deliberately imports the previous fitting and audit implementation
instead of copying or modifying it. Only its output root is redirected in
memory. `PROTOCOL.md` and the run receipt freeze this wrapper before fitting.
`verify_replay.py` independently checks metrics and data/temporal invariants.
`build_review.py`, `qa_review.cjs` and `package_review.py` build and verify outputs.

The package includes this branch under `branch/`, reader artefacts under
`review/`, and the prior data-readiness package under `context/`. It is not a
standalone reproduction of every dissertation experiment: original cached
QLD/VIC inputs, baseline covariate panels, prior national model artefacts and
helper modules remain existing-workspace dependencies recorded by hashes or
paths. Source code paths are intentionally not rewritten after fitting.

## Not completed here

- New label-budget or condition-search experiments.
- All previous weekly experiments or a full replacement of the dissertation.
- Chinese post-2022 acquisition and station-year reconciliation.
- South Australian quality checks and final external-test certification.
- Historical role/instrument harmonisation (old snapshots are intentionally
  retained to isolate the acquisition correction).

Read the 2026-09-30 data-readiness report for acquisition/evidence context and
the 2026-10-01 replay report for the updated model status. The former was a
snapshot before this model replay, not a conflicting claim about current work.
