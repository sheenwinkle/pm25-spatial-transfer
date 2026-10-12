# Fixed-condition replay after NSW year-end acquisition correction

Locked on 2026-10-01 Australia/Sydney before corrected model fitting.

## Question and scope

Quantify the effect of correcting omitted 31 December NSW daily records on the
existing monthly China-Australia conditional-transfer experiment. This is a
retrospective data-version sensitivity analysis, not an untouched external test,
new model search, or proof that a narrower region generalises.

Preserve the previous regional branch, its inputs, models, predictions, city
splits, 48 condition locks, station-role snapshots and bootstrap procedure.
Preserve the completed data-readiness branch. Write only in this new branch and
its output directory. Do not open new SA labels or include 2023-2025 data here.

## Data correction

Rebuild all NSW station-months for 2015-2022 from original daily files plus only
the separately archived missing December 31 records. Preserve original values
and missingness on all overlapping dates. Reuse the original numeric and monthly
completeness screens. Recompute eligibility; reconstruct previously absent
months if any. Reuse the existing site metadata and unique FUA mappings, not
new role classifications, to isolate this one acquisition correction.

Recompute all-network and both regional cohort targets, station counts and
station dispersion. Hold existing covariates fixed and assert that any newly
eligible city-month has matching covariates. Stop on a missing covariate row;
do not impute a new city-month from outcome information. Report city-month
membership changes and target changes. No target trimming or rule revision.

## Model replay

Use the exact prior fit_predict implementation with its original helper module,
hyperparameters, 12-month rolling target calibration, 2018 source cutoff and
three seeds. Use the same fixed development/new-city split for both cohorts.
Retrain Australian national source and shuffled-source models on corrected
2015-2018 targets, cloning their original estimators. Before this, refit each
on the original data and require prediction agreement with its archived model.
Reuse unchanged Chinese national models. Refit regional/matched and local
models through the existing implementation. Save helper and dependency hashes.

Replay all 48 old locked rules and selected model names without condition
search. Recompute their discovery and validation support/gains on corrected
predictions to report whether their early gate changes; do not reselect a
model or rule if it fails. Recompute the same temporal, new-city and weather-grid
audits with the original comparators and clustered bootstrap. Re-evaluation
does not erase previous exposure. There is no multiple-testing-adjusted
confirmation claim.

## Verification and deliverables

Verify hash immutability of original inputs and outputs, corrected daily/monthly
reconstruction, unchanged non-NSW rows and covariates, exact rule/model/split
identity, and time ordering for every rolling fit. Independently recompute
all reported prediction metrics and compare before/after on matching settings.
Report all directions/cohorts/seeds/coverage floors, not only the best change.
The joint screen retains the previous minimum 5% gain, support and bootstrap
requirements in both temporal and new-city evaluations, plus the early gate.

Deliver corrected monthly data/predictions, a paired effect table, a Chinese
review and an English manuscript note. Weekly experiments, annual role
harmonisation, full China post-2022 acquisition, South Australian quality
verification, and a new condition-selection design remain separate tasks.
