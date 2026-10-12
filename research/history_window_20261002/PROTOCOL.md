# Historical-label window sensitivity

Locked 2026-10-02 Australia/Sydney before new fits or scores.

Use only the boundary-corrected 2015-2022 monthly panels and fixed city splits.
Do not modify previous branches or open SA/new external labels. This is a
retrospective exploratory experiment on previously examined years, not fresh
confirmation. The primary population is membership_all; documented_role remains
a sensitivity cohort with its old, unverified cross-year role snapshots.

Compare rolling target-label windows of 1, 3, 6, and 12 months. The primary
window is 3 months and the primary transfer estimator is regional_kernel,
chosen before this run as an existing geographic-source representation model.
No condition search, outcome-based trimming, or model winner selection. Report
both directions and all three existing splits. All 27 existing predictions are
retained, including six local, nine transfer, nine shuffled-source and three
uncalibrated-source estimators; other transfer estimators are secondary.

For each window/configuration, all calibrated estimators and target-fitted
preprocessing receive exactly the same past target labels. Unlabelled target
covariates from the existing pre-2019 source-matching design remain available.
Source models remain trained through 2018. Preserve all model hyperparameters
and the minimum of five target rows for a rolling fit. Fixed parameters may
not be optimal for every smaller sample, so this is not a tuned optimum claim.

Reuse the previous fitting implementation with exactly two AST substitutions:
the lookback in the training mask and its manifest value. Verify both substitutions
and reproduce a 12-month configuration numerically against the corrected replay.
The other 12-month configurations may reuse the verified prior predictions;
record reuse versus new fitting instead of claiming all were refitted.

Report native prediction availability and a common-row intersection across all
four windows for each cohort/direction/split, chosen only by prediction
availability, not scores. The primary comparison uses that common intersection.
Disclose its fraction of every stage's full eligible panel; do not hide loss of
months or cities. No intersection may be described as full-country coverage.

Stages: development cities in 2019, in 2020, and in 2021-2022; held-out cities in
2021-2022; held-out cities additionally outside development weather grids in
2021-2022. For the primary estimator, compare RMSE with all six local and nine
shuffled controls on the identical rows. A descriptive screen requires >=5%
gain against the strongest observed comparator; >=50% stage coverage; and in
every represented evaluation year >=24 rows, >=3 cities and >=6 months. Require
at least two evaluation years for the temporal/new-city joint screen. Apply
the existing two-way city-quarter and weather-grid-quarter bootstrap, 1000
replicates, against each comparator. All lower bounds must exceed zero.
Report joint temporal/new-city passes per split without asserting independence
between splits or multiplicity-adjusted significance. Early-stage results are
reported but are not used to choose the estimator, window or rows.

A window is not a fixed global labelling budget: report per-fit rows, unique
label rows over the entire rolling experiment, training city counts and skipped
months. Do not infer reduced collection cost merely from a shorter window.
The next question about actual fixed label budgets is separate and not solved
by this experiment. Quantify 3-versus-12 differences on identical rows without
claiming significance for the difference from marginal confidence intervals.

Deliver all predictions, metrics, support/availability and label accounting,
primary-screen tables, regression tests, Chinese review, English manuscript
note and an auditable package. No promised R-squared or successful transfer.
