# Post-review positive-control protocol

Recorded: 4 October 2026, before fitting this analysis.

This is a post-hoc diagnostic requested after review. It is not preregistered,
prospective, or independent confirmation of the China--Australia experiment.
Its sole purpose is to test whether the primary pipeline can recover
source-domain value in a more favourable within-country setting while retaining
the same target-label budget and decision standard.

## Fixed population

- Data version: corrected `boundary_regression_20261001` all-network panel.
- Target domain: the 43 Chinese FUAs appearing in the official-membership YRD
  cohort, but with targets taken from the all-network panel so source and target
  use the same station aggregation definition.
- Source domain: all other Chinese FUAs in that panel. Source and target FUA IDs
  must be disjoint.
- Source training years: 2015--2018.
- Target development and evaluation years: 2015--2022.
- Feature set: the twelve `monthly_common` fields used by the primary national
  observed-label experiment. No feature or hyperparameter search is permitted.

## Fixed learner and label budget

- Source learner, six target-only comparators, four source-informed learners,
  and three shuffled-source controls are identical to the primary experiment.
- Development-city eligibility requires at least six target months in 2018.
- For seeds 5993, 5994, and 5995, shuffle the eligible IDs and take half,
  rounding upward and capped at ten. Do not select a seed using outcomes.
- At each prediction month, target-assisted learners receive only labels from
  development FUAs in the immediately preceding 3, 6, or 12 months. Current
  month and held-city labels are forbidden.

## Fixed gate and audits

- Gate discovery: development FUAs in 2019--2020.
- Gate class: at most two inequalities over the same eight gate variables and
  discovery-only predictor quantiles.
- Minimum support: 25% of rows and cities, at least three cities and twelve
  months.
- Discovery requires at least 5% RMSE improvement over the strongest of all six
  local and three shuffled-source controls, plus within-year improvement.
- Frozen audits: development FUAs in 2021--2022 and held-out FUAs in 2021--2022.
- Audit success requires the discovery pass, at least 5% point improvement over
  the strongest comparator on identical accepted rows, support, and positive
  lower clustered-bootstrap bounds under FUA and weather-grid resampling for
  the selected comparator, target mean, and all shuffled controls.
- A configuration is jointly successful only if both temporal and held-city
  audits pass. Full-coverage metrics and accepted coverage remain visible.

## Interpretation

A joint pass would show that the implemented procedure can detect transferable
structure in this within-country control; it would not validate cross-country
transfer. Zero joint passes would expose a pipeline-sensitivity problem and
make the cross-country null even less interpretable. All nine configurations
and all failures must be reported; no post-run changes to this protocol count
as part of the positive control.
