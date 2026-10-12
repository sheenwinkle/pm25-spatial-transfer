# Completion branch: 12 September 2026

Author (provisional): Sheen Wang. Course: COMP5702.

This is an exploratory research revision, not an independently preregistered
experiment. Existing historical outcomes have already been inspected. Never
overwrite earlier releases or label new historical splits as prospective tests.

- [x] Locate original manuscript, archived raw monitoring data and real FUA polygons.
- [x] Recover previous monthly weather experiment and identify its claim boundaries.
- [ ] Rebuild regulatory monitor cohorts with explicit time, quality and spatial provenance.
- [ ] Run monitor-trained, bidirectional transfer with equal-budget local comparators.
- [ ] Quantify conditional-rule coverage, uncertainty and robustness.
- [ ] Integrate English research report, journal adaptation and full Chinese explanation.
- [ ] Verify references, numerical claims, maps, Overleaf compilation and rendered pages.
- [ ] Package reproducibility materials and before/after change record.

Author-side checks remain: Canvas rubric and permitted AI scope, supervisor name,
student ID, degree wording, COMP5704 enrolment relationship and final submission
deadline. Public COMP5702 information is not a substitute for these checks.

## Scientific acceptance boundary

A completed thesis does not require positive cross-country transfer. A claim of
useful conditional transfer requires a discovery-frozen covariate rule, adequate
coverage, improvement over equal-target-label local learners and shuffled-source
controls, and uncertainty assessed by spatial and temporal blocks. Do not remove
samples according to their prediction error or tune on audit outcomes.

## Monitoring protocol, recorded before fitting in this branch

1. Common observed period: 2015--2022. 2023--2024 Australia is not a symmetric
   China--Australia observed-label experiment. Satellite-product labels remain
   a separate benchmark, never relabelled as direct observations.
2. Chinese cached monthly means were formed from days with >=20 valid hours;
   recheck selected raw station-days and all archive MD5 checksums. Recompute
   monthly eligibility as >=ceil(0.75 * calendar days). Preserve raw exclusions.
3. Queensland/Victoria: deduplicate instrument-hour records, preserve plausible
   signed measurements (-10 to 1000 ug/m3), form daily means with >=20 hours,
   then apply the same monthly day fraction. Report negative-value prevalence.
   China used nonnegative hourly values upstream: this asymmetry remains explicit.
4. NSW: rebuild monthly means from archived official daily averages and require
   >=75% calendar days. Hour counts are unavailable, so this cohort does not have
   independently verified 20-hour daily completeness. Report sensitivity excluding NSW.
5. Use supplied calendar dates (native agency reporting day). Do not silently
   assert that a timezone-ambiguous legacy timestamp is UTC. Compare Queensland
   native-day and +10-hour alternatives; carry ambiguity as a limitation if the
   original annual-archive convention cannot be authoritatively established.
6. Primary assignment is station point intersecting a GHS-FUA polygon, with no
   nearest-city fallback. Exclude unmatched/ambiguous stations from primary FUA
   analysis, save them in the audit. Average eligible station means equally per
   FUA-month, retain station counts and station-composition diagnostics.
7. Use the six recovered monthly POWER weather fields plus the existing six
   contextual fields. Source fit 2015--2018; target calibration uses only strictly
   earlier months from development FUAs. Gate discovery 2019--2020; temporal and
   held-city audit 2021--2022. Three fixed city split seeds (5993--5995), half of
   eligible target FUAs for development, capped at ten; eligibility uses >=6
   observed months in 2018 only. Use ceiling for odd city counts. Cities that
   become available later stay in the audit rather than being discarded.
   No audit-dependent seed selection.
8. Compare target mean, ridge, RBF kernel, random forest, Extra Trees and
   LightGBM against source-only, affine calibration, residual boosting and
   source-partition-kernel adaptation. Use 3/6/12-month label windows and matched
   target rows. Three independently shuffled source-label controls.
9. Gates: at most two scalar inequalities, discovery quantiles only; keep >=25%
   of rows and cities, >=3 cities where available and >=12 calendar months.
   Select source learner/rule using discovery only and require >=5% improvement
   over all local comparators and shuffled controls. Evaluate frozen choices even
   if discovery fails. Report full coverage and abstention separately.
10. Bootstrap paired errors over FUA and quarter blocks; seeds overlap and are not
    independent studies. Do not multiply their sample counts. Report R2, RMSE,
    MAE, bias, coverage and local-relative skill together.
11. Prespecified feature sensitivity: add the five existing log-transformed EDGAR
    annual emission fields to common monthly weather/context, on all networks.
    These are annual inventory covariates, not new monthly emission observations.
    Hour-auditable sensitivity uses common features only. Add a source-partition
    model weighted by a balanced logistic domain classifier trained on source and
    pre-2019 development-city covariates; clip importance odds to [0.1, 10].
