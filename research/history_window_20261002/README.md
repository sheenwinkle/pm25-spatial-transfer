# Historical-window sensitivity, 2026-10-02

Primary: three-month target history, regional_kernel, membership_all cohort.
Sensitivity: one/six/twelve months and the original role-snapshot cohort.
The source/target country directions and three city splits are unchanged.

This is a retrospective historical-window experiment, NOT a fixed global
labelling budget or fresh independent confirmation. A shorter window changes
sample count, recency and seasonal composition simultaneously.

## Reproduction

Use the existing scientific Python environment in the original project.
1. `run_windows.py`: fit and save four windows; exactly two AST edits to the
   previous implementation (lookback and manifest value), checked structurally.
2. `evaluate.py`: all 27 model metrics, native and common-row availability,
   primary model screens, and distinct-label/per-fit accounting.
3. `verify.py`: original hashes, twelve-month regression, temporal/city label
   sets, availability intersection, and independent point-metric calculations.
4. `build_review.py`: Chinese/English reports, complete secondary point estimates,
   charts and the 2018 Chinese cached-completeness diagnostic.
5. `qa_review.cjs`: desktop/mobile HTML checks.
6. `package_review.py`: hashes and verified ZIP.

The fitting implementation and source models are imported/read from prior
branches without modification. Data copies under window_*/data are identical
to the boundary-corrected panels. Twelve-month reused manifests retain their
original provenance; complete.json identifies reuse versus new fitting.

## Completed and scope

48 configurations: 37 newly fitted and 11 reused after a twelve-month regression
test. 2,268 rolling windows checked (not all are newly fitted), 12,960 metric rows
independently verified. Primary three-month joint passes: 0/6. All-window primary
joint passes including the sensitivity cohort: 0/48. Thirteen of 432 secondary
new-city point estimates were positive; the maximum 2.28% is below the locked
5% effect threshold. Secondary estimators were not promoted to primary.

The package includes this branch and its reader outputs, not the entire prior
dissertation workspace or all source archives. Existing helper modules and
national source-model files remain external dependencies listed in the run
receipt. Original model outputs and protocols remain unchanged.

## Remaining issue

The current Chinese 2018 cached station-month records have at most 19 valid
days in August and 23 in December, below the required 24. This explains absent
eligible months and one-month calibration gaps in the current panel, but not
their raw-archive cause. No completeness threshold was relaxed. Raw archive
inspection, actual fixed-total-label-budget tests, external validation and full
manuscript integration remain incomplete.
