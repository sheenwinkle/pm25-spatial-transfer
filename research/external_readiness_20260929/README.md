# 2026-09-30 data-readiness audit

This branch preserves all original source files and historical model outputs.
Its locked scope is data/evidence readiness, not model selection. The separate
2026-09-30 amendment documents the newly discovered NSW calendar-boundary bug.

## Reader outputs

- `data_readiness_review_zh.html`: self-contained Chinese review with before/after comparisons.
- `manuscript_data_amendment_en.html`: English proposed manuscript amendment.
- `australia_station_month_2023_2025.csv.gz`: 2,916 rows, 2,801 eligible.
- `australia_fua_month_2023_2025.csv.gz`: 425 city-months from eligible matched stations.
- `observed_station_year_registry_2015_2025.csv`: 1,752 rows, including unresolved evidence.

The source package contains these reader outputs under `review/` and this
branch under `branch/`. It is an audit and data-delivery package, not a standalone
reproduction of the entire dissertation. Large legacy QLD/VIC hourly files,
some previous reports, baseline panels and model artefacts remain external
workspace dependencies; their paths/hashes are retained where used. Do not
misrepresent the package as containing all historical raw data or all models.

## Rebuild in the existing workspace

Use the project scientific Python environment. The scripts depend on pandas,
numpy, scipy/scikit-learn in earlier branches, geopandas and python-calamine;
PDF extraction also used pdfplumber. The report uses matplotlib and Pandoc.
Paths are centralised in `common.py`.

1. `acquire.py`, `acquire_candidates.py`: source acquisition, retry/status ledgers.
2. `extract_metadata.py`, `extract_pdf_metadata.py`: structured/annual evidence.
3. `audit_exposure.py`: historical outcome-exposure ledger.
4. `repair_nsw_boundary.py`: original-cache integrity, official year-end patches,
   overlapping-date comparison, December impacts, existing-row correction candidate.
5. `build_extension.py`: corrected NSW / QLD / VIC monthly extension.
6. `build_historical_registry.py`: year-specific role evidence, with conflicts preserved.
7. `verify_readiness.py`: independent aggregation checks and availability-only SA audit.
8. `build_review.py`, `qa_review.cjs`: human-readable outputs and desktop/mobile verification.
9. `package_review.py`: file hashes and ZIP validation.

Running the acquisition scripts may access public source URLs. Cached year-end
patches are reused only with matching request and response hashes. Original
files are never replaced. Baseline eligibility is recomputed in the correction
candidate, but absent original months are not created by that existing-row table.

## Important interpretation rules

- Historical NSW daily caches omitted December 31 in all 11 years, 2015–2025.
  341 station-date slots were appended; 284 held valid concentrations. Five
  December station-months cross the eligibility threshold. See the amendment.
- The 2023–2025 Australian data are previously exposed retrospective data.
  Reaggregation does not reset exposure. The first 2025 test is not retrospectively
  declared invalid merely because it was subsequently reused.
- 99 extension identifiers are not asserted to be 99 distinct physical sites.
- Of 265 role-supported station-years, 263 have no identified same-year conflict.
  Supported designation is not synonymous with pure urban background.
- SA: 18 archives, 152,040 timestamp slots, 79,096 nonempty PM2.5 fields at 5 sites.
  No SA model fitting/scoring or final holdout certification was performed.
- Chinese 2023–2025 files are three fixed-date availability probes only.
- Old predictive scores were not recomputed in this readiness branch.

The report records the state of this audit. A later model-correction branch,
if present, must be cited separately rather than silently replacing old outputs.
