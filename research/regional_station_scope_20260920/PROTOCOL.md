# Official regional membership and documented station-role sensitivity

Frozen before new model fitting, 21 September 2026. Directory was opened on
20 September for primary-source acquisition. This is an EXPLORATORY extension:
2015-2022 outcomes have already been inspected in earlier branches. It is not
a preregistered confirmation or an independent test of earlier hypotheses.

## Question and scope

Does conditional, few-shot China-Australia PM2.5 transfer improve after fixing
the regional definition and reducing documented monitoring-purpose mismatch?
Absolute fit and incremental benefit from source-country labels are different
endpoints. Do not delete large observed PM2.5 or large model residuals.

Chinese membership uses the 26 prefecture-level cities in the NDRC 2016 Yangtze
River Delta city-cluster plan. Join Chinese administrative city names in the
CNEMC archive; explicitly map the archive's county-level city names to their
documented parent prefectures. English Suzhou and Taizhou are ambiguous and
MUST NOT be join keys. The plan includes eight Anhui cities but not Wenzhou.
This is administrative-membership selection, NOT a newly obtained polygon
boundary, and NOT the same definition used by every Yangtze Delta paper.
Continue using the existing unique same-country GHS-FUA polygon assignment
to define the modelling unit. A prefecture and an FUA are not interchangeable.
Australia retains the existing observed NSW/QLD/VIC FUA panel, not a claim of
nationally representative Australian coverage. No new latitude or coast cutoff.

## Two cohorts, fixed without new outcome inspection

1. `membership_all`: eligible station-months inside the Chinese membership
   above, and all eligible Australian station-months already assigned to FUAs.
2. `documented_role`: restrict China to exact city+station-name matches to the
   ordinary (non-control) column of the MEE 2012 national-network roster. No
   fuzzy matching or inferred aliases. Unmatched names are UNKNOWN, not bad data.
   Australia includes explicitly listed non-roadside NSW NEPM stations (2022
   table 2; exclude Bradfield Highway), QLD neighbourhood PM2.5 locations (2023
   table 80), and VIC NEPM PM2.5 locations (2021 table 1). Matched archived IDs
   are listed explicitly. QLD peak/background locations, NSW Newcastle local
   industrial-purpose network and unknown sites are not in this subset.

The second cohort is a retrospective DOCUMENTED-ROLE SENSITIVITY, not a fully
year-verified urban-background cohort: CN evidence is 2012, NSW 2022, VIC 2021,
QLD 2023. Applying designation snapshots to all available 2015-2022 records
assumes role stability. Store evidence year and this assumption for every
classified site. NEPM can include representative upper-bound, industrial/
residential, and roadside-influenced locations. Never call this a pure
background subset or claim that China/Australia measurement roles are identical.
Record exclusions and unknowns. Original samples remain intact in cohort 1.

Rebuild each FUA-month target as the equal-weight mean of eligible station-month
measurements AFTER selection. Retain the prior observation completeness and
signed-value rules. Join identical existing predictors by country, FUA, year,
month. No interpolated or model-reconstructed PM2.5 becomes ground truth.
Report station count, target change on common rows and lost rows/cities.

## Models and validation

Two cohorts x two transfer directions x three seeds (5993/5994/5995) = 12
configurations. For each target country, determine eligible development cities
from the INTERSECTION of cities with >=6 valid 2018 months in both cohorts.
Shuffle IDs with each seed; choose ceil(n/2), capped at 10, identical in both
cohorts. Other cities remain held out; new post-2018 sites cannot join calibration.

Reuse unchanged model definitions from focused_conditions_20260919:
six target-only controls; affine, residual and tree-partition kernel transfers
for national, regional and covariate-matched source policies; three shuffled
source-label kernel controls per policy; raw predictions as descriptive baselines.
National models are frozen original all-network 2015-2018 models, deliberately
retained as broad-source controls (NOT re-labelled as purpose-aligned).
Regional and matched models are re-trained on this cohort's source-country
2015-2018 labels. Matching retains half the regional source rows closest to
pre-2019 target development covariates, with no target outcome matching.
All six local models and all target calibrators are rebuilt on precisely the
same previous 12 months of development-city labels within each configuration.
No held-out city labels; no future labels. The targets and available labels
differ between cohorts, so an across-cohort gain is a task/sampling sensitivity,
NOT the isolated causal effect of station type. Use the same 15 predictors.

Search at most four inequalities using the prior deterministic 12-wide beam.
Discover in development-city 2019; shortlist and select in 2020; write four
locks for coverage floors 50/25/10/5% before auditing 2021-2022. Retain all
failed choices. Minimum per year: 24 rows, three FUAs, six calendar months.
Audits: temporal development-city, held-out city, held-out POWER grid. Same-row
comparators: strongest of six target-only and nine shuffled-source controls.
Screen: >=5% RMSE gain in both selection years; >=5% temporal AND held-out-city
gain, per-year support, positive unadjusted paired bootstrap lower bounds
against every control for FUA-quarter and POWER-grid-quarter resampling.
These are exploratory screening flags, not multiplicity-adjusted significance.
Twelve configurations yield 48 overlapping coverage settings, not 48 replications.

## QA and deliverables

Check official roster counts/roles, metadata joins, unknowns, station-month
uniqueness, exact target recomputation, fixed covariates and paired city splits.
Verify calibration chronology, source cutoff and same label budget for all
methods, rule masks, every reported metric, and aggregate completion counts.
Provide Chinese report, before/after tables, condition ledger, source URLs,
limitation statement and reproducibility bundle. Do not alter the old releases.
Stop after these fixed configurations; no outcome-dependent expansion of gates.
Any positive result needs new time/city data and year-specific role verification.

## Primary and methodological sources

- NDRC (2016), Yangtze River Delta city-cluster plan, printed page 1:
  https://www.ndrc.gov.cn/xxgk/zcfb/ghwb/201606/W020190905497826154295.pdf
- MEE (2012), national-network station roster, table 3:
  https://www.mee.gov.cn/gkml/hbb/bwj/201204/W020221214657415435410.pdf
- NSW (2022 reporting year), tables 2-3, actual publication later:
  https://www.environment.nsw.gov.au/sites/default/files/2025-02/nsw-air-quality-monitoring-report-2022-NEPM-250027.pdf
- QLD (2023 reporting year), table 80, published 2024:
  https://www.qld.gov.au/__data/assets/pdf_file/0032/68657/air-monitoring-report-2023.pdf
- EPA Victoria (2021 reporting year), table 1, publication 2052 (2022):
  https://www.epa.vic.gov.au/sites/default/files/epa/publications/2052_nepm_compliance_air-quality-report.pdf
- Su, L., et al. (2021), station spatial representativeness, NOT an exact
  replication and NOT the same region definition as NDRC 2016:
  https://html.rhhz.net/hjkxxb/html/20210211001.htm
- Roberts et al. (2017), spatial/temporal validation:
  https://doi.org/10.1111/ecog.02881
- Cawley & Talbot (2010), model-selection bias:
  https://jmlr.org/papers/v11/cawley10a.html
