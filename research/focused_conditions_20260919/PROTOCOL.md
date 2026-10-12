# Focused conditional transfer: exploratory extension, 19 September 2026

## Status and estimand

This extension was designed after inspecting earlier 2015-2022 outcomes. It is
NOT a new preregistered or untouched confirmatory study. It retains observed
monthly FUA-level regulatory PM2.5 in micrograms per cubic metre, both transfer
directions, and the original observation/QC definitions. No observations are
removed because their outcomes or prediction errors are large. The old release
and its 90 configurations remain unchanged.

Question: after explicitly restricting both the geographical learning task and
the meteorological applicability region, is there a reproducible incremental
benefit from source-country labels beyond equally calibrated target-only and
shuffled-source models? Good absolute prediction is separately reported and is
not itself evidence that foreign labels help.

## Geographical scopes, fixed before these model runs

1. `coastal150`: FUA centroid distance to coast <=150 km, both countries.
2. `latitude22_35_coastal150`: scope 1 and absolute latitude 22-35 degrees.
   Latitude is a proxy to be tested, not a declaration of identical climates.
3. `east_window`: China centroid 117-123 E, 28-34 N; Australia centroid
   140-154 E, 25-39 S and coast distance <=150 km. This is an explicit
   rectangular eastern-China window, NOT the administrative Yangtze River
   Delta. It includes some places outside that agglomeration.

These scopes are nested/overlapping and are not independent replications.
Membership uses fixed FUA centroid geography, not the test outcomes.

## Sources, calibration and controls

- Source-country labels: 2015-2018 only. Target city eligibility: >=6 valid
  months in 2018 inside the specified scope. Shuffle eligible IDs with seeds
  5993/5994/5995; development cities = ceil(eligible/2), capped at 10.
- Target calibration: strictly previous 12 months from development cities,
  >=5 labels. No labels from held-out cities. All methods use exactly the same
  target labels; total unique labels and each fit's dates are recorded.
- Three source policies: national (cached original common-mixing models),
  regional (new models trained on the specified scope), and matched (regional
  rows within the median nearest-neighbour distance to pre-2019 target
  development covariates). Matching uses six monthly POWER features plus
  three ERA5 mixing features; the scaler is fitted to regional source rows.
  It never uses target audit outcomes or current PM2.5 for acceptance.
- For each source policy: affine calibration, source residual correction,
  source tree-partition kernel. Three shuffled-source-label kernel controls
  per policy. National negative controls are the original cached models.
- Six target-only models: mean, ridge, RBF kernel ridge, random forest,
  ExtraTrees, LightGBM. Original hyperparameters are retained.
- Same 15 predictors as the previous monthly-plus-mixing branch. The coarse
  1.5-degree/six-hour ERA5 context is not re-labelled as local 3-D measurement.
- 3 scopes x 2 directions x 3 city splits = 18 prediction configurations.
  3 source policies are compared inside each, not called independent trials.

## Conditions and nested selection

Train predictions for 2019-2022 with the causal-in-time calibration above.
Discover gates on development-city 2019 predictions ONLY; shortlist the best
12 gates for each source candidate and each minimum coverage setting.
Choose from that shortlist using development-city 2020, maximizing the worse
of 2019 and 2020 skill when none passes. Among candidates with >=5% relative
RMSE improvement over all six local and all nine shuffled-kernel controls in
BOTH years, prefer fewer clauses, then greater minimum annual coverage.

Numeric gate thresholds are 2019 quantiles .1/.25/.5/.75/.9, plus physical
anchors PBL=500/1000 m, RH=60/80%, wind=2/4 m/s, temperature=10/20/25 C,
and log fire count=0. These anchors are transparent exploratory choices, not
claimed universal cutoffs or thresholds estimated by a cited paper.
Gate variables: harmonized monthly POWER temperature, humidity, wind and
precipitation; ERA5 PBL mean, low-PBL fraction, ventilation; coast distance,
population density, fire detections, and regional-source support distance.
Unknown fire availability becomes missing for the fire gate, not 'no fire'.
At most four inequalities using a deterministic beam of width 12 at each
depth (not an exhaustive global optimizer). Record all evaluated candidates.

Minimum accepted coverage settings: 50%, 25%, 10%, 5%, each relative to the
specified geographical scope. Also report coverage relative to the country's
entire observed panel and the actual coverage, never label a minimum as exact.
Every discovery/validation rule must retain >=24 FUA-months, >=3 FUAs and
>=6 distinct calendar months in each year. These safeguards can make 5% or
10% unattainable in small Australian samples. Do not lower them to force a win.
This permits a seasonal applicability task rather than pretending all-year
coverage; report whether the previous >=25%/12-month support rule also holds.

Rules, source policy and model choice are written to lock files before their
2021-2022 outcomes are evaluated. Development-city temporal, held-out-city,
and held-out POWER-grid audits are reported separately. Audits require >=24
rows, >=3 cities and >=6 months in EACH audit year, plus the coverage floor
in each year. Gains are measured against the strongest comparator on the
SAME accepted rows. Bootstrap RMSE differences against all local and all
shuffled controls, resampling FUA x quarter and POWER-grid x quarter, 1000
replicates. Report descriptive unadjusted 95% intervals. A joint screen needs
discovery eligibility, temporal AND new-city >=5% gain, support and all lower
bounds >0. This is a screening flag, NOT multiplicity-adjusted significance.

Also evaluate the same selected model on every in-scope audit row; report
local-only best RMSE, R2, bias and accepted coverage. Do not compare R2 across
different outcome ranges as though they were interchangeable.

## Methodological grounding, not claims of exact replication

- Shah et al. (2022), ICML, selective regression / risk-coverage trade-offs:
  https://proceedings.mlr.press/v162/shah22a.html
  Narrowing coverage does not guarantee better performance for every subgroup.
- Gupta et al. (2024), ECML-PKDD accepted manuscript, spatial and semantic
  source-target dependencies: https://arxiv.org/abs/2404.07308
  We use similarity screening, not their latent-dependency autoencoder.
- Air pollutant prediction model based on transfer learning two-stage
  attention mechanism (2024), source-site similarity selection:
  https://www.nature.com/articles/s41598-024-57784-7
  Their PM10/MMD/attention experiment is not a PM2.5 China-Australia result;
  our nearest-neighbour screen is not MMD or their network.
- Izzo et al. (2023), ICML, interpretable high-performing subgroups:
  https://proceedings.mlr.press/v202/izzo23a.html
  Our beam search is not an implementation of DDGroup.
- Cawley & Talbot (2010), model-selection bias:
  https://jmlr.org/papers/v11/cawley10a.html
- Roberts et al. (2017), spatial/temporal validation:
  https://doi.org/10.1111/ecog.02881

## Deliverables and stop rule

Complete the 18 fixed configurations and 72 locked coverage settings, retain
failed settings, audit exact rule reproduction and calibration chronology,
and produce a Chinese result report and before/after table. Do not repeatedly
modify this protocol after seeing its audit outcomes. Positive candidates
need a new independent period or external cohort before a confirmatory claim.
