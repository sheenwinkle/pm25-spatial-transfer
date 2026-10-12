# Benchmark-aligned comparison protocol (recorded 2026-10-07, before any fit)

Benchmark (对标): Gupta, Park, Bi, Gupta, Züfle, Wildani & Liu (2024), *Spatial Transfer
Learning for Estimating PM2.5 in Data-Poor Regions*, ECML PKDD 2024, LNCS 14949, 385–400,
doi:10.1007/978-3-031-70378-2_24. Released code: github.com/YongbeeIngkle/spatial-transfer-learning
(cloned 2026-10-07; commit recorded in RUN_LOG).

Purpose: evaluate the benchmark's methods and this report's source-recalibration estimators
on the same data, splits and metrics, following the benchmark's own problem formulation and
evaluation protocol, and then under a stricter later-period protocol that the benchmark does
not use. Nothing here searches for conditions or tunes hyperparameters.

## Problem formulation (benchmark's)
Source domain D_S (fully labelled, m rows) and target domain D_T with few labelled locations
(n << m). Estimate PM2.5 at unlabelled target locations. Unit here: FUA-month (benchmark:
station-day).

## Tasks
- **T1 within-country (analogue of benchmark US-East -> California–Nevada).** Source: Chinese
  FUAs outside the Yangtze River Delta (YRD, 43-FUA official membership). Target: YRD FUAs with
  >= 80% of the 96 months observed (benchmark: >300/365 days).
- **T2 cross-country (analogue of benchmark US -> Lima application, but with ground truth).**
  Source: all Chinese FUAs. Target: Australian FUAs with >= 50% of months observed.
- Data: corrected all-network observed-label panel (NSW 31-December correction), 2015–2022.

## Splits (benchmark's)
For each k, `np.random.seed(1000)` then 20 draws of k training FUAs from the eligible target
set without replacement; all remaining eligible FUAs are test FUAs.
T1: k in {5, 7, 9, 11}. T2: k in {3, 5}.

## Protocols
- **P-B (benchmark protocol):** training-target rows = all months of the k training FUAs;
  test rows = all months of test FUAs (same period); source rows = all source rows.
- **P-S (stricter, this report):** training-target rows = training FUAs, years <= 2020;
  test rows = test FUAs, years 2021–2022; source rows = all source rows. LDF neighbour pools
  for test months therefore contain source rows only.

## Inputs
X = 12 common predictors of the report + month + time index t + FUA centroid lon/lat
(benchmark inputs include day, month and coordinates). Same X for every method.
LDF neighbour-search tags: lon, lat, six monthly weather fields, log population density,
coastal distance, urban-centre share, log fire count.

## Methods (fixed list)
Benchmark reproduction (released code; hyperparameters copied):
- RF, GBR: target-only (paper: "trained on only the target region data").
- NNW, KLIEP, KMM: adapt instance weights, applied **as in the released code** (source features
  multiplied by weights), then GBR on [weighted source; target].
- LDF: 12-neighbour same-month cloud, inverse-distance weighting, two-stage autoencoder
  (released architecture, NAdam 1e-3, 30 epochs, batch 128), latent dim 1; replaces the
  PM2.5 placeholder column. Variants: GBR[LDF], NNW[LDF], KLIEP[LDF], KMM[LDF].
- Sensitivity: NNW and NNW[LDF] with weights passed as `sample_weight` (standard usage).
This report's estimators (thesis implementations, imported unchanged):
- source-affine, source-residual, source-kernel (source LightGBM 128 rounds).
- shuffled-source affine control (source labels permuted, seed 5993).
- six target-only learners (mean, ridge, RBF, RF, Extra Trees, LightGBM).
- source-affine[LDF]: Ridge(alpha=10) on standardised [source prediction, LDF].

## Metrics and primary comparisons (fixed before fitting)
Per split: R2 and RMSE on pooled test rows (benchmark), plus MAE. Report the mean over 20
splits (benchmark) and paired per-split differences.
Primary comparison 1 (T1, P-B): source-affine vs the benchmark's best reported model NNW[LDF].
Primary comparison 2 (T1, P-B): source-affine vs benchmark target-only GBR.
Secondary: same comparisons in T2 and under P-S; shuffled-source control; all other models.
Relative RMSE change = 1 - RMSE_method / RMSE_reference, paired within split.
Splits overlap, so split-level uncertainty is descriptive; win counts out of 20 are reported.
All results are reported whatever their direction.

## Addendum A (recorded 2026-10-07, before any P-M fit; after one P-B split had been inspected)
Motivation: the report's earlier within-China analysis (3/6/12 months of target labels) predates
this benchmark run. This addendum re-tests that label-budget question inside the benchmark design.
It was added after the single P-B timing test (T1, k=5, split 0) had been inspected; that split showed
strong target-only learners under P-B. The addendum is therefore labelled post-hoc in the report.

- **P-M(m), temporal label budget:** training-target rows = the k training FUAs in the last m months of
  2020 (m in {3, 6, 12}); test rows = test FUAs in 2021-2022; source rows = all source rows.
  P-S is the same design with all months <= 2020 (m = 72), so P-M nests P-S.
- Tasks/k: T1 k in {5, 11}; T2 k in {3, 5}. Splits: the same 20 benchmark splits per k.
- Methods (reduced for cost, fixed before fitting): RF, GBR, GBR[LDF] (target-only); NNW and NNW[LDF]
  (released and sample-weight); six local learners; source-affine, source-residual, source-kernel,
  shuffled-source affine, source-affine[LDF]. KLIEP and KMM are not run under P-M.
- Primary comparisons under P-M: source-affine vs NNW[LDF], and source-affine vs target-only GBR,
  at each (task, k, m). Same metrics and reporting rules as above; all directions reported.

## Execution notes
- KMM is computed in a second pass that reuses the cached LDF encodings of the first pass
  (`--pass kmm`); results are identical in definition to a single pass.
- torch's lazily imported optimiser modules crash if first imported after TensorFlow (pulled in
  by adapt); the harness initialises them before importing adapt. No numerical effect.
- Scope change for cost (recorded 2026-10-08 00:05, before any KMM result other than the single
  timing-test split): KMM takes about 3 CPU-minutes per split, so the KMM pass is run only for the
  benchmark protocol on T1 (P-B, k = 5/7/9/11), the setting in which the benchmark reports KMM.
  Under P-S, P-M and T2 the ITL comparison uses NNW and KLIEP.
- Scope change for cost (recorded 2026-10-08 00:25 AEDT, before any later-period or T2 fit): KLIEP
  takes about one CPU-minute per split and is run only under the benchmark protocol (P-B, both
  tasks). Under P-S the instance-transfer comparison uses NNW (released and sample-weight), the
  benchmark's best reported method family.

## Addendum B (recorded 2026-10-08 ~02:05 AEDT, before any fit of this design)
Prompted by the independent fidelity review, which noted that under P-S/P-M every source-based
method (source model, LDF neighbour pools, ITL) has source labels from the test period 2021-2022.
That is the benchmark's design (fully labelled source), but it leaves open whether the
short-history advantage depends on contemporaneous source labels.
- **P-N(h), no contemporaneous source labels:** as P-M(h) (h in {3, 6, 12}; T1 only, k in {5, 11};
  the same 20 splits), but source rows are restricted to years <= 2020 for every method.
- LDF cannot be composed for 2021-2022 test months without same-month labelled rows, so LDF-based
  models are not run. Methods: RF, GBR (target-only); NNW (released and sample-weight) on the
  benchmark's normalised centre features (equivalent to the centre position of the normalised
  cloud: mean/sd over source + training-target rows, pm25 placeholder 0); six local learners;
  source-affine, source-residual, source-kernel and shuffled-source affine with the source model
  refitted on source rows <= 2020.
- Comparisons reported: source-affine vs NNW, vs target-only GBR, vs the per-split best local
  learner (post hoc). This is a sensitivity analysis, labelled post hoc in the report.
