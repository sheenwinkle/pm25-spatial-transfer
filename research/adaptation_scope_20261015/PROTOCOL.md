# Adaptation scope and label coverage: protocol (frozen 15 October 2026, before any target fit)

## Questions

- **Q1 (how much and where to adapt).** With a few scattered target labels, which parts of a source
  PM2.5 model should be adjusted, and by how much? Does an adjustment that is restricted to the weather
  and fire conditions where local evidence exists, and whose size is limited, improve on a global
  intercept-and-slope recalibration, on an unrestricted residual correction and on target-only learners?
- **Q2 (which conditions to label).** For the same number of labelled FUA-months, does choosing the
  labelled months by the weather and fire conditions they cover improve accuracy, compared with the
  last consecutive months, random months and season-spread months?

## Information condition

Targets are monthly FUA-mean PM2.5. At prediction time no PM2.5 value of the predicted FUA is used: test
FUAs have no labels at all. Labels exist only for the labelled FUA-months of each design. Test rows are the
test FUAs in 2021-2022.

## Data and tasks

- Corrected panel `all_networks_model_panel_corrected.csv.gz` (SHA-256 9026749a...a2f03).
- T1: YRD target (30 FUAs observed in >= 80% of months), source = Chinese FUAs outside the 43-FUA YRD
  membership. T2: Australian target (9 FUAs, >= 50% of months), source = all Chinese FUAs. Both are
  development tasks: their labels have been used in earlier analyses.
- T3 (confirmation): region 8 of `regions.json` (24 eligible FUAs in north-west China), drawn by
  `regions.py` from the regions with >= 20 eligible FUAs. Source = all Chinese FUAs outside region 8.
  T3 is run once, after the development results are final, with the settings fixed here.
- Benchmark splits (seed 1000): k = 5, 11 (T1, T3), k = 3, 5 (T2); 20 splits.

## Source model and inputs

LightGBM with the thesis source settings (`run_monitor_models.tree(source=True)`) on the 16 benchmark
inputs, fitted once per task on all source rows. Predictions are clipped at zero.

Condition vector z (8 variables), standardised with the Chinese source means and SDs: monthly mean
temperature, diurnal range (max - min), relative humidity, wind speed, precipitation, log fire count within
100 km, and the two local-season terms.

## Estimators

Baselines: direct transfer (clipped source prediction); source-affine (ridge, alpha 10, on the standardised
source prediction; as in the thesis); source-residual (source prediction + target LightGBM on the residual
with all 16 inputs; as in the thesis); six local learners (mean, ridge, RBF kernel, RF, Extra Trees,
LightGBM; as in the thesis) and their per-split minimum ("best local six"). For the label-history designs the
archived NNW[LDF] and best-eight results are used as published (split-level comparisons only).

**Evidence-controlled adaptation (ECA, main).** Prediction = source-affine + g(z), where g is a zero-mean
Gaussian process on z fitted to the source-affine residuals of the labelled rows (ARD squared-exponential
kernel, signal SD = a_f * mean(y_labelled), noise SD = a_n * mean(y_labelled)). Away from labelled conditions
g returns to zero, so the calibrated source model is kept.
- *Where* (length-scales) is meta-learned on the source side (below) and fixed for the target.
- *How much* (a_f, a_n) is set by the target labels: maximum a posteriori under log-normal priors centred on
  the meta-learned values (SD 1 on the log scale).

Ablations: ECA-fixed (a_f, a_n fixed at the meta-learned values); ECA-no-meta (all hyperparameters by MAP on
the target labels, generic priors: log length-scale ~ N(0, 1.5), log a_f, log a_n ~ N(log 0.1, 1.5));
ECA-unbounded (a_f x 10, a_n / 10, fixed); ECA-isotropic (one meta-learned length-scale, MAP amplitude).

## Meta-learning on the source side

Episodes use the nine regions that are neither YRD nor region 8. For each region c, a source model is fitted on
the other eight regions; region c's eligible FUAs (>= 80% of months) are the pseudo-target. Tasks per region:
k = 5 labelled FUAs drawn at random; label designs (i) m consecutive months ending in a random month from
June 2019 to December 2020 and (ii) m random months from 2019-2020 per labelled FUA; m = 3, 6, 12; four
repetitions each (seed 20261015); test rows = the other eligible FUAs of region c in 2021-2022.
Objective: mean over tasks of log(MSE of ECA-fixed / MSE of source-affine) on test rows; L-BFGS-B over the
log hyperparameters from three starts; the best value is kept. The isotropic variant is meta-learned the
same way. No target-task data are used.

## Q1 designs

- Label history (as the thesis): labels = k labelled FUAs in the last m months of 2020, m = 3, 6, 12.
- Calibration window, m = 3: October-December 2020, April-June 2020, July-September 2020 and
  October-December 2019 (T1 and T2).

## Q2 designs

Pool = observed months of the labelled FUAs in 2019-2020; each labelled FUA receives m = 3 or 6 labels.
Strategies:
- S0 last consecutive months (the label-history design);
- S1 random months (5 seeds);
- S2 season-spread: months taken round-robin over the four seasons in random order (5 seeds);
- S3 condition coverage: greedy max-min distance in z within each FUA's pool;
- S4 target-aware coverage: greedy choice, across the labelled FUAs (quota m each), of the months that most
  reduce the summed posterior variance of g over the conditions of the prediction rows (test FUAs,
  2021-2022; covariates only), using the meta-learned kernel.

Methods evaluated under each strategy: source-affine, ECA and best local six.

## Metrics and decisions

RMSE per split on the test rows; mean paired gain 100 * mean(1 - RMSE_A / RMSE_B) over splits; wins out of 20;
95% intervals from 1,000 crossed FUA-by-calendar-quarter bootstrap draws (seed 20261015) on row-level
predictions, shared across methods. A method is said to improve on another in a cell when the mean paired gain
is positive and its interval lies above zero. Label efficiency: ECA at m months against source-affine and best
local six at 2m months. Results are reported for every cell, whatever their sign.

## Disclosure

Before this protocol, two exploratory checks were run: a covariate-only dissimilarity probe (no PM2.5) and an
error decomposition of YRD and Australia using all of their labels, which showed condition-dependent residuals
after recalibration. They motivated Q1; they are reported as exploratory.

## Amendment 1 (15 October 2026, after the Q1 development runs on T1 and T2, before Q2 and before T3)

Development results showed two things not anticipated above. (a) The unadapted source model ("Direct") is the
most accurate T1 estimator with 3-6 months of labels; it had not been reported in the thesis's benchmark
sections. (b) Among the ECA variants, the fixed meta-learned amplitude (ECA-fixed) is more accurate than the
evidence-updated amplitude (ECA, the pre-specified main variant) in most development cells. Changes, made before
any Q2 or T3 fit:
- Direct transfer is reported in every Q1 and Q2 table (it uses no target labels, so it is the same for all
  label designs and strategies).
- ECA-fixed is added to the Q2 methods and is reported alongside ECA as a co-primary candidate on T3. The T3 run
  decides between them; the choice is not made on development data.
- Direct transfer is also computed for the thesis's same-period (B) and later-period (S) protocols, as a
  descriptive addition to Tables 5.1, 5.2 and 5.5.

## Amendment 2 (15 October 2026, after the Q1 development runs on T1 and T2; before Q2 and before any T3 fit)

Development findings. (a) On T1 with 3-6 months of labels, direct transfer remained more accurate than every
adapted estimator, and ECA and ECA-fixed inherit the error of fitting a global level and slope to a short record
(they correct the output of source-affine). On the 216 within-China pseudo-transfers, source-affine has 8.9% higher
mean MSE than direct transfer. (b) On T2, ECA-fixed is close to source-affine and in the October-December 2019
window (Black Summer) clearly better than every other estimator.

Changes, made before any Q2 or T3 fit:
1. Naming. ECA-fixed (Amendment 1) is renamed ECA-R ("recalibrated"): source-affine followed by the meta-learned
   condition correction. The MAP-amplitude version (the original "ECA") is reported as ECA-R-MAP.
2. New estimator ECA-S ("source-anchored", code name ECA2): y_hat = max{0, f(x) + delta(x)}, delta(x) = a + b s(x)
   + g(z), with s(x) the source prediction centred on its labelled-row mean and divided by its SD over the source
   rows; independent zero-mean Gaussian priors on a and b and a Gaussian-process prior on g (ARD squared-exponential
   kernel on z); prediction = posterior mean of delta given the labelled residuals y - f(x). Every adjustment is
   shrunk toward the unadapted source model, so with no or weak evidence ECA-S returns direct transfer. Its
   signal-to-noise ratios and length-scales are meta-learned on the same 216 pseudo-transfers by minimising the mean
   log ratio of its test MSE to that of direct transfer (`meta_learn2.py`, `theta2.json`). Variants: ECA-S-affine
   (no condition term; own meta-learned prior), ECA-S-isotropic (one length-scale; own prior), ECA-S-MAP (ratios
   re-estimated on the target labels with the noise variance profiled out; log-normal priors, SD 1, centred on the
   meta-learned ratios), ECA-S-no-meta (all ratios and length-scales from the target labels under generic priors)
   and ECA-S-unbounded (meta-learned ratios x 10).
3. Development hypothesis to be confirmed on T3: ECA-S suits a source model that is current and in-domain (it
   adjusts little); ECA-R suits a source model that is out of date or out of domain (it re-estimates level and
   scale first). To test both regimes within the confirmation region, Q1 adds design `historyN` for T1 and T3: the
   history design with the source model fitted on source years up to 2020 only (no contemporaneous source labels,
   as the thesis's N protocol; its source-affine reproduces the archived N results).
4. Q2 keeps S0-S4 as pre-specified (S3 and S4 with the ECA-R condition kernel). Methods: direct transfer,
   source-affine, ECA-R, ECA-S and the six local learners.
5. T3 is run once with these settings: Q1 `history` and `historyN`, and Q2.

Pre-declared confirmatory comparisons on T3 (mean paired RMSE gain over the 20 splits with its 95% crossed bootstrap
interval; "improves" requires the interval to exclude zero):
- current source model (`history`, m = 3, 6, 12; k = 5, 11): ECA-S vs direct transfer; ECA-S vs source-affine;
  source-affine vs direct transfer;
- out-of-date source model (`historyN`): ECA-R vs source-affine; ECA-R vs direct transfer; ECA-S vs direct transfer;
- Q2 (m = 3, 6): S4 vs S0 and S4 vs S1 for ECA-S and ECA-R.
Every other T3 number is descriptive.
