# Exploratory analyses added after the T3 run (15 October 2026)

None of these analyses is confirmatory: every target, including the locked region T3, had already been used. Each
analysis was specified, in its script and in this file, before its results were seen, and its results are reported
whatever they show. No estimation code, setting or prior of the main study is changed.

1. `ablate_groups.py` (where to adapt). The ECA-S prior is re-learnt with one group of conditions removed
   (meteorology: temperature, diurnal range, humidity, wind, precipitation; fire: log fire count; season: the two
   local-season terms; or all conditions) and compared with the full prior on the 216 within-China pseudo-transfers,
   with the region-held-out cross-validation of `meta_learn2.py`. Effect: mean paired RMSE reduction of the full
   prior over the reduced prior on the held-out tasks; 95% interval from 1,000 bootstrap draws of the nine regions.

2. `hybrid_labels.py` (which months to label). New strategy H: each labelled FUA receives its floor(m/2) most recent
   months of 2020 (m = 3: one; m = 6: three); the remaining months are chosen by the target-aware criterion of S4
   (greedy reduction of the posterior variance of the ECA-R condition correction over the prediction rows), after
   conditioning on the recent months. Same tasks, splits, seeds and estimators (source-affine, ECA-R, ECA-S) as
   `run_q2.py`; the six local learners are not refitted. Comparisons: H against S0 (last months), S1 (random) and
   S4 (target-aware), with the crossed FUA-by-quarter bootstrap of `summarise.py`.

3. `mechanism_figures.py`. (a) Monthly mean bias of the source model (prediction minus observation) at the target
   FUAs, 2019-2022, for T1, T2 and T3, and the distance between the bias of each candidate label month and the mean
   test-period bias. (b) The ECA-R condition correction against log fire count in the Australian October-December
   2019 window.

4. Added after item 3(a) had been run (its bias plot is dominated by the seasonal cycle and does not by itself show
   why the most recent months win): `window_scan.py`. For three consecutive labelled months ending in each month from
   March 2019 to December 2020, the mean 2021-2022 RMSE over the 20 benchmark splits (k = 5) of source-affine, ECA-R
   and ECA-S, with direct transfer as the reference. The last window (October-December 2020) is S0 at m = 3.

Notes on the runs. `ablate_groups.py` rebuilds the 216 tasks identically to `meta/meta2_tasks.csv`; its full-prior
cross-validation reproduces the held-out objectives in `meta/theta2.json` to within 0.0002 (first fold -0.1139
against -0.1137; the other two folds agree to four decimals), so the exact-match flag in `ablation_summary.json`,
which uses a tolerance of 1e-6, reads false. `hybrid_labels.py` recomputes S0 and reproduces the S0 predictions of
`results/<task>_q2_predictions.csv.gz` to the six significant digits stored there.

## Post-hoc addition (19 October 2026)

`hybrid_matched.py` and `hybrid_matched_summary.py` were added after a review of the sixth draft. In Australia some
labelled FUAs miss part of late 2020, so the latest-months design (S0) falls short of its budget while H fills its
coverage part; H therefore held one label more than S0 in 50 of the 80 Australian splits (columns `*|n_rows` of
`aus_hybrid_predictions.csv.gz`). The matched run repeats the hybrid selection exactly and cuts each FUA's labels to
the number S0 received (recent months first, then the coverage months in selection order); the splits in which
nothing is cut reproduce `aus_hybrid_predictions.csv.gz` exactly. In `hybrid_matched_gains.csv`, the two ECA-S gains
over S0 with three months remain (15.4% [12.0, 19.8] and 22.3% [18.2, 28.0]), the four source-affine losses remain,
and the ECA-R loss with three months at five FUAs is no longer resolved (-8.6% [-22.2, 1.2]), so H loses to S0 in 6
rather than 7 of the 36 cells. Item 2 above describes the strategy as it was specified; the caption of
`tab_explore_hybrid.tex` states the actual Australian label counts.
