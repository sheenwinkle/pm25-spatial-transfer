# Changes after the T3 run (15 October 2026)

`freeze_T3.txt` records the checksums of the protocol, region partition, library, runners and meta-learned priors
taken before T3 was run. After T3, only reporting code changed:

- `summarise.py`: (1) a bootstrap draw that gives one split's test rows zero total weight, possible with nine
  Australian FUAs, now leaves that split out of the draw's mean instead of returning no interval; T1 and T3 have no
  such draws, so their intervals are unchanged; (2) descriptive ECA-R comparisons were added to the output.
- `make_adapt_report.py` and `check_adapt_claims.py` were written after T3 to build the thesis tables and figure and
  to check every number quoted in the thesis.
- `aoa/distance_check.py` (exploratory) was added after T3. It sets the covariate-only distance of each development
  region from the other eight against the gain from recalibration in that region's 24 pseudo-transfers, using
  `meta/meta2_tasks.csv`; it fits nothing.
- `explore/` (exploratory) was added after T3: a condition-group ablation that re-learns the ECA-S prior on the
  development pseudo-transfers, a hybrid label strategy, a label-window scan and a mechanism figure. Each check was
  specified before its results were seen (`explore/README.md`); the window scan was added after the bias plot, as
  recorded there.

No estimation code, setting or prior changed after T3.
- `run_q2_matched.py` (post hoc, 17 October 2026) was added after a review of the fifth draft. In Australia the
  latest-months design (S0) received fewer labels than its budget in 50 of 80 split-by-budget cells, because some
  labelled FUAs miss part of late 2020; every other strategy, and every strategy in the YRD and the locked region,
  received the full budget. The script cuts S1-S4, FUA by FUA, to the labels S0 received and refits; S0 reproduces
  `results/aus_q2_metrics.csv` exactly and remains the best for source-affine and ECA-R in all four cells.
- The ECA-R weak-shrinkage variant (`ECA-unbounded` in `adapt_lib.eca_predict`) multiplies the signal amplitude by
  10 and divides the noise SD by 10, so its signal-to-noise ratio grows 100-fold; `ECA2-unbounded` (ECA-S)
  multiplies the ratios by 10. The thesis reports both factors.
- 19 October 2026, after a review of the sixth draft: `make_adapt_report.py` and `explore/make_explore_report.py`
  now write the thesis's corrected captions and row labels (the weak-shrinkage factors, and "is assigned" with the
  actual Australian label counts, read from `results/aus_q2_selected.csv` and `explore/aus_hybrid_predictions.csv.gz`).
  No model was refitted; the regenerated `tab_adapt_rmse.tex` and `tab_adapt_confirm.tex` are byte-identical to the
  previous ones. `check_adapt_claims.py` now also requires every generated table to appear verbatim in the thesis.
- `explore/hybrid_matched.py` (post hoc, 19 October 2026) applies the matched-budget check to the hybrid strategy of
  thesis Table C.4; see `explore/README.md`.

