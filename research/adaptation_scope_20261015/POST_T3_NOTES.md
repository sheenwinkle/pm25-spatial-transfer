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
