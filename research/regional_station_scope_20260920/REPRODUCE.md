# Regional membership and station-role sensitivity: reproducibility notes

The data/fit protocol was frozen for the completed 21 September 2026 batch.
The Chinese reading report was finalized on 29 September. This supplement does
not overwrite the preceding thesis or change its reported experiments.

## Offline verification from this bundle

Use a Python environment with the versions in `requirements_snapshot.txt`.
From the bundle root run:

```text
python research/regional_station_scope_20260920/verify_regional.py
python research/regional_station_scope_20260920/spotcheck_predictions.py
```

The first command independently recomputes aggregation, metrics, support and
chronology, and reproduces the bootstrap calculations using the frozen helper.
The second refits each target-side method/calibrator for the first available
2021 month in every configuration and compares all 27 stored predictions.
It loads the source fits without retraining them. It is NOT a claim that every
source model and all 576 rolling windows were retrained in this QA step.

Prepared cohort panels, station-month records with role flags, source fits,
predictions, source selection and calibration audit rows are included. The
necessary unchanged helper scripts and original common-feature panels/national
source fits are included at their expected relative paths.

`run_regional.py` skips completed folders by design. To regenerate model fits,
work in a separate copy of this bundle and move that copy's `results` folder
to a backup location before running it. Do not overwrite the audited originals.
The frozen script checks its own and the protocol's hashes when resuming.

Reconstructing the raw hourly archives is OUTSIDE this bundle. `build_cohorts.py`
documents the original prepared-data build and expects the original project
station metadata at its recorded user-home location. All input records required
to verify the delivered station selections and target means are already
included in `data/station_month_with_roles.csv.gz`. This is a reproducible
prepared-data/model supplement, not an archive of all underlying hourly data.

The optional report scripts use GeoPandas, Matplotlib and the project's Pandoc.
Rendered HTML embeds its three images, so reading it does not require those tools.

## Interpretation constraints

- The Chinese scope is official 2016 city MEMBERSHIP, not an official boundary
  polygon acquisition or a reconstruction of annual administrative changes.
- The role subset uses exact roster matching and explicitly documented Australian
  IDs. Unknown roles are not labelled erroneous measurements or background sites.
- Historical station-purpose continuity was not verified annually. The subset
  is a sensitivity analysis, not a homogeneous, fully verified background network.
- The two cohorts share development cities, not identical target labels or every
  held-out city. Additional common-row comparisons are descriptive post-fit checks.
- National source models retain their original all-network targets. Only regional
  and matched source policies use the newly rebuilt cohort targets.
- 48 coverage settings overlap; only 30 distinct rule/model/split combinations
  occur. No setting passes the joint temporal and new-city screen.
- Earlier project work already used 2015-2022 outcomes. New confirmatory claims
  require independent observations and pre-fixed selection rules.

## Evidence

Primary documents and URL/hash ledgers are in `evidence`. Failed downloads are
recorded, not silently filled. The Zhejiang county-parent table was verified
through the web retrieval tool but direct download returned HTTP 403; the
verified relational facts and URL are separately recorded. Role evidence years
refer to the designation snapshot, not necessarily the publication year.

Only load the joblib model files from this trusted project bundle. As with all
pickle-based model formats, untrusted joblib files can execute code when loaded.
