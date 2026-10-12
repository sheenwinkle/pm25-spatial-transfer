# Reproducing the verification

Run commands from the repository root using Python with NumPy and pandas installed.
The recorded model-fitting environment is in
`research/manuscript_revision_20261004/audit/weighting_check_20261012/environment.json`.
The checks below do not train any models. They do not require the manuscript.

## 1. Check the distributed per-split results

```text
python research/manuscript_revision_20261004/benchmark_replication_20261008/benchmark_experiment/check_claims.py
```

Expected: `139/139 claims pass`. The input CSVs are already in that experiment's
`results/` directory; there is no ZIP to extract. These assertions concern the
archived benchmark summaries, not every statement or figure in the thesis.

## 2. Verify the added loss-weighted comparison

Row-level prediction archives and source data are **not** included. Supply the
original `window_results` directory and the re-run `window_results_sw` directory
from the local analysis archive, or regenerate them using the experiment's
`bench_window_sensitivity.py` and `bench_window_sensitivity_sw.py` with the
corrected panel and membership inputs configured as described in `paths.py`.
Each archive must contain its 200 prediction files and corresponding metrics JSON
files, preserving the `<window>/k<k>_split<ss>/seed<offset>_*` directory layout.

```text
python reproducibility/verify_weighting.py --old "PATH_TO/window_results" --new "PATH_TO/window_results_sw"
```

The script checks exact equality of all original prediction columns and 2,200
original model-score records, recomputes all 2,400 RMSEs, invokes the published
1,000-draw bootstrap summary, and compares all 80 comparisons and 120 model-mean
and rank rows with the distributed reference CSVs. Expected: five of the six
original-seed 2020 hybrid-versus-stronger-NNW intervals have a lower bound above
zero. A missing input or mismatch exits with an error, not a partial pass.

Outputs go to `reproducibility/output/` (ignored by Git), or to `--out DIRECTORY`.
They include `verification.json`, `summary.log` and the regenerated summary CSVs.
This is a saved-result check; the reused bootstrap implementation is not an
independent second statistical implementation, and no new fitting is performed.

For each window, budget and LDF seed, the stronger NNW weighting is the one with
lower mean test RMSE over the 20 splits. Its identity is fixed during bootstrap
resampling. The eight-model local reference instead takes the per-split minimum
and is reselected within each draw. The checks preserve these definitions.

## Refitting versus summary verification

Full refitting additionally requires the corrected modelling panel and FUA
membership data, neither of which is distributed here. The experiment README
lists the expected hashes and configurable input paths. Source acquisition and
processing scripts are retained in the repository, but running the summary check
alone does not demonstrate an end-to-end raw-data rebuild.
