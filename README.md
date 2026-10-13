# pm25-spatial-transfer

Code, protocols, audit records and summary results for the thesis *When Does Source-Region
Information Help? A Benchmark Study of Spatial Transfer for PM2.5 with Scarce Labels in Chinese
and Australian Cities* (Sheen Wang, Research Pathway Project, School of Computer Science,
The University of Sydney, 2026).

The study takes the spatial-transfer benchmark of Gupta et al. (2024) to monthly regulatory
PM2.5 observations: Chinese functional urban areas (FUAs) outside the Yangtze River Delta are the
source for 30 YRD target FUAs, and all Chinese FUAs for nine Australian FUAs.

## Where things are

| Folder | Contents | Thesis |
|---|---|---|
| `research/manuscript_revision_20261004/benchmark_replication_20261008/benchmark_experiment/` | Benchmark-aligned experiment: target-only learners, NNW, KLIEP and KMM with and without LDF, source recalibration, label-history and fixed-window analyses; `make_tables.py`, `make_figures.py`, `check_claims.py`; per-split results in `results/` | Chapters 4 and 5 |
| `research/manuscript_revision_20261004/audit/` | Verification of the window and LDF-seed analysis (`round8_20261009`), text checks, and the re-run with loss-weighted NNW[LDF] (`weighting_check_20261012`) | Section 5.6, Chapter 8 |
| `research/manuscript_revision_20261004/analysis/` | Corrected-panel refits, resolution and control analyses for the conditional audit | Chapter 6, Appendix B |
| `research/thesis_completion_20260912/` | Observed-label panel: station archives, polygon assignment, eligibility rules and audits | Chapter 3 |
| `research/common_monthly_transfer_20260912/` | Recovery of genuine monthly NASA POWER meteorology | Section 3.2 |
| `research/conditional_transfer_20260912/` and later dated folders | Conditional applicability audit and its extensions, each with its own README or PROTOCOL | Chapter 6, Appendix B |
| `upstream_info5993/` | Data-acquisition and panel-building scripts and download records (with checksums) from the preceding project, including SatPM2.5 V6.GL.03, GHS-FUA, EDGAR and NASA POWER | Chapter 3 |
| `research/manuscript_revision_20261004/submission/build_thesis.py` | Wraps the manuscript in the School thesis template | – |

## Verification entry point

See [reproducibility/README.md](reproducibility/README.md) for the saved-result
check and the optional row-level weighting verification. The latter requires
prediction archives that are not included in the repository.

## Data access

Raw and derived data are not included. The sources are the CNEMC archive on Zenodo
(doi:10.5281/zenodo.10836299); the New South Wales, Queensland and Victorian air-quality
services; NASA POWER; ERA5 through WeatherBench2; MODIS active-fire detections; EDGAR v8.1;
GHS-FUA R2019A; Natural Earth; and SatPM2.5 V6.GL.03 from the Atmospheric Composition Analysis
Group, Washington University in St. Louis. The benchmark scripts read the modelling panel from
the location set in `paths.py`.

## Software

Final runs used Python 3.12 on Windows 11 with numpy 2.5.3, pandas 3.0.6, scipy 1.18.1,
scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.14.1, adapt 0.4.5 and tensorflow 2.21.0
(`research/manuscript_revision_20261004/audit/weighting_check_20261012/environment.json`).
Earlier stages record their own versions.

## Notes

- Paths in the earlier stages refer to the original local folder layout; the Windows user name
  is replaced by `USER`.
- Manuscript text, reports and report-generation scripts are not included.
- `MANIFEST.txt` lists every file with its size.
