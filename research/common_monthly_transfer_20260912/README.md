# Common Monthly Transfer Review, 2026-09-12

Start with `REPORT_ZH.html` (self-contained Chinese report with embedded figures) or `REPORT_ZH.md`. `MANUSCRIPT_UPDATE_EN.md` provides proposed replacement methods/results text, not a rewritten full dissertation. All outputs are retrospective and exploratory.

## Completed

- Recovered genuine monthly NASA POWER fields for 1998-2022 from inherited raw responses; retrieved matching 2022-2025 responses and checked the overlap.
- Exported six-field monthly covariates for 1559 FUAs, using 959 meteorological cells, and an analysis-ready 2000-2024 modelling panel with unchanged SatPM labels.
- Ran 96 matched transfer configurations and retained frozen rules, selected row predictions, equal-budget local predictions, source models and metadata.
- Recomputed 5760 saved-prediction metric records, 480 gate-stage records and 32 bootstrap intervals independently of the training metric calculation. Four method tests passed.
- Audited all 48 China-to-Australia configurations against existing regulatory monthly records, with separate quality cohorts and no refitting on monitor labels.
- No configuration met the stated temporal-plus-held-city replication criterion. A temperature interval remains hypothesis-generating, not validated.

## Contents

- `data/processed`: corrected monthly weather and analysis-ready model panel; see DATA_DICTIONARY.md.
- `data/raw/power_tiles_2022_2025`: all 132 new raw API responses and request metadata.
- `data/metadata`: original/new raw file hashes, request plan, overlap checks, coverage and manifest.
- `results`: all eight feature/direction folders, 96 frozen rule files, 96 saved prediction files, model summaries and eight pairs of source/negative-control models.
- `audit`: complete combined result tables, component failure checks, regulatory audit, verification summaries, label budgets and provenance.
- `figures`: three scientific figures, PNG and vector PDF exports.
- `dependency_snapshot`: selected prior helper, prior protocol and city manifests. This is a provenance snapshot; it is not a replacement for the previous experiment directory.
- `release_manifest.json`: SHA-256 entries for packaged files, excluding the manifest itself and the zip.

The release excludes only rebuildable local joblib cache files, Python bytecode, temporary report-screen screenshots and the duplicated long-format weather intermediate. It includes the compact processed grids, FUA weather, analysis-ready modelling panel, new raw downloads and saved predictions. The inherited 1998-2022 raw responses, original regulatory raw records, original project context files and Python environment are not copied; original locations and hashes are retained. This package is an auditable result/data release, **not a completely standalone portable recreation of every upstream source**.

## Local Reproduction

The original work is retained at:

`C:\Users\USER\Documents\ChatGPT\Dissertation Program\research\common_monthly_transfer_20260912`

The science Python environment is:

`C:\Users\USER\.codex\.chatgpt-projects\g-p-69aaacd3007481919301f520a145f3f4\.venv\Scripts\python.exe`

Run the scripts from that original research directory (or configure their documented paths in a separate reproduction copy). `audit_inputs.py` points to the inherited project and the sibling `conditional_transfer_20260912` experiment. `run_models.py` depends on that sibling's helper, protocol, city manifests and cached original FUA panel. Do not mistake execution from the extracted release directory for a fully configured environment.

Pipeline order: audit_inputs.py, collect_monthly_power.py, verify_weather.py, test_methods.py, run_models.py, audit_results.py, make_figures.py, render_report.mjs, package_release.py. The model and collection script hashes are locked in their manifests. Download and model scripts should not be modified while still claiming the same completed run.

LightGBM and scikit-learn provide the model implementations; pandas/numpy/scipy/joblib are used for data handling. The bundled Node runtime and marked package render HTML; Playwright performs local HTML screenshot checks. Exact installed science-package versions are recorded in the release manifest.

## Important Interpretation

The primary target is the inherited SatPM FUA-month mean, not station-measured PM2.5. NASA POWER is reanalysis, not a new airport station network. The acquisition expands covariates through 2025, not labelled PM2.5 into 2025. Current-month inputs make this retrospective estimation, not a one-month-ahead forecast. Source-informed models use ongoing target labels. Rules, repeated partitions, and confidence intervals are exploratory; the whole historical project has not been converted into an untouched external test.

The original manuscript and previous delivery package are unchanged. Review the revised evidence before merging proposed prose into the thesis. No Overleaf compiler setting was changed in this continuation.
