# Source-period restriction: second exploratory pilot, 20 September 2026

This pilot was specified AFTER all 18 fixed-source configurations were known
to fail the joint temporal/new-city screen. It does not replace PROTOCOL.md.
It asks whether restricting the age of source-country labels helps; it is
not a confirmation on untouched data and is not a new data acquisition.

Scope: only the already-defined absolute-latitude 22-35 / coast <=150 km
population, both directions, city split seeds 5993/5994/5995. This is a focused
follow-up to a physically interpretable scope with comparatively modest
Chinese audit deficits. Scope selection is thus partly result-informed.

For each prediction month t, regional source rows must lie in [t-24,t-1].
The matched policy selects the nearest half of these source rows to the
SAME pre-2019 target-development covariates used in the first extension.
No current source-month outcomes are allowed. All six target-only predictions,
national fixed-source predictions and controls, target calibration rows,
covariates and geography are reused unchanged from the first extension.
Target calibration still uses only [t-12,t-1] development-city observations.

Refit regional and matched LightGBM models and their three shuffled-label
controls every month. Use the same affine, residual and partition-kernel
calibration methods and hyperparameters. Keep the original static regional
support-distance feature for gate evaluation, so comparisons on previously
locked rules use identical accepted rows. It is a distance to the reference
2015-2018 source climate, not distance to the current rolling source window.

Two reported analyses:
1. Frozen first-extension rules and model identities, with just the regional
   and matched predictions updated. This directly measures the paired effect
   of the source period on the SAME accepted samples.
2. The same 2019-search / 2020-selection procedure, four coverage floors and
   unchanged support and screening criteria, on the updated predictions.
   Lock before calculating pilot 2021-2022 audit summaries.

Six prediction configurations, 24 newly locked settings, and 24 inherited-rule
comparisons. Repeated settings and overlapping data are not independent tests.
No subsequent changes to this pilot based on its audit outcomes. Keep all
results, including declines. The primary success definition remains the
strict source-information benefit criterion while awaiting author preference.

Rationale (not exact algorithm replication):
Han, E., Huang, C., & Wang, K. (2024). Model Assessment and Selection under
Temporal Distribution Shift. ICML/PMLR 235, 17374-17392.
https://proceedings.mlr.press/v235/han24b.html
Their adaptive window procedure motivates investigating temporal relevance;
we use a transparent fixed 24-month training window, not their estimator.

Technical note: Improving the European air quality forecast of the Copernicus
Atmosphere Monitoring Service using machine learning techniques (2023).
https://acp.copernicus.org/articles/23/5317/2023/
This is European forecast post-processing, not evidence of China-Australia
PM2.5 transfer. Its discussion of adaptation to changing emissions motivates
testing training-period freshness rather than only spatial restriction.
