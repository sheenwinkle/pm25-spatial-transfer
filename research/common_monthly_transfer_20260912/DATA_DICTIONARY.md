# Data Dictionary

## Tables and Keys

| File in data/processed | Unique key | Coverage | Meaning |
| --- | --- | --- | --- |
| fua_monthly_power_1998_2025.csv.gz | eFUA_ID, year, month | 523824 rows; 1559 FUAs; 1998-2025 | Six recovered monthly POWER meteorological fields |
| power_grid_month_1998_2025.csv.gz | power_lon, power_lat, year, month | 959 used native cells | Monthly values before duplication across same-cell FUAs |
| power_grid_annual_1998_2025.csv.gz | power_lon, power_lat, year | 959 used native cells | API annual summary used only in annual ablation arms |
| fua_month_model_panel_2000_2024.csv.gz | country_iso, eFUA_ID, year, month | 467700 rows; 2000-2024 | Inherited SatPM target and all common/annual/aviation modelling features |
| regulatory_monthly_audit_input.csv.gz | site_key, year, month | 4886 inherited station-months | Audit of eligible station summaries; not all rows enter scoring |

## Identifiers and Target

- `country_iso`: CHN or AUS.
- `eFUA_ID`: inherited functional-urban-area identifier, not a new administrative boundary.
- `fua_name`: inherited name.
- `year`, `month`: calendar year and month (1-12). Month 13 in raw responses is an annual statistic, never a model month.
- `t`: year * 12 + month; `quarter_id`: year * 4 + floor((month-1)/3).
- `power_lon`, `power_lat`: matched meteorological-cell coordinates, degrees.
- `weather_grid_id`: factorized cell key in this version; do not join arbitrary releases using this integer without also checking coordinates.
- `power_great_circle_distance_km`: spherical great-circle distance, radius 6371.0088 km, from inherited FUA centroid to matched cell.
- `pm25_mean_ug_m3`: unchanged inherited unweighted SatPM FUA-month mean in micrograms per cubic metre. Not a regulatory-station observation or a causal outcome estimate.

## Monthly Meteorology

| Field | API parameter | Units and interpretation |
| --- | --- | --- |
| monthly_power_temperature_c | T2M | Degrees Celsius, 2-m temperature monthly statistic |
| monthly_power_temperature_max_c | T2M_MAX | Degrees Celsius, API maximum-temperature monthly statistic |
| monthly_power_temperature_min_c | T2M_MIN | Degrees Celsius, API minimum-temperature monthly statistic |
| monthly_power_humidity_pct | RH2M | Percent, 2-m relative humidity |
| monthly_power_wind_mps | WS10M | Metres per second, 10-m wind speed |
| monthly_power_precip_mm_day | PRECTOTCORR | Millimetres per day, monthly mean daily corrected precipitation; NOT monthly accumulation |

Native meteorology is MERRA-2 assimilation on 0.5-degree latitude by 0.625-degree longitude cells. Retain the NASA attribution and raw version/time metadata. Complete coverage refers to this fixed matched-city panel, not all locations in either country. Fill values are treated as missing; final recovered six-field panel passed completeness and broad physical-range checks.

`legacy_annual_` prefixes identify annual API summaries repeated by month with 2022 carried into 2023-2024. `updated_annual_` prefixes use the correct year's annual summary. They are comparison arms, not the recommended monthly predictors. These full-year statistics make both annual arms retrospective, not prospective inputs.

## Other Modelling Features

- `local_season_sin`, `local_season_cos`: inherited month cycle with a six-month Southern Hemisphere alignment; alignment is a modelling convention, not proof of ecological equivalence.
- `log_population_density`: inherited log-transformed density; static context is not a yearly census series.
- `distance_to_coast_km`: inherited coastal distance.
- `urban_centre_area_share`: inherited fraction of urban-centre area.
- `log_fire_count_100km`: inherited log-transformed contemporaneous fire count within 100 km.
- `surface_*`: original aviation/ground-weather aggregation retained separately. Units appear in field names: C, percent, m/s, km, mm and hPa. Availability and dry-visibility valid-day fields document incomplete observations. Missing values are not relabelled as measurements. These are ground summaries, not aircraft profiles.

## Saved Prediction Gates

- `development_city`: belongs to the ten-FUA calibration network for this seed.
- `accepted`: satisfies that configuration's frozen 0-2 predictor inequalities.
- `new_weather_grid`: cell absent from the development network.
- `support_distance`: standardized common-weather/season nearest-source distance; not the exact importance-weighted AOA dissimilarity index from Meyer and Pebesma.
- `calibration_similarity`: maximum fraction of shared source-tree terminal leaves with the preceding target fitting window.
- `selected_prediction`: candidate selected using 2019-2020, not selected again on later outcomes.
- `target_mean`, `target_ridge`, `target_lgbm`, `target_rbf`: equal-budget target-only comparators.
- `shuffled_source_kernel`: source-label permutation control, not a model with zero source information.

`coverage` always refers to rows in that audit stage; `fua_coverage` to FUAs. Neither is land-area coverage. Regulatory `coverage` uses monitor-available rows, with `selected_over_all_target_rows` also retained to disclose the larger denominator. Missing confidence bounds represent unavailable estimates, not evidence of no uncertainty.

## Regulatory Quality

`recomputable_75pct`: valid_days / calendar_days >= 0.75 in inherited QLD/VIC records. `official_monthly_no_day_counts`: NSW official monthly values without locally available day counts. `below_completeness`: not eligible for the primary completeness-auditable supplemental cohort. Never combine these as a single harmonized strict-quality class without further work.

Full original context construction and station matching remain dependencies of the original project. This release adds explicit provenance; it does not claim that every upstream feature or instrument record has been newly validated.
