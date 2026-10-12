# Pre-fit supplemental experiment specification

The main 54 monitor configurations are running. Supplementary extensions below
were specified after partial main outputs were inspected. They are therefore
explicitly exploratory extensions, not a pristine confirmatory experiment.

## Common mixing context

Retrieve ERA5 through WeatherBench2's public conservatively regridded 1.5-degree
six-hourly product for both countries, 2015--2022. This is coarser contextual
information, not an improvement in local spatial resolution. Retain all 1,559
original FUAs (including those without monitors). Download PBLH and 10m u/v wind;
compute monthly mean height, p10 height, fraction below 500m, and mean of
instantaneous PBLH times surface wind magnitude. Do not describe this surface-wind
proxy as a full-column transport integral.

On the all-network monitor cohort, add log1p(mean PBLH), low-PBLH fraction and
log1p(mean ventilation proxy) to the twelve common predictors. Repeat all three
city seeds and 3/6/12-month windows in both directions, with the same comparators
and rule family. PBLH variables do not expand the rule-search family in this
extension, allowing a matched prediction-feature contrast (18 configurations).

## Shifted outer time window

Use common monthly predictors and all-network monitors. Fit source on 2015--2016,
calibrate sequentially, discover rules in 2017--2018, and audit in 2019--2020.
Development-city eligibility is >=6 observed months in 2016; the remaining design
is unchanged. Three seeds, three label windows, both directions (18 configurations).
Compare selected variables and endpoints with the 2019--2020 discovery window;
report instability without redefining any window after scoring.
