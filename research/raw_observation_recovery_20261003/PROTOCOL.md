# Raw observation recovery, 2026-10-03 Australia/Sydney

Search independent acquisition archives and original official records to repair
the identified Chinese August/December 2018 deficits. Preserve existing data,
models and frozen evaluation. Do not infer new ground truth from predictions.

Source hierarchy: official hourly records; research archives of observed hourly
records; independently archived copies of the same station network; separate
observing networks; daily/city aggregates; imputed or reanalysis products.
Archive independence is not measurement independence. Only compatible observed
station-hour records can directly fill a CNEMC station-hour gap. Other observing
sites remain separate stations; daily/city aggregates do not become station hours.

Validate declared provenance, license/access terms, timestamps and timezone,
PM2.5 mass concentration units, site identifiers/locations, duplicates, missing
flags, and overlap agreement. Preserve and quantify conflicts rather than
selecting whichever source improves model performance. No automatic replacement
of existing valid values; no AQI-to-concentration inversion presented as raw data.
Explicitly flag homogenized, corrected, filled, model-estimated or unknown values.

When a source supplies compatible raw measurements, stage missing-only rows and
audit station-day/month eligibility with unchanged >=20 h and ceil(75% days)
thresholds. Do not merge until station/time/unit comparability is established.
Any source-data change must be followed by a separate model-impact experiment,
not reuse of prior model predictions as though retrained.

Do not bypass authentication, paywalls or access controls, create accounts, buy
data, or contact data owners without user authorization. Save available public
source descriptions, request receipts and access limitations. A bounded search
does not prove missing observations do not exist. Document specific unresolved
access requirements and deliver a traceable source ledger and recovery decision.
