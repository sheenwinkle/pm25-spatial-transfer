# Historical station and external-data readiness audit

Locked before this branch processes outcome extensions on 2026-09-29.

## Scope

The research remains China-Australia PM2.5 conditional transferability. Preserve
all previous protocols, predictions, source data, and manuscript results.
This branch does not select models, optimise conditions, or calculate predictive
performance. Data completeness may be inspected without declaring a final test.

## Exposure rule

Downloading a file is distinct from inspecting its labels or model performance.
However, a later change of aggregation (weekly to monthly), model, or station
filter does not reset prior outcome exposure. Australian NSW/QLD/VIC 2025 has
already been evaluated in the 2026-08-29 branch; 2023-2024 are also used in prior
work. They are retrospective extensions, not untouched validation for the next
adaptive investigation. Chinese satellite labels and regulatory labels must be
tracked separately. Absence of a filename match is not proof of non-exposure.

## Historical evidence rule

Keep official report year, publication year, event date, source page, station
identifier, instrument/method, and uncertainty separate. An annual table verifies
the listed year, not all adjacent years. No role is propagated backwards or
forwards by default. Metadata annotations from the secondary CNEMC archive are
event leads, not official verification. Unknown roles remain in the all-station
cohort and are not equated with low-quality observations. Exact city and station
matches only; stripped name annotations may generate candidate matches, never
automatically certify station identity through relocation.

## Extension quality rule

Use the previous monthly convention: QLD/VIC >=20 distinct valid hours per day;
>=ceil(0.75 * calendar days) valid days per station-month. Keep signed instrument
values in [-10,1000] for Australian archives, with explicit out-of-range counts;
these inherited limits are not a declaration that all values inside are correct.
NSW official daily means have unavailable hourly completeness. Preserve units,
native day convention, method precedence and coordinate provenance. Do not fill
missing months, choose stations by concentrations, or trim on prediction errors.
Only a unique same-country FUA polygon match is accepted. Quality eligibility,
geographic eligibility and historical-role certainty are separate fields.

## Candidate new networks

SA/WA official historical data may be inventoried and downloaded with provenance.
Do not fit or select on their outcomes. Bulk availability, pollutant, averaging
period, instrument comparability, role history, licensing and previous project
exposure must be checked first. A 24-hour rolling mean is not an hourly observation
or calendar-day mean. New monitor labels may still overlap previously used
satellite training labels; report this distinction.

## Confirmation gate

There is currently no demonstrated successful transfer rule: the preceding
48-setting audit had zero joint temporal/new-city passes. Do not cherry-pick a
previously failed setting and rename a reused period independent confirmation.
Before any future candidate test is opened, freeze one primary rule and its
development-only selection algorithm, target-label budget, source data, all
comparators, minimum coverage, effect threshold and clustered uncertainty method.
If no candidate passes development, record no-go rather than burn new test labels.

## Deliverables

1. File-hashed exposure ledger and correction to the external-test claim.
2. Station-year evidence/uncertainty registry and event-lead inventory.
3. Australian 2023-2025 monthly extension with quality/provenance summaries.
4. Chinese and new Australian network availability ledger.
5. Human-readable Chinese progress report, evidence package and verification.

All incomplete evidence must remain visibly unresolved. No claim that all
historical roles or all countries' 2023-2025 data are complete is permitted.
