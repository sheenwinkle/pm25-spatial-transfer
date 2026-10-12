# Raw-observation recovery audit

Started 3 October and completed 4 October 2026, Australia/Sydney.

Outcome: bounded public-source search and audit completed; original missing
observations NOT recovered. No observations admitted and no models refitted.
Do not describe candidate rows, processed products, or additional downloads as
new observed training labels. See PROTOCOL.md for admission rules.

## Outputs

The reader reports are in output/research/raw_observation_recovery_20261003.
The Chinese report includes the decision, before/after comparison, source
ledger, figure, and limitations. The English addendum is not an integrated
replacement dissertation. The outreach draft has not been sent.

## Reproduction

Python dependencies: pandas, numpy, xarray, netCDF4, requests and matplotlib.
Use the existing project scientific environment. Original 2018 CNEMC annual
ZIP, prior gap-aware branch, and unchanged model dependencies remain external
to this package; paths and hashes are recorded in audit outputs. The current
code resolves original data under the existing user's project directory.

1. fetch_sources.py caches public initial metadata responses and receipts.
   Additional recorded requests have their exact URLs/methods/forms in evidence.
   Cached errors are preserved as evidence, not silently retried or overwritten.
2. audit_candidate.py reconstructs the candidate comparison and hypothetical
   missing-only eligibility overlay using xarray and wide hourly alignment.
3. finish_audit.py independently repeats the comparison using direct netCDF4
   decoding and long-form joins, tests date rejection, inventories downloaded
   evidence, checks original hashes and writes VERIFICATION.json.
4. build_review.py renders the reports using the existing Pandoc installation.
5. qa_review.cjs checks desktop/mobile HTML with the existing Playwright runtime
   and Chrome. Screenshots were also visually reviewed.
6. package_review.py creates the ZIP and checks every archived file against
   the byte count and SHA-256 manifest.

## Important distinctions

- A 200 response with a wrong observation date is not historical data.
- Two archives of the same instruments are not independent measurements.
- The Li file's zero-offset agreement is diagnostic, not confirmed UTC metadata.
- The hypothetical five-value overlay does not modify any original data.
- The PANGAEA homogenized daily release is not a raw-hourly backup.
- Negative access results apply to tested requests, not all provider services.
- A reconstruction experiment using blocked masks has not yet been conducted.

Third-party attribution and license notes accompany the reader reports.
