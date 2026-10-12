# Positive-control results

Completed and independently verified on 4 October 2026.

## Within-China procedural control

The source domain contains 15,010 observations from 330 non-YRD Chinese FUAs
during 2015--2018. The target contains 3,453 observations from 43 YRD FUAs during
2015--2022. Both domains use the corrected all-network target definition; their
FUA IDs are disjoint. Thirty-nine target FUAs meet the fixed 2018 eligibility
rule.

All nine fixed seed/window configurations meet the discovery selection rule.
Two of nine pass the full held-city audit, none passes the temporal audit, and
none passes both. Temporal relative gain has median -4.18% and range -13.39% to
18.68%; three configurations exceed the 5% point margin, but none has all
required positive clustered-bootstrap lower bounds. Held-city relative gain has
median 4.34% and range -6.07% to 26.18%; four exceed 5% and two pass the full
audit.

This result does not establish that within-China transfer is absent. It shows
that the real-data pipeline cannot use the current joint criterion as a
sensitive positive control: temporal instability or uncertainty prevents every
configuration from passing. This weakens any substantive interpretation of the
cross-country 0/90 count.

## Known-effect synthetic recovery

The second control retains the same covariates, FUA splits, learners, label
budgets, gates and audits. Its target is the nonnegative source-model prediction
plus Gaussian noise from fixed seed 2701, with noise SD equal to 25% of the
source-signal SD. Observed YRD outcomes do not determine the synthetic row
values or noise level.

All nine configurations pass discovery, temporal and held-city audits, and all
nine pass jointly. Median relative gains are 58.53% in discovery, 57.52% in the
temporal audit and 64.53% in the held-city audit. The minimum gains in those
stages are 37.22%, 31.52% and 37.16%, respectively.

The synthetic result demonstrates that the implementation can recover a strong,
stationary injected source effect. It does not demonstrate sensitivity to a 5%
real-world effect, validate the assumed noise model, or provide environmental
evidence for cross-country transfer.

## Combined interpretation

The controls separate a software failure from an inferential-resolution problem.
The pipeline is not mechanically incapable of returning a joint pass, because
the strong known effect yields 9/9. However, the realistic within-China control
still yields 0/9 because of the temporal audit. The cross-country result must
therefore be reported as failure to demonstrate a stable effect under the
current data and criterion, not proof that useful transfer is absent.

`verify_controls.py` independently reapplied every gate, recomputed coverage,
RMSE and relative gain, checked chronological fit records, and verified protocol,
script, prediction and synthetic-label hashes for 72 stage records. All checks
passed. Machine-readable summaries are in `audit/verification.json`,
`audit/summary.csv` and `synthetic_audit/summary.csv`.
