# Known-effect synthetic recovery protocol

Recorded: 4 October 2026, after the frozen within-China procedural control
completed and before any synthetic target was generated or fitted.

The within-China non-YRD-to-YRD control produced discovery passes but no joint
temporal/new-city pass. This second diagnostic asks a narrower software and
evaluation question: can the unchanged pipeline recover a deliberately injected,
stationary source-response signal? It has no environmental interpretation and is
not evidence for China--Australia transfer.

## Fixed construction

- Reuse the corrected all-network panel, non-YRD Chinese source domain, YRD target
  covariates, twelve `monthly_common` features, and disjoint FUA IDs from
  `PROTOCOL.md`.
- Fit the unchanged primary source LightGBM to non-YRD Chinese observations in
  2015--2018.
- For every YRD target row, define the synthetic response as the nonnegative
  source-model prediction plus independent Gaussian noise.
- Fix the noise generator seed at 2701. Fix noise standard deviation at 25% of
  the source-model prediction standard deviation across all target rows. Do not
  use observed YRD outcomes to set row values, select the seed, or tune noise.
- Save the generated labels and their hash before fitting target-assisted models.

## Fixed evaluation

- Run all seeds 5993--5995 and windows 3, 6, and 12 months.
- Use the same learners, shuffled-source controls, city split construction, gate
  search, minimum coverage, 5% improvement margin, temporal/new-city audits,
  and clustered-bootstrap requirements specified in `PROTOCOL.md`.
- Report all nine configurations. Do not alter the noise or decision rule after
  seeing results.

## Interpretation

- At least one joint pass shows that the implementation can recover a strong,
  stationary source effect under its full criterion; it does not establish power
  for 5% real-world effects.
- Zero joint passes means the full criterion lacks even this basic recovery
  sensitivity and the cross-country zero result cannot support a substantive
  no-transfer conclusion.
