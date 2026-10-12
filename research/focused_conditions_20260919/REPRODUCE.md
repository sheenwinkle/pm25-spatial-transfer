# Reproduction of the focused-condition extension

The archive contains sibling folders `focused_conditions_20260919` and
`thesis_completion_20260912`. Keep that relative arrangement. The latter is a
minimal dependency snapshot: harmonized observation/feature panels, original
helper functions, and cached 2015-2018 national source models. It is not a
replacement for the earlier raw-data provenance release.

Python 3.12 and the scientific package versions in `environment.json` are the
tested environment. From `focused_conditions_20260919`, use:

```text
python -m unittest test_focused.py -v
python audit_focused.py
python audit_recency.py
```

Those checks recompute metrics from included predictions, validate sample
selection and chronology, and reproduce the rolling-source nearest-neighbour
selection. They do not need the author's private accounts or network access.

The experiment runners are `run_focused.py` and `run_recency.py`. They retain
completed cached configurations and resume incomplete work. For a full new
fit, use a separate copy of the extracted package and rename the generated
`results`, `audit`, and `recency_results` directories in that copy before
running both scripts. Keep the delivered results intact for comparison.
National source models are the frozen controls from the prior experiment;
regional and matched policies are refitted by these runners.

`PROTOCOL.md` and `RECENCY_PROTOCOL.md` document which decisions were made
after earlier outcomes had been inspected. Re-running does not make the
analysis confirmatory. `build_report.py` is an optional local report builder
and requires Pandoc; the supplied self-contained HTML can be read without it.
