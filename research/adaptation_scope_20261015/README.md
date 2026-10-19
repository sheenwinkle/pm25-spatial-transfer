# Adaptation scope and label selection (thesis Section 7, Appendix C)

How much and where should a source PM2.5 model be adapted with a few scattered target labels, and which months should
those labels cover? Protocol and its two amendments: `PROTOCOL.md`; checksums frozen before the confirmation run:
`freeze_T3.txt`; changes after it: `POST_T3_NOTES.md`.

| File | Role |
|---|---|
| `regions.py`, `regions.json` | k-means partition of non-YRD Chinese FUAs into ten regions; locked confirmation region (8) |
| `adapt_lib.py` | data access (through the benchmark harness), source model, source-affine, ECA-R and ECA-S |
| `meta_learn.py`, `meta/theta.json` | meta-learning of the ECA-R prior on 216 within-China pseudo-transfers |
| `meta_learn2.py`, `meta/theta2.json` | meta-learning of the ECA-S priors (ARD, isotropic, affine-only) |
| `run_q1.py` | how much and where to adapt: designs `history`, `historyN` (source up to 2020), `window` |
| `run_q2.py` | which months to label: strategies S0-S4 |
| `summarise.py` | means, paired gains and crossed FUA-by-quarter bootstrap intervals (`summary/`) |
| `make_adapt_report.py` | thesis Tables 7.1-7.4 and Figure 7.1 (`report/`) |
| `check_adapt_claims.py` | checks every number of the study quoted in the thesis against `summary/`, `meta/`, `aoa/`, and that every generated table (`report/`, `explore/`) appears verbatim in the thesis |
| `aoa/di_probe.py` | covariate-only area-of-applicability shares (no PM2.5 read) |
| `aoa/distance_check.py` | exploratory, after T3: distance from the source data against the gain from recalibration in the nine development regions (thesis Table C.1) |
| `run_q2_matched.py` | post hoc, after the fifth draft: Q2 in Australia with every strategy cut to the labels the latest-months design actually received (matched budget) |
| `explore/hybrid_matched.py`, `explore/hybrid_matched_summary.py` | post hoc, after the sixth draft: the hybrid of thesis Table C.4 cut to the labels the latest-months design received in Australia (matched budget) |
| `explore/` | exploratory, after T3: condition groups, a hybrid label strategy, a label-window scan and the mechanism figure (thesis Figure C.1 and Tables C.2-C.4); see `explore/README.md` |
| `results/` | per-split metrics and row-level predictions for T1 (yrd), T2 (aus) and T3 (locked) |

Run order (paths in `env.sh.example`; the benchmark harness is `research/manuscript_revision_20261004/benchmark_replication_20261008/
benchmark_experiment`, whose README
describes the modelling panel and FUA membership file, which are not distributed):

```
python regions.py <panel> <membership> regions.json
python meta_learn.py meta
python meta_learn2.py meta
python run_q1.py <yrd|aus|locked> <history|historyN|window> meta/theta.json meta/theta2.json results
python run_q2.py <yrd|aus|locked> meta/theta.json meta/theta2.json results
python summarise.py results yrd,aus,locked summary
python make_adapt_report.py results summary report
python aoa/distance_check.py <panel> regions.json meta/meta2_tasks.csv aoa/distance_check.json report/tab_adapt_distance.tex
python run_q2_matched.py aus meta/theta.json meta/theta2.json results
python explore/ablate_groups.py explore
python explore/hybrid_labels.py <yrd|aus|locked> meta/theta.json meta/theta2.json explore
python explore/hybrid_summary.py . explore
python explore/hybrid_matched.py aus meta/theta.json meta/theta2.json explore
python explore/hybrid_matched_summary.py . explore aus
python explore/window_scan.py meta/theta.json meta/theta2.json explore
python explore/mechanism_figures.py meta/theta.json explore
python explore/condition_overlap.py explore
python explore/make_explore_report.py explore
python check_adapt_claims.py . <main.tex>
```

The harness imports torch for the LDF autoencoder, which this study does not use; `adapt_lib.py` substitutes a
minimal stand-in when torch is not installed. Runs used Python 3.13 on Linux with numpy 2.5.3, pandas 3.0.5,
scipy 1.18.1, scikit-learn 1.9.1 and lightgbm 4.7.0. Source-affine reproduces the archived benchmark results
(M and N protocols) to within 1e-14.
