# Run log
- Benchmark repository: github.com/YongbeeIngkle/spatial-transfer-learning, commit
  3d8a25ab671ca7d34d38e16025163342c7a545c6 (2024-09-22), cloned 2026-10-07.
- Software: python 3.13, torch 2.14.1 (CPU), adapt 0.4.5, tensorflow 2.21.0, scikit-learn 1.9.1,
  lightgbm 4.7.0, numpy 2.5.3, cvxopt. OMP_NUM_THREADS=1, torch threads 1. 2 CPUs, 7 GB RAM.
- Data: all_networks_model_panel_corrected.csv.gz (34,639 FUA-months; NSW 31-Dec correction).
  No missing values in the 16 input columns or the label (checked before fitting).
- T1 eligible YRD FUAs (>= 80% of 96 months): 30; source 30,406 rows in 332 FUAs; target rows 2,737.
- 2026-10-07: single-pass timing test, T1/P-B/k=5/split 0 (300 s: LDF 38 s, NNW 13 s, KLIEP 63 s,
  KMM 182 s). File kept as results/yrd_B/singlepass_test_k5_split00.csv and recomputed by the
  core pass; the two are compared as a determinism check.
- First test attempt crashed (segfault) at torch.optim.NAdam after TensorFlow import; fixed by
  initialising torch optimiser modules before importing adapt.
- Core passes launched as two chains (A: YRD k=5,7 B then S, AUS B; B: YRD k=9,11 B then S, AUS S),
  followed by KMM passes.
- Fidelity notes recorded 2026-10-07 after re-reading the released code:
  (1) data_process/allocate.py loops over the dates present in the test set only; source and
      training-target rows of other dates are dropped. Under P-S/P-M (test 2021-2022) this would
      drop every earlier training-target row, so the harness loops over all months instead. Under
      P-B the two loops differ only for months in which no test FUA is observed.
  (2) The authors' README states that they used a locally edited copy of the adapt package,
      distributed separately (Google Drive). The harness uses the public adapt 0.4.5 release;
      instance weights may therefore differ in detail. The sample-weight variant is reported as a
      sensitivity check on how weights enter the final regressor.
  (3) Autoencoder architecture, optimiser, epochs, batch size, loss handling (two backward passes
      before one step), cloud normalisation (pm25 row mean 0 / sd 1, sd 0 -> 1), 12-neighbour
      composition (centre inserted at index 6, zero distance -> inf, inverse raw-distance
      weighting over measure tags) re-checked line by line against model/autoencoder.py,
      data_process/compose.py and data_process/allocate.py.
- Determinism check (2026-10-08): the core pass recomputed T1/P-B/k=5/split 0; all 20 models present
  in both runs have identical R2, RMSE and MAE (maximum absolute difference 0.0) to the earlier
  single-pass test, including the LDF-based models (torch single-threaded, seeded per split).
- Scope changes for compute (see PROTOCOL.md execution notes): KMM only under T1 P-B; KLIEP only
  under P-B (both tasks). Recorded before the corresponding fits.
- 2026-10-08 18:10: container restarted overnight; core and sensitivity runs were complete; KMM pass had finished 18/80 T1 P-B splits (k=5: 2, k=9: 16). Remaining 62 relaunched as two processes (run_kmm_P1.sh, run_kmm_P2.sh); the KMM pass skips splits that already have KMM rows.
- 2026-10-08 19:06: two-pass KMM check: T1/P-B/k=5/split 0 KMM and KMM[LDF] RMSE from the KMM pass (cached LDF codes) equal the single-pass test values (10.283 and 8.993).
