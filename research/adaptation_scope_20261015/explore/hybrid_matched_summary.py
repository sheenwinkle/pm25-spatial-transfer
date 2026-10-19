"""Paired comparison of the matched-budget hybrid with S0 (post hoc, 19 October 2026; see hybrid_matched.py).

Uses the bootstrap functions, seed and draws of hybrid_summary.py, so the intervals are comparable with
hybrid_gains.csv (thesis Table C.4). Checks first that the matched run reproduces the original hybrid predictions
in every split where no label was cut.
Usage: python explore/hybrid_matched_summary.py <study_dir> <out_dir> [task ...]   (default task: aus)
Writes <out_dir>/hybrid_matched_gains.csv.
"""
import sys
from pathlib import Path

_src = Path(__file__).with_name("hybrid_summary.py").read_text()
exec(_src.split("rows_out, means_out = [], []")[0])   # A, OUT, draws_for, split_rmse, gains, H and KEY as defined there

tasks = sys.argv[3:] or ["aus"]
rows_out = []
for task in tasks:
    Q = pd.read_csv(A / f"results/{task}_q2_predictions.csv.gz")
    Hd = pd.read_csv(A / f"explore/{task}_hybrid_predictions.csv.gz")
    Mm = pd.read_csv(A / f"explore/{task}_hybridmatched_predictions.csv.gz")
    meth = ("Source-affine", "ECA-R", "ECA-S")
    hcols = [f"{H}|0|{x}" for x in meth]
    # 1. where nothing was cut, the matched run must equal the original hybrid run
    chk = Hd[KEY + hcols].merge(Mm[KEY + hcols + [f"{H}|0|n_rows", f"{H}|0|n_rows_full"]], on=KEY,
                                suffixes=("", "_m"), validate="one_to_one")
    same = chk[chk[f"{H}|0|n_rows"] == chk[f"{H}|0|n_rows_full"]]
    dev = max(float((same[c] - same[c + "_m"]).abs().max()) for c in hcols)
    assert dev < 1e-3, dev
    print(f"{task}: uncut splits reproduce the original hybrid (max |diff| {dev:.2g})")
    # 2. matched hybrid against S0, with the same draws as hybrid_summary.py
    P = Q.merge(Mm[KEY + hcols], on=KEY, how="left", validate="one_to_one")
    assert P[hcols].notna().all().all()
    fuas, quarters, wf, wq = draws_for(P)
    for (m, k), g in P.groupby(["m", "k"]):
        for method in meth:
            per = []
            for _, s in g.groupby("split"):
                per.append(split_rmse(s, {"S0": f"S0 last consecutive|0|{method}", "H": f"{H}|0|{method}"},
                                      fuas, quarters, wf, wq))
            rows_out.append(dict(task=task, m=m, k=k, method=method, comparison="H (matched) vs S0",
                                 **gains(per, "H", "S0")))
OUT.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(rows_out)
res.to_csv(OUT / "hybrid_matched_gains.csv", index=False, float_format="%.4f")
print(res.round(1).to_string(index=False))
