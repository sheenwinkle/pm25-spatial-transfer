"""Exploratory, added after the T3 run (explore/README.md, item 4): test error by position of a three-month label
window. Labels: the k = 5 labelled FUAs of each benchmark split, three consecutive months ending in each month from
March 2019 to December 2020; test: the other FUAs in 2021-2022. Estimators as in run_q1.py (source-affine, ECA-R,
ECA-S) and direct transfer.
Usage: python explore/window_scan.py <theta.json> <theta2.json> <out_dir>
Writes window_scan.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_lib as al
from adapt_lib import X, Y, core

theta_path, theta2_path, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
th = json.loads(theta_path.read_text())["ARD"]
THETA = dict(ls=np.array(th["ls"]), af=th["af"], an=th["an"])
TH2 = json.loads(theta2_path.read_text())
rows = []
for task in ("yrd", "locked", "aus"):
    src, tgt, eligible = al.load(task)
    f = al.source_model(src)
    tgt["raw"] = core.pos(f.predict(tgt[X]))
    src_sd = float(np.std(core.pos(f.predict(src[X]))))
    Z = al.zmat(tgt)
    t_of = tgt.drop_duplicates(["year", "month"]).set_index(["year", "month"]).t
    ends = [(y, m) for y in (2019, 2020) for m in range(1, 13) if (y, m) >= (2019, 3)]
    for sid, (train_ids, test_ids) in enumerate(al.bh.benchmark_splits(eligible, 5, 20)):
        tst = (tgt.eFUA_ID.isin(test_ids) & tgt.year.ge(2021)).to_numpy()
        b = tgt[tst]; vy = b[Y].to_numpy(); eraw = b.raw.to_numpy()
        for (y, m) in ends:
            te = int(t_of[(y, m)])
            trn = (tgt.eFUA_ID.isin(train_ids) & tgt.t.between(te - 2, te)).to_numpy()
            if trn.sum() < 5:
                continue
            a = tgt[trn]; yl, raw = a[Y].to_numpy(), a.raw.to_numpy()
            aff_l, aff_t = al.affine(raw, yl, eraw)
            g, _ = al.eca_predict("ECA-fixed", THETA, Z[trn], yl - aff_l, Z[tst], yl.mean())
            s_l, s_t = al.scaled_source(raw, eraw, src_sd)
            d2, _ = al.eca2_predict("ECA2", TH2, Z[trn], s_l, yl - raw, Z[tst], s_t)
            for name, p in (("Direct", eraw), ("Source-affine", core.pos(aff_t)), ("ECA-R", core.pos(aff_t + g)),
                            ("ECA-S", core.pos(eraw + d2))):
                rows.append(dict(task=task, split=sid, end=f"{y}-{m:02d}", model=name, rmse=al.rmse(vy, p)))
    print(task, "done", flush=True)
pd.DataFrame(rows).to_csv(OUT / "window_scan.csv", index=False, float_format="%.5g")
