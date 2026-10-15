"""Summaries, paired gains and crossed bootstrap intervals for Q1 and Q2 (see PROTOCOL.md).

Usage: python summarise.py <results_dir> <tasks, comma-separated> <out_dir>
Paired gain of A over B: 100 * mean over splits of (1 - RMSE_A / RMSE_B); intervals from 1,000 crossed
FUA-by-calendar-quarter bootstrap draws of the test rows (seed 20261015), shared by every comparison of a task.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import adapt_lib as al

RES, TASKS, OUT = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
DRAWS = 1000
LOCAL = ["Local mean", "Local ridge", "Local RBF", "Local RF", "Local Extra Trees", "Local LightGBM"]
EIGHT = ["RF (target-only)", "GBR (target-only)"] + LOCAL
Q1_MODELS = ["Direct", "Source-affine", "Source-residual", "ECA", "ECA-fixed", "ECA-no-meta", "ECA-unbounded",
             "ECA-isotropic", "ECA2", "ECA2-MAP", "ECA2-affine", "ECA2-isotropic", "ECA2-no-meta",
             "ECA2-unbounded"] + LOCAL
Q1_PAIRS = [("ECA2", b) for b in ["Direct", "Source-affine", "Source-residual", "Best local six", "ECA2-MAP",
                                  "ECA2-affine", "ECA2-isotropic", "ECA2-no-meta", "ECA2-unbounded", "ECA",
                                  "ECA-fixed"]] + \
           [("ECA2-MAP", b) for b in ["Direct", "Source-affine", "Best local six"]] + \
           [("ECA-fixed", b) for b in ["Direct", "Source-affine", "Best local six", "ECA", "ECA-no-meta",
                                       "ECA-unbounded", "ECA-isotropic"]] + \
           [("Source-affine", "Direct"), ("Best local six", "Direct"), ("Source-affine", "Best local six")]


def draws_for(frame):
    """Crossed FUA-by-quarter multinomial weights for the rows of one task (shared by all comparisons)."""
    fuas = np.array(sorted(frame.eFUA_ID.unique())); quarters = np.array(sorted(frame.quarter_id.unique()))
    rng = np.random.default_rng(al.SEED)
    wf = rng.multinomial(len(fuas), np.full(len(fuas), 1 / len(fuas)), size=DRAWS).astype(float)
    wq = rng.multinomial(len(quarters), np.full(len(quarters), 1 / len(quarters)), size=DRAWS).astype(float)
    return fuas, quarters, wf, wq


def split_rmse(df, cols, fuas, quarters, wf, wq):
    """Point and bootstrap RMSE for each column of one split's rows; list entries are per-split (and per-draw)
    minima over learners."""
    w = wf[:, np.searchsorted(fuas, df.eFUA_ID.to_numpy())] * wq[:, np.searchsorted(quarters, df.quarter_id.to_numpy())]
    tot = w.sum(1)
    tot = np.where(tot > 0, tot, np.nan)
    y = df.observed.to_numpy()
    out = {}
    for name, c in cols.items():
        if isinstance(c, list):
            pts, bts = zip(*[(np.sqrt(np.mean((df[x].to_numpy() - y) ** 2)),
                              np.sqrt(w @ (df[x].to_numpy() - y) ** 2 / tot)) for x in c])
            out[name] = (min(pts), np.min(np.vstack(bts), 0))
        else:
            e = (df[c].to_numpy() - y) ** 2
            out[name] = (np.sqrt(e.mean()), np.sqrt(w @ e / tot))
    return out


def gains(rows, a, b):
    """A draw that gives a split's test rows zero total weight (possible with nine Australian FUAs) leaves that split
    out of the draw's mean; draws with no usable split are dropped and counted."""
    pa = np.array([r[a][0] for r in rows]); pb = np.array([r[b][0] for r in rows])
    ba = np.vstack([r[a][1] for r in rows]); bb = np.vstack([r[b][1] for r in rows])
    g = 100 * np.mean(1 - pa / pb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 1 - ba / bb
    ok = np.isfinite(rel)
    d = 100 * np.where(ok.any(0), np.nansum(np.where(ok, rel, 0), 0) / np.maximum(ok.sum(0), 1), np.nan)
    d = d[np.isfinite(d)]
    return dict(gain=g, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), wins=int((pa < pb).sum()),
                n=len(rows), mean_a=pa.mean(), mean_b=pb.mean(), draws=len(d),
                partial_draws=int((~ok).any(0).sum()))


q1_rows, q1_means, arch_rows, eff_rows, q2_rows, q2_means = [], [], [], [], [], []
for task in TASKS:
    for design in ("history", "historyN", "window"):
        f = RES / f"{task}_{design}_predictions.csv.gz"
        if not f.exists():
            continue
        P = pd.read_csv(f)
        fuas, quarters, wf, wq = draws_for(P)
        cols = {m: m for m in Q1_MODELS if m in P.columns}
        cols["Best local six"] = LOCAL
        per_cell = {}
        for (cell, k), g in P.groupby(["cell", "k"]):
            rows = [split_rmse(s, cols, fuas, quarters, wf, wq) for _, s in g.groupby("split")]
            per_cell[(cell, k)] = (rows, sorted(g.split.unique()))
            for name in cols:
                q1_means.append(dict(task=task, design=design, cell=cell, k=k, model=name,
                                     mean_rmse=np.mean([r[name][0] for r in rows])))
            for a, b in Q1_PAIRS:
                if a in cols and b in cols:
                    q1_rows.append(dict(task=task, design=design, cell=cell, k=k, comparison=f"{a} vs {b}",
                                        **gains(rows, a, b)))
            if design == "history" and task in ("yrd", "aus"):     # archived benchmark results, split level
                for anchor in ("ECA2", "Direct"):
                    ours = {r_split: r[anchor][0] for r_split, r in zip(per_cell[(cell, k)][1], rows)}
                    for ref in ("NNW[LDF]", "Best eight"):
                        a, bvals = [], []
                        for sid, e in ours.items():
                            arch = pd.read_csv(al.BENCH / "results" / f"{task}_{cell}" / f"k{k}_split{sid:02d}.csv")
                            bv = arch[arch.model.isin(EIGHT)].rmse.min() if ref == "Best eight" else \
                                float(arch[arch.model == ref].rmse.iloc[0])
                            a.append(e); bvals.append(bv)
                        a, bvals = np.array(a), np.array(bvals)
                        arch_rows.append(dict(task=task, cell=cell, k=k, comparison=f"{anchor} vs {ref} (archived)",
                                              gain=100 * np.mean(1 - a / bvals), wins=int((a < bvals).sum()),
                                              n=len(a), mean_a=a.mean(), mean_b=bvals.mean()))
        if design in ("history", "historyN"):   # label efficiency: ECA2 with m months against others with 2m months
            for k in sorted(P.k.unique()):
                for m1, m2 in (("M3", "M6"), ("M6", "M12")):
                    if (m1, k) not in per_cell or (m2, k) not in per_cell:
                        continue
                    r1, s1 = per_cell[(m1, k)]; r2, s2 = per_cell[(m2, k)]
                    common = sorted(set(s1) & set(s2))
                    r1 = [r for s, r in zip(s1, r1) if s in common]; r2 = [r for s, r in zip(s2, r2) if s in common]
                    for b in ("Source-affine", "Best local six", "ECA2"):
                        merged = [{"A": a_["ECA2"], "B": b_[b]} for a_, b_ in zip(r1, r2)]
                        eff_rows.append(dict(task=task, design=design, k=k,
                                             comparison=f"ECA2 at {m1} vs {b} at {m2}", **gains(merged, "A", "B")))
    f = RES / f"{task}_q2_predictions.csv.gz"
    if f.exists():
        P = pd.read_csv(f)
        fuas, quarters, wf, wq = draws_for(P)
        strategies = sorted({c.split("|")[0] for c in P.columns if "|" in c})
        for (m, k), g in P.groupby(["m", "k"]):
            q2_means.append(dict(task=task, m=m, k=k, method="Direct", strategy="(no labels)",
                                 mean_rmse=np.mean([np.sqrt(np.mean((s.Direct - s.observed) ** 2))
                                                    for _, s in g.groupby("split")])))
            for method in ("ECA-S", "ECA-R", "Source-affine", "Best local six"):
                rows = []
                for _, s in g.groupby("split"):
                    cols = {"Direct": "Direct"}
                    for st in strategies:
                        reps = sorted({c.split("|")[1] for c in P.columns if c.startswith(st + "|")})
                        for rep in reps:
                            cols[f"{st}|{rep}"] = [f"{st}|{rep}|{x}" for x in LOCAL] if method == "Best local six" \
                                else f"{st}|{rep}|{method}"
                    r = split_rmse(s, cols, fuas, quarters, wf, wq)
                    merged = {"Direct": r["Direct"]}
                    for st in strategies:      # average over the repetitions of random strategies
                        keys = [kk for kk in r if kk.startswith(st + "|")]
                        merged[st] = (np.mean([r[kk][0] for kk in keys]), np.mean([r[kk][1] for kk in keys], 0))
                    rows.append(merged)
                for st in strategies:
                    q2_means.append(dict(task=task, m=m, k=k, method=method, strategy=st,
                                         mean_rmse=np.mean([r[st][0] for r in rows])))
                    q2_rows.append(dict(task=task, m=m, k=k, method=method, comparison=f"{st} vs Direct",
                                        **gains(rows, st, "Direct")))
                s4 = [s for s in strategies if s.startswith("S4")][0]
                for st in strategies:
                    if st != s4:
                        q2_rows.append(dict(task=task, m=m, k=k, method=method, comparison=f"{s4} vs {st}",
                                            **gains(rows, s4, st)))

for name, rows in [("q1_gains", q1_rows), ("q1_means", q1_means), ("q1_archived", arch_rows),
                   ("q1_label_efficiency", eff_rows), ("q2_gains", q2_rows), ("q2_means", q2_means)]:
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
print("wrote", sorted(p.name for p in OUT.glob("*.csv")))
