"""Fill \\PH{key} placeholders in manuscript drafts from the experiment results.

Key grammar
  <task>_<prot>_k<k>_<cand>_vs_<ref>_<stat>   paired comparison; stat in rel|med|wins|r2d|rc|rr|n
  <task>_<prot>_k<k>_<model>_mean_<metric>     mean over splits; metric in r2|rmse|mae
  diag_<T1|T2>_<month_share|month_fua_share>    descriptive diagnostic (percent)
Other keys are left untouched (they are written by hand after reading the results).
Writes the filled text next to the draft (suffix _filled.tex) and a ledger CSV of every value used.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from make_tables import load, with_best
from paths import GENERATED

CODES = {"aff": "Source-affine (ours)", "affldf": "Source-affine[LDF] (ours)", "res": "Source-residual (ours)",
         "ker": "Source-kernel (ours)", "nnwldf": "NNW[LDF]", "nnw": "NNW", "gbr": "GBR (target-only)",
         "gbrldf": "GBR[LDF] (target-only)", "rf": "RF (target-only)", "kliep": "KLIEP", "kliepldf": "KLIEP[LDF]",
         "kmm": "KMM", "kmmldf": "KMM[LDF]", "nnwsw": "NNW (sample-weight)", "nnwldfsw": "NNW[LDF] (sample-weight)",
         "et": "Local Extra Trees", "lgbm": "Local LightGBM", "lrf": "Local RF", "best": "Best local",
         "bestbench": "Best benchmark", "shuf": "Shuffled-source affine (control)", "mean": "Local mean",
         "rbf": "Local RBF", "ridge": "Local ridge"}
PAIR = re.compile(r"^(yrd|aus)_(B|S|M3|M6|M12|N3|N6|N12)_k(\d+)_([a-z]+)_vs_([a-z]+)_(rel|abs|med|wins|r2d|rc|rr|n|maerel|min|max|rom|r2diff)$")
XB = re.compile(r"^(yrd|aus)_k(\d+)_(B|S|M3|M6|M12|N3|N6|N12)([a-z]+)_vs_(B|S|M3|M6|M12|N3|N6|N12)([a-z]+)_(rel|wins)$")
MEAN = re.compile(r"^(yrd|aus)_(B|S|M3|M6|M12|N3|N6|N12)_k(\d+)_([a-z]+)_mean_(r2|rmse|mae)$")
DIAG = re.compile(r"^diag_(T1|T2)_(month_share|month_fua_share)$")


def num(v, nd):
    v = round(float(v), nd)
    if v == 0:
        v = 0.0
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def value(key, d, diag):
    m = PAIR.match(key)
    if m:
        task, prot, k, c, r, stat = m.groups()
        g = d[(d.task == task) & (d.protocol == prot) & (d.k == int(k))]
        w = g.pivot_table(index="split", columns="model", values="rmse")
        r2 = g.pivot_table(index="split", columns="model", values="r2")
        cn, rn = CODES[c], CODES[r]
        if cn not in w or rn not in w:
            return None, None
        ok = w[[cn, rn]].dropna(); rel = 1 - ok[cn] / ok[rn]
        if stat == "maerel":
            wm = g.pivot_table(index="split", columns="model", values="mae")
            raw = 100 * (1 - wm[cn] / wm[rn]).mean()
            return num(raw, 1), float(raw)
        if stat == "rom":      # ratio of mean RMSEs, as in the benchmark's tables
            raw = 100 * (1 - ok[cn].mean() / ok[rn].mean())
            return num(raw, 1), float(raw)
        if stat == "r2diff":   # difference of mean R2 (decimal)
            raw = r2.loc[ok.index, cn].mean() - r2.loc[ok.index, rn].mean()
            return num(raw, 3), float(raw)
        raw = {"rel": 100 * rel.mean(), "abs": abs(100 * rel.mean()), "med": 100 * rel.median(), "min": 100 * rel.min(),
               "max": 100 * rel.max(), "wins": (ok[cn] < ok[rn]).sum(),
               "r2d": 100 * (r2.loc[ok.index, cn] - r2.loc[ok.index, rn]).mean(),
               "rc": ok[cn].mean(), "rr": ok[rn].mean(), "n": len(ok)}[stat]
        txt = str(int(raw)) if stat in ("wins", "n") else num(raw, 1 if stat in ("rel", "abs", "med", "r2d", "min", "max") else 2)
        return txt, float(raw)
    m = XB.match(key)
    if m:
        task, k, pa, ca, pb, cb, stat = m.groups()
        A = d[(d.task == task) & (d.protocol == pa) & (d.k == int(k)) & (d.model == CODES[ca])].set_index("split")
        B = d[(d.task == task) & (d.protocol == pb) & (d.k == int(k)) & (d.model == CODES[cb])].set_index("split")
        if not len(A) or not len(B):
            return None, None
        assert (A.n.sort_index().values == B.n.sort_index().values).all(), f"{key}: test rows differ"
        rel = 1 - A.rmse / B.rmse
        raw = 100 * rel.mean() if stat == "rel" else int((A.rmse < B.rmse).sum())
        return (num(raw, 1) if stat == "rel" else str(raw)), float(raw)
    m = MEAN.match(key)
    if m:
        task, prot, k, c, metric = m.groups()
        g = d[(d.task == task) & (d.protocol == prot) & (d.k == int(k)) & (d.model == CODES[c])]
        if not len(g):
            return None, None
        raw = g[metric].mean()
        return num(raw, 3 if metric == "r2" else 2), float(raw)
    m = DIAG.match(key)
    if m:
        t, s = m.groups()
        raw = 100 * diag[f"{t}_all"][s]
        return num(raw, 1), float(raw)
    return None, None


def main(paths):
    d = with_best(load())
    diag = json.loads((GENERATED / "diagnostics.json").read_text())
    ledger = []
    for p in map(Path, paths):
        s = p.read_text()
        keys = sorted(set(re.findall(r"\\PH\{([^}]*)\}", s)))
        for key in keys:
            txt, raw = value(key, d, diag)
            if txt is None:
                continue
            s = s.replace(f"\\PH{{{key}}}", txt)
            ledger.append({"file": p.name, "key": key, "text": txt, "raw": raw})
        out = p.with_name(p.stem + "_filled.tex"); out.write_text(s)
        left = sorted(set(re.findall(r"\\PH\{([^}]*)\}", s)))
        print(p.name, "filled", len(keys) - len(left), "left", left)
    pd.DataFrame(ledger).to_csv(GENERATED / "fill_ledger.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
