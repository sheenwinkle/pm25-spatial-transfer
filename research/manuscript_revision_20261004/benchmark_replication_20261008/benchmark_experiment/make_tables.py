"""Generate the LaTeX tables and the number ledger of the benchmark-aligned experiments from results/*/.

Every number that the manuscript quotes from these experiments is written to generated/numbers.json;
check_claims.py verifies the manuscript's qualitative claims against the same per-split results.
Round 7: ties at the displayed precision are all set in bold; R-squared is typeset as \\(R^2\\); and in the
no-contemporaneous sensitivity (Table 5) the target-only references (GBR, best local learner) are the fits
of the matching short-history run, because those learners use no source data and a refit changes only the
row order (which moves seeded GBR, random-forest and Extra Trees fits).
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from paths import GENERATED, RESULTS

R = RESULTS
G = GENERATED
G.mkdir(parents=True, exist_ok=True)

GROUPS = [
    ("Benchmark: target only", [("RF (target-only)", "RF"), ("GBR (target-only)", "GBR"), ("GBR[LDF] (target-only)", "GBR[LDF]$^{a}$")]),
    ("Benchmark: instance transfer", [("NNW", "NNW"), ("NNW[LDF]", "NNW[LDF]"), ("KLIEP", "KLIEP"), ("KLIEP[LDF]", "KLIEP[LDF]"),
                                      ("KMM", "KMM"), ("KMM[LDF]", "KMM[LDF]")]),
    ("Benchmark: weights as sample weights", [("NNW (sample-weight)", "NNW-sw"), ("NNW[LDF] (sample-weight)", "NNW[LDF]-sw")]),
    ("Local learners (target only)", [("Local mean", "Mean"), ("Local ridge", "Ridge"), ("Local RBF", "RBF kernel"),
                                      ("Local RF", "Random forest"), ("Local Extra Trees", "Extra Trees"), ("Local LightGBM", "LightGBM")]),
    ("Source recalibration (this report)", [("Source-affine (ours)", "Source-affine"), ("Source-residual (ours)", "Source-residual"),
                                            ("Source-kernel (ours)", "Source-kernel"), ("Source-affine[LDF] (ours)", "Source-affine[LDF]")]),
    ("Control", [("Shuffled-source affine (control)", "Shuffled-source affine")]),
]
LOCAL = ["Local mean", "Local ridge", "Local RBF", "Local RF", "Local Extra Trees", "Local LightGBM"]
TARGET_ONLY = ["RF (target-only)", "GBR (target-only)"] + LOCAL
BENCH = ["RF (target-only)", "GBR (target-only)", "GBR[LDF] (target-only)", "NNW", "NNW[LDF]", "KLIEP", "KLIEP[LDF]",
         "KMM", "KMM[LDF]", "NNW (sample-weight)", "NNW[LDF] (sample-weight)"]
AFF, NNWL, GBR, AFFL = "Source-affine (ours)", "NNW[LDF]", "GBR (target-only)", "Source-affine[LDF] (ours)"


def load():
    fr = []
    for f in sorted(R.glob("*_*/k*_split*.csv")):
        if f.name.startswith("singlepass") or f.parent.name == "summary":
            continue
        d = pd.read_csv(f)
        if "SKIPPED" in set(d.model):
            continue
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["protocol"] = d.protocol.astype(str)
    return d


def with_best(d):
    extra = []
    for _, g in d.groupby(["task", "protocol", "k", "split"]):
        # Keep the legacy result key; its current definition includes all eight target-only learners.
        for name, pool in (("Best local", TARGET_ONLY), ("Best benchmark", BENCH)):
            h = g[g.model.isin(pool)]
            if name == "Best local" and set(h.model) != set(TARGET_ONLY):
                raise ValueError("The per-split post-hoc best reference requires all eight target-only learners")
            r = h.loc[h.rmse.idxmin()].copy(); r["model"] = name; extra.append(r)
    return pd.concat([d, pd.DataFrame(extra)], ignore_index=True)


def fmt(x, nd=3):
    if pd.isna(x):
        return "--"
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    return f"{v:.{nd}f}".replace("-", "$-$")


def bench_table(d, task, prot, ks, caption, label, place="H"):
    g = d[(d.task == task) & (d.protocol == prot)]
    m = g.groupby(["model", "k"]).agg(r2=("r2", "mean"), rmse=("rmse", "mean"), n=("split", "nunique")).reset_index()
    best_r2 = m[m.model.isin([x for _, grp in GROUPS for x, _ in grp])].groupby("k").r2.max()
    best_rm = m[m.model.isin([x for _, grp in GROUPS for x, _ in grp])].groupby("k").rmse.min()
    cols = "l" + "rr" * len(ks)
    head1 = " & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in ks]) + " \\\\"
    cmid = "".join([f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))])
    head2 = "Model & " + " & ".join(["\\(R^2\\) & RMSE"] * len(ks)) + " \\\\"
    lines = [f"\\begin{{table}}[{place}]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", head1, cmid, head2, "\\midrule"]
    for gname, models in GROUPS:
        present = [(k_, n_) for k_, n_ in models if (m.model == k_).any()]
        if not present:
            continue
        lines.append(f"\\multicolumn{{{1 + 2 * len(ks)}}}{{@{{}}l}}{{\\emph{{{gname}}}}} \\\\")
        for key, name in present:
            cells = []
            for k in ks:
                r = m[(m.model == key) & (m.k == k)]
                if not len(r):
                    cells += ["--", "--"]; continue
                r2, rm = r.r2.iloc[0], r.rmse.iloc[0]
                c1 = fmt(r2); c2 = fmt(rm)
                if fmt(r2) == fmt(best_r2[k]): c1 = f"\\textbf{{{c1}}}"
                if fmt(rm) == fmt(best_rm[k]): c2 = f"\\textbf{{{c2}}}"
                cells += [c1, c2]
            lines.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
    nsp = m.groupby("k").n.max().to_dict()
    shown = m[m.model.isin([x for _, grp in GROUPS for x, _ in grp])]
    bad = shown[shown.n != 20]
    if len(bad):
        msg = f"{task}-{prot}: cells with != 20 splits:\n{bad[['model', 'k', 'n']]}"
        if os.environ.get("STRICT") == "1":
            raise AssertionError(msg)
        print("WARNING", msg)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines), {int(k): int(v) for k, v in nsp.items()}


def paired_rows(d, task, prot, k, cand, ref):
    g = d[(d.task == task) & (d.protocol == prot) & (d.k == k)]
    w = g.pivot_table(index="split", columns="model", values="rmse")
    r2 = g.pivot_table(index="split", columns="model", values="r2")
    if cand not in w or ref not in w:
        return None
    ok = w[[cand, ref]].dropna()
    rel = 1 - ok[cand] / ok[ref]
    return {"task": task, "protocol": prot, "k": int(k), "candidate": cand, "reference": ref, "splits": int(len(ok)),
            "rmse_c": float(ok[cand].mean()), "rmse_r": float(ok[ref].mean()),
            "rel_mean_pct": float(100 * rel.mean()), "rel_median_pct": float(100 * rel.median()),
            "wins": int((ok[cand] < ok[ref]).sum()),
            "r2_c": float(r2.loc[ok.index, cand].mean()), "r2_r": float(r2.loc[ok.index, ref].mean()),
            "r2_diff_pts": float(100 * (r2.loc[ok.index, cand] - r2.loc[ok.index, ref]).mean())}


def paired_cross(d, task, k, prot_c, cand, prot_r, ref):
    """Paired reduction of cand (from protocol prot_c) against ref (from protocol prot_r) on the same splits and test rows."""
    a = d[(d.task == task) & (d.protocol == prot_c) & (d.k == k) & (d.model == cand)].set_index("split")
    b = d[(d.task == task) & (d.protocol == prot_r) & (d.k == k) & (d.model == ref)].set_index("split")
    if not len(a) or not len(b):
        return None
    j = a[["rmse", "n"]].join(b[["rmse", "n"]], lsuffix="_c", rsuffix="_r", how="inner")
    assert (j.n_c == j.n_r).all(), "cross-protocol comparison on different test rows"
    rel = 1 - j.rmse_c / j.rmse_r
    return {"task": task, "protocol": f"{prot_c}/{prot_r}", "k": int(k), "candidate": cand, "reference": ref,
            "splits": int(len(j)), "rmse_c": float(j.rmse_c.mean()), "rmse_r": float(j.rmse_r.mean()),
            "rel_mean_pct": float(100 * rel.mean()), "rel_median_pct": float(100 * rel.median()),
            "wins": int((j.rmse_c < j.rmse_r).sum())}


def main():
    d = with_best(load())
    numbers = {"completed": {}}
    for (t, p, k), g in d[~d.model.isin(["Best local", "Best benchmark"])].groupby(["task", "protocol", "k"]):
        numbers["completed"][f"{t}_{p}_k{k}"] = int(g.split.nunique())
    out = {}
    if ((d.task == "yrd") & (d.protocol == "B")).any():
        out["tab_T1_B"], _ = bench_table(d, "yrd", "B", [5, 7, 9, 11],
            "T1 (YRD target), same-period benchmark protocol: mean \\(R^2\\) and RMSE "
            "($\\mu$g/m$^3$) over 20 benchmark splits for $k$ labelled target FUAs, in the format of the benchmark's Table 1. "
            "Bold marks the best value in each column. $^{a}$GBR[LDF] is fitted on target rows only, but its LDF feature is learnt from source and target clouds.", "tab:t1-benchmark",
            place="tbp")
    if ((d.task == "yrd") & (d.protocol == "S")).any():
        out["tab_T1_S"], _ = bench_table(d, "yrd", "S", [5, 7, 9, 11],
            "T1 (YRD target), later-period protocol: labelled FUAs contribute months up to 2020 and test FUAs are "
            "evaluated in 2021--2022. Mean over 20 splits; bold marks the best value in each column. KLIEP and KMM were run "
            "only under the benchmark protocol. $^{a}$GBR[LDF] is fitted on target rows only, but its LDF feature is learnt from source and target clouds.", "tab:t1-later")
    t2 = d[d.task == "aus"]
    if len(t2):
        # T2: same-period and later-period side by side
        rows = []
        for prot, lab in (("B", "Same period"), ("S", "Later period")):
            for k in (3, 5):
                rows.append((prot, k, lab))
        m = t2.groupby(["protocol", "model", "k"]).agg(r2=("r2", "mean"), rmse=("rmse", "mean"), n=("split", "nunique")).reset_index()
        _shown = m[m.protocol.isin(["B", "S"]) & m.model.isin([x for _, grp in GROUPS for x, _ in grp])]
        if not (_shown.n == 20).all():
            msg = f"T2 cells with != 20 splits:\n{_shown[_shown.n != 20]}"
            if os.environ.get("STRICT") == "1":
                raise AssertionError(msg)
            print("WARNING", msg)
        keys = [x for _, grp in GROUPS for x, _ in grp]
        best = m[m.model.isin(keys)].groupby(["protocol", "k"]).agg(r2=("r2", "max"), rmse=("rmse", "min"))
        lines = ["\\begin{table}[H]", "\\centering", "\\footnotesize",
                 "\\caption{T2 (China source, Australian target): mean \\(R^2\\) and RMSE ($\\mu$g/m$^3$) over 20 benchmark splits "
                 "(19 distinct label sets per $k$) under the same-period and later-period protocols. Bold marks the best value in each "
                 "column. KLIEP was run only under the same-period protocol and KMM only for T1. $^{a}$GBR[LDF] is fitted on target rows only, but its LDF feature is learnt from source and target clouds.}",
                 "\\label{tab:t2}", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}l" + "rr" * 4 + "@{}}", "\\toprule",
                 " & \\multicolumn{4}{c}{Same period} & \\multicolumn{4}{c}{Later period} \\\\",
                 "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
                 " & \\multicolumn{2}{c}{$k=3$} & \\multicolumn{2}{c}{$k=5$} & \\multicolumn{2}{c}{$k=3$} & \\multicolumn{2}{c}{$k=5$} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 "Model & \\(R^2\\) & RMSE & \\(R^2\\) & RMSE & \\(R^2\\) & RMSE & \\(R^2\\) & RMSE \\\\", "\\midrule"]
        for gname, models in GROUPS:
            present = [(k_, n_) for k_, n_ in models if (m.model == k_).any()]
            if not present:
                continue
            lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{gname}}}}} \\\\")
            for key, name in present:
                cells = []
                for prot, k, _ in rows:
                    r = m[(m.protocol == prot) & (m.model == key) & (m.k == k)]
                    if not len(r):
                        cells += ["--", "--"]; continue
                    c1, c2 = fmt(r.r2.iloc[0]), fmt(r.rmse.iloc[0])
                    if (prot, k) in best.index:
                        if fmt(r.r2.iloc[0]) == fmt(best.loc[(prot, k), "r2"]): c1 = f"\\textbf{{{c1}}}"
                        if fmt(r.rmse.iloc[0]) == fmt(best.loc[(prot, k), "rmse"]): c2 = f"\\textbf{{{c2}}}"
                    cells += [c1, c2]
                lines.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        out["tab_T2"] = "\n".join(lines)
    # short-history summary (paired vs NNW[LDF] and GBR), h = 3, 6, 12, 72
    cells = []
    for task, ks in (("yrd", (5, 11)), ("aus", (3, 5))):
        for k in ks:
            for h, prot in ((3, "M3"), (6, "M6"), (12, "M12"), (72, "S")):
                a = paired_rows(d, task, prot, k, AFF, NNWL)
                b = paired_rows(d, task, prot, k, AFF, GBR)
                c = paired_rows(d, task, prot, k, AFFL, NNWL)
                e = paired_rows(d, task, prot, k, AFF, "Shuffled-source affine (control)")
                f = paired_rows(d, task, prot, k, AFF, "Best local")
                if a is None:
                    continue
                cells.append({"task": task, "k": k, "h": h, "protocol": prot, "aff_vs_nnwldf": a, "aff_vs_gbr": b,
                              "affldf_vs_nnwldf": c, "aff_vs_shuffled": e, "aff_vs_bestlocal": f})
    numbers["short_history"] = cells
    if cells:
        lines = ["\\begin{table}[H]", "\\centering", "\\footnotesize",
                 "\\caption{Label-history ablation. Labelled FUAs contribute the last $h$ months of 2020 ($h=72$: all months up to 2020, "
                 "the later-period protocol); test FUAs are evaluated in 2021--2022. Aff.\\ is source-affine and Aff.[LDF] source-affine[LDF]. "
                 "Paired entries are mean relative RMSE reductions (\\%) over the 20 splits, with the number of splits won in parentheses; "
                 "positive values favour the first-named estimator. RMSE columns are means in $\\mu$g/m$^3$.}",
                 "\\label{tab:history}", "\\setlength{\\tabcolsep}{3.5pt}",
                 "\\begin{adjustbox}{max width=\\linewidth}",
                 "\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
                 " & & \\multicolumn{4}{c}{Mean RMSE} & \\multicolumn{4}{c}{Paired RMSE reduction, \\% (splits won)} \\\\",
                 "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
                 " & & & & & Aff. & Aff. vs & Aff. vs & Aff.[LDF] vs & Aff. vs \\\\",
                 "Task & $h$ & GBR & NNW[LDF] & Aff. & [LDF] & NNW[LDF] & GBR & NNW[LDF] & shuffled \\\\",
                 "\\midrule"]
        last = None
        for c in cells:
            tk = f"{'T1' if c['task']=='yrd' else 'T2'}, $k={c['k']}$"
            show = tk if tk != last else ""
            if last is not None and tk != last:
                lines.append("\\addlinespace")
            last = tk
            def pr(x):
                return "--" if x is None else f"{fmt(x['rel_mean_pct'], 1)} ({x['wins']})"
            lines.append(f"{show} & {c['h']} & {fmt(c['aff_vs_gbr']['rmse_r'] if c['aff_vs_gbr'] else np.nan, 2)} & "
                         f"{fmt(c['aff_vs_nnwldf']['rmse_r'], 2)} & {fmt(c['aff_vs_nnwldf']['rmse_c'], 2)} & "
                         f"{fmt(c['affldf_vs_nnwldf']['rmse_c'] if c['affldf_vs_nnwldf'] else np.nan, 2)} & "
                         f"{pr(c['aff_vs_nnwldf'])} & {pr(c['aff_vs_gbr'])} & {pr(c['affldf_vs_nnwldf'])} & {pr(c['aff_vs_shuffled'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
        out["tab_history"] = "\n".join(lines)
    # sensitivity without contemporaneous source labels (Addendum B)
    ncells = []
    for k in (5, 11):
        for h, prot in ((3, "N3"), (6, "N6"), (12, "N12")):
            # target-only references come from the matching short-history run (same target rows and test rows;
            # these learners use no source data), so only the source-based methods differ between the two runs
            a = paired_cross(d, "yrd", k, prot, AFF, f"M{h}", GBR)
            if a is None:
                continue
            b = paired_rows(d, "yrd", prot, k, AFF, "NNW")
            c = paired_cross(d, "yrd", k, prot, AFF, f"M{h}", "Best local")
            e = paired_rows(d, "yrd", f"M{h}", k, AFF, GBR)
            ncells.append({"k": k, "h": h, "gbr": a, "nnw": b, "best": c, "with": e})
    numbers["no_contemporaneous"] = ncells
    if ncells:
        lines = ["\\begin{table}[H]", "\\centering", "\\footnotesize",
                 "\\caption{Sensitivity without contemporaneous source labels (T1). As Table~\\ref{tab:history}, but every "
                 "source-based method uses source years up to 2020 only, so no method sees 2021--2022 labels; LDF cannot be composed "
                 "for the test months and NNW is used without it. Target-only learners use no source data, so the GBR and post-hoc best "
                 "references are the fits of the short-history runs (Table~\\ref{tab:history}); a refit would differ only in row order "
                 "and in feature scaling, which trees ignore, yet seeded subsampling would change a split's GBR RMSE by up to "
                 "4.9~$\\mu$g/m$^3$. Post-hoc best means the lowest test RMSE per split among all eight target-only learners: "
                 "benchmark RF and GBR plus the six additional local learners. Entries as in Table~\\ref{tab:history}.}",
                 "\\label{tab:nocontemp}", "\\setlength{\\tabcolsep}{4pt}",
                 "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
                 " & & \\multicolumn{2}{c}{Source-affine RMSE} & \\multicolumn{3}{c}{Paired RMSE reduction of source-affine, \\% (splits won)} \\\\",
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
                 "Task & $h$ & with 2021--22 source & without & vs GBR & vs NNW & vs post-hoc best \\\\", "\\midrule"]
        last = None
        for c in ncells:
            tk = f"T1, $k={c['k']}$"; show = tk if tk != last else ""
            if last is not None and tk != last:
                lines.append("\\addlinespace")
            last = tk
            pr = lambda x: "--" if x is None else f"{fmt(x['rel_mean_pct'], 1)} ({x['wins']})"
            lines.append(f"{show} & {c['h']} & {fmt(c['with']['rmse_c'] if c['with'] else np.nan, 2)} & {fmt(c['gbr']['rmse_c'], 2)} & "
                         f"{pr(c['gbr'])} & {pr(c['nnw'])} & {pr(c['best'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        out["tab_nocontemp"] = "\n".join(lines)
    # same-period paired numbers (benchmark claims and primary comparisons)
    sp = []
    for task, prot, ks in (("yrd", "B", (5, 7, 9, 11)), ("yrd", "S", (5, 7, 9, 11)), ("aus", "B", (3, 5)), ("aus", "S", (3, 5))):
        for k in ks:
            for cand, ref in ((AFF, NNWL), (AFF, GBR), (NNWL, GBR), (AFFL, NNWL), (AFF, "Best local"), (NNWL, "Best local"),
                              ("Best local", GBR), ("NNW[LDF]", "NNW"), ("GBR[LDF] (target-only)", GBR), ("KLIEP[LDF]", "KLIEP"),
                              ("KMM[LDF]", "KMM"), ("NNW (sample-weight)", "NNW"), ("NNW[LDF] (sample-weight)", NNWL),
                              (AFF, "Shuffled-source affine (control)"), ("Source-residual (ours)", NNWL), ("Source-residual (ours)", "Best local"),
                              ("Local Extra Trees", GBR), ("Local Extra Trees", NNWL)):
                r = paired_rows(d, task, prot, k, cand, ref)
                if r is not None:
                    sp.append(r)
    numbers["paired"] = sp
    # model means for every (task, protocol, k)
    mm = d.groupby(["task", "protocol", "k", "model"]).agg(r2=("r2", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"),
                                                            splits=("split", "nunique")).reset_index()
    numbers["means"] = mm.to_dict(orient="records")
    for name, txt in out.items():
        (G / f"{name}.tex").write_text(txt + "\n")
    (G / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    pd.DataFrame(sp).to_csv(G / "paired_selected.csv", index=False)
    print("wrote", list(out), "completed:", numbers["completed"])


if __name__ == "__main__":
    main()
