"""Verify the qualitative claims made in the manuscript against the per-split results.
Prints PASS/FAIL for each claim; exits non-zero if any claim fails.
A block whose input files are missing is reported as FAIL (not silently skipped)."""
import sys
import numpy as np
import pandas as pd
from make_tables import load, with_best, GROUPS, TARGET_ONLY

d = with_best(load())
GROUP_MODELS = [m for _, grp in GROUPS for m, _ in grp]
AFF, AFFL, NL, GBR = "Source-affine (ours)", "Source-affine[LDF] (ours)", "NNW[LDF]", "GBR (target-only)"
results = []


class Missing(Exception):
    pass


def means(task, prot, k, metric="rmse"):
    g = d[(d.task == task) & (d.protocol == prot) & (d.k == k)]
    if not len(g):
        raise Missing(f"{task}-{prot}-k{k}")
    return g.groupby("model")[metric].mean()


def paired(task, prot, k, c, r):
    g = d[(d.task == task) & (d.protocol == prot) & (d.k == k)]
    if not len(g):
        raise Missing(f"{task}-{prot}-k{k}")
    w = g.pivot_table(index="split", columns="model", values="rmse")
    rel = 1 - w[c] / w[r]
    return 100 * rel.mean(), int((w[c] < w[r]).sum()), len(w)


def cross(task, k, prot_c, cand, prot_r, ref):
    """Mean paired reduction (%) of cand in protocol prot_c against ref in protocol prot_r (same splits and test rows)."""
    a = d[(d.task == task) & (d.protocol == prot_c) & (d.k == k) & (d.model == cand)].set_index("split")
    b = d[(d.task == task) & (d.protocol == prot_r) & (d.k == k) & (d.model == ref)].set_index("split")
    if not len(a) or not len(b):
        raise Missing(f"{task}-{prot_c}/{prot_r}-k{k}")
    j = a[["rmse", "n"]].join(b[["rmse", "n"]], lsuffix="_c", rsuffix="_r", how="inner")
    assert (j.n_c == j.n_r).all()
    return 100 * (1 - j.rmse_c / j.rmse_r).mean()


def claim(name, ok, detail=""):
    results.append((name, bool(ok), detail))


# T1 same-period
try:
    for k in (5, 7, 9, 11):
        m = means("yrd", "B", k); m2 = means("yrd", "B", k, "r2")
        gm = m[m.index.isin(GROUP_MODELS)]; gr = m2[m2.index.isin(GROUP_MODELS)]
        claim(f"T1-B k={k}: Extra Trees lowest RMSE and highest R2", gm.idxmin() == "Local Extra Trees" and gr.idxmax() == "Local Extra Trees",
              f"{gm.idxmin()} / {gr.idxmax()}")
        r, w, n = paired("yrd", "B", k, NL, GBR)
        claim(f"T1-B k={k}: NNW[LDF] reduces RMSE vs GBR (positive, small)", 0 < r < 2.5, f"{r:.2f}% wins {w}/{n}")
        r, w, n = paired("yrd", "B", k, "Local Extra Trees", NL)
        claim(f"T1-B k={k}: Extra Trees better than NNW[LDF] on mean RMSE", m["Local Extra Trees"] < m[NL], f"paired {r:.2f}%")
    for k in (5, 11):
        r, w, n = paired("yrd", "B", k, AFF, NL)
        claim(f"T1-B k={k}: source-affine wins no split vs NNW[LDF]", w == 0, f"{r:.2f}% wins {w}")
        r2, _, _ = paired("yrd", "B", k, NL, "NNW")
        claim(f"T1-B k={k}: LDF reduces NNW RMSE (positive)", r2 > 0, f"{r2:.2f}%")
    vals = {k: round(paired("yrd", "B", k, NL, GBR)[0], 1) for k in (5, 7, 9, 11)}
    claim("T1-B: NNW[LDF] vs GBR max over k equals k=5 value (abstract 'at most')", max(vals.values()) == vals[5], str(vals))
    claim("T1-B: NNW[LDF] vs GBR min over k equals k=11 value (conclusion range)", min(vals.values()) == vals[11], str(vals))
    m = means("yrd", "B", 5)
    claim("T1-B k=5: residual and affine[LDF] between NNW[LDF] and affine", m[NL] < m["Source-residual (ours)"] < m[AFF] and m[NL] < m[AFFL] < m[AFF])
    for k in (5, 7, 9, 11):
        m = means("yrd", "B", k)
        for x in ("KLIEP", "KMM"):
            if x in m:
                claim(f"T1-B k={k}: {x} worse than GBR", m[x] > m[GBR], f"{m[x]:.2f} vs {m[GBR]:.2f}")
except Missing as e:
    results.append(('SKIPPED block: ' + '# T1 same-period' + ' (missing ' + str(e) + ')', False, 'input missing; block not checked'))

# T1 later period
try:
    for k in (5, 7, 9, 11):
        m = means("yrd", "S", k); gm = m[m.index.isin(GROUP_MODELS)]
        claim(f"T1-S k={k}: NNW[LDF] lowest RMSE", gm.idxmin() == NL, gm.idxmin())
        locs = ["Local mean", "Local ridge", "Local RBF", "Local RF", "Local Extra Trees", "Local LightGBM"]
        claim(f"T1-S k={k}: all six local learners worse than GBR", all(m[l] > m[GBR] for l in locs))
    r5 = paired("yrd", "S", 5, AFF, GBR)[0]; r11 = paired("yrd", "S", 11, AFF, GBR)[0]
    claim("T1-S: affine better than GBR at k=5, worse at k=11", r5 > 0 and r11 < 0, f"{r5:.2f} / {r11:.2f}")
except Missing as e:
    results.append(('SKIPPED block: ' + '# T1 later period' + ' (missing ' + str(e) + ')', False, 'input missing; block not checked'))

# T1 short history
try:
    for k in (5, 11):
        for h in ("M3", "M6", "M12"):
            r, w, n = paired("yrd", h, k, AFF, GBR)
            claim(f"T1-{h} k={k}: affine better than GBR", r > 0, f"{r:.2f}% ({w}/{n})")
        seq = [means("yrd", p, k)[NL] for p in (("M3", "M6", "M12", "S") if k == 5 else ("M3", "M6", "M12"))]
        claim(f"T1 k={k}: NNW[LDF] RMSE decreases with h (k=5: through h=72; k=11: through h=12)", all(a > b for a, b in zip(seq, seq[1:])), str(np.round(seq, 3)))
        r, w, n = paired("yrd", "M3", k, AFF, "NNW[LDF] (sample-weight)")
        claim(f"T1-M3 k={k}: affine beats NNW[LDF]-sw", r > 0, f"{r:.2f}% ({w}/{n})")
        m = means("yrd", "M3", k); gm = m[m.index.isin(GROUP_MODELS)]
        claim(f"T1-M3 k={k}: affine[LDF] lowest RMSE of all models", gm.idxmin() == AFFL, gm.idxmin())
        r, _, _ = paired("yrd", "M3", k, NL, "NNW")
        claim(f"T1-M3 k={k}: LDF reduces NNW RMSE", r > 0, f"{r:.2f}%")
        for h in ("M12",):
            r, w, n = paired("yrd", h, k, AFF, NL)
            claim(f"T1-{h} k={k}: NNW[LDF] at least as accurate as affine", r <= 0, f"{r:.2f}%")
        r, w, n = paired("yrd", "S", k, AFF, NL)
        claim(f"T1-S k={k}: NNW[LDF] better than affine", r < 0, f"{r:.2f}%")
    r3 = paired("yrd", "M3", 5, NL, GBR)[0]
    claim("T1-M3 k=5: NNW[LDF] improves on GBR", r3 > 0, f"{r3:.2f}%")
except Missing as e:
    results.append(('SKIPPED block: ' + '# T1 short history' + ' (missing ' + str(e) + ')', False, 'input missing; block not checked'))

# sensitivity without contemporaneous source labels
try:
    for k in (5, 11):
        # target-only references are the short-history (M) fits: they use no source data (see make_tables.paired_cross)
        with_g = paired("yrd", "M3", k, AFF, GBR)[0]; without_g = cross("yrd", k, "N3", AFF, "M3", GBR)
        with_b = paired("yrd", "M3", k, AFF, "Best local")[0]; without_b = cross("yrd", k, "N3", AFF, "M3", "Best local")
        claim(f"N3 k={k}: positive gain remains against GBR and the eight-model per-split post-hoc best",
              without_g > 0 and without_b > 0, f"GBR {with_g:.1f}->{without_g:.1f}; post-hoc best {with_b:.1f}->{without_b:.1f}")
        expected = {5: (19.6, 66), 11: (6.1, 44)}[k]
        claim(f"N3 k={k}: corrected post-hoc-best gain and retained fraction",
              round(without_b, 1) == expected[0] and round(100 * without_b / with_b) == expected[1],
              f"gain {without_b:.1f}%; retained {100 * without_b / with_b:.1f}%")
        claim(f"N3 k={k}: affine RMSE rises without contemporaneous labels", means("yrd", "N3", k)[AFF] > means("yrd", "M3", k)[AFF])
        claim(f"N6 k={k}: advantage vs GBR smaller than at h=3", cross("yrd", k, "N6", AFF, "M6", GBR) < cross("yrd", k, "N3", AFF, "M3", GBR))
except Missing as e:
    results.append(('SKIPPED block: ' + '# sensitivity without contemporaneous source labels' + ' (missing ' + str(e) + ')', False, 'input missing; block not checked'))

# T2
try:
    for prot, ks in (("B", (3, 5)), ("S", (3, 5)), ("M3", (3, 5)), ("M6", (3, 5)), ("M12", (3, 5))):
        for k in ks:
            m = means("aus", prot, k)
            if NL not in m:
                continue
            best_to = m[[x for x in TARGET_ONLY if x in m]].min()
            claim(f"T2-{prot} k={k}: NNW[LDF] RMSE > 1.3 x best target-only and > GBR", m[NL] > 1.3 * best_to and m[NL] > m[GBR],
                  f"NNW[LDF] {m[NL]:.2f}, best target-only {best_to:.2f}")
    for k in (3, 5):
        m = means("aus", "B", k); gm = m[m.index.isin(GROUP_MODELS)]
        claim(f"T2-B k={k}: best model is target-only", gm.idxmin() in TARGET_ONLY, gm.idxmin())
    m = means("aus", "S", 3); r, w, n = paired("aus", "S", 3, AFF, "Local Extra Trees")
    claim("T2-S k=3: affine level with Extra Trees (best target-only), paired <= 0, wins <= 10",
          m[TARGET_ONLY].idxmin() == "Local Extra Trees" and r <= 0 and w <= 10, f"{r:.1f}% wins {w}")
    m = means("aus", "S", 5); gm = m[m.index.isin(GROUP_MODELS)]
    claim("T2-S k=5: target-only GBR best; affine worse", gm.idxmin() == GBR and m[AFF] > m[GBR], gm.idxmin())
    for k in (3, 5):
        m2 = means("aus", "S", k, "r2"); gm2 = m2[m2.index.isin(GROUP_MODELS)]
        claim(f"T2-S k={k}: every method negative mean R2", (gm2 < 0).all(), f"max {gm2.max():.3f} ({gm2.idxmax()})")
        m = means("aus", "M3", k)
        claim(f"T2-M3 k={k}: local RBF better than affine and is best target-only", m["Local RBF"] < m[AFF] and m[TARGET_ONLY].idxmin() == "Local RBF",
              f"{m[TARGET_ONLY].idxmin()}")
        m = means("aus", "M6", k); gm = m[m.index.isin(GROUP_MODELS)]
        claim(f"T2-M6 k={k}: affine lowest of all models; RBF best target-only", gm.idxmin() == AFF and m[TARGET_ONLY].idxmin() == "Local RBF",
              f"{gm.idxmin()} / {m[TARGET_ONLY].idxmin()}")
        m = means("aus", "M12", k)
        claim(f"T2-M12 k={k}: local mean best target-only; kernel below affine; affine below local mean",
              m[TARGET_ONLY].idxmin() == "Local mean" and m["Source-kernel (ours)"] < m[AFF] < m["Local mean"], f"{m[TARGET_ONLY].idxmin()}")
        for prot in ("B", "S", "M3"):
            m = means("aus", prot, k); bt = m[TARGET_ONLY].idxmin()
            r, w, n = paired("aus", prot, k, AFF, bt)
            claim(f"T2-{prot} k={k}: affine does not improve on best target-only (paired <= 0, wins <= 10)",
                  r <= 0 and w <= 10, f"{r:.1f}% wins {w} vs {bt}")
    a3 = round(paired("aus", "M12", 3, AFF, "Local mean")[0], 1); a5 = round(paired("aus", "M12", 5, AFF, "Local mean")[0], 1)
    a6 = [round(paired("aus", "M6", k, AFF, "Local RBF")[0], 1) for k in (3, 5)]
    claim("T2: largest margin over best target-only is the M12 k=5 value ('at most')", a5 >= max([a3] + a6), f"M12 {a3}/{a5}, M6 {a6}")

except Missing as e:
    results.append(('SKIPPED block: ' + '# T2' + ' (missing ' + str(e) + ')', False, 'input missing; block not checked'))

try:
    ok = all(means("aus", h, k)[AFFL] > means("aus", h, k)[AFF] for h in ("M3", "M6", "M12") for k in (3, 5))
    claim("T2 short histories: affine[LDF] worse than affine at every budget", ok)
except Missing as e:
    results.append(("SKIPPED T2 affine[LDF] check (missing " + str(e) + ")", False, "input missing; block not checked"))

SOURCE_USING = ["NNW", "NNW[LDF]", "NNW (sample-weight)", "NNW[LDF] (sample-weight)", "KLIEP", "KLIEP[LDF]", "KMM", "KMM[LDF]",
                "GBR[LDF] (target-only)", AFF, AFFL, "Source-residual (ours)", "Source-kernel (ours)"]
try:
    for prot in ("S", "M3", "M6", "M12"):
        for k in ((5, 7, 9, 11) if prot == "S" else (5, 11)):
            m = means("yrd", prot, k); gm = m[m.index.isin(GROUP_MODELS)]
            claim(f"T1-{prot} k={k}: most accurate model uses source information", gm.idxmin() in SOURCE_USING, gm.idxmin())
    for k in (5, 11):
        m = means("yrd", "M3", k)
        claim(f"T1-M3 k={k}: recalibration (alone or with LDF) most accurate", m[m.index.isin(GROUP_MODELS)].idxmin() in (AFF, AFFL))
        for prot in ("M12", "S"):
            m = means("yrd", prot, k)
            claim(f"T1-{prot} k={k}: NNW[LDF] at least as accurate as source-affine", m[NL] <= m[AFF], f"{m[NL]:.2f} vs {m[AFF]:.2f}")
    vals = [paired("yrd", "B", k, NL, GBR)[0] for k in (5, 7, 9, 11)]
    claim("T1-B: NNW[LDF] improves on GBR by less than 2% at every k", max(vals) < 2, str(np.round(vals, 2)))
except Missing as e:
    results.append(("SKIPPED source-information checks (missing " + str(e) + ")", False, "input missing; block not checked"))

try:
    cells = [(h, k) for h in ("M3", "M6", "M12") for k in (5, 11)]
    best = [means("yrd", h, k)[lambda x: x.index.isin(GROUP_MODELS)].idxmin() for h, k in cells]
    claim("T1 short history: affine[LDF] most accurate in 5 of 6 cells (NNW[LDF] in M12 k=11)",
          sum(b == AFFL for b in best) == 5 and best[cells.index(("M12", 11))] == NL, str(best))
    flips = []
    for h, k in (("M6", 5), ("M6", 11), ("M12", 5), ("M12", 11)):
        a = paired("yrd", h, k, AFF, NL)[0]; b = paired("yrd", h, k, AFF, "NNW[LDF] (sample-weight)")[0]
        flips.append(((h, k), np.sign(a) != np.sign(b)))
    claim("T1: sign flips with weighting mode exactly at (M6,11) and (M12,5)", [c for c, f in flips if f] == [("M6", 11), ("M12", 5)], str(flips))
    claim("T1-B k=5: sample weights raise RMSE for NNW and NNW[LDF]",
          paired("yrd", "B", 5, "NNW (sample-weight)", "NNW")[0] < 0 and paired("yrd", "B", 5, "NNW[LDF] (sample-weight)", NL)[0] < 0)
    seq = [means("yrd", p_, 5)[AFF] for p_ in ("M3", "M6", "M12", "S")]
    claim("T1 k=5: affine changes little to 12 months (<5% spread) and is worse with full record",
          (max(seq[:3]) - min(seq[:3])) / min(seq[:3]) < 0.05 and seq[3] > max(seq[:3]), str(np.round(seq, 2)))
    for k in (5, 11):
        def xb(pa, ca, pb, cb):
            A = d[(d.task == "yrd") & (d.protocol == pa) & (d.k == k) & (d.model == ca)].set_index("split")
            B = d[(d.task == "yrd") & (d.protocol == pb) & (d.k == k) & (d.model == cb)].set_index("split")
            return 100 * (1 - A.rmse / B.rmse).mean()
        claim(f"k={k}: 3-month affine beats best local trained on 12 months and on full record",
              xb("M3", AFF, "M12", "Best local") > 0 and xb("M3", AFF, "S", "Best local") > 0)
        claim(f"k={k}: without contemporaneous labels, comparison with full-record eight-model post-hoc best",
              round(xb("N3", AFF, "S", "Best local"), 1) == {5: -3.8, 11: 0.9}[k],
              f"{xb('N3', AFF, 'S', 'Best local'):.1f}%")
    lowers = sum(means("aus", p_, k)[NL] < means("aus", p_, k)["NNW"] for p_ in ("B", "S", "M3", "M6", "M12") for k in (3, 5))
    claim("T2: LDF lowers NNW's RMSE in most (>=6/10) cells", lowers >= 6, f"{lowers}/10")
    for p_ in ("B", "S", "M3", "M6", "M12"):
        for k in (3, 5):
            m = means("aus", p_, k)
            claim(f"T2-{p_} k={k}: plain NNW also far worse than best target-only (>1.3x)", m["NNW"] > 1.3 * m[TARGET_ONLY].min())
    kk = [round(paired("aus", "M12", k, "Source-kernel (ours)", "Local mean")[0], 1) for k in (3, 5)]
    claim("T2-M12: source-kernel beats local mean by more than affine; k=5 value is the max ('up to')",
          all(kk[i] > paired("aus", "M12", k, AFF, "Local mean")[0] for i, k in enumerate((3, 5))) and kk[1] >= kk[0], str(kk))
    m6 = [paired("aus", "M6", k, "Source-kernel (ours)", "Local RBF")[0] for k in (3, 5)]
    claim("T2-M6: source-kernel not better than RBF (so 'source-kernel up to' refers to M12)", max(m6) < kk[1], str(np.round(m6, 1)))
    vals = [paired("aus", "M6", 3, AFF, "Local RBF")[0], paired("aus", "M6", 5, AFF, "Local RBF")[0],
            paired("aus", "M12", 3, AFF, "Local mean")[0], paired("aus", "M12", 5, AFF, "Local mean")[0]]
    claim("T2: 1.2-5.3 range endpoints are min (M6 k=3) and max (M12 k=5)", min(vals) == vals[0] and max(vals) == vals[3], str(np.round(vals, 1)))
    rom = [paired_rom for paired_rom in [100 * (1 - means("yrd", "B", k)[NL] / means("yrd", "B", k)[GBR]) for k in (5, 7, 9, 11)]]
    claim("T1-B: ratio-of-means range endpoints are k=11 (min) and k=5 (max)", min(rom) == rom[3] and max(rom) == rom[0], str(np.round(rom, 2)))
except Missing as e:
    results.append(("SKIPPED review-round checks (missing " + str(e) + ")", False, "input missing; block not checked"))

try:
    claim("T1-B k=5: LDF lowers KMM's RMSE (paired > 0)", paired("yrd", "B", 5, "KMM[LDF]", "KMM")[0] > 0)
    claim("T1-B k=5: LDF raises KLIEP's RMSE (paired < 0)", paired("yrd", "B", 5, "KLIEP[LDF]", "KLIEP")[0] < 0)
    for k in (5, 7, 9, 11):
        m = means("yrd", "B", k)
        claim(f"T1-B k={k}: KLIEP, KLIEP[LDF], KMM, KMM[LDF] all worse than GBR",
              all(m[x] > m[GBR] for x in ("KLIEP", "KLIEP[LDF]", "KMM", "KMM[LDF]")))
except Missing as e:
    results.append(("SKIPPED KMM checks (missing " + str(e) + ")", False, "input missing; block not checked"))

bad = [r for r in results if not r[1]]
for name, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
print(f"\n{len(results) - len(bad)}/{len(results)} claims pass")
sys.exit(1 if bad else 0)
