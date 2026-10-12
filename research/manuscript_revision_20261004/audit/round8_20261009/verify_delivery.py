"""Verify round-8 manuscript/assets against generated results and archived inputs."""
import difflib
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
EXP = REV / "benchmark_replication_20261008/benchmark_experiment"
MANUSCRIPT = REV / "manuscript/main.tex"
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def digest(data):
    return hashlib.sha256(data).hexdigest()


text = MANUSCRIPT.read_text(encoding="utf-8")
generated = sorted((EXP / "generated").glob("tab_*.tex"))
generated.append(HERE / "window_summary/tab_window_sensitivity.tex")
for path in generated:
    expected = path.read_text(encoding="utf-8").strip()
    label = re.search(r"\\label\{([^}]+)\}", expected).group(1)
    loc = text.index("\\label{" + label + "}")
    start = text.rfind(r"\begin{table}", 0, loc)
    end = text.index(r"\end{table}", loc) + len(r"\end{table}")
    check("generated table exactly matches manuscript: " + path.name,
          text[start:end].strip() == expected)

for name in ("fig_k_ablation.png", "fig_history_ablation.png"):
    check("generated figure matches manuscript: " + name,
          (EXP / "generated" / name).read_bytes() == (REV / "manuscript/figures" / name).read_bytes())

labels = re.findall(r"\\label\{([^}]+)\}", text)
refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", text)
check("no duplicate labels", len(labels) == len(set(labels)))
check("all cross-references defined", not (set(refs) - set(labels)))
check("no Figure W/X/Y/Z placeholders", not re.search(r"Figure\s+[WXYZ]\b", text))
paragraphs = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"\n\s*\n", text)]
paragraphs = [s for s in paragraphs if len(s) > 160 and not re.search(r"\\(?:begin|end|item|toprule)", s)]
check("no duplicate long prose paragraphs", len(paragraphs) == len(set(paragraphs)))
check("AI writing and new analyses disclosed",
      "Declaration of Generative AI Use" in text
      and "running the post-hoc window-position, LDF-seed and clustered" in text)
check("agreed defensive sentences removed", all(s not in text for s in (
    "This declaration records actual use.",
    "These are descriptive comparisons, not audit-based deployment selection.",
    "twelve geographical replications")))
check("incorrect retention wording absent", all(s not in text for s in (
    "most of the advantage remains", "more than half remains")))

with ZipFile(EXP / "results_per_split.zip") as archive:
    names = [n for n in archive.namelist() if n.endswith(".csv")]
    same = [n for n in names if (EXP / n).is_file()
            and digest((EXP / n).read_bytes()) == digest(archive.read(n))]
check("archived result CSVs unchanged", len(names) == len(same) == 601, len(same))

validation = json.loads((HERE / "window_summary/validation.json").read_text())
check("200 complete fits and 2200 reconstructed RMSEs",
      validation["completed_fits"] == 200 and validation["recomputed_rmse"] == 2200)
check("corrected panel hash retained",
      digest((EXP / "data/all_networks_model_panel_corrected.csv.gz").read_bytes())
      == validation["panel_sha256"])
rows = json.loads((HERE / "window_summary/window_sensitivity_summary.json").read_text())
lookup = {(r["window"], r["k"], r["seed_offset"]): r for r in rows}
check("all planned summary cells present", set(lookup) == {
    (w, k, s) for w in ("2020Q4", "2020Q2", "2020Q3", "2019Q4")
    for k in (5, 11) for s in ([0, 10000] if w == "2020Q4" else [0])})
check("label counts fixed across all windows",
      all(r["train_rows_min"] == r["train_rows_max"] == 3 * r["k"] for r in rows))
check("1000 valid crossed draws for each cell",
      all(r["bootstrap_replicates"] == r["valid_replicates"] == 1000 for r in rows))
check("original-window replay intervals include zero at both budgets",
      all(lookup["2020Q4", k, 0]["ci_low_pct"] < 0 < lookup["2020Q4", k, 0]["ci_high_pct"]
          for k in (5, 11)))
check("July-September intervals above zero at both budgets",
      all(lookup["2020Q3", k, 0]["ci_low_pct"] > 0 for k in (5, 11)))
check("2019 same-season comparison reverses at both budgets",
      all(lookup["2019Q4", k, 0]["ci_high_pct"] < 0 for k in (5, 11)))
check("all tested windows retain positive point gains versus eight-model reference",
      all(r["gain_vs_posthoc_best_pct"] > 0 for r in rows))

before = (HERE / "before/manuscript/main.tex").read_text(encoding="utf-8")
(HERE / "main_round8.diff").write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile="before/manuscript/main.tex", tofile="manuscript/main.tex")), encoding="utf-8")
report = {"passed": sum(c["pass"] for c in checks), "total": len(checks),
          "main_tex_sha256": digest(MANUSCRIPT.read_bytes()), "checks": checks}
(HERE / "delivery_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
lines = [("PASS " if c["pass"] else "FAIL ") + c["name"] for c in checks]
lines.append(f"{report['passed']}/{report['total']} delivery checks pass")
(HERE / "delivery_verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if all(c["pass"] for c in checks) else 1)
