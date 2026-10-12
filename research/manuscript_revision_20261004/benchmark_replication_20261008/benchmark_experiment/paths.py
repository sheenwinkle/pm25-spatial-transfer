"""Portable paths for the benchmark-aligned experiment bundle."""

import os
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("DISSERTATION_ROOT", str(EXP_ROOT.parents[3]))
).resolve()
RESULTS = Path(os.environ.get("BENCH_RESULTS_DIR", str(EXP_ROOT / "results"))).resolve()
GENERATED = Path(os.environ.get("BENCH_GENERATED_DIR", str(EXP_ROOT / "generated"))).resolve()
THESIS = Path(
    os.environ.get(
        "THESIS_COMPLETION_DIR",
        str(PROJECT_ROOT / "research" / "thesis_completion_20260912"),
    )
).resolve()
PANEL = Path(
    os.environ.get(
        "BENCH_PANEL",
        str(EXP_ROOT / "data" / "all_networks_model_panel_corrected.csv.gz"),
    )
).resolve()
MEMBER = Path(
    os.environ.get(
        "BENCH_MEMBERSHIP",
        str(
            PROJECT_ROOT
            / "research"
            / "boundary_regression_20261001"
            / "data"
            / "membership_all_panel.csv.gz"
        ),
    )
).resolve()


def require(path: Path, variable: str) -> Path:
    """Return an input path or raise an actionable configuration error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}. Set {variable} to its location."
        )
    return path
