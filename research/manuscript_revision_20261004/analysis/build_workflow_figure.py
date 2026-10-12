"""Render the observed-label study workflow as vector and raster figures."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "research" / "manuscript_revision_20261004" / "manuscript" / "figures"


def box(ax, center, y, width, height, title, body, color):
    x = center - width / 2
    ax.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor=color, linewidth=2.0))
    ax.text(center, y + height * .72, title, ha="center", va="center", fontsize=11.2,
            fontweight="bold", color="#17262C")
    ax.text(center, y + height * .34, body, ha="center", va="center", fontsize=8.7,
            color="#34434A", linespacing=1.35)
    return (x, y, width, height)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color="#52646B", connectionstyle="arc3,rad=0"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.4, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    b1 = box(ax, 1.9, 3.65, 3.35, 1.12, "1  Harmonise data",
             "Monthly PM2.5 targets and covariates\nKeep both country directions separate", "#3478A8")
    b2 = box(ax, 6.7, 3.65, 3.65, 1.12, "2  Fit under equal label budgets",
             "6 local + 4 source-informed learners\nsource-only and 3 shuffled-source controls", "#258878")
    b3 = box(ax, 11.35, 3.65, 3.75, 1.12, "3  Discover and freeze",
             "Development cities, 2019--2020\nChoose model and at most 2 observable rules", "#D58A36")

    box(ax, 4.0, 1.95, 4.2, 1.12, "4a  Temporal audit",
        "Same development cities, 2021--2022\nNo rule or learner reselection", "#527D9A")
    box(ax, 9.4, 1.95, 4.2, 1.12, "4b  Held-city audit",
        "Cities held out of discovery, 2021--2022\nReport coverage and shared-grid support", "#B56555")
    box(ax, 6.7, .28, 10.4, 1.0, "Joint evidence requires both audits to pass",
        "Frozen model and rule; minimum support; prespecified RMSE gain; positive clustered-bootstrap bounds\nA failed joint procedure is not proof of zero estimator-level source signal.", "#3B4C54")

    arrow(ax, (b1[0] + b1[2] + .08, 4.21), (b2[0] - .08, 4.21))
    arrow(ax, (b2[0] + b2[2] + .08, 4.21), (b3[0] - .08, 4.21))
    arrow(ax, (10.1, 3.58), (4.9, 3.1))
    arrow(ax, (12.0, 3.58), (8.5, 3.1))
    arrow(ax, (4.5, 1.88), (5.8, 1.3))
    arrow(ax, (8.9, 1.88), (7.6, 1.3))

    ax.text(6.7, .02, "Primary regulatory panel; overlapping city splits are sensitivity analyses, not independent replications.",
            ha="center", va="bottom", fontsize=8.2, color="#52646B")
    fig.tight_layout(pad=.3)
    fig.savefig(OUTPUT / "observed_label_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "observed_label_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
