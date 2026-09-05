#!/usr/bin/env python3
"""Create the evidence-bound vector charts used by the technical report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "figures" / "figure_data.json").read_text(encoding="utf-8"))
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#6B7280"


def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=3)


def progress() -> None:
    values = DATA["official_progress"]["correct"]
    labels = ["Baseline", "Movie batch", "Complement", "Protected"]
    fig, ax = plt.subplots(figsize=(6.3, 2.1))
    x = list(range(len(values)))
    ax.plot(x, values, color=BLUE, marker="o", linewidth=2.2, markersize=5)
    ax.fill_between(x, values, [260] * len(values), color=BLUE, alpha=0.08)
    for index, value in enumerate(values):
        ax.annotate(f"{value}/538", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color="#111827")
    ax.set_xticks(x, labels)
    ax.set_ylim(255, 350)
    ax.set_ylabel("Correct predictions", fontsize=8)
    ax.set_xlabel("System stage", fontsize=8)
    style(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(ROOT / "figures" / "official_progress.pdf", bbox_inches="tight")
    plt.close(fig)


def factors() -> None:
    values = DATA["factor_audit"]["correct"]
    labels = DATA["factor_audit"]["labels"]
    fig, ax = plt.subplots(figsize=(6.3, 2.45))
    y = list(range(len(values)))
    colors = [BLUE, GREY, GREY, ORANGE, ORANGE, ORANGE]
    bars = ax.barh(y, values, color=colors, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, f"{value}/60", va="center", fontsize=8, color="#111827")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 35)
    ax.set_xlabel("Semantically correct answers", fontsize=8)
    style(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(ROOT / "figures" / "factor_audit.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    progress()
    factors()
