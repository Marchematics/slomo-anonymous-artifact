#!/usr/bin/env python3
"""Create compact, evidence-bound vector figures for the technical report.

The dashboard uses direct labels, small multiples, and restrained scientific
styling inspired by figures4papers. It is original plotting code and consumes
only the released values in ``figures/figure_data.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "figures" / "figure_data.json").read_text(encoding="utf-8"))
BLUE = "#0072B2"
SKY = "#56B4E9"
GREY = "#8A919B"
DARK = "#1F2933"
GRID = "#D9DEE5"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.4,
        "axes.titlesize": 6.8,
        "axes.labelsize": 6.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def style(ax, *, ylim: tuple[float, float], ticks: list[float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA3AF")
    ax.spines["bottom"].set_color("#9AA3AF")
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.2)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)


def annotate_bars(ax, bars, *, offset: float) -> None:
    for bar, value in zip(bars, [bar.get_height() for bar in bars]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{int(value)}",
            ha="center",
            va="bottom",
            color=DARK,
            fontsize=5.8,
            fontweight="medium",
        )


def panel_title(ax, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left", pad=3, fontweight="bold")


def progress(ax) -> None:
    values = DATA["official_progress"]["correct"]
    labels = ["Base", "Batch", "+Comp.", "Final"]
    x = list(range(len(values)))
    ax.plot(x, values, color=BLUE, marker="o", linewidth=1.5, markersize=3.6)
    ax.fill_between(x, values, [260] * len(values), color=BLUE, alpha=0.08)
    for index, value in enumerate(values):
        ax.annotate(
            str(value),
            (index, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=5.8,
            color=DARK,
            fontweight="medium",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Correct / 538")
    panel_title(ax, "a", "Development trajectory")
    style(ax, ylim=(258, 365), ticks=[270, 310, 350])


def bar_panel(
    ax,
    labels: list[str],
    values: list[int],
    colors: list[str],
    *,
    label: str,
    title: str,
    ylim: tuple[float, float],
    ticks: list[float],
    ylabel: str | None = None,
) -> None:
    x = list(range(len(values)))
    bars = ax.bar(x, values, color=colors, width=0.66, edgecolor="none")
    annotate_bars(ax, bars, offset=(ylim[1] - ylim[0]) * 0.025)
    ax.set_xticks(x, labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    panel_title(ax, label, title)
    style(ax, ylim=ylim, ticks=ticks)


def dashboard() -> None:
    factor = DATA["factor_audit"]["correct"]
    source = DATA["source_comparison"]
    fig, axes = plt.subplots(1, 5, figsize=(7.45, 1.85), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=1 / 72, wspace=0.12, hspace=0.0)

    progress(axes[0])
    bar_panel(
        axes[1],
        ["Joint", "Indep."],
        [factor[0], factor[3]],
        [BLUE, GREY],
        label="b",
        title="Joint context",
        ylim=(0, 32),
        ticks=[0, 15, 30],
        ylabel="Correct / 60",
    )
    bar_panel(
        axes[2],
        ["No\nframes", "Chrono16"],
        [factor[1], factor[0]],
        [GREY, BLUE],
        label="c",
        title="Visual evidence",
        ylim=(0, 32),
        ticks=[0, 15, 30],
    )
    bar_panel(
        axes[3],
        ["Chrono16", "Target16", "Dense48"],
        [factor[0], factor[4], factor[5]],
        [GREY, SKY, BLUE],
        label="d",
        title="Visual allocation",
        ylim=(0, 32),
        ticks=[0, 15, 30],
    )
    bar_panel(
        axes[4],
        ["Luna", "VL-F", "VL+", "3.8M", "K3"],
        source["correct"],
        [BLUE, GREY, GREY, GREY, GREY],
        label="e",
        title="Source study (141Q)",
        ylim=(0, 82),
        ticks=[0, 40, 80],
        ylabel="Correct / 141",
    )

    fig.savefig(ROOT / "figures" / "evidence_dashboard.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def architecture_box(ax, x: float, y: float, width: float, height: float, label: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=0.8,
        edgecolor="#64748B",
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=DARK, fontsize=7.2, fontweight="medium")


def architecture_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#64748B",
        )
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.45, 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    architecture_box(ax, 0.15, 0.83, 1.48, 1.2, "SF20K movie\nand all questions", "#F2F2F2")
    architecture_box(ax, 2.18, 1.72, 1.82, 0.72, "Timestamped VTT\nASR transcript", "#E8F4FB")
    architecture_box(ax, 2.18, 0.54, 1.82, 0.72, "16 centered uniform\nchronological frames", "#E8F4FB")
    architecture_box(ax, 4.62, 0.83, 2.03, 1.2, "Joint movie request\nshared evidence + all Qs", "#E8F4FB")
    architecture_box(ax, 7.22, 0.83, 1.2, 1.2, "GPT-5.6-\nLuna\nzero-shot", "#FFF2E5")
    architecture_box(ax, 8.86, 0.83, 1.02, 1.2, "JSON bank\n→ CSV", "#E9F6EE")

    architecture_arrow(ax, (1.63, 1.43), (2.18, 2.08))
    architecture_arrow(ax, (1.63, 1.43), (2.18, 0.90))
    architecture_arrow(ax, (4.0, 2.08), (4.62, 1.57))
    architecture_arrow(ax, (4.0, 0.90), (4.62, 1.29))
    architecture_arrow(ax, (6.65, 1.43), (7.22, 1.43))
    architecture_arrow(ax, (8.42, 1.43), (8.86, 1.43))

    ax.text(3.08, 2.75, "Evidence construction", ha="center", va="center", fontsize=6.8, color="#475569", fontweight="bold")
    ax.text(5.63, 2.75, "Movie-level inference", ha="center", va="center", fontsize=6.8, color="#475569", fontweight="bold")
    ax.text(9.36, 2.75, "Validated output", ha="center", va="center", fontsize=6.8, color="#475569", fontweight="bold")
    fig.savefig(ROOT / "figures" / "system_architecture.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    dashboard()
    architecture()
