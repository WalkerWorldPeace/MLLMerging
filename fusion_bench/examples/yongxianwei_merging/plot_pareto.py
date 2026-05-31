"""Render the accuracy--cost Pareto figure used in tpami2026/sections/09_analysis.tex.

Numbers are taken from Table~\\ref{tab:efficiency} (wall-clock and peak GPU memory)
and from the per-benchmark accuracy tables in sections 7-8 of the manuscript.
This script writes ``tpami2026/figure/pareto.pdf``.

Re-run when any of the cited numbers change. The data dictionary below is the
single source of truth for the figure.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter


# ---- Data (single source of truth) ----------------------------------------
# Each benchmark: list of (method_name, wall_clock_seconds, accuracy_percent).
# Wall-clock from Table~\ref{tab:efficiency} (sec.~\ref{sec:efficiency}); accuracy
# from Tables~\ref{tab:clip-b32}, \ref{tab:llama}, \ref{tab:intern}, \ref{tab:qwen}.

BENCHMARKS = {
    "CLIP-ViT-B/32": {
        "iterative_label": "WUDI",
        "rows": [
            ("WUDI",   86.30, 84.63),   # T=300 from tab:wudi-iters
            ("SWUDI",   2.81, 85.55),
            ("SWUDI-A",  2.85, 85.53),
        ],
    },
    "InternVL2.5-1B": {
        "iterative_label": "OptMerge",
        "rows": [
            ("OptMerge", 552.3, 56.18),  # tab:efficiency (9.2 min, 69.0x)
            ("SWUDI",      8.0, 56.56),
            ("SWUDI-A",     8.0, 56.33),
        ],
    },
    "Qwen2-VL-7B": {
        "iterative_label": "OptMerge",
        "rows": [
            ("OptMerge", 13608.0, 62.62),  # 3.78 h
            ("SWUDI",      487.8, 61.93),
            ("SWUDI-A",     487.6, 62.72),
        ],
    },
    "Llama-3.2-3B": {
        "iterative_label": "OptMerge",
        "rows": [
            ("OptMerge", 5126.1, 43.37),
            ("SWUDI",      70.86, 44.10),
            ("SWUDI-A",     70.78, 43.46),
        ],
    },
}

# Closed-form solvers nearly tie on wall-clock. When their actual runtimes are
# visually indistinguishable on the log axis, separate them by a small symmetric
# offset whose direction follows the measured runtime: the faster solver remains
# on the left and the slower one on the right.
CLOSE_RUNTIME_RATIO = 1.08
PAIR_OFFSET = 1.035


# ---- Style ----------------------------------------------------------------
METHOD_STYLE = {
    "WUDI":     dict(color="#5B5B5B", marker="s", facecolor="white"),
    "OptMerge": dict(color="#5B5B5B", marker="s", facecolor="white"),
    "SWUDI":    dict(color="#1F77B4", marker="o", facecolor="#1F77B4"),
    "SWUDI-A":   dict(color="#D62728", marker="^", facecolor="#D62728"),
}


def _human_time(seconds: float) -> str:
    """Format seconds for x-axis ticks: '1s', '10s', '1m', '10m', '1h'."""
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds/60:g}m"
    return f"{seconds/3600:g}h"


# Human-friendly tick positions (seconds).
TICK_SECONDS = [1, 10, 60, 600, 3600, 36000]
TICK_LABELS = ["1s", "10s", "1m", "10m", "1h", "10h"]


def make_figure(out_path: Path) -> None:
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        9,
        "axes.labelsize":   9,
        "axes.titlesize":   9.5,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8.5,
        "axes.linewidth":   0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })

    n = len(BENCHMARKS)
    fig, axes = plt.subplots(
        nrows=1, ncols=n,
        figsize=(2.5 * n, 2.65),
        sharey=False,
        constrained_layout=False,
    )

    for ax, (name, info) in zip(axes, BENCHMARKS.items()):
        rows = [(str(m), float(w), float(a)) for m, w, a in info["rows"]]
        display_x = {method: wc for method, wc, _ in rows}
        closed_rows = [(method, wc) for method, wc, _ in rows if method in {"SWUDI", "SWUDI-A"}]
        if len(closed_rows) == 2:
            (m1, w1), (m2, w2) = closed_rows
            if max(w1, w2) / min(w1, w2) < CLOSE_RUNTIME_RATIO:
                faster, slower = ((m1, w1), (m2, w2)) if w1 <= w2 else ((m2, w2), (m1, w1))
                center = math.sqrt(w1 * w2)
                display_x[faster[0]] = center / PAIR_OFFSET
                display_x[slower[0]] = center * PAIR_OFFSET

        plotted = {}
        for method, wc, acc in rows:
            wc_disp = display_x[method]
            style = METHOD_STYLE[method]
            ax.plot(
                wc_disp, acc,
                marker=style["marker"],
                markersize=8.5,
                markeredgecolor=style["color"],
                markerfacecolor=style["facecolor"],
                markeredgewidth=1.1,
                linestyle="None",
                zorder=3,
            )
            plotted[method] = (wc_disp, acc, wc)

        # Pareto-improvement arrow: iterative -> SWUDI-A (or SWUDI fallback).
        # The arrow uses original (unjittered) wall-clock for the speedup factor.
        iterative_method = info["iterative_label"]
        target_method = "SWUDI-A" if "SWUDI-A" in plotted else "SWUDI"
        it_disp_x, it_y, it_wc = plotted[iterative_method]
        tg_disp_x, tg_y, tg_wc = plotted[target_method]
        speedup = it_wc / tg_wc

        ax.annotate(
            "",
            xy=(tg_disp_x, tg_y),
            xytext=(it_disp_x, it_y),
            arrowprops=dict(
                arrowstyle="->",
                color="#777777",
                linewidth=1.0,
                shrinkA=10,
                shrinkB=10,
                connectionstyle="arc3,rad=-0.22",
            ),
            zorder=2,
        )

        # Speedup callout at arrow's apex (geometric midpoint on log scale).
        x_mid = math.sqrt(it_disp_x * tg_disp_x)
        y_mid = 0.5 * (it_y + tg_y)
        ax.text(
            x_mid, y_mid + 0.32,
            rf"${speedup:.0f}\!\times$ faster",
            ha="center", va="bottom",
            color="#333333", fontsize=8.5, style="italic",
            zorder=4,
        )

        # Axes
        ax.set_xscale("log")
        ax.set_xlabel("Wall-clock time")
        ax.set_title(name, pad=4)

        # Y range: tight crop around method scores
        ys = [r[2] for r in rows]
        y_lo, y_hi = min(ys), max(ys)
        pad = max(0.55, 0.35 * (y_hi - y_lo))
        ax.set_ylim(y_lo - pad, y_hi + pad * 1.2)

        # X range: ~half-decade pad each side
        xs = [r[1] for r in rows]
        x_lo, x_hi = min(xs), max(xs)
        ax.set_xlim(x_lo / 4.0, x_hi * 4.0)

        # X ticks: human-readable units (1s, 10s, 1m, 10m, 1h, 10h)
        x_lo_lim, x_hi_lim = ax.get_xlim()
        visible = [(s, lab) for s, lab in zip(TICK_SECONDS, TICK_LABELS)
                   if x_lo_lim <= s <= x_hi_lim]
        if visible:
            tick_pos, tick_lab = zip(*visible)
            ax.xaxis.set_major_locator(FixedLocator(list(tick_pos)))
            ax.xaxis.set_major_formatter(FixedFormatter(list(tick_lab)))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", labelbottom=True, pad=2)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Grid
        ax.grid(True, which="major", axis="both", linestyle=":",
                color="#BBBBBB", linewidth=0.6, alpha=0.7, zorder=0)

    axes[0].set_ylabel("Average accuracy (%)")

    # Shared legend at top
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markeredgecolor="#5B5B5B", markerfacecolor="white",
               markersize=8.5, markeredgewidth=1.1,
               label="Iterative WUDI / OptMerge"),
        Line2D([0], [0], marker="o", linestyle="None",
               markeredgecolor="#1F77B4", markerfacecolor="#1F77B4",
               markersize=8.5, markeredgewidth=1.1,
               label="SWUDI (closed-form)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markeredgecolor="#D62728", markerfacecolor="#D62728",
               markersize=8.5, markeredgewidth=1.1,
               label="SWUDI-A (closed-form)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.5,
        columnspacing=2.4,
    )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.18, wspace=0.32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=240,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    out = repo / "tpami2026" / "figure" / "pareto.pdf"
    make_figure(out)
