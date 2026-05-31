"""Generate publication-quality figures for the paper.

Reads JSON outputs from outputs/yongxianwei_merging/theory_diagnostics/* and
renders seven figures (PDF + PNG) into the same directory:

    figures/figure_A_theory_chain.pdf       (Fig 1: closed-form vs real interference)
    figures/figure_B_optimizer.pdf          (Fig 2: Adam = implicit spectral filter)
    figures/figure_C_spectral_rank.pdf      (Fig 3: noise amplification + adaptive rank)
    figures/figure_S1_filter_risk.pdf       (App: filter / risk / boundary rho, §10.5)
    figures/figure_S2_rank_diagnostics.pdf  (App: rank by kind / fit traj / capture-gap)
    figures/figure_S3_diagnostics_1.pdf     (App: input-subspace + SGD/Landweber identity)
    figures/figure_S4_diagnostics_2.pdf     (App: per-layer R² heatmap + K-rule scatter)

Figs 1/2/3 were tightened in the manuscript revision: Fig 1 stays 1×2 horizontal at
``figsize=(6.2, 2.35)``; Figs 2 and 3 are 2×1 single-column at ``(3.25, 3.45)``
and ``(3.25, 3.35)`` respectively. Diagnostic panels removed from Figs 1/2/3
were split across two horizontal appendix figures (S3 and S4) so neither needs
aggressive scaling on the page.

Design choices:
- No suptitles; figure-level explanation lives in the LaTeX caption.
- Figs 2/3 avoid in-frame legends in favour of direct in-place labels.
- Larger fonts in legends and annotations are deliberately avoided to keep the
  single-column figures readable at native size.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repository roots
import os as _os
_REPO_ROOT = Path(_os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3])))
THEORY = _REPO_ROOT / "outputs" / "yongxianwei_merging" / "theory_diagnostics"
FIG_DIR = THEORY / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style — tightened for double-column journal:
#   - main-text figures should survive width<=0.84\textwidth or \columnwidth
#   - smaller fonts; thinner lines; shorter titles; minimal in-figure text
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "lines.linewidth": 1.6,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,  # editable vector text in Illustrator
})

C_TASKVEC = "#d62728"
C_HEAD = "#1f77b4"

C_METHOD = {
    "sum": "#7f7f7f",
    "closed_form": "#d62728",
    "iwudi_t10": "#ff9896",
    "iwudi_t100": "#ff7f0e",
    "iwudi_t300": "#bcbd22",
    "swudi_r0_65": "#2ca02c",
    "aswudi_sqrt": "#1f77b4",
    "wiener": "#9467bd",
    "drop_all": "#000000",
}
LABEL_METHOD = {
    "sum": "Task-arith.",
    "closed_form": "Closed-form",
    "iwudi_t10": "IWUDI $t{=}10$",
    "iwudi_t100": "IWUDI $t{=}100$",
    "iwudi_t300": "IWUDI $t{=}300$",
    "swudi_r0_65": "SWUDI $r{=}0.65$",
    "aswudi_sqrt": "SWUDI-A $\\sqrt{\\lambda}$",
    "wiener": "Wiener",
    "drop_all": "Drop-all",
}


def _save(fig, name: str):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {pdf}")
    print(f"[fig] wrote {png}")


def _smart_annotate_points(ax, points, labels, colors, *, fontsize=6.6,
                           reserved=(), min_radius=24, max_radius=108):
    """Place point labels with a lightweight repulsion / blank-space heuristic.

    Candidate text positions are generated in display space around each point.
    They are scored by (i) staying inside the axes, (ii) avoiding markers,
    already placed labels and reserved text regions, (iii) preferring the central
    blank area, and (iv) keeping connector lines short. The final labels are
    drawn with thin guide lines so they can occupy open space rather than the
    crowded corners of the panel.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer).expanded(0.96, 0.90)
    dpi_scale = fig.dpi / 72.0

    point_boxes = []
    for x, y in points.values():
        px, py = ax.transData.transform((x, y))
        point_boxes.append(matplotlib.transforms.Bbox.from_bounds(px - 5, py - 5, 10, 10))

    def approx_box(cx, cy, text):
        lines = text.split("\n")
        width = max(len(s) for s in lines) * fontsize * dpi_scale * 0.55 + 5
        height = len(lines) * fontsize * dpi_scale * 1.20 + 4
        return matplotlib.transforms.Bbox.from_bounds(cx - width / 2, cy - height / 2, width, height)

    placed = list(reserved)
    center = np.array(ax_box.get_points()).mean(axis=0)
    angles = np.deg2rad([0, 25, 45, 70, 110, 135, 160, 200, 225, 250, 290, 315, 340])
    radii = np.linspace(min_radius, max_radius, 5)

    # Place crowded lower-left labels first, then closed-form and task arithmetic.
    order = sorted(points, key=lambda k: (points[k][1], points[k][0]))
    for key in order:
        x, y = points[key]
        p = np.array(ax.transData.transform((x, y)))
        best = None
        for r in radii:
            for a in angles:
                q = p + r * np.array([np.cos(a), np.sin(a)])
                box = approx_box(q[0], q[1], labels[key])
                if not ax_box.contains(*box.get_points()[0]) or not ax_box.contains(*box.get_points()[1]):
                    continue
                overlap = 0.0
                for b in point_boxes + placed:
                    inter = matplotlib.transforms.Bbox.intersection(box, b)
                    if inter is not None:
                        overlap += inter.width * inter.height
                # Prefer central blank space but avoid very long connectors.
                center_penalty = 0.003 * np.linalg.norm(q - center)
                length_penalty = 0.015 * r
                score = overlap + center_penalty + length_penalty
                if best is None or score < best[0]:
                    best = (score, q, box)
        if best is None:
            # Conservative fallback: upper-right offset in data coordinates.
            tx, ty = x * 1.5, y * 1.25
            box = approx_box(*ax.transData.transform((tx, ty)), labels[key])
        else:
            _, q, box = best
            tx, ty = ax.transData.inverted().transform(q)
        placed.append(box)
        ax.annotate(
            labels[key], xy=(x, y), xytext=(tx, ty), textcoords="data",
            color=colors[key], fontsize=fontsize, weight="semibold",
            ha="center", va="center", zorder=8,
            arrowprops=dict(arrowstyle="-", color=colors[key], lw=0.65,
                            alpha=0.9, shrinkA=2.0, shrinkB=3.5,
                            connectionstyle="arc3,rad=0.08"),
        )


# ---------------------------------------------------------------------------
# Figure A — Theory chain (§10.1 / §10.7 / §10.2)
# ---------------------------------------------------------------------------

def figure_A():
    """Figure 1: Why the exact pseudoinverse is insufficient.

    1x2 layout sized for ``width=0.84\\textwidth`` (2-col):
      (a) Cumulative WUDI proxy reduction vs retained rank.
      (b) Layer-median real interference vs WUDI proxy across method classes.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(6.2, 2.35),
        gridspec_kw={"width_ratios": [1.15, 0.95], "wspace": 0.38},
    )
    ax_proxy, ax_scatter = axes

    # ---- (a) Cumulative proxy reduction --------------------------------
    rows = json.loads((THEORY / "exp_10_7_contribution/per_layer.json").read_text())
    K_target_grid = np.linspace(0.0, 1.0, 41)
    norm_curves = []
    K_aswudi_ratios = []
    for r in rows:
        d = r["shape"][1]
        K = np.array(r["proxy_curve_K"], dtype=float) / d
        P = np.array(r["proxy_curve_P"], dtype=float)
        P_zero = r["proxy_zero"]
        if P_zero <= 0:
            continue
        order = np.argsort(K)
        norm_curves.append(np.interp(K_target_grid, K[order], (P / P_zero)[order]))
        K_aswudi_ratios.append(r["K_aswudi"] / d)
    norm_curves = np.array(norm_curves)
    median = np.median(norm_curves, axis=0)
    p25 = np.percentile(norm_curves, 25, axis=0)
    p75 = np.percentile(norm_curves, 75, axis=0)
    ax_proxy.fill_between(K_target_grid, p25, p75, color=C_HEAD, alpha=0.20,
                          label=r"25--75\% band")
    ax_proxy.plot(K_target_grid, median, "-", color=C_HEAD, linewidth=1.8,
                  label="median (72 layers)")
    K_asw_med = float(np.median(K_aswudi_ratios))
    # Red dashed line + in-line label (no legend entry — annotation carries it)
    ax_proxy.axvline(K_asw_med, color=C_TASKVEC, ls="--", lw=1.2)
    ax_proxy.text(K_asw_med + 0.03, 0.27,
                  "SWUDI-A cut\n" + r"retains $\geq 98\%$",
                  fontsize=7, weight="semibold", color="black",
                  bbox=dict(facecolor="white", edgecolor=C_TASKVEC,
                            boxstyle="round,pad=0.18", alpha=0.9))
    ax_proxy.set_xlabel(r"retained rank ratio $K/d$")
    ax_proxy.set_ylabel(r"remaining WUDI proxy")
    ax_proxy.set_yscale("log")
    ax_proxy.set_xlim(0, 1)
    ax_proxy.set_ylim(1e-4, 1.2)
    ax_proxy.set_title("(a) Head directions explain proxy reduction")
    ax_proxy.legend(loc="lower left", framealpha=0.85, fontsize=6.5,
                    handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
                    handlelength=1.4)

    # ---- (b) proxy vs real interference (layer-medianed) ---------------
    blob = json.loads((THEORY / "exp_10_2_proxy_calibration/real_subset.json").read_text())
    layer_keys = list(blob["per_layer"].keys())
    method_keys = blob["method_keys"]
    keep_methods = [m for m in ("sum", "closed_form", "iwudi_t300", "aswudi_sqrt")
                    if m in method_keys]
    short_label = {
        "sum": "Task Arithmetic",
        "closed_form": "Closed-form",
        "iwudi_t300": "Iterative",
        "aswudi_sqrt": "SWUDI/SWUDI-A",
    }
    method_P, method_I = {}, {}
    for k in keep_methods:
        Ps = [blob["per_layer"][ln]["proxies"][k] for ln in layer_keys]
        Is = [blob["per_layer"][ln]["real_interference"][k] for ln in layer_keys]
        method_P[k] = float(np.median(Ps))
        method_I[k] = float(np.median(Is))
    for k in keep_methods:
        ax_scatter.scatter(method_P[k], method_I[k], color=C_METHOD[k],
                           marker="o", s=46, edgecolor="black", linewidth=0.5,
                           zorder=5)

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(r"WUDI proxy $\mathcal{P}(\tau)$")
    ax_scatter.set_ylabel(r"real interference $\hat I(\tau)$")
    ax_scatter.set_title("(b) Proxy minimum is not real optimum")

    # Reserve the axis-level guidance text first so method labels do not cover it.
    lower_txt = ax_scatter.text(0.66, 0.075, "lower is better",
                                transform=ax_scatter.transAxes, fontsize=6.3,
                                color="0.35", style="italic",
                                ha="center", va="center", zorder=7)

    reserved_boxes = []
    fig.canvas.draw()
    reserved_boxes.append(lower_txt.get_window_extent(fig.canvas.get_renderer()).expanded(1.10, 1.25))

    # Keep the proxy-minimum callout fixed near the central blank region and
    # reserve it before solving the method-label placement problem.
    if "closed_form" in method_P:
        proxy_ann = ax_scatter.annotate(
            "proxy\nminimum",
            xy=(method_P["closed_form"], method_I["closed_form"]),
            xytext=(method_P["closed_form"] * 1.35,
                    method_I["closed_form"] * 1.55),
            fontsize=6.3, weight="semibold",
            ha="center", va="bottom", zorder=8,
            arrowprops=dict(arrowstyle="->", color=C_METHOD["closed_form"],
                            lw=0.85, shrinkA=1.5, shrinkB=2.5))
        fig.canvas.draw()
        reserved_boxes.append(proxy_ann.get_window_extent(fig.canvas.get_renderer()).expanded(1.05, 1.20))

    # Smart anti-overlap labels: candidate positions are scored in display space
    # and connected back to points with guide lines.
    _smart_annotate_points(
        ax_scatter,
        {k: (method_P[k], method_I[k]) for k in keep_methods},
        short_label,
        {k: C_METHOD[k] for k in keep_methods},
        fontsize=6.4,
        reserved=reserved_boxes,
        min_radius=28,
        max_radius=118,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.20,
                        wspace=0.38)
    _save(fig, "figure_A_theory_chain")


# ---------------------------------------------------------------------------
# Figure B — Optimizer trajectory (§10.3)
# ---------------------------------------------------------------------------

def figure_B():
    """Figure 2: Iterative optimization is implicit spectral filtering.

    2x1 vertical layout, sized for ``width=0.88--0.92\\columnwidth``:
      (a) Adam empirical filter at three checkpoints overlaid on the
          fitted exponential filter.
      (b) Median (across layers) filter-fit R^2 vs optimization step.
    """
    data = json.loads((THEORY / "exp_10_3_optimizer/per_layer.json").read_text())
    layer_names = list(data.keys())
    target = "vision_model.encoder.layers.4.mlp.fc1.weight"
    if target not in layer_names:
        target = layer_names[len(layer_names) // 2]

    fig, axes = plt.subplots(2, 1, figsize=(3.25, 3.45))
    ax_adm, ax_traj = axes

    blob = data[target]
    lam = np.array(blob["spectra"]["lam"])
    keep = min(len(lam), 256)
    lam_keep = lam[:keep]

    # ---- (a) Adam empirical filter (3 checkpoints) ----------------------
    adam = blob["trajectories"]["adam"]
    desired_steps = [10, 50, 300]
    plot_steps_a = [s for s in desired_steps if s in adam["steps"]]
    if not plot_steps_a:
        plot_steps_a = adam["steps"][:3]
    cmap_a = plt.cm.plasma
    step_handles_a = []
    for i, step in enumerate(plot_steps_a):
        idx = adam["steps"].index(step)
        h_emp = np.array(adam["h_emp"][idx])
        t_eff = adam["fit"][idx].get("t_eff")
        if t_eff is None or math.isnan(t_eff):
            continue
        h_theo = 1 - np.exp(-t_eff * np.clip(lam_keep, 0, None))
        color = cmap_a(0.20 + 0.65 * i / max(len(plot_steps_a) - 1, 1))
        ax_adm.scatter(lam_keep, h_emp, color=color, s=6, alpha=0.45,
                       edgecolor="none")
        ax_adm.plot(np.sort(lam_keep), h_theo[np.argsort(lam_keep)],
                    "-", color=color, lw=1.5, alpha=0.95)
        step_handles_a.append(plt.Line2D([0], [0], color=color, lw=1.6,
                                          label=f"step {step}"))
    # Single legend (Adam step). Empirical-vs-fit distinction explained in caption.
    ax_adm.legend(handles=step_handles_a,
                  loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  fontsize=6.0, framealpha=0.85, title="Adam step",
                  title_fontsize=6.2, handletextpad=0.4, handlelength=1.4,
                  borderpad=0.25, labelspacing=0.25)
    ax_adm.set_xscale("log")
    ax_adm.set_xlabel(r"eigenvalue $\lambda_k$")
    ax_adm.set_ylabel(r"empirical filter $\hat h_{k,n}$")
    ax_adm.set_ylim(-0.05, 1.05)
    ax_adm.set_title("(a) Adam behaves as a spectral filter")

    # ---- (b) Median filter-fit R^2 vs step (Adam, IQR band) -------------
    common_steps = set(data[layer_names[0]]["trajectories"]["adam"]["steps"])
    for ln in layer_names[1:]:
        common_steps &= set(data[ln]["trajectories"]["adam"]["steps"])
    common_steps = sorted(s for s in common_steps if s > 0)

    R2_matrix = np.zeros((len(layer_names), len(common_steps)))
    for i, ln in enumerate(layer_names):
        adm_layer = data[ln]["trajectories"]["adam"]
        for j, step in enumerate(common_steps):
            adm_idx = adm_layer["steps"].index(step)
            R2_matrix[i, j] = adm_layer["fit"][adm_idx]["r2"]
    R2_med = np.median(R2_matrix, axis=0)
    R2_p25 = np.percentile(R2_matrix, 25, axis=0)
    R2_p75 = np.percentile(R2_matrix, 75, axis=0)

    ax_traj.fill_between(common_steps, R2_p25, R2_p75, color=C_HEAD, alpha=0.22)
    ax_traj.plot(common_steps, R2_med, "-o", color=C_HEAD, lw=1.6, markersize=3)
    ax_traj.axhline(0.9, color=C_TASKVEC, ls="--", lw=1.0)
    if 50 in common_steps:
        ax_traj.axvline(50, color="0.5", ls=":", lw=0.9)
        ax_traj.text(55, 0.30, r"$\sim 50$ steps",
                     fontsize=6.8, ha="left", va="center")
    # Tag the R^2=0.9 dashed line directly so we don't need a legend.
    ax_traj.text(common_steps[-1] * 0.7, 0.94, r"$R^2 = 0.9$",
                 color=C_TASKVEC, fontsize=6.8, ha="right", va="bottom")
    ax_traj.set_xscale("log")
    ax_traj.set_xlim(left=max(1, common_steps[0]))
    ax_traj.set_xlabel("optimization step")
    ax_traj.set_ylabel(r"filter-fit $R^2$ (Adam)")
    ax_traj.set_ylim(-0.10, 1.05)
    ax_traj.set_title("(b) Filter fit improves after early steps")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, top=0.94, bottom=0.12)
    _save(fig, "figure_B_optimizer")


# ---------------------------------------------------------------------------
# Figure C — Spectral noise + rank rules (§10.4 + §10.6)
# ---------------------------------------------------------------------------

def figure_C():
    """Figure 3: Noise amplification and adaptive rank selection.

    2x1 vertical layout, sized for ``width=0.88--0.92\\columnwidth``:
      (a) Amplified noise scale vs eigenvalue (binned median + IQR band).
      (b) Two CLIP layer spectra with SWUDI-A rank-rule cuts overlaid.
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 3.35),
                             gridspec_kw={"height_ratios": [1.05, 0.95]})
    ax_amp, ax_panel = axes

    # ---- (a) Noise amplification — binned median + IQR band -------------
    rows = json.loads((THEORY / "exp_10_4_noise/per_layer.json").read_text())
    pooled_lam, pooled_amp = [], []
    for r in rows:
        lam = np.array(r["lam"])
        amp = np.array(r["nu_over_lam_sq"])
        m = (lam > 1e-8) & (amp > 1e-30) & np.isfinite(amp)
        pooled_lam.append(lam[m])
        pooled_amp.append(amp[m])
    pooled_lam = np.concatenate(pooled_lam)
    pooled_amp = np.concatenate(pooled_amp)
    log_lam = np.log10(pooled_lam)
    log_amp = np.log10(pooled_amp)
    bins = np.linspace(log_lam.min(), log_lam.max(), 22)
    centres = 0.5 * (bins[:-1] + bins[1:])
    med, p25, p75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (log_lam >= lo) & (log_lam < hi)
        if m.sum() < 10:
            med.append(np.nan); p25.append(np.nan); p75.append(np.nan)
            continue
        med.append(np.median(log_amp[m]))
        p25.append(np.percentile(log_amp[m], 25))
        p75.append(np.percentile(log_amp[m], 75))
    centres = np.array(centres)
    med = np.array(med); p25 = np.array(p25); p75 = np.array(p75)
    valid = ~np.isnan(med)

    ax_amp.fill_between(10 ** centres[valid], 10 ** p25[valid], 10 ** p75[valid],
                        color=C_HEAD, alpha=0.22)
    ax_amp.plot(10 ** centres[valid], 10 ** med[valid], "-o", color=C_HEAD,
                lw=1.6, markersize=3)
    alpha2_y = 10 ** float(np.median(med[valid]))
    ax_amp.axhline(alpha2_y, color=C_TASKVEC, ls="--", lw=1.0, alpha=0.8)
    # In-place labels (no legend)
    ax_amp.text(10 ** centres[valid][-2] * 0.7, alpha2_y * 1.5,
                r"$\alpha = 2$",
                color=C_TASKVEC, fontsize=6.8, ha="right", va="bottom")
    ax_amp.text(0.04, 0.93,
                r"smaller $\lambda$ $\Rightarrow$ larger amplified noise",
                transform=ax_amp.transAxes,
                fontsize=7, weight="semibold", va="top",
                bbox=dict(facecolor="white", edgecolor="grey",
                          boxstyle="round,pad=0.18", alpha=0.85))
    ax_amp.set_xscale("log")
    ax_amp.set_yscale("log")
    ax_amp.set_xlabel(r"eigenvalue $\lambda_k$")
    ax_amp.set_ylabel(r"amplified noise $\hat\nu_k^{\,2}/\lambda_k^2$")
    ax_amp.set_title("(a) Small eigenvalues amplify noise")

    # ---- (b) Two archetype spectra with SWUDI-A rank-rule cuts ----------
    rows = json.loads((THEORY / "exp_10_6_rank/per_layer.json").read_text())
    pick_a = next((r for r in rows if "layers.0.mlp.fc1.weight" in r["layer"]), None)
    pick_b = next((r for r in rows if "layers.11.mlp.fc2.weight" in r["layer"]), None)
    if pick_a is None or pick_b is None:
        sort_layers = sorted(rows, key=lambda r: r["K_sqrt_ratio"])
        pick_a = sort_layers[-1]
        pick_b = sort_layers[0]

    archetype_label = {id(pick_a): "CLIP early MLP",
                       id(pick_b): "CLIP late MLP"}
    archetype_color = {id(pick_a): "#1f77b4", id(pick_b): "#d62728"}

    rule_marker = {"K_sqrt": "o", "K_GD": "s"}
    # Track first-curve marker positions for in-place rule labels
    first_curve_markers: Dict[str, tuple] = {}

    for r in (pick_a, pick_b):
        lam = np.array(r["lambda"])
        lam = lam[lam > 0]
        lam = np.sort(lam)[::-1]
        d = r["shape"][1]
        ks = np.arange(1, len(lam) + 1) / d
        lam_norm = lam / lam[0]
        col = archetype_color[id(r)]
        ax_panel.plot(ks, lam_norm, "-", color=col, lw=1.6)
        # Direct curve label, anchored near the right end of each line
        idx_label = int(0.65 * len(lam))
        ax_panel.text(ks[idx_label], lam_norm[idx_label] * 1.7,
                      archetype_label[id(r)], color=col, fontsize=6.8,
                      ha="left", va="bottom", weight="semibold")
        for rule_key, marker in rule_marker.items():
            K = r[rule_key]
            if K <= 0 or K > len(lam):
                continue
            ax_panel.scatter(K / d, lam_norm[K - 1], color=col,
                             marker=marker, s=42, edgecolor="black",
                             linewidth=0.6, zorder=10)
            if r is pick_a and rule_key not in first_curve_markers:
                first_curve_markers[rule_key] = (K / d, lam_norm[K - 1])

    # Direct rule labels next to the markers on the first (blue) curve
    rule_label_text = {"K_sqrt": r"$K_{\rm psqrt}$", "K_GD": r"$K_{\rm Gavish}$"}
    for rule_key, (xm, ym) in first_curve_markers.items():
        ax_panel.text(xm + 0.025, ym * 1.5, rule_label_text[rule_key],
                      color="black", fontsize=6.8, ha="left", va="bottom")

    ax_panel.set_yscale("log")
    ax_panel.set_xlim(0, 1)
    ax_panel.set_xlabel(r"normalized index $k/d$")
    ax_panel.set_ylabel(r"$\lambda_k / \lambda_1$")
    ax_panel.set_title("(b) Different spectra need different cuts")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, top=0.94, bottom=0.12)
    _save(fig, "figure_C_spectral_rank")


# ---------------------------------------------------------------------------
# Figure S1 — Wiener / risk / boundary rho   (full-width, 3 panels)
# ---------------------------------------------------------------------------

def figure_S1():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax_h, ax_risk, ax_rho = axes

    # ---- (a) filter overlay ------------------------------------------
    rows = json.loads((THEORY / "exp_10_5_wiener/per_layer.json").read_text())
    target = None
    for r in rows:
        if r["K_aswudi"] <= r["head_tail_keep"] and r["K_swudi"] <= r["head_tail_keep"]:
            target = r
            break
    if target is None:
        target = min(rows, key=lambda r: r["K_aswudi"])
    h_w = np.array(target["h_wiener_head"])
    K_aswudi = target["K_aswudi"]
    K_swudi = target["K_swudi"]
    d = target["shape"][1]
    keep = target["head_tail_keep"]
    layer_short_name = "/".join(target["layer"].split(".")[2:5])
    K_axis = np.arange(1, keep + 1)
    h_cf = np.ones_like(K_axis, dtype=float)
    ax_h.plot(K_axis, h_cf, "-", color=C_METHOD["closed_form"], lw=1.6, label="closed-form ($h{=}1$)")
    ax_h.plot(K_axis, h_w, "-", color=C_METHOD["wiener"], lw=1.8, label="empirical Wiener")
    if K_swudi <= keep:
        ax_h.step(K_axis, np.where(K_axis <= K_swudi, 1.0, 0.0), where="post",
                  color=C_METHOD["swudi_r0_65"], lw=1.6,
                  label=fr"SWUDI ($K{{=}}{K_swudi}$)")
    else:
        ax_h.axhline(1.0, color=C_METHOD["swudi_r0_65"], lw=1.6, ls=(0, (3, 1)),
                     alpha=0.7,
                     label=fr"SWUDI ($K{{=}}{K_swudi}{{>}}{keep}$; no cut in view)")
    if K_aswudi <= keep:
        ax_h.step(K_axis, np.where(K_axis <= K_aswudi, 1.0, 0.0), where="post",
                  color=C_METHOD["aswudi_sqrt"], lw=2.0,
                  label=fr"SWUDI-A ($K{{=}}{K_aswudi}$)")
    else:
        ax_h.axhline(1.0, color=C_METHOD["aswudi_sqrt"], lw=2.0, ls=(0, (3, 1)),
                     alpha=0.7,
                     label=fr"SWUDI-A ($K{{=}}{K_aswudi}{{>}}{keep}$; no cut in view)")
    ax_h.set_xlabel(r"index $k$ (head, capped at 256)")
    ax_h.set_ylabel(r"filter coefficient $h_k$")
    ax_h.set_xlim(0, keep)
    ax_h.set_ylim(-0.05, 1.10)
    ax_h.set_title(f"(a) Filter shapes\n({layer_short_name}, $d{{=}}{d}$)")
    ax_h.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # ---- (b) predicted-risk bars -------------------------------------
    summary = json.loads((THEORY / "exp_10_5_wiener/summary.json").read_text())
    methods = ["closed_form", "iwudi_t300", "swudi_r0_65", "aswudi_sqrt", "wiener", "drop_all"]
    risks = [summary["method_risk_summary"][m]["total"] for m in methods]
    colours = [C_METHOD[m] for m in methods]
    bars = ax_risk.bar(range(len(methods)), risks, color=colours,
                       edgecolor="black", linewidth=0.5)
    # Highlight Wiener and SWUDI-A
    for i, (b, r, m) in enumerate(zip(bars, risks, methods)):
        if m in ("wiener", "aswudi_sqrt", "closed_form"):
            ax_risk.text(b.get_x() + b.get_width() / 2, r * 1.10, f"{r:.0f}",
                         ha="center", fontsize=10, weight="semibold")
        else:
            ax_risk.text(b.get_x() + b.get_width() / 2, r * 1.10, f"{r:.0f}",
                         ha="center", fontsize=8.5, color="grey")
    ax_risk.set_yscale("log")
    ax_risk.set_xticks(range(len(methods)))
    ax_risk.set_xticklabels([LABEL_METHOD[m] for m in methods], rotation=30, ha="right",
                            fontsize=9)
    ax_risk.set_ylabel("total predicted Bayes risk (log)")
    ax_risk.set_ylim(top=ax_risk.get_ylim()[1] * 1.6)  # head-room for labels
    ax_risk.set_title("(b) Predicted Bayes risk (diagnostic)\n"
                      "lowest-risk filter $\\neq$ best merged accuracy")

    # ---- (c) boundary rho --------------------------------------------
    K_aswudi_arr = []
    rho_at_K = []
    rho_after_K = []
    for r in rows:
        K_aswudi_arr.append(r["K_aswudi"])
        rho_at_K.append(r["boundary_rho_at_K"])
        rho_after_K.append(r["boundary_rho_after_K"])
    layer_idx = np.arange(len(rows))
    med_rho = float(np.median(rho_at_K))
    ax_rho.scatter(layer_idx, rho_at_K, s=22, color="#1f77b4", alpha=0.85,
                   edgecolor="black", linewidth=0.3,
                   label=r"$\rho_{K_{A}}$")
    ax_rho.scatter(layer_idx, rho_after_K, s=22, color="#ff7f0e", alpha=0.85,
                   marker="^", edgecolor="black", linewidth=0.3,
                   label=r"$\rho_{K_{A}+1}$")
    ax_rho.axhline(1.0, color=C_TASKVEC, ls="--", lw=1.4, label=r"$\rho=1$ (theory cut)")
    ax_rho.axhline(med_rho, color="black", ls=":", lw=1.0,
                   label=fr"median $\rho_K = {med_rho:.1f}$")
    ax_rho.set_yscale("log")
    ax_rho.set_xlabel("layer index")
    ax_rho.set_ylabel(r"boundary SNR $\rho$")
    ax_rho.set_title("(c) SNR around the cutoff\n"
                     fr"no sharp SNR gap at the cut (median $\rho_K \approx {med_rho:.1f}$)")
    ax_rho.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout(w_pad=2.0)
    _save(fig, "figure_S1_filter_risk")


# ---------------------------------------------------------------------------
# Figure S2 — rank by kind / fit traj / capture-gap heatmap (full width)
# ---------------------------------------------------------------------------

def figure_S2():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax_kind, ax_traj, ax_capgap = axes

    # ---- (a) per-kind rank --------------------------------------------
    by_kind = json.loads((THEORY / "exp_10_6_rank/by_kind.json").read_text())
    kinds = ["attn_q", "attn_k", "attn_v", "attn_out", "mlp_fc1", "mlp_fc2"]
    K_sqrt_vals = [by_kind[k]["K_sqrt_mean"] for k in kinds]
    K_lam_vals = [by_kind[k]["K_lambda_mean"] for k in kinds]
    K_gd_vals = [by_kind[k]["K_GD_mean"] for k in kinds]
    x = np.arange(len(kinds))
    w = 0.27
    ax_kind.bar(x - w, K_sqrt_vals, w, color="#1f77b4", edgecolor="black", linewidth=0.4,
                label=r"$K_{\rm psqrt}$")
    ax_kind.bar(x, K_lam_vals, w, color="#ff7f0e", edgecolor="black", linewidth=0.4,
                label=r"$K_{\lambda}$")
    ax_kind.bar(x + w, K_gd_vals, w, color="#2ca02c", edgecolor="black", linewidth=0.4,
                label=r"$K_{\rm Gavish}$")
    ax_kind.axhline(0.65, color="grey", ls=":", lw=1.0, label="SWUDI tuned $r{=}0.65$")
    ax_kind.set_xticks(x)
    ax_kind.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=8.5)
    ax_kind.set_ylabel(r"mean $K/d$")
    ax_kind.set_ylim(0, 0.95)
    ax_kind.set_title("(a) Rank rules by layer type\n"
                      r"$K_{\rm psqrt}\approx$ tuned SWUDI; mlp.fc2 most aggressive")
    ax_kind.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)

    # ---- (b) Fit-R² trajectory --------------------------------------
    optblob = json.loads((THEORY / "exp_10_3_optimizer/per_layer.json").read_text())
    layer_names = list(optblob.keys())
    cmap = plt.cm.tab10
    layer_handles = []
    for i, ln in enumerate(layer_names):
        steps = optblob[ln]["trajectories"]["adam"]["steps"]
        r2 = [f["r2"] for f in optblob[ln]["trajectories"]["adam"]["fit"]]
        ax_traj.plot(steps, r2, "-o", color=cmap(i), markersize=5, linewidth=1.6)
        sgd_steps = optblob[ln]["trajectories"]["sgd"]["steps"]
        sgd_r2 = [f["r2"] for f in optblob[ln]["trajectories"]["sgd"]["fit"]]
        ax_traj.plot(sgd_steps, sgd_r2, ":", color=cmap(i), linewidth=1.4, alpha=0.7)
        layer_handles.append(plt.Line2D([0], [0], marker="o", color=cmap(i),
                                         markersize=6, lw=1.6,
                                         label=f"L{ln.split('.')[3]}"))
    kind_handles = [
        plt.Line2D([0], [0], color="grey", lw=2, marker="o", markersize=5, label="Adam (solid)"),
        plt.Line2D([0], [0], color="grey", lw=1.5, ls=":", label="SGD (dotted, ≈1.0)"),
    ]
    # Both legends placed outside the data area (right of the panel),
    # stacked vertically so they never cover the dip near step 1.
    leg_layer = ax_traj.legend(handles=layer_handles,
                                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                fontsize=8.5, framealpha=0.9,
                                title="layer", title_fontsize=8.5,
                                handletextpad=0.4)
    ax_traj.add_artist(leg_layer)
    ax_traj.legend(handles=kind_handles,
                   loc="lower left", bbox_to_anchor=(1.02, 0.0),
                   fontsize=8.5, framealpha=0.9,
                   title="optimiser", title_fontsize=8.5,
                   handletextpad=0.4)
    ax_traj.set_xscale("symlog")
    ax_traj.set_xlabel("optimization step")
    ax_traj.set_ylabel("filter-fit $R^2$")
    ax_traj.set_ylim(-0.5, 1.05)
    ax_traj.set_title("(b) Optimizer–filter fit\n"
                      r"SGD matches the filter; Adam grows filter-like")

    # ---- (c) capture-gap heatmap -------------------------------------
    cap_rows = json.loads((THEORY / "exp_10_1_input_subspace/capture_at_d_over_4.json").read_text())
    tasks = sorted({c["task"] for c in cap_rows})
    layers = sorted({c["layer"] for c in cap_rows})
    layer_short = {
        layers[0]: "early (L0)",
        layers[1]: "mid (L6)",
        layers[2]: "last (L11)",
    } if len(layers) == 3 else {ln: ln.split(".")[3] for ln in layers}
    M = np.zeros((len(tasks), len(layers)))
    for c in cap_rows:
        i = tasks.index(c["task"])
        j = layers.index(c["layer"])
        M[i, j] = c["gap"]
    im = ax_capgap.imshow(M, cmap="RdBu_r", vmin=-0.05, vmax=0.45, aspect="auto",
                          interpolation="nearest")
    ax_capgap.set_xticks(range(len(layers)))
    ax_capgap.set_xticklabels([layer_short[l] for l in layers], rotation=15, fontsize=9)
    ax_capgap.set_yticks(range(len(tasks)))
    ax_capgap.set_yticklabels(tasks, fontsize=9)
    ax_capgap.set_title("(c) Task-vector capture gap\n"
                        "task-vec capture $-$ random capture")
    for i in range(len(tasks)):
        for j in range(len(layers)):
            ax_capgap.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                           fontsize=9.5,
                           color="white" if abs(M[i, j]) > 0.25 else "black")
    cb = fig.colorbar(im, ax=ax_capgap, fraction=0.05, pad=0.04)
    cb.set_label("capture gap")
    cb.ax.tick_params(labelsize=8)
    ax_capgap.grid(False)

    fig.tight_layout(w_pad=3.0)  # extra w_pad reserves space for (b)'s side-legend
    _save(fig, "figure_S2_rank_diagnostics")


# ---------------------------------------------------------------------------
# Figure S3 / S4 — Diagnostic panels relocated from main-text Figs 1/2/3.
# Split into two 1x2 horizontal figures so each can be rendered at
# ``width=0.82\textwidth`` without aggressive scaling.
# ---------------------------------------------------------------------------

def figure_S3():
    """Appendix Fig S3 (assumption + optimizer diagnostics).

    (a) Input-energy capture vs retained rank — relocated from main Fig 1.
    (b) SGD vs Landweber filter (exact identity) — relocated from main Fig 2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.45))
    ax_cap, ax_sgd = axes

    # ---- (a) Input-energy capture (former Figure 1(a)) ------------------
    rows = json.loads((THEORY / "exp_10_1_input_subspace/per_pair.json").read_text())
    layer_label = {
        "vision_model.encoder.layers.0.mlp.fc1.weight": "early (L0)",
        "vision_model.encoder.layers.6.mlp.fc1.weight": "mid (L6)",
        "vision_model.encoder.layers.11.mlp.fc1.weight": "last (L11)",
    }
    layer_color = {
        "vision_model.encoder.layers.0.mlp.fc1.weight": "#1f77b4",
        "vision_model.encoder.layers.6.mlp.fc1.weight": "#2ca02c",
        "vision_model.encoder.layers.11.mlp.fc1.weight": "#d62728",
    }
    by_layer: Dict[str, Dict[str, list]] = {}
    for r in rows:
        if r["layer"] not in layer_label:
            continue
        bl = by_layer.setdefault(r["layer"], {"K_ratio": [], "tv": [], "rand": []})
        d = r["shape"][1]
        for K, c, rmean in zip(r["K_grid"], r["capture_taskvec"], r["capture_random_mean"]):
            bl["K_ratio"].append(K / d)
            bl["tv"].append(c)
            bl["rand"].append(rmean)
    rand_curves = []
    K_common = None
    for ln, data_l in by_layer.items():
        K = np.array(data_l["K_ratio"])
        order = np.argsort(K)
        K_s = K[order]
        tv_s = np.array(data_l["tv"])[order]
        rd_s = np.array(data_l["rand"])[order]
        unique_K, idx, counts = np.unique(K_s, return_inverse=True, return_counts=True)
        tv_avg = np.bincount(idx, weights=tv_s) / counts
        rd_avg = np.bincount(idx, weights=rd_s) / counts
        ax_cap.plot(unique_K, tv_avg, "-", color=layer_color[ln],
                    label=layer_label[ln], linewidth=1.8)
        rand_curves.append((unique_K, rd_avg))
        if K_common is None or len(unique_K) < len(K_common):
            K_common = unique_K
    rand_mean = np.mean([np.interp(K_common, k, r) for k, r in rand_curves], axis=0)
    ax_cap.plot(K_common, rand_mean, "--", color="black", linewidth=1.2,
                alpha=0.85, label="random subspace")
    ax_cap.plot([0, 1], [0, 1], ":", color="grey", lw=0.7, alpha=0.5)
    ax_cap.set_xlabel(r"retained rank ratio $K/d$")
    ax_cap.set_ylabel(r"input-energy capture")
    ax_cap.set_title("(a) Task-vector subspace captures input energy")
    ax_cap.set_xlim(0, 1)
    ax_cap.set_ylim(0, 1)
    ax_cap.legend(loc="lower right", framealpha=0.85, fontsize=6.0,
                  handletextpad=0.4, borderpad=0.25, labelspacing=0.25,
                  handlelength=1.4)

    # ---- (b) SGD exact Landweber filter (former Figure 2(a)) ------------
    data = json.loads((THEORY / "exp_10_3_optimizer/per_layer.json").read_text())
    layer_names = list(data.keys())
    target = "vision_model.encoder.layers.4.mlp.fc1.weight"
    if target not in layer_names:
        target = layer_names[len(layer_names) // 2]
    blob = data[target]
    lam = np.array(blob["spectra"]["lam"])
    keep = min(len(lam), 256)
    lam_keep = lam[:keep]
    eta = blob["lr_sgd"]
    sgd = blob["trajectories"]["sgd"]
    plot_steps = [s for s in sgd["steps"] if s in (1, 5, 50, 200)]
    cmap_s = plt.cm.viridis
    step_handles = []
    for i, step in enumerate(plot_steps):
        idx = sgd["steps"].index(step)
        h_emp = np.array(sgd["h_emp"][idx])
        n_eff = sgd["fit"][idx].get("n_eff")
        if n_eff is None or math.isnan(n_eff):
            n_eff = step
        h_theo = 1 - np.power(np.clip(1 - eta * lam_keep, -1, 1), n_eff)
        color = cmap_s(0.15 + 0.7 * i / max(len(plot_steps) - 1, 1))
        ax_sgd.scatter(lam_keep, h_emp, color=color, s=8, alpha=0.5,
                       edgecolor="none")
        ax_sgd.plot(np.sort(lam_keep), h_theo[np.argsort(lam_keep)],
                    "-", color=color, lw=1.5, alpha=0.95)
        step_handles.append(plt.Line2D([0], [0], color=color, lw=1.6,
                                        label=f"step {step}"))
    ax_sgd.legend(handles=step_handles, loc="upper left", fontsize=6.0,
                  framealpha=0.85, title="step", title_fontsize=6.2,
                  handletextpad=0.4, borderpad=0.25, labelspacing=0.25,
                  handlelength=1.4)
    ax_sgd.set_xscale("log")
    ax_sgd.set_xlabel(r"eigenvalue $\lambda_k$")
    ax_sgd.set_ylabel(r"empirical filter $\hat h_{k,n}$")
    ax_sgd.set_ylim(-0.05, 1.05)
    ax_sgd.set_title("(b) SGD matches Landweber exactly")

    fig.tight_layout(w_pad=1.4)
    _save(fig, "figure_S3_diagnostics_1")


def figure_S4():
    """Appendix Fig S4 (fit-quality + rank-rule diagnostics).

    (a) Per-layer filter-fit R^2 heatmap (SGD + Adam) — relocated from main Fig 2.
    (b) K_psqrt / K_Gavish layer-wise scatter — relocated from main Fig 3.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.45))
    ax_heat, ax_scatter = axes

    # ---- (a) Per-layer filter-fit R^2 heatmap ---------------------------
    data = json.loads((THEORY / "exp_10_3_optimizer/per_layer.json").read_text())
    layers = list(data.keys())
    short = {ln: f"L{ln.split('.')[3]}" for ln in layers}
    common_steps = set(data[layers[0]]["log_steps"])
    for ln in layers[1:]:
        common_steps &= set(data[ln]["log_steps"])
    common_steps = sorted(s for s in common_steps if s > 0)
    M_sgd = np.zeros((len(layers), len(common_steps)))
    M_adm = np.zeros((len(layers), len(common_steps)))
    for i, ln in enumerate(layers):
        for j, step in enumerate(common_steps):
            sgd_idx = data[ln]["trajectories"]["sgd"]["steps"].index(step)
            adm_idx = data[ln]["trajectories"]["adam"]["steps"].index(step)
            M_sgd[i, j] = data[ln]["trajectories"]["sgd"]["fit"][sgd_idx]["r2"]
            M_adm[i, j] = data[ln]["trajectories"]["adam"]["fit"][adm_idx]["r2"]
    M = np.vstack([M_sgd, M_adm])
    im = ax_heat.imshow(M, aspect="auto", cmap="YlGn", vmin=0.4, vmax=1.0,
                        alpha=0.85)
    ax_heat.set_xticks(range(len(common_steps)))
    ax_heat.set_xticklabels(common_steps, rotation=45, ha="right", fontsize=6.5)
    ax_heat.set_yticks(range(len(layers) + len(layers)))
    ax_heat.set_yticklabels([f"SGD {short[ln]}" for ln in layers] +
                            [f"Adam {short[ln]}" for ln in layers], fontsize=6.5)
    ax_heat.axhline(len(layers) - 0.5, color="black", lw=0.8)
    ax_heat.set_xlabel("optimization step")
    ax_heat.set_title("(a) Per-layer filter-fit $R^2$")
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.02)
    cb.set_label(r"$R^2$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax_heat.grid(False)

    # ---- (b) K_psqrt vs K_Gavish layer-wise scatter ---------------------
    rows = json.loads((THEORY / "exp_10_6_rank/per_layer.json").read_text())
    def kind(name):
        if "mlp.fc1" in name: return "mlp_fc1"
        if "mlp.fc2" in name: return "mlp_fc2"
        if "self_attn.q_proj" in name: return "attn_q"
        if "self_attn.k_proj" in name: return "attn_k"
        if "self_attn.v_proj" in name: return "attn_v"
        if "self_attn.out_proj" in name: return "attn_out"
        return "other"
    KIND_COLOR = {
        "mlp_fc1": "#1f77b4", "mlp_fc2": "#ff7f0e",
        "attn_q": "#2ca02c", "attn_k": "#d62728",
        "attn_v": "#9467bd", "attn_out": "#8c564b",
    }
    for k_ in KIND_COLOR:
        xs = [r["K_sqrt_ratio"] for r in rows if kind(r["layer"]) == k_]
        ys = [r["K_GD_ratio"] for r in rows if kind(r["layer"]) == k_]
        ax_scatter.scatter(xs, ys, color=KIND_COLOR[k_], s=28,
                           edgecolor="black", linewidth=0.3,
                           label=k_.replace("_", " "))
    sx = np.array([r["K_sqrt_ratio"] for r in rows])
    sy = np.array([r["K_GD_ratio"] for r in rows])
    corr = float(np.corrcoef(sx, sy)[0, 1])
    ax_scatter.set_xlabel(r"$K_{\rm psqrt} / d$")
    ax_scatter.set_ylabel(r"$K_{\rm Gavish} / d$")
    ax_scatter.set_title(fr"(b) Two rank rules: Pearson $\rho={corr:+.2f}$")
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 0.45)
    ax_scatter.legend(loc="lower center",
                      bbox_to_anchor=(0.5, 1.06),
                      ncol=3, frameon=False,
                      fontsize=6.2, title_fontsize=6.5,
                      title="layer kind", columnspacing=0.7,
                      handletextpad=0.3, borderpad=0.2,
                      labelspacing=0.25)

    fig.tight_layout(w_pad=1.6)
    _save(fig, "figure_S4_diagnostics_2")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    figure_A()
    figure_B()
    figure_C()
    figure_S1()
    figure_S2()
    figure_S3()
    figure_S4()
    # Remove now-stale combined supplementary figures
    for stale_name in ("figure_S1_supplementary", "figure_S3_dropped_panels"):
        for ext in (".pdf", ".png"):
            f = (FIG_DIR / stale_name).with_suffix(ext)
            if f.exists():
                f.unlink()
                print(f"[fig] removed stale {f}")


if __name__ == "__main__":
    main()
