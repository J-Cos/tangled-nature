#!/usr/bin/env python3
"""
Visualize causal emergence analysis results.

Multi-panel figure:
  Row 1: N and S time series
  Row 2: EI at each scale over time (overlaid)
  Row 3: Scale of maximum EI over time (categorical heatstrip)
  Row 4: Determinism and degeneracy components
"""

import os
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap


def load_results(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    csv_path = os.path.join(results_dir, "causal_emergence.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    data = load_results(csv_path)
    if not data:
        return

    gens = [(d["window_start"] + d["window_end"]) / 2 for d in data]

    scales = ["micro", "meso1", "meso2", "macro"]
    scale_colors = {"micro": "#E41A1C", "meso1": "#377EB8",
                     "meso2": "#4DAF4A", "macro": "#FF7F00"}
    scale_labels = {"micro": "Micro (species)", "meso1": "Meso-1 (SAD shape)",
                     "meso2": "Meso-2 (N,S)", "macro": "Macro (N only)"}

    # ── Figure ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 1, hspace=0.30, height_ratios=[0.7, 1, 0.4, 1])

    # ── Row 1: N and S ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ns = [d["mean_n"] for d in data]
    ss = [d["mean_s"] for d in data]
    ax1.plot(gens, ns, color="#333333", lw=0.8, alpha=0.8)
    ax1.set_ylabel("N", color="#333333")
    ax1.set_title("A  Population & Richness Dynamics", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.15)

    ax1b = ax1.twinx()
    ax1b.plot(gens, ss, color="#1B7837", lw=0.8, alpha=0.8)
    ax1b.set_ylabel("S", color="#1B7837")

    # ── Row 2: EI at each scale ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for scale in scales:
        ei = [d[f"ei_{scale}"] for d in data]
        ax2.plot(gens, ei, color=scale_colors[scale], lw=1.2, alpha=0.8,
                  label=scale_labels[scale])
    ax2.set_ylabel("Effective Information (bits)")
    ax2.set_title("B  Effective Information Across Scales", fontweight="bold",
                   fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.15)

    # ── Row 3: Scale of max EI (categorical strip) ───────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    scale_to_idx = {s: i for i, s in enumerate(scales)}
    max_scale_idx = [scale_to_idx[d["max_ei_scale"]] for d in data]
    cmap = ListedColormap([scale_colors[s] for s in scales])

    # Draw as colored strip
    for i in range(len(gens)):
        ax3.axvspan(gens[max(0, i-1)] if i > 0 else gens[0],
                     gens[i],
                     color=scale_colors[data[i]["max_ei_scale"]], alpha=0.8)

    ax3.set_yticks([])
    ax3.set_title("C  Scale of Maximum EI", fontweight="bold", fontsize=12)
    # Legend
    for s in scales:
        ax3.plot([], [], color=scale_colors[s], lw=8,
                  label=scale_labels[s])
    ax3.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))

    # ── Row 4: Determinism & Degeneracy ──────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for scale in scales:
        det = [d[f"det_{scale}"] for d in data]
        deg = [d[f"deg_{scale}"] for d in data]
        ax4.plot(gens, det, color=scale_colors[scale], lw=1.0, alpha=0.6,
                  label=f"{scale_labels[scale]} det" if scale == scales[0] else "")
        ax4.plot(gens, deg, color=scale_colors[scale], lw=1.0, alpha=0.6,
                  ls="--")

    ax4.set_ylabel("Bits")
    ax4.set_xlabel("Generation")
    ax4.set_title("D  Determinism (solid) & Degeneracy (dashed)",
                   fontweight="bold", fontsize=12)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for s in scales:
        legend_elements.append(
            Line2D([0], [0], color=scale_colors[s], lw=1.5,
                    label=scale_labels[s]))
    legend_elements.append(
        Line2D([0], [0], color="gray", lw=1.5, label="Determinism"))
    legend_elements.append(
        Line2D([0], [0], color="gray", lw=1.5, ls="--", label="Degeneracy"))
    ax4.legend(handles=legend_elements, fontsize=8, ncol=3, loc="upper right")
    ax4.grid(True, alpha=0.15)

    # ── Title ────────────────────────────────────────────────────
    fig.suptitle("Causal Emergence in the Tangled Nature Model",
                  fontsize=15, fontweight="bold", y=0.995)

    out_png = os.path.join(results_dir, "Figure_CausalEmergence.png")
    out_pdf = os.path.join(results_dir, "Figure_CausalEmergence.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ {out_png} / .pdf saved")


if __name__ == "__main__":
    main()
