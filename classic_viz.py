#!/usr/bin/env python3
"""
Visualize single-patch TNM output — comprehensive multi-panel figure.

Layout (2 rows × 3 cols + standalone TaNa heatmap):
  Row 1: N time series | S time series | SAD (rank-abundance, final)
  Row 2: N histogram | Phase space (N vs S) | p_off proxy (N/R time series)

Plus a standalone TaNa genome × time heatmap.
"""

import json
import sys
import numpy as np
from collections import defaultdict


def load_data(fname):
    """Load JSONL and return snapshots with species data."""
    snapshots = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") in ("snapshot", "qess"):
                snapshots.append(d)
    return snapshots


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "classic_output.jsonl"
    snapshots = load_data(fname)

    if not snapshots:
        print("No snapshots found!")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    gens = [s["gen"] for s in snapshots]
    ns = [s["n"] for s in snapshots]
    ss = [s["s"] for s in snapshots]

    # ── Figure 1: Multi-panel overview (2 rows × 3 cols) ──────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Single-Patch TNM Dynamics", fontsize=16, fontweight="bold", y=0.98)

    # Colors
    c_pop = "#2166AC"   # blue
    c_spp = "#B2182B"   # red
    c_sad = "#D6604D"   # salmon
    c_hist = "#4393C3"  # light blue
    c_phase = "#762A83" # purple

    # ── Panel (0,0): Population N over time ─────────────────────────
    ax = axes[0, 0]
    ax.plot(gens, ns, color=c_pop, linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population (N)")
    ax.set_title("Population dynamics", fontweight="bold", fontsize=11, loc="left")
    ax.grid(True, alpha=0.2)
    # Add mean line
    if len(ns) > 10:
        late_ns = ns[len(ns)//2:]
        ax.axhline(np.mean(late_ns), color=c_pop, linestyle="--", alpha=0.4,
                    label=f"late mean = {np.mean(late_ns):.0f}")
        ax.legend(fontsize=8)

    # ── Panel (0,1): Species richness S over time ───────────────────
    ax = axes[0, 1]
    ax.plot(gens, ss, color=c_spp, linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Species (S)")
    ax.set_title("Species richness", fontweight="bold", fontsize=11, loc="left")
    ax.grid(True, alpha=0.2)
    if len(ss) > 10:
        late_ss = ss[len(ss)//2:]
        ax.axhline(np.mean(late_ss), color=c_spp, linestyle="--", alpha=0.4,
                    label=f"late mean = {np.mean(late_ss):.1f}")
        ax.legend(fontsize=8)

    # ── Panel (0,2): SAD rank-abundance (final snapshot) ────────────
    ax = axes[0, 2]
    final = snapshots[-1]
    if "species" in final:
        species = final["species"]
        abundances = sorted([c for _, c in species], reverse=True)
        ranks = np.arange(1, len(abundances) + 1)
        ax.bar(ranks, abundances, color=c_sad, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Abundance")
        ax.set_title(f"SAD (gen {final['gen']}, S={len(species)})",
                      fontweight="bold", fontsize=11, loc="left")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No species data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("SAD", fontweight="bold", fontsize=11, loc="left")

    # ── Panel (1,0): Population histogram ───────────────────────────
    ax = axes[1, 0]
    ax.hist(ns, bins=50, color=c_hist, edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_xlabel("Population (N)")
    ax.set_ylabel("Frequency")
    ax.set_title("N distribution", fontweight="bold", fontsize=11, loc="left")
    ax.grid(True, alpha=0.2)
    ax.axvline(np.mean(ns), color="black", linestyle="--", alpha=0.5,
               label=f"mean = {np.mean(ns):.0f}")
    ax.legend(fontsize=8)

    # ── Panel (1,1): Phase space (N vs S) ───────────────────────────
    ax = axes[1, 1]
    # Color by time (early = light, late = dark)
    colors = np.linspace(0.2, 1.0, len(gens))
    ax.scatter(ns, ss, c=colors, cmap="Purples", s=8, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Population (N)")
    ax.set_ylabel("Species (S)")
    ax.set_title("Phase space (N vs S)", fontweight="bold", fontsize=11, loc="left")
    ax.grid(True, alpha=0.2)

    # ── Panel (1,2): Coefficient of variation over time ─────────────
    ax = axes[1, 2]
    window = min(50, len(ns) // 4) if len(ns) > 20 else max(3, len(ns) // 4)
    if window >= 3:
        rolling_cv = []
        rolling_gens = []
        for i in range(window, len(ns)):
            chunk = ns[i-window:i]
            mean_c = np.mean(chunk)
            if mean_c > 0:
                cv = np.std(chunk) / mean_c
            else:
                cv = 0
            rolling_cv.append(cv)
            rolling_gens.append(gens[i])
        ax.plot(rolling_gens, rolling_cv, color="#1B7837", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"CV(N) (window={window})")
        ax.set_title("Population stability", fontweight="bold", fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)
        # Mark qESS threshold if useful
        ax.axhline(0.05, color="red", linestyle=":", alpha=0.3, label="qESS threshold (0.05)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("Figure_ClassicTNM.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure_ClassicTNM.pdf", dpi=300, bbox_inches="tight")
    print("✓ Figure_ClassicTNM.png / .pdf saved")
    plt.close(fig)

    # ── Figure 2: TaNa genome × time heatmap ──────────────────────────
    # Build genome abundances per snapshot
    gen_abundances = []
    gens_list = []
    has_species = False
    for snap in snapshots:
        if "species" in snap:
            has_species = True
            ab = {}
            for genome, count in snap["species"]:
                ab[int(genome)] = count
            gen_abundances.append(ab)
            gens_list.append(snap["gen"])

    if not has_species:
        print("⚠ No species data in snapshots — TaNa heatmap skipped.")
        print("  Rerun with --out <file> to include species data.")
        return

    # Build ordered list of genomes by first appearance
    genome_first_seen = {}
    for t, ab in enumerate(gen_abundances):
        for g in ab:
            if g not in genome_first_seen:
                genome_first_seen[g] = t

    ordered_genomes = sorted(genome_first_seen.keys(),
                             key=lambda g: genome_first_seen[g])
    genome_to_row = {g: row for row, g in enumerate(ordered_genomes)}
    n_observed = len(ordered_genomes)
    n_gens = len(gens_list)

    # Build compact heatmap
    heatmap = np.zeros((n_observed, n_gens))
    for t, ab in enumerate(gen_abundances):
        for g, c in ab.items():
            if g in genome_to_row:
                heatmap[genome_to_row[g], t] = c

    # Scale figure height
    fig_h = max(6, min(20, n_observed * 0.04 + 2))
    fig_tana = plt.figure(figsize=(14, fig_h))
    ax_tana = fig_tana.add_subplot(111)

    if heatmap.max() > 0:
        masked = np.ma.masked_where(heatmap == 0, heatmap)
        im = ax_tana.imshow(
            masked, aspect="auto", cmap="hot",
            norm=LogNorm(vmin=1, vmax=max(heatmap.max(), 2)),
            interpolation="none",
            extent=[gens_list[0], gens_list[-1], n_observed, 0]
        )
        ax_tana.set_facecolor("black")
        plt.colorbar(im, ax=ax_tana, fraction=0.02, pad=0.02, label="Abundance")

    ax_tana.set_xlabel("Generation", fontsize=12)
    ax_tana.set_ylabel("Species (ordered by first appearance)", fontsize=12)
    ax_tana.set_title(
        f"TaNa Genome × Time — {n_observed} observed species",
        fontweight="bold", fontsize=13
    )

    plt.savefig("Figure_ClassicTNM_tana.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure_ClassicTNM_tana.pdf", dpi=300, bbox_inches="tight")
    print(f"✓ Figure_ClassicTNM_tana.png / .pdf saved ({n_observed} species)")
    plt.close(fig_tana)


if __name__ == "__main__":
    main()
