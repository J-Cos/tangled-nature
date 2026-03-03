#!/usr/bin/env python3
"""
Visualize Spatial TNM — species-level analysis.

Layout (3 rows × 4 cols):
  Row 1: Density maps of 4 most abundant species at final snapshot
  Row 2: SAD (patch-scale) at 4 time points
  Row 3: Genome × time heatmap (TaNa-style) | Landscape SAD (final) |
         Rank-abundance | Occupancy over time
"""

import json
import sys
import numpy as np
from collections import defaultdict


def load_data(fname):
    snapshots = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") == "spatial_snapshot":
                snapshots.append(d)
    return snapshots


def pick_time_points(snapshots):
    """Pick snapshots at initial, ~1/3, ~2/3, and final generation."""
    min_gen = min(s["gen"] for s in snapshots)
    max_gen = max(s["gen"] for s in snapshots)
    span = max_gen - min_gen
    targets = [min_gen, min_gen + span // 3, min_gen + 2 * span // 3, max_gen]
    result = []
    for t in targets:
        best = min(snapshots, key=lambda s: abs(s["gen"] - t))
        result.append(best)
    return result


def get_landscape_abundance(snapshot):
    """Sum species abundances across all patches → {genome: total_count}."""
    totals = defaultdict(int)
    for p in snapshot["patches"]:
        sp = p.get("species", {})
        for genome, count in sp.items():
            totals[int(genome)] += count
    return totals


def get_patch_sads(snapshot):
    """Get list of per-patch abundance vectors (sorted descending)."""
    sads = []
    for p in snapshot["patches"]:
        sp = p.get("species", {})
        counts = sorted(sp.values(), reverse=True)
        if counts:
            sads.append(counts)
    return sads


def build_species_grid(snapshot, genome_id, grid_w, grid_h):
    """Build grid of abundance for a specific species."""
    grid = np.zeros((grid_h, grid_w))
    for p in snapshot["patches"]:
        sp = p.get("species", {})
        count = sp.get(str(genome_id), 0)
        grid[p["y"], p["x"]] = count
    return grid


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "spatial_output.jsonl"
    snapshots = load_data(fname)

    if not snapshots:
        print("No spatial snapshots found!")
        return

    # Check if species data is available
    if "species" not in snapshots[-1]["patches"][0]:
        print("No species data in snapshots — rerun simulation with updated code.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    # Grid dimensions
    last = snapshots[-1]
    grid_w = max(p["x"] for p in last["patches"]) + 1
    grid_h = max(p["y"] for p in last["patches"]) + 1

    # Pick 4 temporal snapshots
    time_snaps = pick_time_points(snapshots)

    # ── Figure 2: Species analysis (3 rows × 4 cols) ──────────────
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35,
                           height_ratios=[1, 1, 1])

    # ── Row 1: Top-4 species density maps (final snapshot) ────────
    landscape_ab = get_landscape_abundance(last)
    top4 = sorted(landscape_ab.items(), key=lambda x: -x[1])[:4]

    for col, (genome_id, total_count) in enumerate(top4):
        ax = fig.add_subplot(gs[0, col])
        grid = build_species_grid(last, genome_id, grid_w, grid_h)
        vmax = max(grid.max(), 1)
        im = ax.imshow(grid, cmap="magma", interpolation="nearest",
                       vmin=0, vmax=vmax)
        ax.set_title(f"Genome {genome_id}\n(N={total_count})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Top species\n(density maps)", fontsize=11,
                          fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 2: Patch-level SAD at 4 time points ───────────────────
    for col, snap in enumerate(time_snaps):
        ax = fig.add_subplot(gs[1, col])
        sads = get_patch_sads(snap)
        if sads:
            max_rank = max(len(s) for s in sads)
            rank_data = defaultdict(list)
            for sad in sads:
                for rank, count in enumerate(sad):
                    rank_data[rank].append(count)

            ranks = sorted(rank_data.keys())
            medians = [np.median(rank_data[r]) for r in ranks]
            q25 = [np.percentile(rank_data[r], 25) for r in ranks]
            q75 = [np.percentile(rank_data[r], 75) for r in ranks]

            x = [r + 1 for r in ranks]
            ax.fill_between(x, q25, q75, alpha=0.3, color="#2166AC")
            ax.plot(x, medians, color="#2166AC", linewidth=2)
            ax.set_yscale("log")
            ax.set_xlabel("Rank")
            if col == 0:
                ax.set_ylabel("Abundance (log)", fontsize=11,
                              fontweight="bold")
            ax.set_title(f"Gen {snap['gen']}\nPatch SAD", fontsize=9)
            ax.grid(True, alpha=0.2)

    # ── Row 3: SAD + Occupancy panels ─────────────────────────────

    # Collect landscape abundances for time series
    gen_abundances = []
    gens_list = []
    for snap in snapshots:
        ab = get_landscape_abundance(snap)
        gen_abundances.append(ab)
        gens_list.append(snap["gen"])

    # Panel 3a-b: Landscape SAD (final, log-log rank-abundance)
    ax_lsad = fig.add_subplot(gs[2, 0:2])
    landscape_counts = sorted(landscape_ab.values(), reverse=True)
    ranks = np.arange(1, len(landscape_counts) + 1)
    ax_lsad.scatter(ranks, landscape_counts, s=15, c="#D6604D",
                    edgecolors="black", linewidth=0.3, alpha=0.8)
    ax_lsad.set_xscale("log")
    ax_lsad.set_yscale("log")
    ax_lsad.set_xlabel("Rank (log)")
    ax_lsad.set_ylabel("Abundance (log)")
    ax_lsad.set_title("Landscape SAD (final)",
                      fontweight="bold", fontsize=10, loc="left")
    ax_lsad.grid(True, alpha=0.2)

    # Panel 3c-d: Species occupancy over time
    ax_occ = fig.add_subplot(gs[2, 2:4])
    n_species_over_time = [len(ab) for ab in gen_abundances]

    n_patches = grid_w * grid_h
    widespread = []
    for snap in snapshots:
        counts = defaultdict(int)
        for p in snap["patches"]:
            sp = p.get("species", {})
            for g in sp:
                counts[g] += 1
        n_wide = sum(1 for g, c in counts.items() if c > n_patches * 0.1)
        widespread.append(n_wide)

    ax_occ.plot(gens_list, n_species_over_time, color="#762A83",
                linewidth=1.5, label="Total species (γ)")
    ax_occ.plot(gens_list, widespread, color="#1B7837",
                linewidth=1.5, label=">10% patches")
    ax_occ.set_xlabel("Generation")
    ax_occ.set_ylabel("Number of species")
    ax_occ.set_title("Species occupancy",
                      fontweight="bold", fontsize=10, loc="left")
    ax_occ.legend(fontsize=8)
    ax_occ.grid(True, alpha=0.2)

    fig.suptitle(
        f"Spatial TNM Species Analysis: {grid_w}×{grid_h} Grid",
        fontsize=15, fontweight="bold", y=0.99
    )

    plt.savefig("Figure_SpatialTNM_species.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure_SpatialTNM_species.pdf", dpi=300, bbox_inches="tight")
    print("✓ Figure_SpatialTNM_species.png / .pdf saved")
    plt.close(fig)

    # ── Figure 3: TaNa heatmap (standalone, observed genomes only) ─
    # Build ordered list of genomes by first appearance
    genome_first_seen = {}  # genome_id → first time index
    for t, ab in enumerate(gen_abundances):
        for g in ab:
            if g not in genome_first_seen:
                genome_first_seen[g] = t

    # Sort genomes by first appearance time
    ordered_genomes = sorted(genome_first_seen.keys(),
                             key=lambda g: genome_first_seen[g])
    genome_to_row = {g: row for row, g in enumerate(ordered_genomes)}
    n_observed = len(ordered_genomes)
    n_gens = len(gens_list)

    # Build compact heatmap: n_observed rows × n_gens columns
    heatmap = np.zeros((n_observed, n_gens))
    for t, ab in enumerate(gen_abundances):
        for g, c in ab.items():
            if g in genome_to_row:
                heatmap[genome_to_row[g], t] = c

    # Scale figure height to number of observed genomes
    fig_h = max(6, min(20, n_observed * 0.04 + 2))
    fig_tana = plt.figure(figsize=(14, fig_h))
    ax_tana = fig_tana.add_subplot(111)

    if heatmap.max() > 0:
        # 'hot' colormap: black → red → yellow → white
        # abundance=1 is dark red, blending with absent (black)
        masked = np.ma.masked_where(heatmap == 0, heatmap)
        im = ax_tana.imshow(
            masked, aspect="auto", cmap="hot",
            norm=LogNorm(vmin=1, vmax=max(heatmap.max(), 2)),
            interpolation="none",
            extent=[gens_list[0], gens_list[-1], n_observed, 0]
        )
        ax_tana.set_facecolor("black")
        plt.colorbar(im, ax=ax_tana, fraction=0.02, pad=0.02,
                     label="Abundance")

    ax_tana.set_xlabel("Generation", fontsize=12)
    ax_tana.set_ylabel("Species (ordered by first appearance)", fontsize=12)
    ax_tana.set_title(
        f"TaNa Genome × Time — {n_observed} observed species",
        fontweight="bold", fontsize=13
    )

    plt.savefig("Figure_SpatialTNM_tana.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure_SpatialTNM_tana.pdf", dpi=300, bbox_inches="tight")
    print("✓ Figure_SpatialTNM_tana.png / .pdf saved")
    plt.close(fig_tana)


if __name__ == "__main__":
    main()
