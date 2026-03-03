#!/usr/bin/env python3
"""
Genome heatmaps for high-rank harvest forks.

4×1 single-column figure showing species abundance heatmaps
for randomly selected forks where a dominant species (rank 1–5) was harvested.
Each heatmap shows genome (y-axis) vs generation (x-axis) with abundance as color,
with a red dashed line marking harvest onset at gen 200.
"""

import json
import os
import sys
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


def load_fork_species(fname):
    """Load per-generation species data from fork JSONL. Returns list of dicts."""
    snapshots = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess"):
                continue
            sp = d.get("species")
            if not sp:
                continue
            species = {int(g): int(c) for g, c in sp}
            snapshots.append({
                "gen": d["gen"],
                "n": d.get("n", sum(species.values())),
                "s": len(species),
                "species": species,
            })
    return snapshots


def get_target_genome(state_file, rank):
    """Get the genome ID for the given rank from the qESS state."""
    with open(state_file) as f:
        state = json.load(f)
    sp = state.get("species", {})
    sorted_sp = sorted(sp.items(), key=lambda x: int(x[1]), reverse=True)
    if rank <= len(sorted_sp):
        return int(sorted_sp[rank - 1][0]), int(sorted_sp[rank - 1][1])
    return None, None


def build_heatmap_matrix(snapshots):
    """Build genome × time heatmap matrix."""
    all_genomes = set()
    for s in snapshots:
        all_genomes.update(s["species"].keys())

    # Sort by total abundance (most abundant at top)
    genome_totals = {}
    for s in snapshots:
        for g, c in s["species"].items():
            genome_totals[g] = genome_totals.get(g, 0) + c
    sorted_genomes = sorted(all_genomes, key=lambda g: -genome_totals.get(g, 0))
    genome_to_idx = {g: i for i, g in enumerate(sorted_genomes)}

    gens = [s["gen"] for s in snapshots]
    matrix = np.zeros((len(sorted_genomes), len(snapshots)))
    for col, s in enumerate(snapshots):
        for g, c in s["species"].items():
            matrix[genome_to_idx[g], col] = c

    return matrix, gens, sorted_genomes


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    state_dir = os.path.join(script_dir, "states")
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "results")

    # Pick ranks 1–4 from a single random sim
    random.seed(42)
    sid = random.randint(1, 32)
    picks = [(sid, rank) for rank in range(1, 5)]

    print(f"Selected forks: {picks}")

    # ── Figure: 4 rows × 1 col ──────────────────────────────────
    fig = plt.figure(figsize=(14, 20))
    gs = gridspec.GridSpec(4, 1, hspace=0.30)

    for i, (sid, rank) in enumerate(picks):
        fname = os.path.join(data_dir, f"sim_{sid:02d}_rank{rank:02d}.jsonl")
        state_file = os.path.join(state_dir, f"sim_{sid:02d}.json")

        snapshots = load_fork_species(fname)
        if not snapshots:
            continue

        target_genome, target_abd = get_target_genome(state_file, rank)
        matrix, gens, sorted_genomes = build_heatmap_matrix(snapshots)
        rel_gens = [g - gens[0] for g in gens]

        # Find target genome row index
        target_row = None
        if target_genome in sorted_genomes:
            target_row = sorted_genomes.index(target_genome)

        ax = fig.add_subplot(gs[i, 0])

        # Mask zeros for log scale
        masked = np.ma.masked_where(matrix == 0, matrix)
        vmin = max(1, masked.min()) if masked.count() > 0 else 1
        vmax = max(vmin + 1, matrix.max())

        im = ax.imshow(masked, aspect="auto", interpolation="nearest",
                        cmap="inferno",
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        extent=[rel_gens[0], rel_gens[-1], matrix.shape[0], 0])

        # Harvest onset line
        ax.axvline(200, color="cyan", ls="--", lw=1.5, alpha=0.8)

        # Mark target genome with arrow
        if target_row is not None:
            ax.annotate(f"← harvested (genome {target_genome})",
                         xy=(rel_gens[-1], target_row + 0.5),
                         xytext=(rel_gens[-1] + 5, target_row + 0.5),
                         fontsize=8, color="cyan", fontweight="bold",
                         verticalalignment="center",
                         arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))

        ax.set_xlabel("Generation (from fork start)")
        ax.set_ylabel("Genome (ranked)")
        title = (f"Sim {sid}, Rank {rank} harvest "
                 f"(genome {target_genome}, initial N={target_abd})")
        ax.set_title(title, fontweight="bold", fontsize=11)

        cb = plt.colorbar(im, ax=ax, label="Abundance", shrink=0.8, pad=0.02)

    fig.suptitle("Genome Heatmaps: High-Rank Species Harvesting (25%)",
                  fontsize=14, fontweight="bold", y=0.995)

    out_png = os.path.join(results_dir, "Figure_GenomeHeatmaps.png")
    out_pdf = os.path.join(results_dir, "Figure_GenomeHeatmaps.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ {out_png} / .pdf saved")


if __name__ == "__main__":
    main()
