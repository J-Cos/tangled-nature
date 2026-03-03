#!/usr/bin/env python3
"""
Ensemble runner for Spatial TNM simulations.

Runs multiple simulations in parallel, collects per-snapshot and final-step
metrics (including core/cloud species classification), and produces a
multi-panel summary figure.

Usage:
    python3 ensemble_run.py --grid-size 2 --n-sims 100 --max-gen 10000
    python3 ensemble_run.py --grid-size 5 --n-sims 20 --parallel 8
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Ensemble Spatial TNM runner")
    p.add_argument("--grid-size", type=int, default=2)
    p.add_argument("--n-sims", type=int, default=100)
    p.add_argument("--max-gen", type=int, default=10000)
    p.add_argument("--output-interval", type=int, default=200)
    p.add_argument("--p-move", type=float, default=0.01)
    p.add_argument("--l", type=int, default=14)
    p.add_argument("--r", type=float, default=5.0)
    p.add_argument("--binary", type=str,
                   default="./target/release/tangled-nature")
    p.add_argument("--out-dir", type=str, default="/tmp/ensemble")
    p.add_argument("--parallel", type=int, default=0,
                   help="Parallel workers (0 = all CPUs, 1 = serial)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Worker function — top-level so ProcessPoolExecutor can pickle it
# ──────────────────────────────────────────────────────────────────────────────
def run_single_sim(sim_idx, binary, grid_size, p_move, l_val, r_val,
                   max_gen, output_interval, out_dir, seed):
    """Run one sim, parse JSONL, return (sim_idx, metrics, elapsed)."""
    out_file = os.path.join(out_dir, f"sim_{sim_idx:04d}.jsonl")

    cmd = [
        binary, "--spatial",
        "--grid-size", str(grid_size),
        "--p-move", str(p_move),
        "--l", str(l_val),
        "--r", str(r_val),
        "--max-gen", str(max_gen),
        "--output-interval", str(output_interval),
        "--qess-threshold", "0",
        "--no-viz",
        "--seed", str(seed),
        "--out", out_file,
    ]

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return (sim_idx, None, elapsed)

    snapshots = []
    final_summary = None
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "spatial_snapshot":
                snapshots.append(obj)
            elif obj.get("type") == "spatial_final":
                final_summary = obj

    metrics = extract_final_metrics(snapshots, final_summary)
    return (sim_idx, metrics, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics extraction  (now includes core/cloud classification)
# ──────────────────────────────────────────────────────────────────────────────
def extract_final_metrics(snapshots, final_summary):
    """Extract key metrics from the final snapshot.

    Core/cloud classification follows the TNM literature
    (Christensen et al. 2002, Jensen & Sibani):
      - A species is "core" if its global abundance > mean abundance (N/γ)
      - Cloud = everything else
    We also compute a persistence-based core using occupancy across
    the last 50% of snapshots.
    """
    if not snapshots:
        return None

    final = snapshots[-1]
    n_patches = len(final["patches"])

    total_n = final["total_n"]
    mean_n = final["mean_n"]
    mean_s = final["mean_s"]
    gamma = final["gamma_s"]
    beta = gamma / mean_s if mean_s > 0 else 0

    # Per-patch distributions
    patch_ns = [p["n"] for p in final["patches"]]
    patch_ss = [p["s"] for p in final["patches"]]
    cv_n = (np.std(patch_ns) / np.mean(patch_ns)
            if np.mean(patch_ns) > 0 else 0)

    # ── SAD: aggregate species counts across all patches ─────────────────
    global_species = Counter()
    for p in final["patches"]:
        if "species" in p:
            for genome_id, count in p["species"].items():
                global_species[genome_id] += count
    abundances = sorted(global_species.values(), reverse=True)

    # ── Core/cloud classification (abundance threshold) ──────────────────
    if gamma > 0 and total_n > 0:
        mean_abundance = total_n / gamma  # = N / γ
        core_species = {sp: n for sp, n in global_species.items()
                        if n >= mean_abundance}
        n_core = len(core_species)
        n_core_pop = sum(core_species.values())
        f_core = n_core_pop / total_n  # fraction of N held by core
        cloud_richness = gamma - n_core
    else:
        n_core = 0
        f_core = 0.0
        cloud_richness = 0

    # ── Persistence-based core (last 50% of snapshots) ───────────────────
    n_snaps = len(snapshots)
    half = n_snaps // 2
    late_snaps = snapshots[half:]
    presence_count = Counter()  # genome_id → # snapshots present in
    for snap in late_snaps:
        present_in_snap = set()
        for p in snap["patches"]:
            if "species" in p:
                for genome_id in p["species"]:
                    present_in_snap.add(genome_id)
        for gid in present_in_snap:
            presence_count[gid] += 1

    n_late = len(late_snaps)
    if n_late > 0:
        # "persistent core" = present in >50% of late snapshots
        persistent_core = {gid for gid, cnt in presence_count.items()
                           if cnt > n_late * 0.5}
        n_persistent = len(persistent_core)
    else:
        n_persistent = 0

    # ── Spatial occupancy of core species ─────────────────────────────────
    if n_core > 0:
        core_ids = set(core_species.keys())
        patches_with_core = 0
        for p in final["patches"]:
            if "species" in p:
                if core_ids & set(p["species"].keys()):
                    patches_with_core += 1
        core_spatial_occ = patches_with_core / n_patches
    else:
        core_spatial_occ = 0.0

    # ── Cumulative observed species over time ────────────────────────────
    all_observed = set()
    cum_species = []
    for snap in snapshots:
        for p in snap["patches"]:
            if "species" in p:
                for genome_id in p["species"]:
                    all_observed.add(genome_id)
        cum_species.append((snap["gen"], len(all_observed)))

    return {
        "total_n": total_n,
        "mean_n": mean_n,
        "mean_s": mean_s,
        "gamma": gamma,
        "beta": beta,
        "cv_n": cv_n,
        "n_patches": n_patches,
        "abundances": abundances,
        "cum_species": cum_species,
        "total_observed": len(all_observed),
        # Core/cloud metrics
        "n_core": n_core,
        "f_core": f_core,
        "cloud_richness": cloud_richness,
        "n_persistent": n_persistent,
        "core_spatial_occ": core_spatial_occ,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (4×4 grid: 16 panels)
# ──────────────────────────────────────────────────────────────────────────────
def plot_ensemble(all_metrics, args):
    """Create a multi-panel ensemble summary figure with core/cloud metrics."""

    fig = plt.figure(figsize=(22, 22))
    fig.suptitle(
        f"Spatial TNM Ensemble: {len(all_metrics)} sims "
        f"\u00d7 {args.grid_size}\u00d7{args.grid_size} grid "
        f"\u00d7 {args.max_gen} gens\n"
        f"L={args.l}, R={args.r}, p_move={args.p_move}",
        fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 4, hspace=0.40, wspace=0.35,
                           top=0.94, bottom=0.04, left=0.06, right=0.96)

    n = len(all_metrics)
    line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    def hist_panel(pos, key, color, xlabel, title, fmt=".0f"):
        ax = fig.add_subplot(gs[pos])
        vals = [m[key] for m in all_metrics]
        ax.hist(vals, bins=min(20, max(5, n // 3)), color=color,
                edgecolor="white", alpha=0.85)
        med = np.median(vals)
        ax.axvline(med, color="red", ls="--", lw=1.5,
                   label=f"median={med:{fmt}}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # ── Row 1: Population & diversity distributions ──────────────────────
    hist_panel((0, 0), "total_n", "#2196F3", "Total N",
               "Total Population")
    hist_panel((0, 1), "mean_n", "#4CAF50", "Mean N per patch",
               "Patch Population Density", ".1f")
    hist_panel((0, 2), "mean_s", "#FF9800",
               "Mean S per patch (\u03b1)", "\u03b1 Diversity", ".1f")
    hist_panel((0, 3), "gamma", "#9C27B0",
               "\u03b3 (unique genomes)", "\u03b3 Diversity")

    # ── Row 2: Spatial & compositional metrics ───────────────────────────
    hist_panel((1, 0), "beta", "#E91E63",
               "\u03b2 = \u03b3 / \u0101", "\u03b2 Diversity (Whittaker)", ".1f")
    hist_panel((1, 1), "cv_n", "#00BCD4",
               "CV(N) across patches", "Spatial Heterogeneity (N)", ".3f")
    hist_panel((1, 2), "total_observed", "#795548",
               "Total observed species", "Genome Exploration (total)")

    # N vs gamma scatter
    ax = fig.add_subplot(gs[1, 3])
    ns = [m["total_n"] for m in all_metrics]
    gs_vals = [m["gamma"] for m in all_metrics]
    ax.scatter(ns, gs_vals, c="#607D8B", alpha=0.6, s=30,
               edgecolor="white", lw=0.5)
    ax.set_xlabel("Total N")
    ax.set_ylabel("\u03b3 diversity")
    ax.set_title("N\u2013\u03b3 Relationship")
    if len(ns) > 2:
        r_val = np.corrcoef(ns, gs_vals)[0, 1]
        ax.text(0.05, 0.95, f"r = {r_val:.2f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold")

    # ── Row 3: Core/cloud metrics ────────────────────────────────────────
    hist_panel((2, 0), "n_core", "#D32F2F",
               "Core species (n\u2095\u2092\u2093\u2091)", "Core Community Size")
    hist_panel((2, 1), "f_core", "#F57C00",
               "Fraction of N in core", "Core Dominance", ".2f")
    hist_panel((2, 2), "cloud_richness", "#7B1FA2",
               "Cloud species (\u03b3 \u2212 n\u2095\u2092\u2093\u2091)",
               "Cloud Richness")
    hist_panel((2, 3), "n_persistent", "#1976D2",
               "Persistent species (>50% occ.)",
               "Temporal Persistence", ".0f")

    # ── Row 4: Trajectories + SAD + core scatter ─────────────────────────

    # Cumulative species spaghetti
    ax = fig.add_subplot(gs[3, 0:2])
    for i, m in enumerate(all_metrics):
        gens = [c[0] for c in m["cum_species"]]
        counts = [c[1] for c in m["cum_species"]]
        ax.plot(gens, counts, color=line_colors[i], alpha=0.4, lw=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative observed species")
    ax.set_title(f"Genome Exploration Trajectories (n={n})")

    # SAD overlay
    ax = fig.add_subplot(gs[3, 2])
    for i, m in enumerate(all_metrics):
        if m["abundances"]:
            ranks = np.arange(1, len(m["abundances"]) + 1)
            ax.plot(ranks, m["abundances"],
                    color=line_colors[i], alpha=0.4, lw=0.8)
    ax.set_xlabel("Species Rank")
    ax.set_ylabel("Abundance")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(f"Rank-Abundance (n={n})")

    # Core size vs gamma scatter — core/total ratio
    ax = fig.add_subplot(gs[3, 3])
    cores = [m["n_core"] for m in all_metrics]
    gammas = [m["gamma"] for m in all_metrics]
    f_cores = [m["f_core"] for m in all_metrics]
    sc = ax.scatter(gammas, cores, c=f_cores, cmap="RdYlGn",
                    alpha=0.7, s=40, edgecolor="white", lw=0.5,
                    vmin=0, vmax=1)
    ax.set_xlabel("\u03b3 diversity")
    ax.set_ylabel("Core species")
    ax.set_title("Core vs \u03b3 (color=f\u2095\u2092\u2093\u2091)")
    plt.colorbar(sc, ax=ax, label="f_core", shrink=0.8)
    # Add 1:1 line
    lim = max(max(gammas) if gammas else 1, max(cores) if cores else 1)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, lw=1)

    # Save
    out_name = (f"Figure_Ensemble_{args.grid_size}x{args.grid_size}"
                f"_{len(all_metrics)}sims")
    plt.savefig(f"{out_name}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_name}.pdf", dpi=200, bbox_inches="tight")
    print(f"\u2713 {out_name}.png / .pdf saved")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n_patches = args.grid_size ** 2
    n_workers = args.parallel if args.parallel > 0 else os.cpu_count()
    n_workers = min(n_workers, args.n_sims)

    # Pre-generate unique seeds
    rng = random.Random(42)
    seeds = [rng.randint(0, 2**63) for _ in range(args.n_sims)]

    print(f"\u2554{'=' * 58}\u2557")
    print(f"\u2551  Ensemble Run: {args.n_sims} sims "
          f"\u00d7 {args.grid_size}\u00d7{args.grid_size} grid "
          f"({n_patches} patches)")
    print(f"\u2551  max_gen={args.max_gen}  L={args.l}  R={args.r}  "
          f"p_move={args.p_move}")
    print(f"\u2551  parallel={n_workers} workers "
          f"(RAYON_NUM_THREADS=1 per sim)")
    print(f"\u2551  out_dir: {out_dir}")
    print(f"\u255a{'=' * 58}\u255d")
    print()

    all_metrics = []
    t_start = time.time()
    done_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run_single_sim,
                i, args.binary, args.grid_size, args.p_move, args.l,
                args.r, args.max_gen, args.output_interval, out_dir,
                seeds[i],
            ): i
            for i in range(args.n_sims)
        }

        for future in as_completed(futures):
            sim_idx, metrics, elapsed = future.result()
            done_count += 1

            if metrics is None:
                print(f"  [{done_count:3d}/{args.n_sims}] sim {sim_idx:4d} "
                      f"FAILED ({elapsed:.1f}s)")
                continue

            all_metrics.append(metrics)
            print(
                f"  [{done_count:3d}/{args.n_sims}] sim {sim_idx:4d}  "
                f"N={metrics['total_n']:6d}  "
                f"\u03b1={metrics['mean_s']:.1f}  "
                f"\u03b3={metrics['gamma']:4d}  "
                f"core={metrics['n_core']:3d} "
                f"({metrics['f_core']:.0%})  "
                f"cloud={metrics['cloud_richness']:4d}  "
                f"persist={metrics['n_persistent']:3d}  "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - t_start
    print(f"\n  Completed {len(all_metrics)}/{args.n_sims} sims "
          f"in {total_time:.1f}s")

    if not all_metrics:
        print("No successful simulations \u2014 nothing to plot.")
        return

    # Summary statistics
    print(f"\n  \u2500\u2500 Summary Statistics "
          f"{'\u2500' * 40}")
    for key in ["total_n", "mean_n", "mean_s", "gamma", "beta", "cv_n",
                "n_core", "f_core", "cloud_richness", "n_persistent",
                "core_spatial_occ", "total_observed"]:
        vals = [m[key] for m in all_metrics]
        print(f"    {key:18s}  mean={np.mean(vals):8.2f}  "
              f"sd={np.std(vals):8.2f}  "
              f"min={np.min(vals):8.2f}  max={np.max(vals):8.2f}")

    # Plot
    print(f"\n  Generating ensemble figure...")
    plot_ensemble(all_metrics, args)


if __name__ == "__main__":
    main()
