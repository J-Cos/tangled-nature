#!/usr/bin/env python3
"""
Analyze harvesting fork experiment results.

Reads fork output JSONL files and computes impact metrics:
- ΔN, ΔS between baseline (first 200 gens) and harvest (last 200 gens) periods
- Whether target species went extinct
- Community-level stability metrics
"""

import json
import os
import sys
import csv
import glob
import numpy as np
from pathlib import Path


def parse_fork_output(fname):
    """Parse a fork JSONL file. Returns list of {gen, n, s} dicts."""
    rows = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess"):
                continue
            rows.append({
                "gen": d.get("gen", 0),
                "n": d.get("n", 0),
                "s": d.get("s", 0),
                "species": d.get("species"),
            })
    return rows


def compute_fork_metrics(rows, harvest_after=200):
    """
    Compute impact metrics from a fork's time series.
    Split into baseline (first harvest_after gens) and harvest (rest) periods.
    """
    if len(rows) < 2:
        return None

    gens = [r["gen"] for r in rows]
    ns = [r["n"] for r in rows]
    ss = [r["s"] for r in rows]

    # Find the baseline/harvest split point (relative to start gen)
    start_gen = gens[0]
    split_gen = start_gen + harvest_after

    baseline_n = [n for g, n in zip(gens, ns) if g < split_gen]
    harvest_n = [n for g, n in zip(gens, ns) if g >= split_gen]
    baseline_s = [s for g, s in zip(gens, ss) if g < split_gen]
    harvest_s = [s for g, s in zip(gens, ss) if g >= split_gen]

    if not baseline_n or not harvest_n:
        return None

    mean_n_base = np.mean(baseline_n)
    mean_n_harv = np.mean(harvest_n)
    mean_s_base = np.mean(baseline_s)
    mean_s_harv = np.mean(harvest_s)

    # Check if target species survived in final snapshot
    final_species = rows[-1].get("species")
    target_survived = None  # will be set by caller

    return {
        "mean_n_baseline": mean_n_base,
        "mean_n_harvest": mean_n_harv,
        "mean_s_baseline": mean_s_base,
        "mean_s_harvest": mean_s_harv,
        "delta_n": mean_n_harv - mean_n_base,
        "delta_s": mean_s_harv - mean_s_base,
        "pct_n_change": (mean_n_harv - mean_n_base) / mean_n_base * 100 if mean_n_base > 0 else 0,
        "pct_s_change": (mean_s_harv - mean_s_base) / mean_s_base * 100 if mean_s_base > 0 else 0,
        "min_n_harvest": min(harvest_n),
        "final_n": ns[-1],
        "final_s": ss[-1],
    }


def extract_top_species(state_file, n_species=32):
    """Extract top N species by abundance from a saved qESS state file."""
    with open(state_file) as f:
        state = json.load(f)
    species = state.get("species", {})
    # Sort by abundance (descending)
    sorted_sp = sorted(species.items(), key=lambda x: x[1], reverse=True)
    return [(int(g), c) for g, c in sorted_sp[:n_species]]


def analyze_all_forks(data_dir, results_dir, n_sims=32, n_targets=32,
                       harvest_after=200):
    """Analyze all fork outputs and produce metrics CSV."""
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []

    for sim_id in range(1, n_sims + 1):
        sim_prefix = f"sim_{sim_id:02d}"

        for rank in range(1, n_targets + 1):
            fname = os.path.join(data_dir, f"{sim_prefix}_rank{rank:02d}.jsonl")
            if not os.path.exists(fname):
                continue

            rows = parse_fork_output(fname)
            metrics = compute_fork_metrics(rows, harvest_after=harvest_after)
            if metrics is None:
                continue

            metrics["sim_id"] = sim_id
            metrics["target_rank"] = rank
            all_metrics.append(metrics)

    if not all_metrics:
        print("No fork data found!")
        return None

    # Write CSV
    fieldnames = [
        "sim_id", "target_rank",
        "mean_n_baseline", "mean_n_harvest", "delta_n", "pct_n_change",
        "mean_s_baseline", "mean_s_harvest", "delta_s", "pct_s_change",
        "min_n_harvest", "final_n", "final_s",
    ]

    out_path = os.path.join(results_dir, "fork_metrics.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_metrics)

    print(f"  → Wrote {len(all_metrics)} rows to {out_path}")
    return all_metrics


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    metrics = analyze_all_forks(data_dir, results_dir)
    if metrics:
        delta_ns = [m["pct_n_change"] for m in metrics]
        print(f"  Mean %ΔN: {np.mean(delta_ns):.1f}% (SD {np.std(delta_ns):.1f}%)")
