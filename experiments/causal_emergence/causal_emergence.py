#!/usr/bin/env python3
"""
Causal Emergence Analysis for the Tangled Nature Model.

Implements Erik Hoel's causal emergence framework:
- Effective Information (EI) = determinism − degeneracy
- Computed from empirical Transition Probability Matrices (TPMs)
- At multiple coarse-graining scales (micro → macro)
- Over sliding windows to track temporal dynamics

Usage:
    python3 causal_emergence.py <input.jsonl> <results_dir> [--window 500] [--step 100]
"""

import json
import os
import sys
import csv
import argparse
import numpy as np
from collections import Counter


# ─── Core Information-Theoretic Functions ────────────────────────

def shannon_entropy(probs):
    """Shannon entropy H(p) in bits, handling zeros."""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def compute_ei(tpm):
    """
    Compute Effective Information from a transition probability matrix.

    EI = determinism − degeneracy

    determinism = log₂(N) − (1/N) Σᵢ H(Tᵢ)
        How certain each state's transition is (avg row entropy)

    degeneracy = log₂(N) − H((1/N) Σᵢ Tᵢ)
        How distinguishable the effects are (entropy of avg column)

    Parameters:
        tpm: N×N numpy array, rows are conditional distributions P(j|i)

    Returns:
        dict with keys: ei, determinism, degeneracy, n_states
    """
    N = tpm.shape[0]
    if N <= 1:
        return {"ei": 0.0, "determinism": 0.0, "degeneracy": 0.0, "n_states": N}

    log2N = np.log2(N)

    # Determinism: log₂(N) − average row entropy
    row_entropies = np.array([shannon_entropy(tpm[i]) for i in range(N)])
    determinism = log2N - np.mean(row_entropies)

    # Degeneracy: log₂(N) − entropy of the average row (stationary effect dist)
    avg_row = np.mean(tpm, axis=0)
    degeneracy = log2N - shannon_entropy(avg_row)

    ei = determinism - degeneracy

    return {
        "ei": ei,
        "determinism": determinism,
        "degeneracy": degeneracy,
        "n_states": N,
    }


# ─── Coarse-Graining Functions ──────────────────────────────────

def coarse_grain_micro(species_dict, top_k=10, n_abundance_bins=5):
    """
    Micro scale: discretize the top-k species abundances.

    Takes top-k species by total historical abundance, bins each count
    into n_abundance_bins using log-scale boundaries.

    Returns: tuple state identifier
    """
    if not species_dict:
        return (0,) * top_k

    sorted_sp = sorted(species_dict.items(), key=lambda x: x[1], reverse=True)
    abundances = [c for _, c in sorted_sp[:top_k]]
    # Pad if fewer than top_k
    while len(abundances) < top_k:
        abundances.append(0)

    # Log-bin abundances: 0, 1-3, 4-15, 16-63, 64+
    def log_bin(a):
        if a == 0:
            return 0
        elif a <= 3:
            return 1
        elif a <= 15:
            return 2
        elif a <= 63:
            return 3
        else:
            return 4

    return tuple(log_bin(a) for a in abundances)


def coarse_grain_meso1(species_dict, n_rank_bins=5):
    """
    Meso-1 scale: SAD shape via binned rank-abundance.

    Bins the rank-abundance distribution into n_rank_bins quantiles
    and returns the proportion in each bin.

    Returns: tuple state identifier
    """
    if not species_dict:
        return (0,) * n_rank_bins

    abundances = sorted(species_dict.values(), reverse=True)
    total = sum(abundances)
    if total == 0:
        return (0,) * n_rank_bins

    S = len(abundances)
    # Split into n_rank_bins groups and compute fraction of total in each
    bin_size = max(1, S // n_rank_bins)
    fractions = []
    for b in range(n_rank_bins):
        start = b * bin_size
        end = start + bin_size if b < n_rank_bins - 1 else S
        frac = sum(abundances[start:end]) / total
        # Discretize fraction into 5 levels
        if frac < 0.05:
            fractions.append(0)
        elif frac < 0.15:
            fractions.append(1)
        elif frac < 0.30:
            fractions.append(2)
        elif frac < 0.50:
            fractions.append(3)
        else:
            fractions.append(4)

    return tuple(fractions)


def coarse_grain_meso2(n, s, n_bins=5, s_bins=5,
                        n_range=(0, 10000), s_range=(0, 100)):
    """
    Meso-2 scale: (N_bin, S_bin) pair.

    Returns: tuple (n_bin, s_bin)
    """
    n_step = (n_range[1] - n_range[0]) / n_bins
    s_step = (s_range[1] - s_range[0]) / s_bins

    n_bin = min(n_bins - 1, max(0, int((n - n_range[0]) / n_step)))
    s_bin = min(s_bins - 1, max(0, int((s - s_range[0]) / s_step)))

    return (n_bin, s_bin)


def coarse_grain_macro(n, n_bins=5, n_range=(0, 10000)):
    """
    Macro scale: total N only, binned.

    Returns: int bin index
    """
    n_step = (n_range[1] - n_range[0]) / n_bins
    return min(n_bins - 1, max(0, int((n - n_range[0]) / n_step)))


# ─── TPM Building ───────────────────────────────────────────────

def build_tpm(states):
    """
    Build an empirical transition probability matrix from a sequence of states.

    Parameters:
        states: list of hashable state identifiers

    Returns:
        tpm: N×N numpy array (rows normalised to sum to 1)
        state_labels: list mapping matrix indices to state labels
    """
    if len(states) < 2:
        return np.array([[1.0]]), [states[0]] if states else [0]

    # Count transitions
    transitions = Counter()
    for i in range(len(states) - 1):
        transitions[(states[i], states[i + 1])] += 1

    # Get unique states in order of first appearance
    seen = set()
    state_labels = []
    for s in states:
        if s not in seen:
            seen.add(s)
            state_labels.append(s)

    N = len(state_labels)
    state_to_idx = {s: i for i, s in enumerate(state_labels)}

    tpm = np.zeros((N, N))
    for (s_from, s_to), count in transitions.items():
        tpm[state_to_idx[s_from], state_to_idx[s_to]] = count

    # Normalise rows
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    tpm = tpm / row_sums

    return tpm, state_labels


# ─── Data Loading ────────────────────────────────────────────────

def load_simulation(fname):
    """Load per-generation snapshots from JSONL. Returns list of dicts."""
    rows = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess"):
                continue
            species = {}
            if d.get("species"):
                species = {int(g): int(c) for g, c in d["species"]}
            rows.append({
                "gen": d.get("gen", 0),
                "n": d.get("n", 0),
                "s": d.get("s", len(species)),
                "species": species,
            })
    return rows


# ─── Main Analysis ───────────────────────────────────────────────

def analyze_causal_emergence(snapshots, window=500, step=100):
    """
    Compute EI at 4 scales over sliding windows.

    Returns list of dicts with window metrics.
    """
    if len(snapshots) < window:
        print(f"Warning: only {len(snapshots)} snapshots, need {window}")
        return []

    # Determine N/S ranges for binning
    all_n = [s["n"] for s in snapshots]
    all_s = [s["s"] for s in snapshots]
    n_range = (max(0, min(all_n) - 100), max(all_n) + 100)
    s_range = (max(0, min(all_s) - 5), max(all_s) + 5)

    # Find persistent top-k genomes (by total abundance across all snapshots)
    genome_totals = Counter()
    for snap in snapshots:
        for g, c in snap["species"].items():
            genome_totals[g] += c
    top_k_genomes = [g for g, _ in genome_totals.most_common(10)]

    results = []
    n_windows = (len(snapshots) - window) // step + 1

    for wi in range(n_windows):
        start = wi * step
        end = start + window
        win = snapshots[start:end]

        gen_start = win[0]["gen"]
        gen_end = win[-1]["gen"]

        # Build state sequences at each scale
        micro_states = []
        meso1_states = []
        meso2_states = []
        macro_states = []

        for snap in win:
            # Micro: top-k genome abundances (using window-local ranking)
            sp = snap["species"]
            # Use globally consistent top_k genomes
            micro_vec = tuple(
                coarse_grain_micro(
                    {g: sp.get(g, 0) for g in top_k_genomes}, top_k=10
                )
            )
            micro_states.append(micro_vec)

            # Meso-1: SAD shape
            meso1_states.append(coarse_grain_meso1(sp))

            # Meso-2: (N_bin, S_bin)
            meso2_states.append(
                coarse_grain_meso2(snap["n"], snap["s"],
                                    n_range=n_range, s_range=s_range)
            )

            # Macro: N_bin only
            macro_states.append(
                coarse_grain_macro(snap["n"], n_range=n_range)
            )

        # Build TPMs and compute EI at each scale
        scales = {}
        for name, states in [("micro", micro_states),
                              ("meso1", meso1_states),
                              ("meso2", meso2_states),
                              ("macro", macro_states)]:
            tpm, labels = build_tpm(states)
            ei_result = compute_ei(tpm)
            scales[name] = ei_result

        # Determine scale of max EI
        max_scale = max(scales, key=lambda k: scales[k]["ei"])

        row = {
            "window_start": gen_start,
            "window_end": gen_end,
            "mean_n": np.mean([s["n"] for s in win]),
            "mean_s": np.mean([s["s"] for s in win]),
        }
        for name in ["micro", "meso1", "meso2", "macro"]:
            row[f"ei_{name}"] = scales[name]["ei"]
            row[f"det_{name}"] = scales[name]["determinism"]
            row[f"deg_{name}"] = scales[name]["degeneracy"]
            row[f"nstates_{name}"] = scales[name]["n_states"]
        row["max_ei_scale"] = max_scale

        results.append(row)

        if wi % 50 == 0:
            print(f"  Window {wi+1}/{n_windows}: gen {gen_start}–{gen_end}"
                  f" | EI: micro={scales['micro']['ei']:.3f}"
                  f" meso1={scales['meso1']['ei']:.3f}"
                  f" meso2={scales['meso2']['ei']:.3f}"
                  f" macro={scales['macro']['ei']:.3f}"
                  f" → max={max_scale}")

    return results


def save_results(results, out_path):
    """Save analysis results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"  → Wrote {len(results)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Causal emergence analysis")
    parser.add_argument("input", help="JSONL simulation output")
    parser.add_argument("results_dir", help="Output directory")
    parser.add_argument("--window", type=int, default=500,
                         help="Sliding window size (gens)")
    parser.add_argument("--step", type=int, default=100,
                         help="Step between windows")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading simulation data...")
    snapshots = load_simulation(args.input)
    print(f"  {len(snapshots)} snapshots loaded")

    if len(snapshots) < args.window:
        print("Not enough data for analysis")
        return

    print(f"Analyzing with window={args.window}, step={args.step}...")
    results = analyze_causal_emergence(snapshots, args.window, args.step)

    out_path = os.path.join(args.results_dir, "causal_emergence.csv")
    save_results(results, out_path)

    # Summary statistics
    if results:
        scale_counts = Counter(r["max_ei_scale"] for r in results)
        print("\nScale of max EI distribution:")
        for scale, count in sorted(scale_counts.items(),
                                     key=lambda x: -x[1]):
            pct = count / len(results) * 100
            print(f"  {scale}: {count} windows ({pct:.0f}%)")


if __name__ == "__main__":
    main()
