#!/usr/bin/env python3
"""
Quake-focused METE analysis for the Tangled Nature Model.

Runs a single long TNM simulation with per-generation species output,
detects endogenous quakes via species turnover (Jaccard distance),
and produces a multi-panel figure showing:
  - TaNa genome heatmap with quake boundaries
  - N/S time series with quake regions shaded
  - Species turnover rate
  - METE R² over time around quakes
  - SAD comparisons at stable vs quake timepoints
"""

import json
import os
import sys
import subprocess
import tempfile
import csv
import numpy as np
from collections import OrderedDict

# ── Configuration ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "max_gen": 100_000,
    "p_mut": 0.025,
    "seed": 42,
    "burnin": 1000,       # skip first N gens for analysis
    "turnover_window": 50,  # smoothing window for turnover
    "quake_threshold_factor": 2.5,  # quake = turnover > factor × median
    "quake_min_gap": 100,  # merge quakes closer than this
    "mete_sample_interval": 5,  # fit METE every N gens (1=every gen, slow)
}


# ── Parsing ──────────────────────────────────────────────────────

def parse_species_jsonl(fname, burnin=1000):
    """
    Parse JSONL with species data. Returns list of dicts:
    [{"gen": int, "n": int, "s": int, "species": {genome: count, ...}}, ...]
    Only includes generations >= burnin that have species data.
    """
    snapshots = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess"):
                continue
            gen = d.get("gen", 0)
            if gen < burnin:
                continue
            species_raw = d.get("species")
            if not species_raw:
                continue
            species = {int(g): int(c) for g, c in species_raw}
            snapshots.append({
                "gen": gen,
                "n": d.get("n", sum(species.values())),
                "s": len(species),
                "species": species,
            })
    return snapshots


# ── Quake Detection ──────────────────────────────────────────────

def jaccard_distance(set_a, set_b):
    """Jaccard distance = 1 - |A ∩ B| / |A ∪ B|. Returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union if union > 0 else 0.0


def compute_turnover(snapshots):
    """Compute per-generation Jaccard turnover between consecutive species sets."""
    turnover = [0.0]  # first gen has no predecessor
    for i in range(1, len(snapshots)):
        set_prev = set(snapshots[i - 1]["species"].keys())
        set_curr = set(snapshots[i]["species"].keys())
        turnover.append(jaccard_distance(set_prev, set_curr))
    return np.array(turnover)


def smooth(arr, window):
    """Simple rolling mean smoother."""
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinkage
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_quakes(turnover, gens, window=50, threshold_factor=2.5, min_gap=100,
                  ns=None):
    """
    Detect quakes as contiguous regions of elevated instability.

    Uses rolling CV of N (population variability) as the primary signal,
    combined with species turnover. At high mutation rates, Jaccard turnover
    is nearly constant — but population crashes still indicate reorganization.

    Returns list of (start_gen, end_gen, peak_gen, peak_signal) tuples.
    """
    # Use rolling CV of N as primary signal if available
    if ns is not None and len(ns) > window:
        signal = np.zeros(len(ns))
        for i in range(window, len(ns)):
            chunk = ns[i - window:i]
            m = np.mean(chunk)
            if m > 0:
                signal[i] = np.std(chunk) / m
        # Also incorporate sudden drops in N
        for i in range(1, len(ns)):
            if ns[i - 1] > 0:
                rel_drop = (ns[i - 1] - ns[i]) / ns[i - 1]
                if rel_drop > 0.15:  # >15% drop in one generation
                    # Boost the signal around this point
                    lo = max(0, i - window // 2)
                    hi = min(len(signal), i + window // 2)
                    signal[lo:hi] = np.maximum(signal[lo:hi], rel_drop)
    else:
        signal = smooth(turnover, window)

    # Threshold: factor × median of non-zero signal
    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return []
    median_s = np.median(nonzero)
    threshold = threshold_factor * median_s

    # Find regions above threshold
    above = signal > threshold
    quake_regions = []
    in_quake = False
    start_idx = 0

    for i in range(len(above)):
        if above[i] and not in_quake:
            start_idx = i
            in_quake = True
        elif not above[i] and in_quake:
            region = signal[start_idx:i]
            peak_idx = start_idx + np.argmax(region)
            quake_regions.append((
                gens[start_idx], gens[i - 1],
                gens[peak_idx], float(signal[peak_idx])
            ))
            in_quake = False

    if in_quake:
        region = signal[start_idx:]
        peak_idx = start_idx + np.argmax(region)
        quake_regions.append((
            gens[start_idx], gens[-1],
            gens[peak_idx], float(signal[peak_idx])
        ))

    # Merge quakes that are too close
    if len(quake_regions) <= 1:
        return quake_regions

    merged = [quake_regions[0]]
    for q in quake_regions[1:]:
        prev = merged[-1]
        if q[0] - prev[1] < min_gap:
            peak = prev if prev[3] >= q[3] else q
            merged[-1] = (prev[0], q[1], peak[2], peak[3])
        else:
            merged.append(q)

    return merged


# ── METE Fitting ─────────────────────────────────────────────────

def fit_mete_batch(snapshots, sample_interval=5):
    """
    Fit METE SAD predictions for snapshots at given interval.
    Uses R/meteR via subprocess for batch efficiency.
    Returns dict {gen: {"r_squared": float, "kl_div": float, ...}}
    """
    # Select snapshots to fit
    selected = [s for i, s in enumerate(snapshots) if i % sample_interval == 0]
    if not selected:
        return {}

    # Write abundance data to temp CSV for R
    tmpdir = tempfile.mkdtemp()
    abd_file = os.path.join(tmpdir, "abundances.csv")
    with open(abd_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "abundance"])
        for s in selected:
            for genome, count in sorted(s["species"].items()):
                w.writerow([s["gen"], count])

    out_file = os.path.join(tmpdir, "metrics.csv")

    # R script for batch METE fitting (parallelized)
    r_script = f"""
suppressPackageStartupMessages(library(meteR))

shannon <- function(n_vec) {{
  n_vec <- n_vec[n_vec > 0]
  p <- n_vec / sum(n_vec)
  -sum(p * log(p))
}}

kl_divergence <- function(obs_p, pred_p) {{
  eps <- 1e-12
  obs_p <- obs_p + eps; pred_p <- pred_p + eps
  obs_p <- obs_p / sum(obs_p); pred_p <- pred_p / sum(pred_p)
  sum(obs_p * log(obs_p / pred_p))
}}

d <- read.csv("{abd_file}", stringsAsFactors=FALSE)
gens <- unique(d$gen)

# Pre-split
abd_by_gen <- split(d$abundance, d$gen)

n_cores <- parallel::detectCores(logical=FALSE)
if (is.na(n_cores) || n_cores < 1) n_cores <- 1

results <- parallel::mclapply(gens, function(g) {{
  abd <- abd_by_gen[[as.character(g)]]
  S0 <- length(abd); N0 <- sum(abd)
  if (S0 < 2 || N0 < S0) return(NULL)
  tryCatch({{
    esf <- meteESF(S0=S0, N0=N0)
    mete_sad <- sad(esf)
    obs_r <- sort(abd, decreasing=TRUE)
    pred_r <- meteDist2Rank(mete_sad)
    len <- min(length(obs_r), length(pred_r))
    obs <- obs_r[1:len]; pred <- pred_r[1:len]
    lo <- log(obs+1); lp <- log(pred+1)
    ss_r <- sum((lo-lp)^2); ss_t <- sum((lo-mean(lo))^2)
    r2 <- ifelse(ss_t>0, 1-ss_r/ss_t, NA)
    kl <- kl_divergence(obs, pred)
    data.frame(gen=g, S=S0, N=N0, r_squared=r2, kl_div=kl,
               shannon_obs=shannon(abd), shannon_pred=shannon(pred_r))
  }}, error=function(e) NULL)
}}, mc.cores=n_cores)

metrics <- do.call(rbind, results[!sapply(results, is.null)])
write.csv(metrics, "{out_file}", row.names=FALSE)
cat(sprintf("Fitted %d/%d snapshots\\n", nrow(metrics), length(gens)))
"""

    r_script_file = os.path.join(tmpdir, "fit.R")
    with open(r_script_file, "w") as f:
        f.write(r_script)

    print(f"  Fitting METE for {len(selected)} snapshots (parallel R)...")
    result = subprocess.run(["Rscript", r_script_file],
                            capture_output=True, text=True, timeout=600)
    if result.stdout:
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    R stderr: {result.stderr[-500:]}")
        return {}

    # Parse results
    metrics = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gen = int(float(row["gen"]))
                metrics[gen] = {
                    "r_squared": float(row["r_squared"]) if row["r_squared"] != "NA" else np.nan,
                    "kl_div": float(row["kl_div"]) if row["kl_div"] != "NA" else np.nan,
                    "shannon_obs": float(row["shannon_obs"]),
                    "shannon_pred": float(row["shannon_pred"]),
                    "S": int(float(row["S"])),
                    "N": int(float(row["N"])),
                }

    return metrics


# ── Visualization ────────────────────────────────────────────────

def generate_figure(snapshots, turnover, quakes, mete_metrics, out_dir):
    """Generate multi-panel quake analysis figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    gens = np.array([s["gen"] for s in snapshots])
    ns = np.array([s["n"] for s in snapshots])
    ss = np.array([s["s"] for s in snapshots])

    # ── Build genome heatmap matrix ──────────────────────────────
    # Collect all genomes that ever appear
    all_genomes = set()
    for s in snapshots:
        all_genomes.update(s["species"].keys())
    all_genomes = sorted(all_genomes)
    genome_to_idx = {g: i for i, g in enumerate(all_genomes)}

    # Downsample for heatmap if too many time points
    max_cols = 2000
    if len(snapshots) > max_cols:
        step = len(snapshots) // max_cols
        hm_indices = list(range(0, len(snapshots), step))
    else:
        hm_indices = list(range(len(snapshots)))

    hm_gens = [gens[i] for i in hm_indices]
    hm_matrix = np.zeros((len(all_genomes), len(hm_indices)))
    for col, idx in enumerate(hm_indices):
        for genome, count in snapshots[idx]["species"].items():
            row = genome_to_idx[genome]
            hm_matrix[row, col] = count

    # Sort genomes by total abundance for better visualization
    total_abd = hm_matrix.sum(axis=1)
    sort_idx = np.argsort(-total_abd)
    hm_matrix = hm_matrix[sort_idx, :]
    # Keep only genomes that appear at least once
    nonzero_rows = hm_matrix.sum(axis=1) > 0
    hm_matrix = hm_matrix[nonzero_rows, :]

    # ── METE metrics arrays ──────────────────────────────────────
    met_gens = sorted(mete_metrics.keys())
    met_r2 = [mete_metrics[g]["r_squared"] for g in met_gens]
    met_kl = [mete_metrics[g]["kl_div"] for g in met_gens]

    # ── Figure layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(5, 2, hspace=0.35, wspace=0.30,
                           height_ratios=[1.5, 0.8, 0.8, 0.8, 1.0])

    # ── Panel A: Genome heatmap (top, full width) ────────────────
    ax_hm = fig.add_subplot(gs[0, :])
    im = ax_hm.imshow(hm_matrix, aspect="auto", interpolation="nearest",
                       cmap="inferno",
                       norm=LogNorm(vmin=max(1, hm_matrix[hm_matrix > 0].min()),
                                    vmax=hm_matrix.max()),
                       extent=[hm_gens[0], hm_gens[-1], hm_matrix.shape[0], 0])
    ax_hm.set_xlabel("Generation")
    ax_hm.set_ylabel("Genome (ranked by abundance)")
    ax_hm.set_title("Species Abundance Heatmap (TaNa)", fontweight="bold", fontsize=13)
    plt.colorbar(im, ax=ax_hm, label="Abundance", shrink=0.6, pad=0.02)

    # Mark quake regions on heatmap
    for q_start, q_end, q_peak, q_val in quakes:
        ax_hm.axvspan(q_start, q_end, alpha=0.15, color="cyan", zorder=5)
        ax_hm.axvline(q_peak, color="cyan", linewidth=0.8, alpha=0.6, zorder=6)

    # ── Panel B: N and S time series ─────────────────────────────
    ax_n = fig.add_subplot(gs[1, 0])
    ax_n.plot(gens, ns, color="#2166AC", linewidth=0.8)
    ax_n.set_xlabel("Generation")
    ax_n.set_ylabel("N")
    ax_n.set_title("Population", fontweight="bold", fontsize=11)
    ax_n.grid(True, alpha=0.2)
    for q_start, q_end, _, _ in quakes:
        ax_n.axvspan(q_start, q_end, alpha=0.15, color="red")

    ax_s = fig.add_subplot(gs[1, 1])
    ax_s.plot(gens, ss, color="#1B7837", linewidth=0.8)
    ax_s.set_xlabel("Generation")
    ax_s.set_ylabel("S")
    ax_s.set_title("Species Richness", fontweight="bold", fontsize=11)
    ax_s.grid(True, alpha=0.2)
    for q_start, q_end, _, _ in quakes:
        ax_s.axvspan(q_start, q_end, alpha=0.15, color="red")

    # ── Panel C: Turnover rate ───────────────────────────────────
    ax_t = fig.add_subplot(gs[2, :])
    smoothed_turnover = smooth(turnover, DEFAULT_CONFIG["turnover_window"])
    ax_t.plot(gens, smoothed_turnover, color="#762A83", linewidth=0.8)
    threshold = DEFAULT_CONFIG["quake_threshold_factor"] * np.median(smoothed_turnover)
    ax_t.axhline(threshold, color="red", linestyle="--", alpha=0.5,
                  label=f"Quake threshold ({DEFAULT_CONFIG['quake_threshold_factor']}× median)")
    for q_start, q_end, _, _ in quakes:
        ax_t.axvspan(q_start, q_end, alpha=0.15, color="red")
    ax_t.set_xlabel("Generation")
    ax_t.set_ylabel("Jaccard Turnover")
    ax_t.set_title("Species Turnover Rate (smoothed)", fontweight="bold", fontsize=11)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.2)

    # ── Panel D: METE R² over time ───────────────────────────────
    ax_r2 = fig.add_subplot(gs[3, 0])
    ax_r2.plot(met_gens, met_r2, color="#D6604D", linewidth=1.0)
    ax_r2.set_xlabel("Generation")
    ax_r2.set_ylabel("R² (rank-abundance)")
    ax_r2.set_title("METE SAD Fit Quality", fontweight="bold", fontsize=11)
    ax_r2.set_ylim(-0.1, 1.1)
    ax_r2.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.9")
    for q_start, q_end, _, _ in quakes:
        ax_r2.axvspan(q_start, q_end, alpha=0.15, color="red")
    ax_r2.legend(fontsize=8)
    ax_r2.grid(True, alpha=0.2)

    ax_kl = fig.add_subplot(gs[3, 1])
    ax_kl.plot(met_gens, met_kl, color="#4393C3", linewidth=1.0)
    ax_kl.set_xlabel("Generation")
    ax_kl.set_ylabel("KL Divergence")
    ax_kl.set_title("METE Prediction Error", fontweight="bold", fontsize=11)
    for q_start, q_end, _, _ in quakes:
        ax_kl.axvspan(q_start, q_end, alpha=0.15, color="red")
    ax_kl.grid(True, alpha=0.2)

    # ── Panel E: SAD comparisons at stable vs quake timepoints ───
    # Pick 2 stable and 2 quake timepoints
    stable_gens_available = [g for g in met_gens
                              if all(g < q[0] or g > q[1] for q in quakes)]
    quake_gens_available = []
    for q_start, q_end, q_peak, _ in quakes:
        closest = min(met_gens, key=lambda g: abs(g - q_peak))
        if closest not in quake_gens_available:
            quake_gens_available.append(closest)

    # Pick one stable and one quake for side-by-side SAD
    pick_stable = stable_gens_available[len(stable_gens_available) // 2] if stable_gens_available else None
    pick_quake = quake_gens_available[0] if quake_gens_available else None

    for col, (label, pick_gen, color) in enumerate([
        ("Stable (qESS)", pick_stable, "#2166AC"),
        ("Quake (reorganization)", pick_quake, "#D6604D"),
    ]):
        ax_sad = fig.add_subplot(gs[4, col])
        if pick_gen is None:
            ax_sad.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_sad.transAxes)
            continue

        # Find snapshot closest to pick_gen
        snap = min(snapshots, key=lambda s: abs(s["gen"] - pick_gen))
        abd = sorted(snap["species"].values(), reverse=True)
        S0 = len(abd)
        N0 = sum(abd)
        ranks = np.arange(1, S0 + 1)

        ax_sad.bar(ranks, abd, alpha=0.6, color=color,
                    label=f"Observed (gen {snap['gen']})", width=0.8)

        # METE prediction
        if S0 > 1 and N0 > S0:
            try:
                # Log-series-like METE rank-abundance
                beta = np.log(1 + N0 / S0) / S0
                pred = N0 * np.exp(-beta * ranks) / np.sum(np.exp(-beta * ranks))
                ax_sad.plot(ranks, pred, "k-", linewidth=2, label="METE prediction")
            except Exception:
                pass

        r2_val = mete_metrics.get(pick_gen, {}).get("r_squared", np.nan)
        ax_sad.set_xlabel("Rank")
        ax_sad.set_ylabel("Abundance")
        ax_sad.set_title(f"{label}: S={S0}, N={N0}, R²={r2_val:.2f}",
                          fontweight="bold", fontsize=10)
        ax_sad.legend(fontsize=8)
        ax_sad.grid(True, alpha=0.2)

    # ── Title ────────────────────────────────────────────────────
    n_quakes = len(quakes)
    fig.suptitle(
        f"TNM Quake Analysis: {n_quakes} quakes detected in {gens[-1] - gens[0]:.0f} generations",
        fontsize=15, fontweight="bold", y=0.99
    )

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "Figure_QuakeAnalysis.png")
    pdf = os.path.join(out_dir, "Figure_QuakeAnalysis.pdf")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"✓ {png} / .pdf saved")


# ── Main ─────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
    binary = os.path.join(project_dir, "target", "release", "tangled-nature")
    out_dir = os.path.join(script_dir, "quake_results")
    os.makedirs(out_dir, exist_ok=True)

    cfg = DEFAULT_CONFIG.copy()

    # Parse CLI overrides
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-").replace("-", "_")
            if key in cfg:
                cfg[key] = type(cfg[key])(val)

    sim_file = os.path.join(out_dir, "quake_sim.jsonl")

    print("=" * 64)
    print("  Quake-Focused METE Analysis")
    print(f"  {cfg['max_gen']} gens | p_mut={cfg['p_mut']} | seed={cfg['seed']}")
    print(f"  Burnin: {cfg['burnin']} gens | METE sample: every {cfg['mete_sample_interval']} gens")
    print("=" * 64)

    # ── Step 1: Run simulation ───────────────────────────────────
    skip_sim = "--skip-sim" in sys.argv
    if skip_sim and os.path.exists(sim_file):
        print("\n[Step 1] Skipping simulation (using existing data)")
        print(f"  File: {sim_file} ({os.path.getsize(sim_file) / 1e6:.1f} MB)")
    else:
        print("\n[Step 1] Running TNM simulation...")
        cmd = [
            binary,
            "--seed", str(cfg["seed"]),
            "--max-gen", str(cfg["max_gen"]),
            "--p-mut", str(cfg["p_mut"]),
            "--output-interval", "100",   # stdout monitoring every 100
            "--species-interval", "1",    # species to file EVERY gen
            "--qess-threshold", "0.0",    # don't stop at qESS
            "--no-viz",
            "--out", sim_file,
        ]
        print(f"  Command: {' '.join(cmd)}")
        # Redirect stdout to devnull (TNM prints snapshots to stdout)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}): {result.stderr[-500:]}")
            return
        print(f"  Simulation complete. Output: {sim_file}")
        print(f"  File size: {os.path.getsize(sim_file) / 1e6:.1f} MB")

    # ── Step 2: Parse species data ───────────────────────────────
    print("\n[Step 2] Parsing species data...")
    snapshots = parse_species_jsonl(sim_file, burnin=cfg["burnin"])
    print(f"  Loaded {len(snapshots)} post-burnin snapshots")
    if len(snapshots) < 100:
        print("  ERROR: Too few snapshots for analysis")
        return

    # ── Step 3: Compute turnover and detect quakes ───────────────
    print("\n[Step 3] Computing species turnover and detecting quakes...")
    turnover = compute_turnover(snapshots)
    gens = np.array([s["gen"] for s in snapshots])
    ns = np.array([s["n"] for s in snapshots])

    quakes = detect_quakes(
        turnover, gens,
        window=cfg["turnover_window"],
        threshold_factor=cfg["quake_threshold_factor"],
        min_gap=cfg["quake_min_gap"],
        ns=ns,
    )

    print(f"  Detected {len(quakes)} quakes:")
    for i, (qs, qe, qp, qv) in enumerate(quakes):
        print(f"    Quake {i + 1}: gen {qs}-{qe} (peak at {qp}, turnover={qv:.3f})")

    # ── Step 4: Fit METE ─────────────────────────────────────────
    print("\n[Step 4] Fitting METE predictions...")
    mete_metrics = fit_mete_batch(snapshots, sample_interval=cfg["mete_sample_interval"])
    print(f"  Fitted {len(mete_metrics)} snapshots")

    if mete_metrics:
        r2_vals = [m["r_squared"] for m in mete_metrics.values() if not np.isnan(m["r_squared"])]
        print(f"  Mean R²: {np.mean(r2_vals):.3f} (SD {np.std(r2_vals):.3f})")

        # Compare R² during quakes vs stable
        quake_r2 = []
        stable_r2 = []
        for g, m in mete_metrics.items():
            if np.isnan(m["r_squared"]):
                continue
            in_quake = any(qs <= g <= qe for qs, qe, _, _ in quakes)
            if in_quake:
                quake_r2.append(m["r_squared"])
            else:
                stable_r2.append(m["r_squared"])

        if quake_r2 and stable_r2:
            print(f"  R² during quakes:  {np.mean(quake_r2):.3f} (n={len(quake_r2)})")
            print(f"  R² during stable:  {np.mean(stable_r2):.3f} (n={len(stable_r2)})")
            print(f"  Difference: {np.mean(stable_r2) - np.mean(quake_r2):.3f}")

    # ── Step 5: Generate figure ──────────────────────────────────
    print("\n[Step 5] Generating figure...")
    generate_figure(snapshots, turnover, quakes, mete_metrics, out_dir)

    # ── Save metrics ─────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "quake_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "n", "s", "turnover", "r_squared", "kl_div"])
        for s, t in zip(snapshots, turnover):
            m = mete_metrics.get(s["gen"], {})
            w.writerow([
                s["gen"], s["n"], s["s"],
                f"{t:.6f}",
                f"{m.get('r_squared', '')}" if m else "",
                f"{m.get('kl_div', '')}" if m else "",
            ])
    print(f"  Metrics saved to {metrics_path}")

    print("\n" + "=" * 64)
    print(f"  Analysis complete! Results in {out_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
