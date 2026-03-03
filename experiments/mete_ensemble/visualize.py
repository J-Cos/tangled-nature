#!/usr/bin/env python3
"""
Visualize METE analysis results from TNM ensemble.

Generates a multi-panel figure showing:
  Row 1: N and S time series (ensemble mean ± SD)
  Row 2: METE R² and KL divergence over time
  Row 3: Example SAD comparisons at 4 time points
  Row 4: Quake detection overlay
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_csv(path):
    """Load CSV returning dict of arrays."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            result[key] = np.array([float(v) if v != "NA" else np.nan for v in vals])
        except ValueError:
            result[key] = np.array(vals)
    return result


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    # Load data
    ts = load_csv(os.path.join(results_dir, "timeseries.csv"))
    metrics = load_csv(os.path.join(results_dir, "mete_metrics.csv"))
    sad = load_csv(os.path.join(results_dir, "sad_snapshots.csv"))

    if not ts or not metrics:
        print("Error: missing data files in", results_dir)
        return

    # ── Aggregate time series by generation ──────────────────────
    unique_gens = np.sort(np.unique(ts["gen"]))
    ts_mean_n, ts_sd_n = [], []
    ts_mean_s, ts_sd_s = [], []

    for g in unique_gens:
        mask = ts["gen"] == g
        ns = ts["n"][mask]
        ss = ts["s"][mask]
        ts_mean_n.append(np.mean(ns))
        ts_sd_n.append(np.std(ns))
        ts_mean_s.append(np.mean(ss))
        ts_sd_s.append(np.std(ss))

    ts_mean_n = np.array(ts_mean_n)
    ts_sd_n = np.array(ts_sd_n)
    ts_mean_s = np.array(ts_mean_s)
    ts_sd_s = np.array(ts_sd_s)

    # ── Aggregate METE metrics by generation ─────────────────────
    met_gens = np.sort(np.unique(metrics["gen"]))
    r2_mean, r2_sd = [], []
    kl_mean, kl_sd = [], []

    for g in met_gens:
        mask = metrics["gen"] == g
        r2 = metrics["r_squared"][mask]
        r2 = r2[~np.isnan(r2)]
        kl = metrics["kl_div"][mask]
        kl = kl[~np.isnan(kl)]
        r2_mean.append(np.mean(r2) if len(r2) > 0 else np.nan)
        r2_sd.append(np.std(r2) if len(r2) > 0 else np.nan)
        kl_mean.append(np.mean(kl) if len(kl) > 0 else np.nan)
        kl_sd.append(np.std(kl) if len(kl) > 0 else np.nan)

    r2_mean = np.array(r2_mean)
    r2_sd = np.array(r2_sd)
    kl_mean = np.array(kl_mean)
    kl_sd = np.array(kl_sd)

    # ── Quake detection: rolling CV of N ─────────────────────────
    def rolling_cv(arr, window=500):
        cv = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            chunk = arr[i - window:i]
            m = np.mean(chunk)
            if m > 0:
                cv[i] = np.std(chunk) / m
        return cv

    cv_n = rolling_cv(ts_mean_n, window=min(500, len(ts_mean_n) // 5))
    cv_threshold = np.nanmedian(cv_n) * 2 if np.any(~np.isnan(cv_n)) else 1.0
    quake_mask = cv_n > cv_threshold

    # ── Figure ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.30,
                           height_ratios=[1, 1, 1.2, 1])

    # ── Row 1: N and S time series ───────────────────────────────
    ax_n = fig.add_subplot(gs[0, 0])
    ax_n.fill_between(unique_gens, ts_mean_n - ts_sd_n, ts_mean_n + ts_sd_n,
                       alpha=0.2, color="#2166AC")
    ax_n.plot(unique_gens, ts_mean_n, color="#2166AC", linewidth=1.5)
    ax_n.set_xlabel("Generation")
    ax_n.set_ylabel("Total N")
    ax_n.set_title("Population (ensemble mean ± SD)", fontweight="bold", fontsize=11)
    ax_n.grid(True, alpha=0.2)

    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.fill_between(unique_gens, ts_mean_s - ts_sd_s, ts_mean_s + ts_sd_s,
                       alpha=0.2, color="#1B7837")
    ax_s.plot(unique_gens, ts_mean_s, color="#1B7837", linewidth=1.5)
    ax_s.set_xlabel("Generation")
    ax_s.set_ylabel("Species Richness S")
    ax_s.set_title("Richness (ensemble mean ± SD)", fontweight="bold", fontsize=11)
    ax_s.grid(True, alpha=0.2)

    # ── Row 2: METE fit metrics ──────────────────────────────────
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_r2.fill_between(met_gens, r2_mean - r2_sd, r2_mean + r2_sd,
                        alpha=0.2, color="#D6604D")
    ax_r2.plot(met_gens, r2_mean, color="#D6604D", linewidth=1.5)
    ax_r2.set_xlabel("Generation")
    ax_r2.set_ylabel("R² (rank-abundance)")
    ax_r2.set_title("METE SAD Fit: R²", fontweight="bold", fontsize=11)
    ax_r2.set_ylim(-0.1, 1.1)
    ax_r2.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.9")
    ax_r2.legend(fontsize=8)
    ax_r2.grid(True, alpha=0.2)

    ax_kl = fig.add_subplot(gs[1, 1])
    ax_kl.fill_between(met_gens, kl_mean - kl_sd, kl_mean + kl_sd,
                        alpha=0.2, color="#762A83")
    ax_kl.plot(met_gens, kl_mean, color="#762A83", linewidth=1.5)
    ax_kl.set_xlabel("Generation")
    ax_kl.set_ylabel("KL Divergence")
    ax_kl.set_title("METE SAD Fit: KL Divergence", fontweight="bold", fontsize=11)
    ax_kl.grid(True, alpha=0.2)

    # ── Row 3: Example SAD comparisons at 4 time points ──────────
    # Pick 4 time points: early, 1/3, 2/3, late
    if len(met_gens) >= 4:
        pick_idx = [0, len(met_gens) // 3, 2 * len(met_gens) // 3, -1]
        pick_gens = [met_gens[i] for i in pick_idx]
    else:
        pick_gens = met_gens[:4]

    for col_idx, pg in enumerate(pick_gens):
        ax_sad = fig.add_subplot(gs[2, 0]) if col_idx < 2 else fig.add_subplot(gs[2, 1])
        # Only use left/right panels
        if col_idx in (0, 2):
            ax_sad = fig.add_subplot(gs[2, col_idx // 2])

        if col_idx in (0, 2):
            # Get observed SAD for first sim at this gen
            mask_sad = (sad["gen"] == pg) & (sad["sim_id"] == sad["sim_id"][0])
            if np.any(mask_sad):
                obs_ab = np.sort(sad["abundance"][mask_sad])[::-1]
                S0 = len(obs_ab)
                N0 = int(np.sum(obs_ab))

                # METE prediction
                try:
                    import subprocess
                    # Use rank indices
                    ranks = np.arange(1, S0 + 1)
                    ax_sad.bar(ranks, obs_ab, alpha=0.6, color="#2166AC",
                               label=f"Observed (S={S0}, N={N0})", width=0.8)

                    # Simple METE prediction: log-series approximation
                    # n(r) ∝ exp(-β*r) where β = ln(1 + N/S)/S
                    if S0 > 1 and N0 > S0:
                        beta = np.log(1 + N0 / S0) / S0
                        pred_ab = N0 * np.exp(-beta * ranks) / np.sum(np.exp(-beta * ranks))
                        ax_sad.plot(ranks, pred_ab, "r-", linewidth=2,
                                    label="METE prediction")

                    ax_sad.set_xlabel("Rank")
                    ax_sad.set_ylabel("Abundance")
                    title_gen = "early" if col_idx == 0 else "late"
                    ax_sad.set_title(f"SAD at gen {int(pg)} ({title_gen})",
                                     fontweight="bold", fontsize=10)
                    ax_sad.legend(fontsize=8)
                    ax_sad.grid(True, alpha=0.2)
                except Exception:
                    pass

    # ── Row 4: Quake detection overlay ───────────────────────────
    ax_q = fig.add_subplot(gs[3, :])
    ax_q2 = ax_q.twinx()

    # N time series
    ax_q.plot(unique_gens, ts_mean_n, color="#2166AC", linewidth=1, alpha=0.6,
              label="N (mean)")
    ax_q.set_xlabel("Generation")
    ax_q.set_ylabel("N", color="#2166AC")

    # METE R² overlay
    ax_q2.plot(met_gens, r2_mean, color="#D6604D", linewidth=1.5, label="METE R²")
    ax_q2.set_ylabel("METE R²", color="#D6604D")
    ax_q2.set_ylim(-0.1, 1.1)

    # Highlight quake regions
    if np.any(quake_mask):
        quake_gens = unique_gens[quake_mask]
        for qg in quake_gens:
            ax_q.axvspan(qg - 50, qg + 50, alpha=0.1, color="red")

    ax_q.set_title("Population dynamics with METE fit quality and quake detection",
                    fontweight="bold", fontsize=11)
    lines1, labels1 = ax_q.get_legend_handles_labels()
    lines2, labels2 = ax_q2.get_legend_handles_labels()
    ax_q.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax_q.grid(True, alpha=0.2)

    # ── Title & save ─────────────────────────────────────────────
    fig.suptitle("METE Predictions vs TNM Dynamics (32-sim ensemble)",
                 fontsize=15, fontweight="bold", y=0.99)

    out_png = os.path.join(results_dir, "Figure_METE_ensemble.png")
    out_pdf = os.path.join(results_dir, "Figure_METE_ensemble.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ {out_png} / .pdf saved")


if __name__ == "__main__":
    main()
