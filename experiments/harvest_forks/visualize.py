#!/usr/bin/env python3
"""
Visualize harvesting fork experiment results.

5×3 multi-panel figure:
  Row 1 (Abundance summary):  %ΔN heatmap    | Impact by rank     | %ΔN distribution (log)
  Row 2 (N trajectories):     Example N traj  | Rank-quartile N    | N volatility by rank
  Row 3 (Richness summary):   %ΔS heatmap    | ΔS by rank         | %ΔS distribution
  Row 4 (S trajectories):     Example S traj  | Rank-quartile S    | S volatility by rank
  Row 5 (Cross-metric):       ΔN vs ΔS        | Impact vs baseline | Rank correlation
"""

import os
import sys
import csv
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_metrics(path):
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


def load_fork_timeseries(fname):
    """Load gen, N, S time series from a fork JSONL file."""
    gens, ns, ss = [], [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess"):
                continue
            gens.append(d["gen"])
            ns.append(d["n"])
            ss.append(d.get("s", 0))
    return np.array(gens), np.array(ns), np.array(ss)


def build_heatmap(metrics, sim_ids, ranks, field):
    hm = np.full((len(sim_ids), len(ranks)), np.nan)
    for m in metrics:
        si = sim_ids.index(int(m["sim_id"]))
        ri = ranks.index(int(m["target_rank"]))
        hm[si, ri] = m[field]
    return hm


def get_fork_path(data_dir, sid, rank):
    return os.path.join(data_dir, f"sim_{sid:02d}_rank{rank:02d}.jsonl")


def harvest_onset_line(ax):
    """Add harvest onset marker at gen 200."""
    ax.axvline(200, color="red", ls="--", lw=1, alpha=0.6, label="Harvest onset")


def collect_trajectories_by_quartile(metrics, data_dir, n_ranks, value="n"):
    """Collect normalized trajectories grouped by rank quartile."""
    quartile_bounds = [1, n_ranks // 4, n_ranks // 2, 3 * n_ranks // 4, n_ranks]
    quartile_labels = ["Rank 1–8 (dominant)", "Rank 9–16", "Rank 17–24",
                        "Rank 25–32 (rare)"]
    quartile_colors = ["#B2182B", "#EF8A62", "#67A9CF", "#2166AC"]
    result = []

    for qi in range(4):
        lo, hi = quartile_bounds[qi], quartile_bounds[qi + 1]
        all_traces = []
        common_gens = None
        for m in metrics:
            if lo <= int(m["target_rank"]) <= hi:
                fname = get_fork_path(data_dir, int(m["sim_id"]),
                                       int(m["target_rank"]))
                if os.path.exists(fname):
                    gens, ns, ss = load_fork_timeseries(fname)
                    vals = ns if value == "n" else ss
                    if len(vals) > 5:
                        rel_g = gens - gens[0]
                        v0 = np.mean(vals[:min(5, len(vals))])
                        if v0 > 0:
                            all_traces.append((rel_g, vals / v0 * 100))
                            if common_gens is None:
                                common_gens = rel_g

        if all_traces and common_gens is not None:
            min_len = min(len(t[0]) for t in all_traces)
            aligned = np.array([t[1][:min_len] for t in all_traces])
            result.append({
                "label": quartile_labels[qi],
                "color": quartile_colors[qi],
                "gens": common_gens[:min_len],
                "mean": np.mean(aligned, axis=0),
                "sd": np.std(aligned, axis=0),
            })
    return result


def compute_temporal_volatility(metrics, data_dir, harvest_after=200, value="n"):
    """Compute CV of N or S during the harvest period for each fork."""
    vols = {}  # rank -> list of CVs
    for m in metrics:
        rank = int(m["target_rank"])
        fname = get_fork_path(data_dir, int(m["sim_id"]), rank)
        if not os.path.exists(fname):
            continue
        gens, ns, ss = load_fork_timeseries(fname)
        vals = ns if value == "n" else ss
        rel_g = gens - gens[0]
        harvest_vals = vals[rel_g >= harvest_after]
        if len(harvest_vals) > 2:
            mu = np.mean(harvest_vals)
            if mu > 0:
                cv = np.std(harvest_vals) / mu
                vols.setdefault(rank, []).append(cv)
    return vols


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    metrics_path = os.path.join(results_dir, "fork_metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found")
        return

    metrics = load_metrics(metrics_path)
    if not metrics:
        return

    sim_ids = sorted(set(int(m["sim_id"]) for m in metrics))
    ranks = sorted(set(int(m["target_rank"]) for m in metrics))
    n_sims, n_ranks = len(sim_ids), len(ranks)

    hm_n = build_heatmap(metrics, sim_ids, ranks, "pct_n_change")
    hm_s = build_heatmap(metrics, sim_ids, ranks, "pct_s_change")

    rank_mean_dn, rank_sd_dn = [], []
    rank_mean_ds, rank_sd_ds = [], []
    for r in ranks:
        vn = [m["pct_n_change"] for m in metrics if int(m["target_rank"]) == r]
        vs = [m["pct_s_change"] for m in metrics if int(m["target_rank"]) == r]
        rank_mean_dn.append(np.mean(vn));  rank_sd_dn.append(np.std(vn))
        rank_mean_ds.append(np.mean(vs));  rank_sd_ds.append(np.std(vs))

    delta_n_all = [m["pct_n_change"] for m in metrics]
    delta_s_all = [m["pct_s_change"] for m in metrics]

    # ── Figure (5×3) ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 28))
    gs = gridspec.GridSpec(5, 3, hspace=0.38, wspace=0.35,
                            height_ratios=[1, 0.9, 1, 0.9, 0.9])

    # ═══════════════════ ROW 1: ABUNDANCE SUMMARY ════════════════

    # R1C1: %ΔN heatmap
    ax = fig.add_subplot(gs[0, 0])
    vmax = max(abs(np.nanmin(hm_n)), abs(np.nanmax(hm_n)), 1)
    im = ax.imshow(hm_n, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("Simulation ID")
    ax.set_title("A  Population Impact (%ΔN)", fontweight="bold", fontsize=11)
    ax.set_xticks(range(0, n_ranks, max(1, n_ranks // 6)))
    ax.set_xticklabels([ranks[i] for i in range(0, n_ranks, max(1, n_ranks // 6))])
    ax.set_yticks(range(0, n_sims, max(1, n_sims // 6)))
    ax.set_yticklabels([sim_ids[i] for i in range(0, n_sims, max(1, n_sims // 6))])
    plt.colorbar(im, ax=ax, label="%ΔN", shrink=0.8)

    # R1C2: Mean %ΔN by rank
    ax = fig.add_subplot(gs[0, 1])
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranks)))
    ax.bar(ranks, rank_mean_dn, yerr=rank_sd_dn, alpha=0.8,
            color=colors_bar, capsize=2, width=0.8, edgecolor="white", lw=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("Mean %ΔN")
    ax.set_title("B  Abundance Impact by Rank", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15)

    # R1C3: %ΔN distribution (log Y)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(delta_n_all, bins=40, alpha=0.7, color="#D6604D", edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(np.mean(delta_n_all), color="#2166AC", lw=2,
                label=f"Mean={np.mean(delta_n_all):.1f}%")
    ax.set_yscale("log")
    ax.set_xlabel("%ΔN")
    ax.set_ylabel("Count (log)")
    ax.set_title("C  Abundance Impact Distribution", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ═══════════════════ ROW 2: N TRAJECTORIES ═══════════════════

    # R2C1: Example N trajectories
    ax = fig.add_subplot(gs[1, 0])
    sorted_by_n = sorted(metrics, key=lambda m: m["pct_n_change"])
    picks = [sorted_by_n[0], sorted_by_n[-1], sorted_by_n[len(sorted_by_n) // 2]]
    labels_ts = ["Most negative", "Most positive", "Median"]
    colors_ts = ["#D6604D", "#2166AC", "#808080"]
    for pick, label, col in zip(picks, labels_ts, colors_ts):
        sid, rank = int(pick["sim_id"]), int(pick["target_rank"])
        fname = get_fork_path(data_dir, sid, rank)
        if os.path.exists(fname):
            gens, ns, _ = load_fork_timeseries(fname)
            if len(gens) > 0:
                ax.plot(gens - gens[0], ns, color=col, lw=1.2, alpha=0.8,
                         label=f"{label} ({pick['pct_n_change']:+.0f}%)")
    harvest_onset_line(ax)
    ax.set_xlabel("Generation (from fork start)")
    ax.set_ylabel("N")
    ax.set_title("D  Example N Trajectories", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.15)

    # R2C2: Rank-quartile N trajectories
    ax = fig.add_subplot(gs[1, 1])
    n_quart = collect_trajectories_by_quartile(metrics, data_dir, n_ranks, "n")
    for q in n_quart:
        ax.plot(q["gens"], q["mean"], color=q["color"], lw=1.5, label=q["label"])
        ax.fill_between(q["gens"], q["mean"] - q["sd"], q["mean"] + q["sd"],
                         alpha=0.1, color=q["color"])
    harvest_onset_line(ax)
    ax.axhline(100, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Generation (from fork start)")
    ax.set_ylabel("N (% of initial)")
    ax.set_title("E  N Trajectory by Rank Quartile", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.15)

    # R2C3: N volatility (CV) during harvest by rank
    ax = fig.add_subplot(gs[1, 2])
    n_vols = compute_temporal_volatility(metrics, data_dir, value="n")
    vol_means = [np.mean(n_vols.get(r, [0])) for r in ranks]
    vol_sds = [np.std(n_vols.get(r, [0])) for r in ranks]
    ax.bar(ranks, vol_means, yerr=vol_sds, alpha=0.7, color="#4393C3",
            capsize=2, width=0.8)
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("CV(N) during harvest")
    ax.set_title("F  N Volatility During Harvest", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15)

    # ═══════════════════ ROW 3: RICHNESS SUMMARY ═════════════════

    # R3C1: %ΔS heatmap
    ax = fig.add_subplot(gs[2, 0])
    vmax_s = max(abs(np.nanmin(hm_s)), abs(np.nanmax(hm_s)), 1)
    im = ax.imshow(hm_s, aspect="auto", cmap="PiYG", vmin=-vmax_s, vmax=vmax_s,
                    interpolation="nearest")
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("Simulation ID")
    ax.set_title("G  Richness Impact (%ΔS)", fontweight="bold", fontsize=11)
    ax.set_xticks(range(0, n_ranks, max(1, n_ranks // 6)))
    ax.set_xticklabels([ranks[i] for i in range(0, n_ranks, max(1, n_ranks // 6))])
    ax.set_yticks(range(0, n_sims, max(1, n_sims // 6)))
    ax.set_yticklabels([sim_ids[i] for i in range(0, n_sims, max(1, n_sims // 6))])
    plt.colorbar(im, ax=ax, label="%ΔS", shrink=0.8)

    # R3C2: ΔS by rank
    ax = fig.add_subplot(gs[2, 1])
    ax.bar(ranks, rank_mean_ds, yerr=rank_sd_ds, alpha=0.7,
            color="#1B7837", capsize=2, width=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("Mean %ΔS")
    ax.set_title("H  Richness Impact by Rank", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15)

    # R3C3: %ΔS distribution
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(delta_s_all, bins=40, alpha=0.7, color="#1B7837", edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(np.mean(delta_s_all), color="#762A83", lw=2,
                label=f"Mean={np.mean(delta_s_all):.1f}%")
    ax.set_xlabel("%ΔS")
    ax.set_ylabel("Count")
    ax.set_title("I  Richness Impact Distribution", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ═══════════════════ ROW 4: S TRAJECTORIES ═══════════════════

    # R4C1: Example S trajectories
    ax = fig.add_subplot(gs[3, 0])
    sorted_by_s = sorted(metrics, key=lambda m: m["pct_s_change"])
    picks_s = [sorted_by_s[0], sorted_by_s[-1], sorted_by_s[len(sorted_by_s) // 2]]
    for pick, label, col in zip(picks_s, labels_ts, colors_ts):
        sid, rank = int(pick["sim_id"]), int(pick["target_rank"])
        fname = get_fork_path(data_dir, sid, rank)
        if os.path.exists(fname):
            gens, _, ss = load_fork_timeseries(fname)
            if len(gens) > 0:
                ax.plot(gens - gens[0], ss, color=col, lw=1.2, alpha=0.8,
                         label=f"{label} ({pick['pct_s_change']:+.0f}%)")
    harvest_onset_line(ax)
    ax.set_xlabel("Generation (from fork start)")
    ax.set_ylabel("S")
    ax.set_title("J  Example S Trajectories", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.15)

    # R4C2: Rank-quartile S trajectories
    ax = fig.add_subplot(gs[3, 1])
    s_quart = collect_trajectories_by_quartile(metrics, data_dir, n_ranks, "s")
    for q in s_quart:
        ax.plot(q["gens"], q["mean"], color=q["color"], lw=1.5, label=q["label"])
        ax.fill_between(q["gens"], q["mean"] - q["sd"], q["mean"] + q["sd"],
                         alpha=0.1, color=q["color"])
    harvest_onset_line(ax)
    ax.axhline(100, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Generation (from fork start)")
    ax.set_ylabel("S (% of initial)")
    ax.set_title("K  S Trajectory by Rank Quartile", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.15)

    # R4C3: S volatility (CV) during harvest by rank
    ax = fig.add_subplot(gs[3, 2])
    s_vols = compute_temporal_volatility(metrics, data_dir, value="s")
    vol_means_s = [np.mean(s_vols.get(r, [0])) for r in ranks]
    vol_sds_s = [np.std(s_vols.get(r, [0])) for r in ranks]
    ax.bar(ranks, vol_means_s, yerr=vol_sds_s, alpha=0.7, color="#762A83",
            capsize=2, width=0.8)
    ax.set_xlabel("Target Species Rank")
    ax.set_ylabel("CV(S) during harvest")
    ax.set_title("L  S Volatility During Harvest", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15)

    # ═══════════════════ ROW 5: CROSS-METRIC ═════════════════════

    # R5C1: ΔN vs ΔS scatter
    ax = fig.add_subplot(gs[4, 0])
    sc = ax.scatter(delta_n_all, delta_s_all,
                     c=[int(m["target_rank"]) for m in metrics],
                     cmap="viridis", alpha=0.5, s=15, edgecolors="none")
    ax.set_xlabel("%ΔN (abundance)")
    ax.set_ylabel("%ΔS (richness)")
    ax.set_title("M  Abundance vs Richness Impact", fontweight="bold", fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    plt.colorbar(sc, ax=ax, label="Target Rank")
    ax.grid(True, alpha=0.15)

    # R5C2: %ΔN vs baseline N (does initial pop size predict impact?)
    ax = fig.add_subplot(gs[4, 1])
    baseline_ns = [m["mean_n_baseline"] for m in metrics]
    sc2 = ax.scatter(baseline_ns, delta_n_all,
                      c=[int(m["target_rank"]) for m in metrics],
                      cmap="viridis", alpha=0.5, s=15, edgecolors="none")
    ax.set_xlabel("Baseline N (mean)")
    ax.set_ylabel("%ΔN")
    ax.set_title("N  Impact vs Community Size", fontweight="bold", fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    plt.colorbar(sc2, ax=ax, label="Target Rank")
    ax.grid(True, alpha=0.15)

    # R5C3: Per-sim mean impact (which communities are most sensitive?)
    ax = fig.add_subplot(gs[4, 2])
    sim_mean_dn = []
    sim_mean_ds = []
    for sid in sim_ids:
        sim_metrics = [m for m in metrics if int(m["sim_id"]) == sid]
        sim_mean_dn.append(np.mean([m["pct_n_change"] for m in sim_metrics]))
        sim_mean_ds.append(np.mean([m["pct_s_change"] for m in sim_metrics]))
    x = np.arange(len(sim_ids))
    w = 0.35
    ax.bar(x - w/2, sim_mean_dn, w, alpha=0.7, color="#D6604D", label="%ΔN")
    ax.bar(x + w/2, sim_mean_ds, w, alpha=0.7, color="#1B7837", label="%ΔS")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Simulation ID")
    ax.set_ylabel("Mean % Change")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([sim_ids[i] for i in range(0, len(sim_ids), 4)])
    ax.set_title("O  Community Sensitivity", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # ── Title & save ─────────────────────────────────────────────
    fig.suptitle(
        f"Species Harvesting Experiment: {len(metrics)} forks "
        f"({n_sims} sims × {n_ranks} species, 25% harvest rate)",
        fontsize=15, fontweight="bold", y=0.995
    )

    out_png = os.path.join(results_dir, "Figure_HarvestForks.png")
    out_pdf = os.path.join(results_dir, "Figure_HarvestForks.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ {out_png} / .pdf saved")


if __name__ == "__main__":
    main()
