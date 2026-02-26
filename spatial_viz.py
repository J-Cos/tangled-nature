#!/usr/bin/env python3
"""
Visualize Spatial TNM output — comprehensive multi-panel figure.

Layout (4 rows × 4 cols):
  Row 1: N heatmaps at 4 time points (temporal evolution of spatial structure)
  Row 2: S heatmaps at 4 time points
  Row 3: Time series — Total N | α diversity | γ diversity | β diversity
  Row 4: Final-state analysis — N histogram | S histogram | N–S scatter | β(Whittaker) explanation
"""

import json
import sys
import numpy as np


def load_data(fname):
    snapshots = []
    final = None
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") == "spatial_snapshot":
                snapshots.append(d)
            elif d.get("type") == "spatial_final":
                final = d
    return snapshots, final


def pick_time_points(snapshots):
    """Pick snapshots at gen 0, ~1/3, ~2/3, and final generation."""
    max_gen = max(s["gen"] for s in snapshots)
    targets = [0, max_gen // 3, 2 * max_gen // 3, max_gen]
    result = []
    for t in targets:
        best = min(snapshots, key=lambda s: abs(s["gen"] - t))
        result.append(best)
    return result


def build_grid(snapshot, key, grid_w, grid_h):
    grid = np.zeros((grid_h, grid_w))
    for p in snapshot["patches"]:
        grid[p["y"], p["x"]] = p[key]
    return grid


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "spatial_output.jsonl"
    snapshots, final = load_data(fname)

    if not snapshots:
        print("No spatial snapshots found!")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Grid dimensions
    last = snapshots[-1]
    grid_w = max(p["x"] for p in last["patches"]) + 1
    grid_h = max(p["y"] for p in last["patches"]) + 1
    n_patches = grid_w * grid_h

    # Pick 4 temporal snapshots for heatmaps
    time_snaps = pick_time_points(snapshots)

    # ── Compute time series ────────────────────────────────────────
    gens = [s["gen"] for s in snapshots]
    total_n = [s["total_n"] for s in snapshots]
    gamma_s = [s["gamma_s"] for s in snapshots]
    alpha_s = [s["mean_s"] for s in snapshots]
    beta_s = [s["gamma_s"] / s["mean_s"] if s["mean_s"] > 0 else 0
              for s in snapshots]

    # Global color ranges for consistent heatmaps
    # For N: use range across all time points
    all_n = [p["n"] for s in time_snaps for p in s["patches"]]
    n_vmin, n_vmax = min(all_n), max(all_n)
    # For S: use post-transient range for ALL heatmaps
    # Gen 0 S (~1000) >> post-transient S (~30), so it renders as uniform bright
    post_s = [p["s"] for s in time_snaps if s["gen"] > 0 for p in s["patches"]]
    s_vmin, s_vmax = min(post_s), max(post_s)

    # ── Figure layout: 4 rows × 4 cols ────────────────────────────
    fig = plt.figure(figsize=(18, 18))

    # Use gridspec for flexible layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 4, hspace=0.35, wspace=0.35,
                           height_ratios=[1, 1, 0.8, 0.8])

    # ── Row 1: N heatmaps at 4 time points ────────────────────────
    for col, snap in enumerate(time_snaps):
        ax = fig.add_subplot(gs[0, col])
        n_grid = build_grid(snap, "n", grid_w, grid_h)
        im = ax.imshow(n_grid, cmap="YlOrRd", interpolation="nearest",
                        vmin=n_vmin, vmax=n_vmax)
        ax.set_title(f"Gen {snap['gen']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Population (N)", fontsize=11, fontweight="bold")

    # Colorbar for N row
    cax = fig.add_axes([0.92, 0.73, 0.012, 0.15])
    plt.colorbar(im, cax=cax, label="N")

    # ── Row 2: S heatmaps at 4 time points ────────────────────────
    im2 = None
    for col, snap in enumerate(time_snaps):
        ax = fig.add_subplot(gs[1, col])
        s_grid = build_grid(snap, "s", grid_w, grid_h)
        im2 = ax.imshow(s_grid, cmap="viridis", interpolation="nearest",
                         vmin=s_vmin, vmax=s_vmax)
        ax.set_title(f"Gen {snap['gen']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Richness (S)", fontsize=11, fontweight="bold")

    # Colorbar for S row (post-transient scale)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=s_vmin, vmax=s_vmax))
    cax2 = fig.add_axes([0.92, 0.51, 0.012, 0.15])
    plt.colorbar(sm, cax=cax2, label="S")

    # ── Row 3: Time series (separate panels) ──────────────────────
    # Filter to gen >= 100 so y-axes aren't dominated by the initial transient
    mask = [i for i, g in enumerate(gens) if g >= 100]
    gens_f = [gens[i] for i in mask]
    total_n_f = [total_n[i] for i in mask]
    alpha_s_f = [alpha_s[i] for i in mask]
    gamma_s_f = [gamma_s[i] for i in mask]
    beta_s_f = [beta_s[i] for i in mask]

    # Panel: Total N
    ax_n = fig.add_subplot(gs[2, 0])
    ax_n.plot(gens_f, total_n_f, color="#2166AC", linewidth=1.5)
    ax_n.set_xlabel("Generation")
    ax_n.set_ylabel("Total N")
    ax_n.set_title("Total population", fontweight="bold", fontsize=10, loc="left")
    ax_n.grid(True, alpha=0.2)
    ax_n.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Panel: α diversity (mean per-patch S)
    ax_a = fig.add_subplot(gs[2, 1])
    ax_a.plot(gens_f, alpha_s_f, color="#1B7837", linewidth=1.5)
    ax_a.set_xlabel("Generation")
    ax_a.set_ylabel("Mean S per patch")
    ax_a.set_title("α diversity", fontweight="bold", fontsize=10, loc="left")
    ax_a.grid(True, alpha=0.2)

    # Panel: γ diversity (total unique species)
    ax_g = fig.add_subplot(gs[2, 2])
    ax_g.plot(gens_f, gamma_s_f, color="#D6604D", linewidth=1.5)
    ax_g.set_xlabel("Generation")
    ax_g.set_ylabel("Unique genomes")
    ax_g.set_title("γ diversity", fontweight="bold", fontsize=10, loc="left")
    ax_g.grid(True, alpha=0.2)

    # Panel: β diversity (Whittaker's β = γ / ᾱ)
    ax_b = fig.add_subplot(gs[2, 3])
    ax_b.plot(gens_f, beta_s_f, color="#762A83", linewidth=1.5)
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel("β = γ / ᾱ")
    ax_b.set_title("β diversity (Whittaker)", fontweight="bold", fontsize=10, loc="left")
    ax_b.grid(True, alpha=0.2)

    # ── Row 4: Final-state distributions ──────────────────────────

    # N distribution across patches
    ax_nh = fig.add_subplot(gs[3, 0])
    patch_n = [p["n"] for p in last["patches"]]
    ax_nh.hist(patch_n, bins=20, color="#2166AC", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax_nh.set_xlabel("Patch N")
    ax_nh.set_ylabel("Count")
    ax_nh.set_title("N distribution (final)", fontweight="bold", fontsize=10, loc="left")
    ax_nh.axvline(np.mean(patch_n), color="red", linestyle="--", linewidth=1.0,
                   label=f"mean={np.mean(patch_n):.0f}")
    ax_nh.legend(fontsize=8)
    ax_nh.grid(True, alpha=0.2)

    # S distribution across patches
    ax_sh = fig.add_subplot(gs[3, 1])
    patch_s = [p["s"] for p in last["patches"]]
    ax_sh.hist(patch_s, bins=20, color="#1B7837", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax_sh.set_xlabel("Patch S")
    ax_sh.set_ylabel("Count")
    ax_sh.set_title("S distribution (final)", fontweight="bold", fontsize=10, loc="left")
    ax_sh.axvline(np.mean(patch_s), color="red", linestyle="--", linewidth=1.0,
                   label=f"mean={np.mean(patch_s):.0f}")
    ax_sh.legend(fontsize=8)
    ax_sh.grid(True, alpha=0.2)

    # N–S scatter
    ax_ns = fig.add_subplot(gs[3, 2])
    sc = ax_ns.scatter(patch_n, patch_s, c=patch_s, cmap="viridis", s=40,
                        edgecolors="black", linewidth=0.5, alpha=0.8)
    corr = np.corrcoef(patch_n, patch_s)[0, 1]
    ax_ns.set_xlabel("Patch N")
    ax_ns.set_ylabel("Patch S")
    ax_ns.set_title(f"N–S correlation (r={corr:.2f})", fontweight="bold", fontsize=10, loc="left")
    ax_ns.grid(True, alpha=0.2)
    # Trend line
    z = np.polyfit(patch_n, patch_s, 1)
    xfit = np.linspace(min(patch_n), max(patch_n), 100)
    ax_ns.plot(xfit, np.polyval(z, xfit), "r--", linewidth=1.0, alpha=0.6)

    # β over space: pairwise Jaccard-like proxy (CV of S across patches)
    ax_cv = fig.add_subplot(gs[3, 3])
    # Compute spatial CV of N and S over time
    cv_n_over_time = []
    cv_s_over_time = []
    for s in snapshots:
        ns = [p["n"] for p in s["patches"]]
        ss = [p["s"] for p in s["patches"]]
        cv_n_over_time.append(np.std(ns) / np.mean(ns) if np.mean(ns) > 0 else 0)
        cv_s_over_time.append(np.std(ss) / np.mean(ss) if np.mean(ss) > 0 else 0)
    ax_cv.plot(gens, cv_n_over_time, color="#2166AC", linewidth=1.5, label="CV(N)")
    ax_cv.plot(gens, cv_s_over_time, color="#1B7837", linewidth=1.5, label="CV(S)")
    ax_cv.set_xlabel("Generation")
    ax_cv.set_ylabel("Coefficient of Variation")
    ax_cv.set_title("Spatial heterogeneity", fontweight="bold", fontsize=10, loc="left")
    ax_cv.legend(fontsize=8)
    ax_cv.grid(True, alpha=0.2)

    # ── Suptitle ──────────────────────────────────────────────────
    fig.suptitle(
        f"Spatial TNM Metacommunity: {grid_w}×{grid_h} Grid, p_move={0.01}",
        fontsize=15, fontweight="bold", y=0.99
    )

    plt.savefig("Figure_SpatialTNM.png", dpi=200, bbox_inches="tight")
    plt.savefig("Figure_SpatialTNM.pdf", dpi=300, bbox_inches="tight")
    print("✓ Figure_SpatialTNM.png / .pdf saved")


if __name__ == "__main__":
    main()
