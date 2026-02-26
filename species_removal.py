#!/usr/bin/env python3
"""
Species Removal Experiment.

1. Burn-in one large ecosystem to stable qESS.
2. Fork copies, each with a different NUMBER of whole species removed
   (1, 2, 3, ... S-1), chosen randomly.
3. Run each fork at baseline R until qESS or max_gen,
   tracking N and S every output_interval generations.
4. Plot recovery trajectories.

Target runtime: <5 minutes.
"""

import json
import subprocess
import time
import random
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
BINARY = Path("target/release/tangled-nature")
STATE_DIR = Path("states_species_removal")

L = 25                     # 5x more genomes than L=20
W = 10.0
R_BASELINE = 300.0         # 3x scale-up → N≈2000, S≈55
P_KILL = 0.2
N_INIT = 1500
THETA = 0.25
P_MUT = 0.001
OUTPUT_INTERVAL = 1        # every 1 gen for high-resolution trajectories
QESS_WINDOW = 5000
QESS_THRESH = 0.05

BURNIN_MAX_GEN = 20000     # qESS typically by 5000
RECOVERY_MAX_GEN = 150     # very short run to focus on immediate recovery

# ── Helpers ─────────────────────────────────────────────────────────────

def remove_species(state_path, n_remove, output_path, rng_seed):
    """Load state, remove `n_remove` randomly chosen species entirely."""
    with open(state_path) as f:
        state = json.load(f)

    species = state["species"]
    rng = random.Random(rng_seed)

    genomes = list(species.keys())
    s_before = len(genomes)
    n_before = sum(species.values())

    if n_remove >= s_before:
        n_remove = s_before - 1  # keep at least 1

    to_remove = set(rng.sample(genomes, n_remove))
    new_species = {g: c for g, c in species.items() if g not in to_remove}

    state["species"] = new_species
    n_after = sum(new_species.values())
    s_after = len(new_species)

    with open(output_path, "w") as f:
        json.dump(state, f)

    return n_before, n_after, s_before, s_after


def run_recovery(seed, state_in, output_interval=OUTPUT_INTERVAL):
    """Run recovery at baseline R, return list of (gen, N, S) tuples."""
    cmd = [
        str(BINARY),
        "--seed", str(seed),
        "--l", str(L),
        "--w", str(W),
        "--r", str(R_BASELINE),
        "--p-kill", str(P_KILL),
        "--n-init", str(N_INIT),
        "--max-gen", str(RECOVERY_MAX_GEN),
        "--output-interval", str(output_interval),
        "--p-mut", str(P_MUT),
        "--theta", str(THETA),
        "--qess-window", str(QESS_WINDOW),
        "--qess-threshold", str(QESS_THRESH),
        "--state-in", state_in,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 and "Extinction" in result.stderr:
        # Extinct
        pass

    trajectory = []
    gen_offset = None
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                d = json.loads(line)
                g = d.get("gen", d.get("generation", 0))
                if gen_offset is None:
                    gen_offset = g
                trajectory.append({
                    "gen": g - (gen_offset or 0),
                    "N": d.get("n", 0),
                    "S": d.get("s", 0),
                })
            except json.JSONDecodeError:
                pass

    return trajectory


def run_fork(n_remove, burnin_state, fork_id):
    """Remove species and run recovery for one fork."""
    state_dir = STATE_DIR / f"fork_{fork_id:03d}"
    state_dir.mkdir(parents=True, exist_ok=True)

    perturbed_state = str(state_dir / "perturbed.json")
    n_before, n_after, s_before, s_after = remove_species(
        burnin_state, n_remove, perturbed_state, rng_seed=fork_id * 37 + 7
    )

    trajectory = run_recovery(
        seed=fork_id * 1000 + 1,
        state_in=perturbed_state,
    )

    # Prepend the post-removal state as t=0
    trajectory.insert(0, {
        "gen": 0,
        "N": n_after,
        "S": s_after,
    })

    return {
        "n_remove": n_remove,
        "fork_id": fork_id,
        "n_before": n_before,
        "n_after": n_after,
        "s_before": s_before,
        "s_after": s_after,
        "trajectory": trajectory,
    }


def main():
    t_start = time.time()

    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    STATE_DIR.mkdir(parents=True)

    print("═" * 60)
    print("  Species Removal Experiment")
    print("═" * 60)

    # ── Phase 1: Burn-in ────────────────────────────────────────────
    print("[Phase 1] Burning in a single large ecosystem...")
    burnin_state = str(STATE_DIR / "burnin.json")

    cmd = [
        str(BINARY),
        "--seed", "12345",
        "--l", str(L),
        "--w", str(W),
        "--r", str(R_BASELINE),
        "--p-kill", str(P_KILL),
        "--n-init", str(N_INIT),
        "--max-gen", str(BURNIN_MAX_GEN),
        "--output-interval", str(BURNIN_MAX_GEN),
        "--p-mut", str(P_MUT),
        "--theta", str(THETA),
        "--qess-window", str(QESS_WINDOW),
        "--qess-threshold", str(QESS_THRESH),
        "--state-out", burnin_state,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  BURN-IN FAILED: {result.stderr[:200]}")
        return

    for line in reversed(result.stdout.strip().split('\n')):
        if line.strip().startswith('{'):
            burnin_out = json.loads(line.strip())
            break

    baseline_n = burnin_out["n"]
    baseline_s = burnin_out["s"]
    elapsed_burnin = time.time() - t_start
    print(f"  Baseline: N={baseline_n}, S={baseline_s}, qESS={burnin_out.get('qess', '?')}")
    print(f"  Burn-in: {elapsed_burnin:.1f}s")

    # List species by abundance for context
    with open(burnin_state) as f:
        bs = json.load(f)
    sp = bs["species"]
    sp_sorted = sorted(sp.items(), key=lambda x: -x[1])
    print(f"  Species abundances: {[c for _, c in sp_sorted]}")
    print()

    # ── Phase 2: Forked species removal ─────────────────────────────
    # Remove 1, 2, 3, ... S-1 species
    removal_counts = list(range(1, baseline_s))
    print(f"[Phase 2] Running {len(removal_counts)} forks (remove 1 to {baseline_s-1} species)...")
    results = []

    with ProcessPoolExecutor(max_workers=min(32, len(removal_counts))) as pool:
        futs = {}
        for i, n_rm in enumerate(removal_counts):
            fut = pool.submit(run_fork, n_rm, burnin_state, i)
            futs[fut] = n_rm

        for fut in as_completed(futs):
            n_rm = futs[fut]
            try:
                r = fut.result()
                traj = r["trajectory"]
                final_n = traj[-1]["N"] if traj else 0
                final_s = traj[-1]["S"] if traj else 0
                done = len([f for f in futs if f.done()])
                elapsed = time.time() - t_start
                print(f"  [{done}/{len(futs)} done, {elapsed:.0f}s] "
                      f"Remove {n_rm:2d}/{baseline_s}: "
                      f"N {r['n_before']}→{r['n_after']}→{final_n} "
                      f"| S {r['s_before']}→{r['s_after']}→{final_s} "
                      f"| {len(traj)} steps")
                results.append(r)
            except Exception as e:
                print(f"  Remove {n_rm}: FAILED ({e})")

    elapsed_forks = time.time() - t_start - elapsed_burnin
    print(f"  Forks complete: {elapsed_forks:.1f}s")
    print()

    # ── Phase 3: Figures ────────────────────────────────────────────
    print("[Phase 3] Generating figures...")
    generate_figures(results, baseline_n, baseline_s)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s")


def generate_figures(results, baseline_n, baseline_s):
    """Generate recovery trajectory plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results.sort(key=lambda r: r["n_remove"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Species Removal: Recovery from Targeted Species Extinction",
                 fontsize=14, fontweight="bold", y=0.98)

    cmap = plt.colormaps.get_cmap("RdYlGn_r").resampled(len(results))

    # Find the time when most trajectories have stabilised (for zooming)
    # Use 90th percentile of "time to reach 80% of final N" as zoom limit
    recovery_times = []
    for r in results:
        traj = r["trajectory"]
        if len(traj) < 3:
            continue
        final_n = traj[-1]["N"]
        threshold = 0.8 * final_n if final_n > 0 else 1
        for t in traj:
            if t["N"] >= threshold:
                recovery_times.append(t["gen"])
                break
    zoom_x = 50  # show first 50 generations

    # ── Panel A: N trajectory (ZOOMED) ──────────────────────────────
    ax = axes[0, 0]
    for i, r in enumerate(results):
        traj = r["trajectory"]
        if not traj:
            continue
        gen = [t["gen"] for t in traj]
        n_vals = [t["N"] / baseline_n for t in traj]
        ax.plot(gen, n_vals, color=cmap(i), alpha=0.7, linewidth=1.2)
        ax.scatter([gen[0]], [n_vals[0]], color=cmap(i), s=20, zorder=5,
                   edgecolors="black", linewidth=0.3)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("N / N₀")
    ax.set_title("A. Population recovery (zoomed)", fontweight="bold", loc="left")
    ax.set_xlim(-10, zoom_x)
    ax.set_ylim(-0.05, 1.4)
    ax.grid(True, alpha=0.2)

    # ── Panel B: S trajectory (ZOOMED) ──────────────────────────────
    ax = axes[0, 1]
    for i, r in enumerate(results):
        traj = r["trajectory"]
        if not traj:
            continue
        gen = [t["gen"] for t in traj]
        s_vals = [t["S"] / baseline_s for t in traj]
        ax.plot(gen, s_vals, color=cmap(i), alpha=0.7, linewidth=1.2)
        ax.scatter([gen[0]], [s_vals[0]], color=cmap(i), s=20, zorder=5,
                   edgecolors="black", linewidth=0.3)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("S / S₀")
    ax.set_title("B. Diversity recovery (zoomed)", fontweight="bold", loc="left")
    ax.set_xlim(-10, zoom_x)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=1, vmax=baseline_s - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[0, :].tolist(), pad=0.02, fraction=0.03)
    cbar.set_label("Species removed", fontsize=10)

    # ── Panel C: Final N/N₀ vs species removed ──────────────────────
    ax = axes[1, 0]
    n_removed = [r["n_remove"] for r in results]
    final_n_ratio = []
    for r in results:
        traj = r["trajectory"]
        if traj:
            final_n_ratio.append(traj[-1]["N"] / baseline_n)
        else:
            final_n_ratio.append(0)

    ax.scatter(n_removed, final_n_ratio,
               c=[cmap(i) for i in range(len(results))],
               s=40, edgecolors="black", linewidth=0.5, zorder=3)
    ax.plot(n_removed, final_n_ratio, color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=0.8,
               label="Collapse (25%)")
    ax.set_xlabel(f"Species removed (of {baseline_s})")
    ax.set_ylabel("Final N / N₀")
    ax.set_title("C. Dose-response: population", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_xlim(0, baseline_s)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    # ── Panel D: Final S/S₀ vs species removed ──────────────────────
    ax = axes[1, 1]
    final_s_ratio = []
    for r in results:
        traj = r["trajectory"]
        if traj:
            final_s_ratio.append(traj[-1]["S"] / baseline_s)
        else:
            final_s_ratio.append(0)

    ax.scatter(n_removed, final_s_ratio,
               c=[cmap(i) for i in range(len(results))],
               s=40, edgecolors="black", linewidth=0.5, zorder=3)
    ax.plot(n_removed, final_s_ratio, color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    # Draw the "no recovery" reference line (S_final = S_before - removed)
    ref_removed = list(range(1, baseline_s))
    ref_s_ratio = [(baseline_s - r) / baseline_s for r in ref_removed]
    ax.plot(ref_removed, ref_s_ratio, color="red", linestyle="--",
            linewidth=0.8, alpha=0.6, label="No recovery line")

    ax.set_xlabel(f"Species removed (of {baseline_s})")
    ax.set_ylabel("Final S / S₀")
    ax.set_title("D. Dose-response: diversity", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_xlim(0, baseline_s)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig("Figure_SpeciesRemoval.png", dpi=200, bbox_inches="tight")
    plt.savefig("Figure_SpeciesRemoval.pdf", dpi=300, bbox_inches="tight")
    print("  ✓ Figure_SpeciesRemoval.png / .pdf saved")


if __name__ == "__main__":
    main()
