#!/usr/bin/env python3
"""
Pulse Perturbation Experiment.

1. Burn-in one large ecosystem to stable qESS.
2. Fork 32 copies, each with a different kill fraction
   (2% to 96% in 3% steps).
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
STATE_DIR = Path("states_pulse")

L = 20
W = 10.0
R_BASELINE = 100.0
P_KILL = 0.2
N_INIT = 500
THETA = 0.25
P_MUT = 0.001
OUTPUT_INTERVAL = 10          # fine resolution for early recovery dynamics
QESS_WINDOW = 5000
QESS_THRESH = 0.05

BURNIN_MAX_GEN = 50000      # long burn-in for a rich community
RECOVERY_MAX_GEN = 30000    # enough to see recovery

# Kill fractions: 2% to 96% in 3% steps = 32 levels
KILL_FRACTIONS = [round(x / 100, 2) for x in range(2, 97, 3)]

# ── Helpers ─────────────────────────────────────────────────────────────

def apply_kill(state_path, kill_fraction, output_path, rng_seed):
    """Load state, randomly kill `kill_fraction` of individuals, save."""
    with open(state_path) as f:
        state = json.load(f)

    species = state["species"]
    rng = random.Random(rng_seed)

    # For each species, kill each individual with probability kill_fraction
    new_species = {}
    for genome, count in species.items():
        survivors = 0
        for _ in range(count):
            if rng.random() > kill_fraction:
                survivors += 1
        if survivors > 0:
            new_species[genome] = survivors

    state["species"] = new_species
    n_before = sum(species.values())
    n_after = sum(new_species.values())

    with open(output_path, "w") as f:
        json.dump(state, f)

    return n_before, n_after, len(species), len(new_species)


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
    if result.returncode != 0:
        return []

    # Parse ALL JSON lines to get trajectory
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
                    "gen": g - (gen_offset or 0),  # relative time from perturbation
                    "N": d.get("n", 0),
                    "S": d.get("s", 0),
                })
            except json.JSONDecodeError:
                pass

    return trajectory


def run_fork(kill_frac, burnin_state, fork_id):
    """Apply kill and run recovery for one fork."""
    state_dir = STATE_DIR / f"fork_{fork_id:03d}"
    state_dir.mkdir(parents=True, exist_ok=True)

    perturbed_state = str(state_dir / "perturbed.json")
    n_before, n_after, s_before, s_after = apply_kill(
        burnin_state, kill_frac, perturbed_state, rng_seed=fork_id * 37
    )

    trajectory = run_recovery(
        seed=fork_id * 1000 + 1,
        state_in=perturbed_state,
    )

    # Prepend the post-kill state as t=0 so we see the initial drop
    trajectory.insert(0, {
        "gen": 0,
        "N": n_after,
        "S": s_after,
    })

    return {
        "kill_frac": kill_frac,
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
    print("  Pulse Perturbation Experiment")
    print("═" * 60)
    print(f"  {len(KILL_FRACTIONS)} kill fractions: {KILL_FRACTIONS[0]}–{KILL_FRACTIONS[-1]}")
    print(f"  L={L}, R={R_BASELINE}, recovery gen={RECOVERY_MAX_GEN}")
    print()

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
        "--output-interval", str(BURNIN_MAX_GEN),  # only final output
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

    # Parse burn-in result
    for line in reversed(result.stdout.strip().split('\n')):
        if line.strip().startswith('{'):
            burnin_out = json.loads(line.strip())
            break

    baseline_n = burnin_out["n"]
    baseline_s = burnin_out["s"]
    elapsed_burnin = time.time() - t_start
    print(f"  Baseline: N={baseline_n}, S={baseline_s}, qESS={burnin_out.get('qess', '?')}")
    print(f"  Burn-in: {elapsed_burnin:.1f}s")
    print()

    # ── Phase 2: Forked perturbation ────────────────────────────────
    print(f"[Phase 2] Running {len(KILL_FRACTIONS)} forks...")
    results = []

    with ProcessPoolExecutor(max_workers=32) as pool:
        futs = {}
        for i, kf in enumerate(KILL_FRACTIONS):
            fut = pool.submit(run_fork, kf, burnin_state, i)
            futs[fut] = kf

        for fut in as_completed(futs):
            kf = futs[fut]
            try:
                r = fut.result()
                traj = r["trajectory"]
                final_n = traj[-1]["N"] if traj else 0
                print(f"  Kill {kf:.0%}: N {r['n_before']}→{r['n_after']}→{final_n} "
                      f"| S {r['s_before']}→{r['s_after']}→{traj[-1]['S'] if traj else 0} "
                      f"| {len(traj)} gen steps")
                results.append(r)
            except Exception as e:
                print(f"  Kill {kf:.0%}: FAILED ({e})")

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
    import matplotlib.cm as cm

    results.sort(key=lambda r: r["kill_frac"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pulse Perturbation: Recovery from Random Individual Removal",
                 fontsize=14, fontweight="bold", y=0.98)

    cmap = cm.get_cmap("RdYlGn_r", len(results))

    # ── Panel A: N trajectory (ZOOMED to first 500 gen) ───────────────
    ax = axes[0, 0]
    for i, r in enumerate(results):
        traj = r["trajectory"]
        if not traj:
            continue
        gen = [t["gen"] for t in traj]
        n_vals = [t["N"] / baseline_n for t in traj]
        ax.plot(gen, n_vals, color=cmap(i), alpha=0.7, linewidth=1.2)
        # Mark t=0 point
        ax.scatter([gen[0]], [n_vals[0]], color=cmap(i), s=15, zorder=5)

    # Reference lines for expected survival levels
    for frac, label in [(0.05, "5% survive"), (0.25, "25%"), (0.50, "50%"),
                         (0.75, "75%")]:
        ax.axhline(frac, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.text(510, frac, label, fontsize=7, color="grey", va="center")

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("N / N₀")
    ax.set_title("A. Population recovery (zoomed)", fontweight="bold", loc="left")
    ax.set_xlim(-10, 600)
    ax.set_ylim(-0.05, 1.3)
    ax.grid(True, alpha=0.2)

    # ── Panel B: S trajectory over time ─────────────────────────────
    ax = axes[0, 1]
    for i, r in enumerate(results):
        traj = r["trajectory"]
        if not traj:
            continue
        gen = [t["gen"] for t in traj]
        s_vals = [t["S"] / baseline_s for t in traj]
        ax.plot(gen, s_vals, color=cmap(i), alpha=0.7, linewidth=1.0)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("S / S₀")
    ax.set_title("B. Diversity recovery trajectories", fontweight="bold", loc="left")
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=KILL_FRACTIONS[0]*100,
                                                   vmax=KILL_FRACTIONS[-1]*100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[0, :].tolist(), pad=0.02, fraction=0.03)
    cbar.set_label("Kill fraction (%)", fontsize=10)

    # ── Panel C: Final N/N₀ vs kill fraction (dose-response) ───────
    ax = axes[1, 0]
    kill_fracs = [r["kill_frac"] * 100 for r in results]
    final_n_ratio = []
    for r in results:
        traj = r["trajectory"]
        if traj:
            final_n_ratio.append(traj[-1]["N"] / baseline_n)
        else:
            final_n_ratio.append(0)

    ax.scatter(kill_fracs, final_n_ratio, c=[cmap(i) for i in range(len(results))],
               s=40, edgecolors="black", linewidth=0.5, zorder=3)
    ax.plot(kill_fracs, final_n_ratio, color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=0.8,
               label="Collapse (25%)")
    ax.set_xlabel("Kill fraction (%)")
    ax.set_ylabel("Final N / N₀")
    ax.set_title("C. Dose-response: recovery outcome", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    # ── Panel D: Final S/S₀ vs kill fraction ────────────────────────
    ax = axes[1, 1]
    final_s_ratio = []
    for r in results:
        traj = r["trajectory"]
        if traj:
            final_s_ratio.append(traj[-1]["S"] / baseline_s)
        else:
            final_s_ratio.append(0)

    ax.scatter(kill_fracs, final_s_ratio, c=[cmap(i) for i in range(len(results))],
               s=40, edgecolors="black", linewidth=0.5, zorder=3)
    ax.plot(kill_fracs, final_s_ratio, color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Kill fraction (%)")
    ax.set_ylabel("Final S / S₀")
    ax.set_title("D. Dose-response: diversity outcome", fontweight="bold", loc="left")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig("Figure_PulsePerturbation.png", dpi=200, bbox_inches="tight")
    plt.savefig("Figure_PulsePerturbation.pdf", dpi=300, bbox_inches="tight")
    print("  ✓ Figure_PulsePerturbation.png / .pdf saved")


if __name__ == "__main__":
    main()
