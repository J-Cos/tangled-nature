#!/usr/bin/env python3
"""
Forked Quench Experiment: EVO vs ECO under adiabatic stress.

Burns in ecosystems with mutations ON, then forks each into:
  - EVO (p_mut=0.001): evolutionary rescue active
  - ECO (p_mut=0.0):   frozen interaction network, no evolutionary rescue

Applies forward stress press to both and compares trajectories.
Produces matplotlib figures comparing N, S, D_KL between treatments.

Target runtime: <5 minutes.
"""

import json
import subprocess
import sys
import time
import csv
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
BINARY = Path("target/release/tangled-nature")
STATE_DIR = Path("states_quench")

# Fast demo parameters
L = 20
W = 10.0
R_BASELINE = 100.0
P_KILL = 0.2
N_INIT = 100
THETA = 0.25
MAX_GEN = 5000      # fast equilibration
QESS_WINDOW = 500
QESS_THRESH = 0.05
OUTPUT_INTERVAL = 5000

# Stress protocol
DELTA_R = -5.0       # coarse steps for speed
R_MIN = 5.0

N_REPLICATES = 8     # 8 reps × 2 treatments = 16 parallel jobs

# ── Helpers ─────────────────────────────────────────────────────────────

def run_step(seed, r_value, p_mut, state_in=None, state_out=None):
    """Run a single TNM step, return parsed final-line JSON."""
    cmd = [
        str(BINARY),
        "--seed", str(seed),
        "--l", str(L),
        "--w", str(W),
        "--r", str(r_value),
        "--p-kill", str(P_KILL),
        "--n-init", str(N_INIT),
        "--max-gen", str(MAX_GEN),
        "--output-interval", str(OUTPUT_INTERVAL),
        "--p-mut", str(p_mut),
        "--theta", str(THETA),
        "--qess-window", str(QESS_WINDOW),
        "--qess-threshold", str(QESS_THRESH),
        "--output-j",
    ]
    if state_in:
        cmd += ["--state-in", state_in]
    if state_out:
        cmd += ["--state-out", state_out]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None

    # Parse last JSON line
    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return None


def compute_dkl(sad_dict):
    """Compute D_KL(observed || METE log-series)."""
    if not sad_dict or len(sad_dict) < 2:
        return float('nan')
    counts = np.array(list(sad_dict.values()), dtype=float)
    S = len(counts)
    N = counts.sum()
    if N < 2 or S < 2:
        return float('nan')

    # Observed proportions
    p_obs = counts / N

    # METE log-series: solve for beta
    from scipy.optimize import brentq
    def constraint(beta):
        if beta <= 0 or beta >= 1:
            return 1e10
        k = np.arange(1, int(N) + 1)
        weights = beta**k / k
        return weights.sum() * S / N - 1.0

    try:
        beta = brentq(constraint, 1e-12, 1 - 1e-12)
    except (ValueError, RuntimeError):
        return float('nan')

    # METE predicted proportions for observed abundance classes
    k = np.arange(1, int(N) + 1)
    raw = beta**k / k
    norm = raw.sum()
    p_mete_full = raw / norm

    # For each observed species with abundance n_i, the METE probability is p_mete_full[n_i - 1]
    abundances = counts.astype(int)
    p_mete = np.array([p_mete_full[min(a-1, len(p_mete_full)-1)] for a in abundances])
    p_mete = p_mete / p_mete.sum()  # renormalize

    # D_KL
    mask = (p_obs > 0) & (p_mete > 0)
    return float(np.sum(p_obs[mask] * np.log(p_obs[mask] / p_mete[mask])))


def compute_lambda2(j_matrix, species_list):
    """Compute algebraic connectivity λ₂ of the normalised Laplacian."""
    if not j_matrix or len(species_list) < 2:
        return float('nan')
    S = len(species_list)
    A = np.zeros((S, S))
    for i, si in enumerate(species_list):
        for j, sj in enumerate(species_list):
            key = f"{si},{sj}"
            if key in j_matrix and abs(j_matrix[key]) > 0:
                A[i, j] = 1.0
    np.fill_diagonal(A, 0)
    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        return 0.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(S) - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def run_treatment(rep_id, treatment, p_mut, burnin_state):
    """Run one replicate under one treatment (EVO or ECO)."""
    rows = []
    state_dir = STATE_DIR / f"rep{rep_id:03d}_{treatment}"
    state_dir.mkdir(parents=True, exist_ok=True)

    prev_state = burnin_state
    r_value = R_BASELINE
    step = 0

    while r_value > R_MIN:
        r_value = max(r_value + DELTA_R, R_MIN)
        step += 1
        state_file = str(state_dir / f"step_{step:03d}.json")
        mu = 1.0 / r_value

        out = run_step(
            seed=rep_id * 1000 + step + (5000 if treatment == "ECO" else 0),
            r_value=r_value,
            p_mut=p_mut,
            state_in=prev_state,
            state_out=state_file,
        )

        if out is None:
            rows.append({
                "Replicate": rep_id, "Treatment": treatment,
                "Step": step, "Mu": mu, "R": r_value,
                "N": 0, "S": 0, "METE_DKL": float('nan'),
                "Lambda_2": float('nan'),
            })
            break

        # Compute metrics
        species = out.get("species", {})
        dkl = compute_dkl(species)

        j_mat = out.get("j_matrix", {})
        sp_list = list(species.keys())
        lam2 = compute_lambda2(j_mat, sp_list)

        rows.append({
            "Replicate": rep_id, "Treatment": treatment,
            "Step": step, "Mu": mu, "R": r_value,
            "N": out.get("n", 0), "S": out.get("s", 0),
            "METE_DKL": dkl, "Lambda_2": lam2,
        })

        prev_state = state_file
        if out.get("n", 0) < 5:
            break  # collapsed

    return rows


def main():
    t_start = time.time()

    # Clean up
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    STATE_DIR.mkdir(parents=True)

    print("═" * 60)
    print("  Forked Quench Experiment: EVO vs ECO")
    print("═" * 60)
    print(f"  {N_REPLICATES} replicates × 2 treatments")
    print(f"  L={L}, R₀={R_BASELINE}, ΔR={DELTA_R}, R_min={R_MIN}")
    print(f"  max_gen={MAX_GEN}, p_mut: EVO=0.001, ECO=0.0")
    print()

    # ── Phase 1: Burn-in ────────────────────────────────────────────
    print("[Phase 1] Burning in replicates at baseline R...")
    burnin_states = {}

    for rep in range(N_REPLICATES):
        state_file = str(STATE_DIR / f"burnin_{rep:03d}.json")
        out = run_step(
            seed=rep * 100,
            r_value=R_BASELINE,
            p_mut=0.001,
            state_out=state_file,
        )
        if out:
            burnin_states[rep] = state_file
            print(f"  Rep {rep}: N={out['n']}, S={out['s']}, qESS={out.get('qess', '?')}")
        else:
            print(f"  Rep {rep}: FAILED")

    elapsed_burnin = time.time() - t_start
    print(f"  Burn-in complete: {len(burnin_states)} reps in {elapsed_burnin:.0f}s")
    print()

    # ── Phase 2: Forked stress press ────────────────────────────────
    print("[Phase 2] Running forked stress press (EVO + ECO)...")
    all_rows = []

    # Record burn-in baseline N per replicate
    baseline_n = {}
    for rep, sf in burnin_states.items():
        with open(sf) as f:
            state = json.load(f)
        baseline_n[rep] = sum(state.get("abundances", {}).values())

    jobs = []
    for rep, state_file in burnin_states.items():
        jobs.append((rep, "EVO", 0.001, state_file))
        jobs.append((rep, "ECO", 0.0, state_file))

    with ProcessPoolExecutor(max_workers=min(16, len(jobs))) as pool:
        futs = {pool.submit(run_treatment, *j): j for j in jobs}
        for fut in as_completed(futs):
            rep, treatment, _, _ = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"  Rep {rep} {treatment}: {len(rows)} stress steps")
            except Exception as e:
                print(f"  Rep {rep} {treatment}: FAILED ({e})")

    elapsed_stress = time.time() - t_start - elapsed_burnin
    print(f"  Stress press complete: {len(all_rows)} data points in {elapsed_stress:.0f}s")
    print()

    # ── Phase 3: Save CSV ───────────────────────────────────────────
    csv_path = "results_quench.csv"
    fieldnames = ["Replicate", "Treatment", "Step", "Mu", "R", "N", "S",
                  "METE_DKL", "Lambda_2"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["Replicate"], r["Treatment"], r["Step"])):
            writer.writerow(row)
    print(f"  Saved: {csv_path} ({len(all_rows)} rows)")

    # ── Phase 4: Figures ────────────────────────────────────────────
    print("[Phase 3] Generating figures...")
    generate_figures(all_rows, baseline_n)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s")


def generate_figures(all_rows, baseline_n):
    """Generate comparison plots: EVO vs ECO."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Organise data
    evo_data = [r for r in all_rows if r["Treatment"] == "EVO"]
    eco_data = [r for r in all_rows if r["Treatment"] == "ECO"]

    # Group by replicate
    def group_by_rep(data):
        reps = {}
        for r in data:
            reps.setdefault(r["Replicate"], []).append(r)
        for k in reps:
            reps[k].sort(key=lambda x: x["Step"])
        return reps

    evo_reps = group_by_rep(evo_data)
    eco_reps = group_by_rep(eco_data)

    # Compute means per step
    def mean_by_step(data, field):
        steps = {}
        for r in data:
            steps.setdefault(r["Step"], []).append(r[field])
        return (
            sorted(steps.keys()),
            [np.nanmean(steps[s]) for s in sorted(steps.keys())],
            [np.nanstd(steps[s]) / max(1, np.sqrt(len(steps[s]))) for s in sorted(steps.keys())],
        )

    def mu_by_step(data):
        steps = {}
        for r in data:
            steps.setdefault(r["Step"], []).append(r["Mu"])
        return [np.mean(steps[s]) for s in sorted(steps.keys())]

    # Colours
    c_evo = "#2176AE"  # blue
    c_eco = "#D64933"  # red

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Forked Quench: Evolutionary Rescue (EVO) vs Ecological Quench (ECO)",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── Panel A: N vs μ ─────────────────────────────────────────────
    ax = axes[0, 0]
    for rep, rows in evo_reps.items():
        mu = [r["Mu"] for r in rows]
        n = [r["N"] for r in rows]
        ax.plot(mu, n, color=c_evo, alpha=0.2, linewidth=0.8)
    for rep, rows in eco_reps.items():
        mu = [r["Mu"] for r in rows]
        n = [r["N"] for r in rows]
        ax.plot(mu, n, color=c_eco, alpha=0.2, linewidth=0.8)

    # Means
    steps_evo, n_mean_evo, n_se_evo = mean_by_step(evo_data, "N")
    steps_eco, n_mean_eco, n_se_eco = mean_by_step(eco_data, "N")
    mu_evo = mu_by_step(evo_data)
    mu_eco = mu_by_step(eco_data)

    ax.plot(mu_evo, n_mean_evo, color=c_evo, linewidth=2.5, label="EVO (mutations ON)")
    ax.fill_between(mu_evo,
                     [m - s for m, s in zip(n_mean_evo, n_se_evo)],
                     [m + s for m, s in zip(n_mean_evo, n_se_evo)],
                     color=c_evo, alpha=0.15)
    ax.plot(mu_eco, n_mean_eco, color=c_eco, linewidth=2.5, label="ECO (mutations OFF)")
    ax.fill_between(mu_eco,
                     [m - s for m, s in zip(n_mean_eco, n_se_eco)],
                     [m + s for m, s in zip(n_mean_eco, n_se_eco)],
                     color=c_eco, alpha=0.15)

    ax.set_xlabel("Abiotic stress, μ = 1/R")
    ax.set_ylabel("Population size, N")
    ax.set_title("A. Population under stress", fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel B: S vs μ ─────────────────────────────────────────────
    ax = axes[0, 1]
    steps_evo, s_mean_evo, s_se_evo = mean_by_step(evo_data, "S")
    steps_eco, s_mean_eco, s_se_eco = mean_by_step(eco_data, "S")

    for rep, rows in evo_reps.items():
        ax.plot([r["Mu"] for r in rows], [r["S"] for r in rows],
                color=c_evo, alpha=0.2, linewidth=0.8)
    for rep, rows in eco_reps.items():
        ax.plot([r["Mu"] for r in rows], [r["S"] for r in rows],
                color=c_eco, alpha=0.2, linewidth=0.8)

    ax.plot(mu_evo, s_mean_evo, color=c_evo, linewidth=2.5, label="EVO")
    ax.plot(mu_eco, s_mean_eco, color=c_eco, linewidth=2.5, label="ECO")
    ax.set_xlabel("Abiotic stress, μ = 1/R")
    ax.set_ylabel("Species richness, S")
    ax.set_title("B. Diversity under stress", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel C: D_KL vs μ ──────────────────────────────────────────
    ax = axes[1, 0]
    steps_evo, dkl_mean_evo, dkl_se_evo = mean_by_step(evo_data, "METE_DKL")
    steps_eco, dkl_mean_eco, dkl_se_eco = mean_by_step(eco_data, "METE_DKL")

    for rep, rows in evo_reps.items():
        ax.plot([r["Mu"] for r in rows], [r["METE_DKL"] for r in rows],
                color=c_evo, alpha=0.2, linewidth=0.8)
    for rep, rows in eco_reps.items():
        ax.plot([r["Mu"] for r in rows], [r["METE_DKL"] for r in rows],
                color=c_eco, alpha=0.2, linewidth=0.8)

    ax.plot(mu_evo, dkl_mean_evo, color=c_evo, linewidth=2.5, label="EVO")
    ax.plot(mu_eco, dkl_mean_eco, color=c_eco, linewidth=2.5, label="ECO")
    ax.set_xlabel("Abiotic stress, μ = 1/R")
    ax.set_ylabel("D_KL (METE)")
    ax.set_title("C. Structural divergence from METE", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel D: N normalised ───────────────────────────────────────
    ax = axes[1, 1]
    # Show N/N_baseline to compare collapse thresholds
    for rep, rows in evo_reps.items():
        nb = baseline_n.get(rep, 1) or 1
        ax.plot([r["Mu"] for r in rows], [r["N"] / nb for r in rows],
                color=c_evo, alpha=0.2, linewidth=0.8)
    for rep, rows in eco_reps.items():
        nb = baseline_n.get(rep, 1) or 1
        ax.plot([r["Mu"] for r in rows], [r["N"] / nb for r in rows],
                color=c_eco, alpha=0.2, linewidth=0.8)

    # Means
    def norm_n_by_step(data, baseline):
        steps = {}
        for r in data:
            nb = baseline.get(r["Replicate"], 1) or 1
            steps.setdefault(r["Step"], []).append(r["N"] / nb)
        return [np.nanmean(steps[s]) for s in sorted(steps.keys())]

    ax.plot(mu_evo, norm_n_by_step(evo_data, baseline_n),
            color=c_evo, linewidth=2.5, label="EVO")
    ax.plot(mu_eco, norm_n_by_step(eco_data, baseline_n),
            color=c_eco, linewidth=2.5, label="ECO")
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=0.8,
               label="Collapse threshold (25%)")
    ax.axhline(0.80, color="grey", linestyle=":", linewidth=0.8,
               label="Pre-collapse (80%)")
    ax.set_xlabel("Abiotic stress, μ = 1/R")
    ax.set_ylabel("N / N₀")
    ax.set_title("D. Normalised population decline", fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Figure_ForkedQuench.png", dpi=200, bbox_inches="tight")
    plt.savefig("Figure_ForkedQuench.pdf", dpi=300, bbox_inches="tight")
    print("  ✓ Figure_ForkedQuench.png / .pdf saved")


if __name__ == "__main__":
    main()
