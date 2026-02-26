#!/usr/bin/env python3
"""
Tangled Nature Model — Adiabatic Hysteresis Orchestrator

Drives 32 parallel TNM replicate simulations through an adiabatic stress
protocol (burn-in → forward press → reverse press), computes macroecological
metrics at each qESS checkpoint, and outputs results.csv.

Usage:
    python3 orchestrator.py --mode demo    # 2 replicates, 3 stress steps
    python3 orchestrator.py --mode full    # 32 replicates, full protocol
    python3 orchestrator.py --mode full --workers 16  # override worker count
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import root_scalar
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
BINARY = PROJECT_ROOT / "target" / "release" / "tangled-nature"
STATE_DIR = PROJECT_ROOT / "states"
RESULTS_CSV = PROJECT_ROOT / "results.csv"


@dataclass
class ExperimentConfig:
    """All tunables for the experiment."""
    # Mode
    mode: str = "demo"
    workers: int = 32

    # TNM parameters (calibrated for ~60s per step)
    l: int = 15
    w: float = 10.0
    r: float = 100.0       # baseline carrying capacity R₀
    p_kill: float = 0.2
    p_mut: float = 0.001
    theta: float = 0.25
    n_init: int = 100

    # Adiabatic stress protocol (Arthur et al. MNRAS 2024)
    # Stress = increasing μ = 1/R (decreasing R)
    delta_r: float = -1.0   # each step changes R by this amount (negative = more stress)
    collapse_fraction: float = 0.25  # N < 25% of baseline → stop forward press
    r_min: float = 5.0      # minimum R to avoid numerical instability

    # qESS detection
    qess_window: int = 1000
    qess_threshold: float = 0.10
    max_gen_per_step: int = 10_000
    output_interval: int = 500

    # Demo overrides (applied when mode == "demo")
    demo_replicates: int = 2
    demo_l: int = 10
    demo_w: float = 33.0
    demo_r: float = 143.0
    demo_n_init: int = 100
    demo_max_gen: int = 5000
    demo_stress_steps: int = 3
    demo_qess_window: int = 500
    demo_qess_threshold: float = 0.10

    def apply_demo(self):
        """Override for demo mode."""
        self.l = self.demo_l
        self.w = self.demo_w
        self.r = self.demo_r
        self.n_init = self.demo_n_init
        self.max_gen_per_step = self.demo_max_gen
        self.qess_window = self.demo_qess_window
        self.qess_threshold = self.demo_qess_threshold


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = True):
    fmt = "[%(asctime)s] %(levelname)-7s [Rep%(rep)s] %(message)s"

    class RepFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'rep'):
                record.rep = '??'
            return True

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    handler.addFilter(RepFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def log(rep_id, msg, level=logging.INFO):
    logging.getLogger().log(level, msg, extra={'rep': f'{rep_id:02d}'})


# ---------------------------------------------------------------------------
# METE D_KL computation
# ---------------------------------------------------------------------------

def compute_mete_dkl(abundances: list[int]) -> float:
    """Compute KL divergence between observed SAD and METE prediction.

    The METE truncated log-series:
        p(n) = (1/Z) * exp(-β*n) / n,  for n = 1..N
    with constraint: S1(β)/S0(β) = N/S
    where S1 = Σ exp(-βn), S0 = Σ exp(-βn)/n.
    """
    abundances = [a for a in abundances if a > 0]
    if len(abundances) < 2:
        return 0.0

    S = len(abundances)
    N = sum(abundances)
    target_ratio = N / S

    def constraint(beta):
        ns = np.arange(1, N + 1, dtype=np.float64)
        exp_terms = np.exp(-beta * ns)
        s1 = np.sum(exp_terms)
        s0 = np.sum(exp_terms / ns)
        if s0 == 0:
            return 1e10
        return s1 / s0 - target_ratio

    # Solve for beta
    try:
        result = root_scalar(constraint, bracket=[1e-12, 20.0],
                             method='brentq', xtol=1e-10, maxiter=200)
        beta = result.root
    except (ValueError, RuntimeError):
        # If solver fails, return NaN
        return float('nan')

    # Compute predicted SAD
    ns = np.arange(1, N + 1, dtype=np.float64)
    log_p_pred = -beta * ns - np.log(ns)
    log_z = np.log(np.sum(np.exp(log_p_pred)))
    log_p_pred -= log_z  # normalize

    # Observed SAD as probability distribution over abundance bins
    obs_counts = np.zeros(N + 1)
    for a in abundances:
        obs_counts[a] += 1
    p_obs = obs_counts[1:] / S  # p_obs[i] = fraction of species with abundance i+1

    # D_KL = Σ p_obs * log(p_obs / p_pred) over bins where p_obs > 0
    p_pred = np.exp(log_p_pred)
    dkl = 0.0
    for i in range(len(p_obs)):
        if p_obs[i] > 0 and p_pred[i] > 0:
            dkl += p_obs[i] * np.log(p_obs[i] / p_pred[i])

    return max(dkl, 0.0)


# ---------------------------------------------------------------------------
# Algebraic connectivity λ₂
# ---------------------------------------------------------------------------

def compute_lambda2(j_matrix: list[list[float]]) -> float:
    """Compute Fiedler value (2nd smallest eigenvalue) of the normalized
    graph Laplacian from the interaction matrix J.

    A_ij = 1 if |J_ij| > 0 (unweighted adjacency).
    L_norm = I - D^{-1/2} A D^{-1/2}.
    """
    J = np.array(j_matrix, dtype=np.float64)
    s = J.shape[0]
    if s < 2:
        return 0.0

    # Unweighted adjacency (exclude self-loops)
    A = (np.abs(J) > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    # Degree vector
    d = A.sum(axis=1)

    # Handle isolated nodes (degree 0)
    if np.any(d == 0):
        return 0.0  # disconnected graph

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(s) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigenvalues (sorted ascending)
    eigenvalues = eigh(L_norm, eigvals_only=True)
    # λ₂ is the second smallest
    return float(eigenvalues[1])


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------

def run_binary(
    seed: int, cfg: ExperimentConfig, r_value: float,
    state_in: Optional[str] = None, state_out: Optional[str] = None,
    output_j: bool = True, rep_id: int = 0,
) -> dict:
    """Run the TNM binary and parse its JSON-lines output.

    Stress is applied via the --r flag (carrying capacity).
    Lower R = higher stress (Arthur et al. MNRAS 2024).

    Returns dict with keys: qess_reached, snapshots, qess_data, stderr.
    """
    cmd = [
        str(BINARY),
        "--seed", str(seed),
        "--l", str(cfg.l),
        "--w", str(cfg.w),
        "--r", str(r_value),
        "--p-kill", str(cfg.p_kill),
        "--n-init", str(cfg.n_init),
        "--max-gen", str(cfg.max_gen_per_step),
        "--output-interval", str(cfg.output_interval),
        "--p-mut", str(cfg.p_mut),
        "--theta", str(cfg.theta),
        "--qess-window", str(cfg.qess_window),
        "--qess-threshold", str(cfg.qess_threshold),
    ]
    if state_in:
        cmd.extend(["--state-in", state_in])
    if state_out:
        cmd.extend(["--state-out", state_out])
    if output_j:
        cmd.append("--output-j")

    log(rep_id, f"  CMD: {' '.join(cmd[-8:])}", logging.DEBUG)

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=3600,  # 1 hour max per step
    )

    # Parse output
    snapshots = []
    qess_data = None
    final_data = None

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj["type"] == "snapshot":
            snapshots.append(obj)
        elif obj["type"] == "qess":
            qess_data = obj
        elif obj["type"] == "final":
            final_data = obj

    qess_reached = final_data.get("qess", False) if final_data else False

    return {
        "qess_reached": qess_reached,
        "snapshots": snapshots,
        "qess_data": qess_data,
        "final": final_data,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Adiabatic protocol for one replicate
# ---------------------------------------------------------------------------

def run_replicate(rep_id: int, cfg: ExperimentConfig) -> list[dict]:
    """Run the full adiabatic protocol for one replicate.

    Stress protocol follows Arthur et al. (MNRAS 2024):
    - Forward press: decrease R (increase μ = 1/R)
    - Reverse press: increase R back to R₀

    Returns list of result rows for the CSV.
    """
    seed = 1000 + rep_id * 137  # deterministic, spread-out seeds
    results = []
    rep_state_dir = STATE_DIR / f"rep_{rep_id:02d}"
    rep_state_dir.mkdir(parents=True, exist_ok=True)

    r_current = cfg.r  # start at baseline R₀

    log(rep_id, f"═══ STARTING REPLICATE {rep_id:02d} (seed={seed}) ═══")

    # ------------------------------------------------------------------
    # Phase 0: Burn-in at baseline R
    # ------------------------------------------------------------------
    state_file = str(rep_state_dir / "state_burnin.json")

    log(rep_id, f"Phase 0: BURN-IN at R={r_current:.2f} (μ=1/R={1/r_current:.6f})")
    t0 = time.time()

    out = run_binary(seed, cfg, r_current, state_out=state_file,
                     output_j=True, rep_id=rep_id)

    elapsed = time.time() - t0
    if not out["qess_reached"]:
        log(rep_id, f"  ⚠ BURN-IN: qESS NOT reached in {cfg.max_gen_per_step} "
            f"generations ({elapsed:.1f}s). Proceeding anyway.", logging.WARNING)
    else:
        n = out["qess_data"]["n"] if out["qess_data"] else "?"
        s = out["qess_data"]["s"] if out["qess_data"] else "?"
        log(rep_id, f"  ✓ BURN-IN qESS: N={n}, S={s}, CV={out['qess_data'].get('cv', '?'):.4f}"
            f" ({elapsed:.1f}s)")

    # Record baseline
    if out["qess_data"]:
        n_baseline = out["qess_data"]["n"]
        row = compute_metrics(rep_id, "Burn-in", r_current, out["qess_data"])
        results.append(row)
        log(rep_id, f"  Metrics: D_KL={row['METE_DKL']:.4f}, λ₂={row['Lambda_2']:.4f}")
    else:
        n_baseline = cfg.n_init
        log(rep_id, "  ⚠ No qESS data for burn-in, using n_init as baseline",
            logging.WARNING)

    collapse_threshold = max(int(n_baseline * cfg.collapse_fraction), 1)
    log(rep_id, f"  Baseline N₀={n_baseline}, collapse threshold={collapse_threshold}")

    # ------------------------------------------------------------------
    # Phase 1: Forward press (decrease R → increase stress)
    # ------------------------------------------------------------------
    max_stress_steps = cfg.demo_stress_steps if cfg.mode == "demo" else 9999
    step_count = 0

    log(rep_id, f"Phase 1: FORWARD PRESS (ΔR={cfg.delta_r})")

    prev_state = state_file
    while step_count < max_stress_steps:
        r_current += cfg.delta_r  # delta_r is negative → R decreases
        if r_current < cfg.r_min:
            log(rep_id, f"  ★ R reached minimum ({cfg.r_min}). Stopping forward press.")
            r_current = cfg.r_min
            break
        step_count += 1
        state_file = str(rep_state_dir / f"state_fwd_{step_count:04d}.json")

        mu_current = 1.0 / r_current
        log(rep_id, f"  Forward step {step_count}: R={r_current:.2f} (μ=1/R={mu_current:.6f})")
        t0 = time.time()

        out = run_binary(seed + step_count, cfg, r_current,
                         state_in=prev_state, state_out=state_file,
                         output_j=True, rep_id=rep_id)
        elapsed = time.time() - t0

        if out["qess_data"]:
            n_now = out["qess_data"]["n"]
            s_now = out["qess_data"]["s"]
            cv = out["qess_data"].get("cv", "?")
            qess_tag = "qESS" if out["qess_reached"] else "no-qESS"
            log(rep_id, f"    → N={n_now}, S={s_now}, CV={cv:.4f} [{qess_tag}] "
                f"({elapsed:.1f}s)")

            row = compute_metrics(rep_id, "Forward", r_current, out["qess_data"])
            results.append(row)
            log(rep_id, f"    Metrics: D_KL={row['METE_DKL']:.4f}, "
                f"λ₂={row['Lambda_2']:.4f}")

            if n_now < collapse_threshold:
                log(rep_id, f"  ★ NEAR-COLLAPSE at R={r_current:.2f} (N={n_now} < "
                    f"{collapse_threshold}). Stopping forward press.")
                break
        else:
            log(rep_id, f"    → No data (returncode={out['returncode']}, "
                f"{elapsed:.1f}s)", logging.WARNING)
            if out["returncode"] == 2:  # extinction
                log(rep_id, f"  ★ EXTINCTION at R={r_current:.2f}. "
                    "Reversing from last viable state.")
                break

        prev_state = state_file

    r_min_reached = r_current

    # ------------------------------------------------------------------
    # Phase 2: Reverse press (increase R back → hysteresis test)
    # ------------------------------------------------------------------
    log(rep_id, f"Phase 2: REVERSE PRESS from R={r_min_reached:.2f} "
        f"(ΔR={-cfg.delta_r})")

    step_count = 0
    while r_current < cfg.r and step_count < max_stress_steps:
        r_current -= cfg.delta_r  # delta_r is negative, so -delta_r is positive
        r_current = min(r_current, cfg.r)
        step_count += 1
        state_file = str(rep_state_dir / f"state_rev_{step_count:04d}.json")

        mu_current = 1.0 / r_current
        log(rep_id, f"  Reverse step {step_count}: R={r_current:.2f} (μ=1/R={mu_current:.6f})")
        t0 = time.time()

        out = run_binary(seed + 10000 + step_count, cfg, r_current,
                         state_in=prev_state, state_out=state_file,
                         output_j=True, rep_id=rep_id)
        elapsed = time.time() - t0

        if out["qess_data"]:
            n_now = out["qess_data"]["n"]
            s_now = out["qess_data"]["s"]
            cv = out["qess_data"].get("cv", "?")
            qess_tag = "qESS" if out["qess_reached"] else "no-qESS"
            log(rep_id, f"    → N={n_now}, S={s_now}, CV={cv:.4f} [{qess_tag}] "
                f"({elapsed:.1f}s)")

            row = compute_metrics(rep_id, "Reverse", r_current, out["qess_data"])
            results.append(row)
            log(rep_id, f"    Metrics: D_KL={row['METE_DKL']:.4f}, "
                f"λ₂={row['Lambda_2']:.4f}")
        else:
            log(rep_id, f"    → No data ({elapsed:.1f}s)", logging.WARNING)

        prev_state = state_file

    log(rep_id, f"═══ REPLICATE {rep_id:02d} COMPLETE: {len(results)} data points ═══")
    return results


def compute_metrics(rep_id: int, phase: str, r_value: float, qess_data: dict) -> dict:
    """Extract N, S, compute METE D_KL and λ₂ from qESS output.

    The CSV column 'Mu' reports μ = 1/R to match Arthur et al. (MNRAS 2024).
    """
    n = qess_data["n"]
    s = qess_data["s"]
    mu = 1.0 / r_value  # Arthur et al. notation: μ = 1/R

    # Abundances from species list
    abundances = [count for _, count in qess_data.get("species", [])]

    # METE D_KL
    try:
        dkl = compute_mete_dkl(abundances) if len(abundances) >= 2 else float('nan')
    except Exception:
        dkl = float('nan')

    # λ₂ from J matrix
    j_data = qess_data.get("j_matrix")
    if j_data and "matrix" in j_data:
        try:
            lam2 = compute_lambda2(j_data["matrix"])
        except Exception:
            lam2 = float('nan')
    else:
        lam2 = float('nan')

    return {
        "Replicate": rep_id,
        "Phase": phase,
        "Mu": round(mu, 6),
        "N": n,
        "S": s,
        "METE_DKL": round(dkl, 6) if not np.isnan(dkl) else float('nan'),
        "Lambda_2": round(lam2, 6) if not np.isnan(lam2) else float('nan'),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TNM Adiabatic Hysteresis Orchestrator")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                        help="demo: 2 replicates, small scale. full: 32 replicates.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override worker count (default: 32 for full, 2 for demo)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results.csv)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="Verbose logging (default)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Reduce logging")
    args = parser.parse_args()

    setup_logging(verbose=not args.quiet)
    logger = logging.getLogger()

    cfg = ExperimentConfig(mode=args.mode)
    if args.mode == "demo":
        cfg.apply_demo()

    n_replicates = cfg.demo_replicates if args.mode == "demo" else 32
    n_workers = args.workers or (cfg.demo_replicates if args.mode == "demo" else 32)
    out_csv = Path(args.output) if args.output else RESULTS_CSV

    # Banner
    logger.info("╔══════════════════════════════════════════════════════════╗",
                extra={'rep': '--'})
    logger.info("║  Tangled Nature Model — Adiabatic Hysteresis Protocol   ║",
                extra={'rep': '--'})
    logger.info("╚══════════════════════════════════════════════════════════╝",
                extra={'rep': '--'})
    logger.info(f"Mode: {args.mode.upper()} | Replicates: {n_replicates} | "
                f"Workers: {n_workers}", extra={'rep': '--'})
    logger.info(f"Parameters: L={cfg.l}, W={cfg.w}, R₀={cfg.r:.1f}, "
                f"P_KILL={cfg.p_kill}, ΔR={cfg.delta_r}", extra={'rep': '--'})
    logger.info(f"qESS: window={cfg.qess_window}, threshold={cfg.qess_threshold}, "
                f"max_gen={cfg.max_gen_per_step}", extra={'rep': '--'})
    logger.info(f"Output: {out_csv}", extra={'rep': '--'})
    logger.info("", extra={'rep': '--'})

    # Check binary
    if not BINARY.exists():
        logger.error(f"Binary not found: {BINARY}. Run 'cargo build --release'.",
                     extra={'rep': '--'})
        sys.exit(1)

    # Create state directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Run replicates in parallel
    all_results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_replicate, i, cfg): i
            for i in range(n_replicates)
        }

        for future in as_completed(futures):
            rep_id = futures[future]
            try:
                rows = future.result()
                all_results.extend(rows)
                logger.info(f"Replicate {rep_id:02d} finished: {len(rows)} rows",
                            extra={'rep': f'{rep_id:02d}'})
            except Exception as e:
                logger.error(f"Replicate {rep_id:02d} FAILED: {e}",
                             extra={'rep': f'{rep_id:02d}'})

    t_total = time.time() - t_start

    # Write CSV
    if all_results:
        all_results.sort(key=lambda r: (r["Replicate"], r["Phase"], r["Mu"]))
        fieldnames = ["Replicate", "Phase", "Mu", "N", "S", "METE_DKL", "Lambda_2"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logger.info("", extra={'rep': '--'})
        logger.info(f"╔══ COMPLETE ══════════════════════════════════════════╗",
                    extra={'rep': '--'})
        logger.info(f"║ Total rows: {len(all_results):>6}  |  Time: {t_total:>7.1f}s "
                    f"              ║", extra={'rep': '--'})
        logger.info(f"║ Output: {str(out_csv):<46}║", extra={'rep': '--'})
        logger.info(f"╚═════════════════════════════════════════════════════╝",
                    extra={'rep': '--'})
    else:
        logger.error("No results collected!", extra={'rep': '--'})
        sys.exit(1)


if __name__ == "__main__":
    main()
