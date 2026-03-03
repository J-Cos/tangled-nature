#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# TNM Adiabatic Hysteresis — Overnight Full-Scale Run
# 250 replicates, Arthur et al. (MNRAS 2024) stress protocol
# Expected runtime: ~4–5 hours on 32 cores
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="results_overnight_${TIMESTAMP}.csv"
LOG="overnight_${TIMESTAMP}.log"

echo "╔══════════════════════════════════════════════════════════╗" | tee "$LOG"
echo "║  TNM Overnight Run — Started $(date)  ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  250 replicates | L=20 | R₀=100 | ΔR=−1 | max_gen=200k" | tee -a "$LOG"
echo "  Output: $RESULTS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Phase 1: Simulations ──────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 1: Running 250 replicates..." | tee -a "$LOG"

rm -rf states/

python3 -c "
import orchestrator as o
import time, csv, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure
cfg = o.ExperimentConfig(mode='full')
cfg.l = 20
cfg.w = 10.0
cfg.r = 100.0
cfg.delta_r = -1.0        # fine resolution: 95 fwd + 95 rev steps
cfg.r_min = 5.0
cfg.max_gen_per_step = 200_000
cfg.qess_window = 5000
cfg.qess_threshold = 0.05
cfg.output_interval = 10000
cfg.workers = 32

N_REP = 250

o.setup_logging(verbose=False)
logger = logging.getLogger()
logger.info(f'Overnight run: {N_REP} reps, L={cfg.l}, R₀={cfg.r}, ΔR={cfg.delta_r}, max_gen={cfg.max_gen_per_step}',
            extra={'rep':'--'})

o.STATE_DIR.mkdir(parents=True, exist_ok=True)
all_results = []
completed = 0
t0 = time.time()

with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
    futs = {pool.submit(o.run_replicate, i, cfg): i for i in range(N_REP)}
    for fut in as_completed(futs):
        rep_id = futs[fut]
        try:
            rows = fut.result()
            all_results.extend(rows)
            completed += 1
            elapsed = time.time() - t0
            rate = elapsed / completed
            eta = rate * (N_REP - completed)
            logger.info(f'Rep {rep_id:03d} done: {len(rows)} rows | '
                        f'{completed}/{N_REP} complete | '
                        f'ETA: {eta/3600:.1f}h',
                        extra={'rep': f'{rep_id:03d}'})
        except Exception as e:
            completed += 1
            logger.error(f'Rep {rep_id:03d} FAILED: {e}', extra={'rep': f'{rep_id:03d}'})

# Write CSV
out_csv = '$RESULTS'
fieldnames = ['Replicate','Phase','Mu','N','S','METE_DKL','Lambda_2']
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(all_results, key=lambda r: (r['Replicate'], r['Mu'])):
        writer.writerow(row)

total = time.time() - t0
print(f'\\n=== SIMULATIONS COMPLETE: {len(all_results)} rows in {total/3600:.1f}h ===')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Phase 1 complete." | tee -a "$LOG"

# ── Phase 2: R Analysis ──────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 2: Running R analysis..." | tee -a "$LOG"
Rscript analysis.R "$RESULTS" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Phase 2 complete." | tee -a "$LOG"

# ── Phase 3: Figures ──────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 3: Generating figures..." | tee -a "$LOG"
Rscript figures.R "$RESULTS" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Phase 3 complete." | tee -a "$LOG"

# ── Done ──────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "╔══════════════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  ALL DONE — $(date)  ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "  Results: $RESULTS" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "  Figures: Figure_1_Hysteresis.png, Figure_2_EWS.png, Figure_3_Synthesis.png" | tee -a "$LOG"
