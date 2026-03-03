#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# METE Ensemble Experiment
#
# Runs 32 non-spatial TNM simulations, extracts SAD data,
# fits METE predictions, and generates visualizations.
#
# Hardware: 32 CPUs available for parallel simulation
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="$PROJECT_DIR/target/release/tangled-nature"
EXPERIMENT_DIR="$SCRIPT_DIR"

N_SIMS=32
MAX_GEN=100000
P_MUT=0.025
OUTPUT_INTERVAL=1        # every gen to stdout (N/S only)
SPECIES_INTERVAL=100     # species-level data to file every 100 gens

DATA_DIR="$EXPERIMENT_DIR/data"
RESULTS_DIR="$EXPERIMENT_DIR/results"

echo "================================================================"
echo "  METE Ensemble Experiment"
echo "  ${N_SIMS} simulations × ${MAX_GEN} gens | p_mut=${P_MUT}"
echo "  Species output every ${SPECIES_INTERVAL} gens"
echo "================================================================"
echo ""

# ── Step 0: Build ────────────────────────────────────────────────
echo "[Step 0] Building release binary..."
cd "$PROJECT_DIR"
cargo build --release 2>&1 | tail -2
echo ""

# ── Step 1: Run unit tests ───────────────────────────────────────
echo "[Step 1] Running unit tests..."
echo "  Python preprocessing tests:"
python3 -m unittest "$EXPERIMENT_DIR/test_preprocess.py" -v 2>&1 | tail -10
echo ""
echo "  R METE analysis tests:"
Rscript "$EXPERIMENT_DIR/test_mete_analysis.R" 2>&1
echo ""

# ── Step 2: Run simulations in parallel ──────────────────────────
echo "[Step 2] Running ${N_SIMS} simulations in parallel..."
mkdir -p "$DATA_DIR"

# Function to run a single simulation
run_sim() {
    local sim_id=$1
    local seed=$((1000 + sim_id))
    local out_file="$DATA_DIR/sim_$(printf '%02d' $sim_id).jsonl"

    "$BIN" \
        --seed "$seed" \
        --max-gen "$MAX_GEN" \
        --p-mut "$P_MUT" \
        --output-interval "$OUTPUT_INTERVAL" \
        --species-interval "$SPECIES_INTERVAL" \
        --qess-threshold 0.0 \
        --no-viz \
        --out "$out_file" \
        > /dev/null 2>&1

    echo "  sim_$(printf '%02d' $sim_id) done (seed=$seed)"
}

export -f run_sim
export BIN DATA_DIR MAX_GEN P_MUT OUTPUT_INTERVAL SPECIES_INTERVAL

START_TIME=$(date +%s)

# Run all sims in parallel (one per CPU)
seq 1 "$N_SIMS" | xargs -P "$N_SIMS" -I {} bash -c 'run_sim {}'

END_TIME=$(date +%s)
SIM_ELAPSED=$((END_TIME - START_TIME))
echo "  All ${N_SIMS} simulations completed in ${SIM_ELAPSED}s"
echo ""

# ── Step 3: Preprocess ───────────────────────────────────────────
echo "[Step 3] Preprocessing JSONL outputs..."
mkdir -p "$RESULTS_DIR"
python3 "$EXPERIMENT_DIR/preprocess.py" "$DATA_DIR" "$RESULTS_DIR"
echo ""

# ── Step 4: METE analysis (R) ────────────────────────────────────
echo "[Step 4] Running METE analysis (meteR)..."
Rscript "$EXPERIMENT_DIR/mete_analysis.R" "$RESULTS_DIR"
echo ""

# ── Step 5: Visualization ────────────────────────────────────────
echo "[Step 5] Generating visualizations..."
python3 "$EXPERIMENT_DIR/visualize.py" "$RESULTS_DIR"
echo ""

echo "================================================================"
echo "  Experiment complete!"
echo "  Simulations: ${SIM_ELAPSED}s"
echo "  Data:    ${DATA_DIR}/"
echo "  Results: ${RESULTS_DIR}/"
echo "  Figure:  ${RESULTS_DIR}/Figure_METE_ensemble.png"
echo "================================================================"
