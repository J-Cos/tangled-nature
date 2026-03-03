#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# Causal Emergence Experiment
#
# Runs a single 100k-gen TNM simulation with per-gen species output,
# then computes effective information at multiple scales over time.
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="$PROJECT_DIR/target/release/tangled-nature"

DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"

MAX_GEN=10000
OUTPUT_INTERVAL=1        # N/S every gen (stdout)
SPECIES_INTERVAL=1       # species-level output every gen (file)
WINDOW=500
STEP=100

echo "================================================================"
echo "  Causal Emergence in the Tangled Nature Model"
echo "  ${MAX_GEN} generations | window=${WINDOW} step=${STEP}"
echo "================================================================"
echo ""

# ── Step 0: Build ────────────────────────────────────────────────
echo "[Step 0] Building release binary..."
cd "$PROJECT_DIR"
cargo build --release 2>&1 | tail -2
echo ""

# ── Step 1: Unit tests ───────────────────────────────────────────
echo "[Step 1] Running unit tests..."
python3 -m unittest "$SCRIPT_DIR/test_causal_emergence.py" -v 2>&1 | tail -10
echo ""

# ── Step 2: Run simulation ───────────────────────────────────────
echo "[Step 2] Running ${MAX_GEN}-gen simulation..."
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

SIM_OUT="$DATA_DIR/sim_100k.jsonl"

"$BIN" \
    --seed 42 \
    --max-gen "$MAX_GEN" \
    --p-mut 0.025 \
    --output-interval "$OUTPUT_INTERVAL" \
    --species-interval "$SPECIES_INTERVAL" \
    --qess-threshold 0.0 \
    --no-viz \
    --out "$SIM_OUT" \
    > /dev/null 2>&1 || true  # exits 1 when max_gen reached

N_LINES=$(wc -l < "$SIM_OUT")
echo "  Simulation output: ${N_LINES} lines → ${SIM_OUT}"
echo ""

# ── Step 3: Analyze ──────────────────────────────────────────────
echo "[Step 3] Computing causal emergence..."
python3 "$SCRIPT_DIR/causal_emergence.py" \
    "$SIM_OUT" "$RESULTS_DIR" \
    --window "$WINDOW" --step "$STEP"
echo ""

# ── Step 4: Visualize ────────────────────────────────────────────
echo "[Step 4] Generating visualizations..."
python3 "$SCRIPT_DIR/visualize.py" "$RESULTS_DIR"
echo ""

echo "================================================================"
echo "  Experiment complete!"
echo "  Results: ${RESULTS_DIR}/"
echo "  Figure:  ${RESULTS_DIR}/Figure_CausalEmergence.png"
echo "================================================================"
