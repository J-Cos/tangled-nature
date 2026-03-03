#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# Harvesting Fork Experiment (32×32)
#
# Phase 1: Run 32 sims to qESS, save states
# Phase 2: For each sim, harvest top 32 species (1,024 forks total)
# Phase 3: Analyze and visualize
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="$PROJECT_DIR/target/release/tangled-nature"

N_SIMS=32
N_TARGETS=32
FORK_GENS=400
HARVEST_AFTER=200
HARVEST_RATE=0.25
OUTPUT_INTERVAL=1       # every gen
SPECIES_INTERVAL=10     # species data every 10 gens

STATE_DIR="$SCRIPT_DIR/states"
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "================================================================"
echo "  Harvesting Fork Experiment"
echo "  ${N_SIMS} sims → ${N_TARGETS} species each → ${N_SIMS}×${N_TARGETS} = $((N_SIMS * N_TARGETS)) forks"
echo "  Each fork: ${FORK_GENS} gens (${HARVEST_AFTER} baseline + $((FORK_GENS - HARVEST_AFTER)) harvest)"
echo "  Harvest rate: ${HARVEST_RATE} (25%)"
echo "================================================================"
echo ""

# ── Step 0: Build ────────────────────────────────────────────────
echo "[Step 0] Building release binary..."
cd "$PROJECT_DIR"
cargo build --release 2>&1 | tail -2
echo ""

# ── Step 1: Unit tests ───────────────────────────────────────────
echo "[Step 1] Running unit tests..."
echo "  Rust:"
cargo test 2>&1 | tail -3
echo "  Python:"
python3 -m unittest "$SCRIPT_DIR/test_analyze.py" -v 2>&1 | tail -8
echo ""

# ── Step 2: Run 32 sims to qESS ─────────────────────────────────
echo "[Step 2] Running ${N_SIMS} simulations to qESS..."
mkdir -p "$STATE_DIR" "$DATA_DIR"

run_burnin() {
    local sim_id=$1
    local seed=$((2000 + sim_id))
    local state_out="$STATE_DIR/sim_$(printf '%02d' $sim_id).json"

    "$BIN" \
        --seed "$seed" \
        --max-gen 200000 \
        --p-mut 0.025 \
        --output-interval 1000 \
        --qess-threshold 0.05 \
        --no-viz \
        --state-out "$state_out" \
        --out /dev/null \
        > /dev/null 2>&1

    echo "  sim_$(printf '%02d' $sim_id) → qESS (seed=$seed)"
}

export -f run_burnin
export BIN STATE_DIR

START_TIME=$(date +%s)
seq 1 "$N_SIMS" | xargs -P "$N_SIMS" -I {} bash -c 'run_burnin {}'
BURNIN_TIME=$(($(date +%s) - START_TIME))
echo "  Burn-in complete in ${BURNIN_TIME}s"
echo ""

# ── Step 3: Extract top species and run forks ────────────────────
echo "[Step 3] Running $((N_SIMS * N_TARGETS)) harvesting forks..."

# Python helper to extract top species genomes from state file
extract_genomes() {
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    state = json.load(f)
species = state.get('species', {})
sorted_sp = sorted(species.items(), key=lambda x: int(x[1]), reverse=True)
for i, (g, c) in enumerate(sorted_sp[:int(sys.argv[2])]):
    print(f'{g} {c}')
" "$1" "$2"
}

run_fork() {
    local sim_id=$1
    local rank=$2
    local genome=$3
    local state_file="$STATE_DIR/sim_$(printf '%02d' $sim_id).json"
    local out_file="$DATA_DIR/sim_$(printf '%02d' $sim_id)_rank$(printf '%02d' $rank).jsonl"

    "$BIN" \
        --state-in "$state_file" \
        --max-gen "$FORK_GENS" \
        --output-interval "$OUTPUT_INTERVAL" \
        --species-interval "$SPECIES_INTERVAL" \
        --qess-threshold 0.0 \
        --no-viz \
        --harvest-genome "$genome" \
        --harvest-rate "$HARVEST_RATE" \
        --harvest-after "$HARVEST_AFTER" \
        --out "$out_file" \
        > /dev/null 2>&1
}

export -f run_fork
export BIN STATE_DIR DATA_DIR FORK_GENS OUTPUT_INTERVAL SPECIES_INTERVAL HARVEST_RATE HARVEST_AFTER

START_TIME=$(date +%s)

# Build a task list file: sim_id rank genome
TASK_FILE=$(mktemp)
for sim_id in $(seq 1 "$N_SIMS"); do
    state_file="$STATE_DIR/sim_$(printf '%02d' $sim_id).json"
    if [ ! -f "$state_file" ]; then
        echo "  WARNING: $state_file not found, skipping sim $sim_id"
        continue
    fi
    # Extract top N_TARGETS species genomes
    python3 -c "
import json, sys
with open('$state_file') as f:
    state = json.load(f)
sp = state.get('species', {})
for i, (g, c) in enumerate(sorted(sp.items(), key=lambda x: int(x[1]), reverse=True)[:$N_TARGETS]):
    print(f'$sim_id {i+1} {g}')
"
done > "$TASK_FILE"

N_TASKS=$(wc -l < "$TASK_FILE")
echo "  Generated $N_TASKS fork tasks"

# Run all forks in parallel (32 at a time)
run_single_fork() {
    local sim_id=$1
    local rank=$2
    local genome=$3
    local state_file="$STATE_DIR/sim_$(printf '%02d' $sim_id).json"
    local out_file="$DATA_DIR/sim_$(printf '%02d' $sim_id)_rank$(printf '%02d' $rank).jsonl"

    "$BIN" \
        --state-in "$state_file" \
        --max-gen "$FORK_GENS" \
        --output-interval "$OUTPUT_INTERVAL" \
        --species-interval "$SPECIES_INTERVAL" \
        --qess-threshold 0.0 \
        --no-viz \
        --harvest-genome "$genome" \
        --harvest-rate "$HARVEST_RATE" \
        --harvest-after "$HARVEST_AFTER" \
        --out "$out_file" \
        > /dev/null 2>&1 || true  # TNM exits 1 when max_gen reached (expected)
}
export -f run_single_fork
export BIN STATE_DIR DATA_DIR FORK_GENS OUTPUT_INTERVAL SPECIES_INTERVAL HARVEST_RATE HARVEST_AFTER

cat "$TASK_FILE" | xargs -P "$N_SIMS" -n 3 bash -c 'run_single_fork "$@"' _
rm -f "$TASK_FILE"

FORK_TIME=$(($(date +%s) - START_TIME))
N_FORKS=$(ls "$DATA_DIR"/sim_*_rank*.jsonl 2>/dev/null | wc -l)
echo "  Completed ${N_FORKS} forks in ${FORK_TIME}s"
echo ""

# ── Step 4: Analyze ──────────────────────────────────────────────
echo "[Step 4] Analyzing fork results..."
mkdir -p "$RESULTS_DIR"
python3 "$SCRIPT_DIR/analyze.py" "$DATA_DIR" "$RESULTS_DIR"
echo ""

# ── Step 5: Visualize ────────────────────────────────────────────
echo "[Step 5] Generating visualizations..."
python3 "$SCRIPT_DIR/visualize.py" "$RESULTS_DIR"
echo ""

echo "================================================================"
echo "  Experiment complete!"
echo "  Burn-in: ${BURNIN_TIME}s | Forks: ${FORK_TIME}s"
echo "  Forks: ${N_FORKS}"
echo "  Results: ${RESULTS_DIR}/"
echo "  Figure:  ${RESULTS_DIR}/Figure_HarvestForks.png"
echo "================================================================"
