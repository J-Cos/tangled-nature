# Tangled Nature Model — Spatial Metacommunity Simulator

A high-performance Rust implementation of the [Tangled Nature Model](https://iopscience.iop.org/article/10.1088/1361-6404/aaee8f/meta) (Jensen 2018), extended with a spatial metacommunity grid, qESS-based burn-in, and state save/load for fork-and-stress experiments.

## Features

- **Single-patch TNM** — classic well-mixed Tangled Nature dynamics
- **Spatial mode** — 2D grid of coupled patches with nearest-neighbour migration
- **Independent initialization** — `--independent-init` gives each patch its own random species pool (shared JEngine)
- **qESS detection** — automatic detection of quasi-Evolutionary Stable States via CV of total N and γ-diversity
- **State save/load** — checkpoint at qESS, fork into stress scenarios from identical starting conditions
- **Visualization** — automated multi-panel figures via `spatial_viz.py` and `spatial_viz2.py`
- **Deterministic** — seeded `ChaCha12Rng` for full reproducibility

## Quick Start

```bash
# Build
cargo build --release

# Run single-patch simulation (defaults: L=10, N_init=100, 200k gens)
./target/release/tangled-nature

# Run spatial metacommunity (10×10 grid, p_move=0.01)
./target/release/tangled-nature \
  --spatial --grid-size 10 --p-move 0.01 \
  --max-gen 200000 --output-interval 500 \
  --state-out state_burnin.json \
  --out burnin.jsonl
```

## Burn-In → Fork Workflow

The primary workflow is to run a spatial metacommunity to qESS, checkpoint it, then fork stress scenarios from the same starting state:

```bash
# Step 1: Burn-in to qESS (auto-detects and saves state)
./target/release/tangled-nature \
  --spatial --grid-size 10 --p-move 0.01 \
  --max-gen 200000 --output-interval 500 \
  --state-out qess_checkpoint.json \
  --out burnin.jsonl

# Step 2: Fork null scenario from checkpoint
./target/release/tangled-nature \
  --spatial --grid-size 10 --p-move 0.01 \
  --max-gen 5000 --output-interval 100 \
  --state-in qess_checkpoint.json \
  --out null_scenario.jsonl

# Step 3: Fork stress scenario (e.g. reduced carrying capacity)
./target/release/tangled-nature \
  --spatial --grid-size 10 --p-move 0.01 \
  --max-gen 5000 --output-interval 100 \
  --r 80 \
  --state-in qess_checkpoint.json \
  --out stress_scenario.jsonl
```

## CLI Parameters

| Flag                 | Default                | Description                                         |
| -------------------- | ---------------------- | --------------------------------------------------- |
| `--seed`             | system time            | RNG seed for reproducibility                        |
| `--l`                | 10                     | Genome length (bits); `2^L` possible genomes        |
| `--n-init`           | 100                    | Initial population per patch                        |
| `--max-gen`          | 200000                 | Maximum generations to run                          |
| `--theta`            | 0.25                   | J matrix sparsity (fraction of non-zero entries)    |
| `--p-kill`           | 0.2                    | Per-step kill probability                           |
| `--w`                | 33.0                   | Interaction weight in fitness function              |
| `--r`                | 143.0                  | Carrying capacity (stress target)                   |
| `--p-mut`            | 0.001                  | Mutation rate per locus per reproduction            |
| `--output-interval`  | 100                    | Generations between JSONL snapshots                 |
| `--qess-window`      | 5000                   | Rolling window for qESS CV calculation              |
| `--qess-threshold`   | 0.05                   | CV threshold for qESS detection                     |
| `--state-in`         | —                      | Load simulation state from JSON checkpoint          |
| `--state-out`        | —                      | Save state to JSON when qESS is reached             |
| `--output-j`         | false                  | Dump J matrix to stdout                             |
| `--spatial`          | false                  | Enable spatial metacommunity mode                   |
| `--grid-size`        | 10                     | Grid width and height (grid is square)              |
| `--p-move`           | 0.01                   | Per-individual migration probability per generation |
| `--independent-init` | false                  | Each patch gets its own random initial species pool |
| `--no-viz`           | false                  | Skip auto-visualization after spatial runs          |
| `--out`              | `spatial_output.jsonl` | Output JSONL file path                              |

## Fitness Function

Following Arthur et al. (MNRAS 2024):

```
p_off(i) = σ( W · H(i)/N − N/R )
```

where `H(i) = Σ_j J(i,j) · n_j` is the summed interaction strength, `N` is total population, `R` is carrying capacity, and `σ` is the logistic sigmoid.

## Output Format

### Spatial JSONL

Each line is a JSON object with:

```json
{
  "gen": 5000,
  "mean_n": 853.2,
  "mean_s": 12.1,
  "gamma_s": 47,
  "patches": [
    {"x": 0, "y": 0, "n": 845, "s": 11},
    {"x": 1, "y": 0, "n": 870, "s": 13},
    ...
  ]
}
```

### State Checkpoint (JSON)

The `--state-out` file captures the complete simulation state:
- `generation`: current generation count
- `grid_w`, `grid_h`: grid dimensions
- `p_move`: migration probability
- `l`: genome length
- `j_locus_matrices`: full J matrix (locus-level 2×2 matrices)
- `patch_states[]`: per-patch species composition (`BTreeMap<genome_id, count>`), RNG state, and seed
- `migration_rng`: migration RNG state

## Visualization

Two visualization scripts generate comprehensive figures:

```bash
# Spatial structure & diversity (auto-invoked unless --no-viz)
python3 spatial_viz.py burnin.jsonl

# Species-level analysis + TaNa genome heatmap
python3 spatial_viz2.py burnin.jsonl
```

**`spatial_viz.py` — Figure layout (4×4 grid):**

| Row | Panels                                                                            |
| --- | --------------------------------------------------------------------------------- |
| 1   | N heatmaps at 4 time points                                                       |
| 2   | S (richness) heatmaps at 4 time points                                            |
| 3   | Time series: Total N · α diversity · γ diversity · β diversity                    |
| 4   | Final state: N histogram · S histogram · N–S scatter · Spatial heterogeneity (CV) |

**`spatial_viz2.py` — Species analysis (3 rows × 4 cols + TaNa heatmap):**

| Row | Panels                                                               |
| --- | -------------------------------------------------------------------- |
| 1   | Top-4 species density maps across the spatial grid (final state)     |
| 2   | Patch-level SAD at 4 time points (rank-abundance with IQR)           |
| 3   | Landscape SAD (log-log rank-abundance) · Species occupancy over time |

Additionally generates a standalone **TaNa heatmap** (`Figure_SpatialTNM_tana.png`) showing all observed genomes (y-axis, ordered by first appearance) × time (x-axis) with log-scale abundance coloring. Use `--output-interval 1` for maximum temporal resolution.

## Source Layout

```
src/
├── main.rs       # CLI entry point, mode dispatch
├── config.rs     # Parameter struct and --flag parsing
├── model.rs      # Core TNM: JEngine, Simulation, QessDetector
├── spatial.rs    # SpatialGrid: 2D grid of patches with migration
├── state.rs      # SimState / SpatialState serialization
└── output.rs     # JSONL snapshot formatting
```

## Tests

### Rust unit tests (25 tests)

```bash
cargo test
```

Tests cover:
- **Config**: default values, L range validation
- **JEngine**: deterministic J matrix generation, J symmetry, finite values
- **Simulation**: population growth, reproduction probability, qESS detection, mutation range, state roundtrip with h_cache verification, weighted sampling
- **State**: JSON serialization roundtrip for `SimState`
- **Spatial**: toroidal neighbours, population conservation during migration, contagion stress ordering, ramp scaling, independent init (shared vs diverse patches, JEngine consistency)

### Python integration tests

```bash
# Spatial metrics equivalence test
python3 tests/test_equivalence.py

# Diversity and output metrics
python3 tests/test_metrics.py
```

## Key Algorithms

### qESS Detection

The simulation detects quasi-Evolutionary Stable States using a rolling-window coefficient-of-variation (CV) approach:
1. Total N across all patches is tracked in a sliding window (default 5000 gens)
2. γ-diversity (unique genomes across all patches) tracked in a parallel window
3. qESS is declared when both `CV(N) < threshold` and `CV(γ) < 2×threshold`
4. A minimum burn-in of 10,000 generations prevents false triggers during the initial transient

### Migration

Nearest-neighbour migration on a 2D grid with periodic (toroidal) boundaries:
1. Each individual migrates with probability `p_move` per generation
2. Destination is a random von Neumann neighbour (up/down/left/right)
3. Migration processes sequentially through patches (order doesn't matter due to low p_move)

### Interaction Matrix

The J matrix is built from `L` independent 2×2 locus matrices (θ-sparse), giving `J(a,b) = Σ_k M_k[a_k][b_k]` where `a_k` is bit `k` of genome `a`. This allows O(L) computation and compact serialization.

## Citation

Jensen, H. J. (2018). *The Tangled Nature Model*. European Journal of Physics, 39(6), 064002.
