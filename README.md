# Tangled Nature Model — Spatial Metacommunity Simulator

A high-performance Rust implementation of the [Tangled Nature Model](https://iopscience.iop.org/article/10.1088/1361-6404/aaee8f/meta) (Jensen 2018), extended with a spatial metacommunity grid, qESS-based burn-in, and state save/load for fork-and-stress experiments.

The J matrix uses the dense independent-random-entry approach matching the original reference implementation: `J[i][k]` and `J[k][i]` are sampled independently from `U(-1,1)` with probability Θ.

## Features

- **Single-patch TNM** — classic well-mixed Tangled Nature dynamics
- **Spatial mode** — 2D grid of coupled patches with nearest-neighbour migration (rayon parallelism across all cores)
- **Independent initialization** — `--independent-init` gives each patch its own random species pool (shared JEngine)
- **qESS detection** — automatic detection of quasi-Evolutionary Stable States via CV of total N and γ-diversity
- **State save/load** — checkpoint at qESS, fork into stress scenarios from identical starting conditions
- **Visualization** — automated multi-panel figures via `classic_viz.py` (single-patch) and `spatial_viz.py` / `spatial_viz2.py` (spatial)
- **Deterministic** — seeded `ChaCha12Rng` for full reproducibility
- **Optimized** — zero-allocation h_cache updates and batched reproduction for ~2× single-patch throughput

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

| Flag                 | Default                | Description                                                          |
| -------------------- | ---------------------- | -------------------------------------------------------------------- |
| `--seed`             | system time            | RNG seed for reproducibility                                         |
| `--l`                | 10                     | Genome length (bits); `2^L` possible genomes                         |
| `--n-init`           | 100                    | Initial population per patch                                         |
| `--max-gen`          | 200000                 | Maximum generations to run                                           |
| `--theta`            | 0.25                   | J matrix sparsity (fraction of non-zero entries)                     |
| `--p-kill`           | 0.2                    | Per-step kill probability                                            |
| `--w`                | 33.0                   | Interaction weight in fitness function                               |
| `--r`                | 143.0                  | Carrying capacity (stress target)                                    |
| `--p-mut`            | 0.001                  | Mutation rate per locus per reproduction                             |
| `--output-interval`  | 100                    | Generations between JSONL snapshots                                  |
| `--species-interval` | 0                      | Interval for species-level file output (0 = same as output-interval) |
| `--qess-window`      | 5000                   | Rolling window for qESS CV calculation                               |
| `--qess-threshold`   | 0.05                   | CV threshold for qESS detection                                      |
| `--state-in`         | —                      | Load simulation state from JSON checkpoint                           |
| `--state-out`        | —                      | Save state to JSON when qESS is reached                              |
| `--output-j`         | false                  | Dump J matrix to stdout                                              |
| `--spatial`          | false                  | Enable spatial metacommunity mode                                    |
| `--grid-size`        | 10                     | Grid width and height (grid is square)                               |
| `--p-move`           | 0.01                   | Per-individual migration probability per generation                  |
| `--independent-init` | false                  | Each patch gets its own random initial species pool                  |
| `--no-viz`           | false                  | Skip auto-visualization after spatial runs                           |
| `--harvest-genome`   | —                      | Genome ID to harvest (enables harvesting mode)                       |
| `--harvest-rate`     | 0.0                    | Fraction of target species removed each generation                   |
| `--harvest-after`    | 0                      | Generations to wait before harvesting begins                         |
| `--out`              | `spatial_output.jsonl` | Output JSONL file path                                               |

## Fitness Function

Following Jensen (2018):

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
- `j_dense`: full dense J matrix (`2^L × 2^L` independent entries)
- `patch_states[]`: per-patch species composition (`BTreeMap<genome_id, count>`), RNG state, and seed
- `migration_rng`: migration RNG state

## Visualization

Three visualization scripts generate comprehensive figures:

```bash
# Single-patch dynamics + TaNa heatmap (auto-invoked unless --no-viz)
python3 classic_viz.py output.jsonl

# Spatial structure & diversity (auto-invoked for spatial runs)
python3 spatial_viz.py burnin.jsonl

# Species-level analysis
python3 spatial_viz2.py burnin.jsonl
```

**`classic_viz.py` — Single-patch analysis (2×3 grid + TaNa heatmap):**

| Row | Panels                                                    |
| --- | --------------------------------------------------------- |
| 1   | N time series · S time series · SAD (rank-abundance)      |
| 2   | N histogram · Phase space (N vs S) · Population stability |

Plus a standalone **TaNa heatmap** showing all `2^L` possible genomes (y-axis) × time (x-axis) with log-scale abundance coloring (paper-style, Jensen 2018).

**`spatial_viz.py` — Spatial structure (4×4 grid):**

| Row | Panels                                                                            |
| --- | --------------------------------------------------------------------------------- |
| 1   | N heatmaps at 4 time points                                                       |
| 2   | S (richness) heatmaps at 4 time points                                            |
| 3   | Time series: Total N · α diversity · γ diversity · β diversity                    |
| 4   | Final state: N histogram · S histogram · N–S scatter · Spatial heterogeneity (CV) |

**`spatial_viz2.py` — Species analysis (3 rows × 4 cols):**

| Row | Panels                                                               |
| --- | -------------------------------------------------------------------- |
| 1   | Top-4 species density maps across the spatial grid (final state)     |
| 2   | Patch-level SAD at 4 time points (rank-abundance with IQR)           |
| 3   | Landscape SAD (log-log rank-abundance) · Species occupancy over time |

## Experiments

Three experiments are included in `experiments/`, each self-contained with analysis, tests, and visualisation:

### Harvest Forks (`experiments/harvest_forks/`)

Systematic study of how harvesting individual species at different abundance ranks affects community dynamics. Runs 32 independent burn-in simulations to qESS, then forks each into 32 harvesting scenarios (one per species rank), applying 25% harvest rate after 200 generations.

- **Rust features used**: `--harvest-genome`, `--harvest-rate`, `--harvest-after`, `--species-interval`
- **Output**: 5×3 panel figure (abundance/richness impacts by rank, trajectories, cross-metric analysis) + 4×1 genome heatmap figure
- **Run**: `bash experiments/harvest_forks/run.sh`

### METE Ensemble (`experiments/mete_ensemble/`)

Tests whether the Tangled Nature Model's species abundance distributions conform to Maximum Entropy Theory of Ecology (METE) predictions. Runs 32 replicate simulations and compares empirical SADs against METE log-series predictions using KL divergence.

- **Analysis**: Python preprocessing + R-based METE fitting (requires `meteR` package)
- **Extended**: Quake-focused analysis tracks how METE conformance breaks down during population crashes
- **Output**: Ensemble METE figure + quake analysis figure
- **Run**: `bash experiments/mete_ensemble/run.sh`

### Causal Emergence (`experiments/causal_emergence/`)

Applies Erik Hoel's causal emergence framework to identify scales of maximum causal explanatory power in TNM dynamics. Computes effective information (EI = determinism − degeneracy) from empirical transition probability matrices at 4 hierarchical scales over sliding windows.

| Scale  | Description                            |
| ------ | -------------------------------------- |
| Micro  | Top-10 species abundances (log-binned) |
| Meso-1 | SAD shape (rank-abundance discretised) |
| Meso-2 | (N, S) pair (binned)                   |
| Macro  | Total N only                           |

- **Output**: 4-panel figure (N/S dynamics, EI across scales, max-EI scale strip, determinism/degeneracy decomposition)
- **Run**: `bash experiments/causal_emergence/run.sh`

## Source Layout

```
src/
├── main.rs       # CLI entry point, mode dispatch
├── config.rs     # Parameter struct and --flag parsing
├── model.rs      # Core TNM: JEngine, Simulation, QessDetector
├── spatial.rs    # SpatialGrid: 2D grid of patches with migration
├── state.rs      # SimState / SpatialState serialization
└── output.rs     # JSONL snapshot formatting

experiments/
├── harvest_forks/     # Species harvesting impact study
├── mete_ensemble/     # METE conformance analysis
└── causal_emergence/  # Multi-scale causal information analysis
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

The J matrix is a dense `2^L × 2^L` matrix where each off-diagonal pair `J[i][k]` and `J[k][i]` is sampled independently from `U(-1,1)` with probability Θ (otherwise zero). This matches the reference implementation exactly and gives O(1) lookup with ~8 MB memory for L=10.

## Performance

Single-patch mode is inherently sequential (each micro-step depends on the previous RNG/population state), but the hot path is heavily optimized:

- **Zero-allocation h_cache updates**: `add_individual()`, `remove_individual()`, and `update_h_cache_for_delta()` iterate h_cache in-place with direct `j_dense` slice access, eliminating millions of per-call `Vec` allocations.
- **Batched reproduction**: The `reproduce()` method merges the 3-operation sequence (add child1, add child2, remove parent) into a single O(S) h_cache pass.
- **Spatial parallelism**: Patch stepping uses `rayon::par_iter_mut()` to utilize all available CPU cores.

| Benchmark (5000 non-spatial gens) | Time      |
| --------------------------------- | --------- |
| Before optimization               | 9.9s      |
| After optimization                | 4.6s      |
| **Speedup**                       | **2.15×** |

For multi-core speedup on single-patch workloads, run independent replicates in parallel (e.g., via `ensemble_run.py`).

## Citation

Jensen, H. J. (2018). *The Tangled Nature Model*. European Journal of Physics, 39(6), 064002.

## Reference Implementation

The `reference/` directory contains the original Rust port by Clem von Stengel (2020), used to validate output equivalence.
