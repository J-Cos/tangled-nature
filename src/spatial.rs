// ---------------------------------------------------------------------------
// Spatial TNM: 2D grid of coupled TNM patches with migration.
// ---------------------------------------------------------------------------

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::time::Instant;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::config::Config;
use crate::model::{JEngine, QessDetector, Simulation};
use crate::state::{PatchState, SpatialState};

/// A 2D grid of TNM patches, coupled by nearest-neighbour migration.
pub struct SpatialGrid {
    pub patches: Vec<Simulation>,
    pub grid_w: usize,
    pub grid_h: usize,
    pub p_move: f64,
    pub max_gen: usize,
    pub output_interval: usize,
    pub out_file: String,
    pub state_out: Option<String>,
    pub generation: usize,
    migration_rng: ChaCha12Rng,
    qess_n: QessDetector,
    qess_gamma: QessDetector,
    // Press stressor
    stress_r: f64,
    stress_duration: usize,
    stress_cells: Vec<usize>,     // indices of stressed patches
    original_r: f64,              // R before stress was applied
    stress_applied: bool,
    // Ramp stressor
    stress_ramp: usize,           // interval (gens) between adding each new stressed cell
    stress_ramp_before: usize,     // gens of baseline before ramp starts
    stress_ramp_order: Vec<usize>, // shuffled order of cells to stress
    stress_ramp_count: usize,      // how many cells have been stressed so far
    no_viz: bool,
}

#[derive(Clone)]
struct Migration {
    from: usize,
    to: usize,
    genome: u64,
    count: u64,
}

impl SpatialGrid {
    /// Create a grid_w × grid_h grid of TNM patches.
    /// Each patch gets a unique RNG seed derived from base_seed.
    pub fn new(
        grid_w: usize,
        grid_h: usize,
        p_move: f64,
        base_config: &Config,
    ) -> Self {
        let n_patches = grid_w * grid_h;
        let mut patches = Vec::with_capacity(n_patches);

        // Create one base simulation to generate the universal JEngine
        let base_sim = Simulation::new(base_config.clone());

        for i in 0..n_patches {
            let patch_seed = base_config.seed + i as u64 * 97;
            if base_config.independent_init {
                // Each patch gets its own random initial species pool
                let mut patch_config = base_config.clone();
                patch_config.seed = patch_seed;
                let mut patch_sim = Simulation::new(patch_config);
                // Share the same JEngine (interaction matrix) across all patches
                patch_sim.j_engine = base_sim.j_engine.clone();
                patch_sim.rebuild_h_cache();
                patches.push(patch_sim);
            } else {
                // All patches start with identical species (cloned from base)
                let mut patch_sim = base_sim.clone();
                patch_sim.rng = ChaCha12Rng::seed_from_u64(patch_seed);
                patch_sim.config.seed = patch_seed;
                patches.push(patch_sim);
            }
        }

        let migration_rng = ChaCha12Rng::seed_from_u64(base_config.seed + 999_999);

        let stress_cells = base_config.stress_cells.clone()
            .unwrap_or_else(|| Self::compute_central_cells(grid_w, grid_h));

        SpatialGrid {
            patches,
            grid_w,
            grid_h,
            p_move,
            max_gen: base_config.max_gen as usize,
            output_interval: base_config.output_interval as usize,
            out_file: base_config.out_file.clone(),
            state_out: base_config.state_out.clone(),
            generation: 0,
            migration_rng,
            qess_n: QessDetector::new(base_config.qess_window, base_config.qess_threshold),
            // γ-diversity is inherently noisier; use 2× threshold
            qess_gamma: QessDetector::new(base_config.qess_window, base_config.qess_threshold * 2.0),
            stress_r: base_config.stress_r,
            stress_duration: base_config.stress_duration,
            stress_cells,
            original_r: base_config.r,
            stress_applied: false,
            stress_ramp: base_config.stress_ramp,
            stress_ramp_before: base_config.stress_ramp_before,
            stress_ramp_order: Self::compute_contagion_order(
                grid_w, grid_h, base_config.stress_sigma,
                &mut ChaCha12Rng::seed_from_u64(base_config.seed + 777),
            ),
            stress_ramp_count: 0,
            no_viz: base_config.no_viz,
        }
    }

    /// Reconstruct a SpatialGrid from a saved SpatialState.
    pub fn from_state(state: SpatialState, config: &Config) -> Self {
        let j_engine = JEngine::from_dense(state.l, state.j_dense);
        let mut patches = Vec::with_capacity(state.patch_states.len());
        for ps in state.patch_states {
            let mut patch_config = config.clone();
            patch_config.seed = ps.seed;
            let sim = Simulation::from_patch_state(patch_config, j_engine.clone(), ps.species, ps.rng);
            patches.push(sim);
        }
        let stress_cells = config.stress_cells.clone()
            .unwrap_or_else(|| Self::compute_central_cells(state.grid_w, state.grid_h));

        SpatialGrid {
            patches,
            grid_w: state.grid_w,
            grid_h: state.grid_h,
            p_move: state.p_move,
            max_gen: config.max_gen as usize,
            output_interval: config.output_interval as usize,
            out_file: config.out_file.clone(),
            state_out: config.state_out.clone(),
            generation: state.generation,
            migration_rng: state.migration_rng,
            qess_n: QessDetector::new(config.qess_window, config.qess_threshold),
            qess_gamma: QessDetector::new(config.qess_window, config.qess_threshold * 2.0),
            stress_r: config.stress_r,
            stress_duration: config.stress_duration,
            stress_cells,
            original_r: config.r,
            stress_applied: false,
            stress_ramp: config.stress_ramp,
            stress_ramp_before: config.stress_ramp_before,
            stress_ramp_order: Self::compute_contagion_order(
                state.grid_w, state.grid_h, config.stress_sigma,
                &mut ChaCha12Rng::seed_from_u64(config.seed + 777),
            ),
            stress_ramp_count: 0,
            no_viz: config.no_viz,
        }
    }

    /// Compute the indices of the central 4 cells in a W×H grid.
    fn compute_central_cells(w: usize, h: usize) -> Vec<usize> {
        let cx = w / 2;
        let cy = h / 2;
        let mut cells = Vec::new();
        for dy in [cy - 1, cy] {
            for dx in [cx - 1, cx] {
                cells.push(dy * w + dx);
            }
        }
        cells
    }

    /// Pre-compute stress order via distance-weighted contagion.
    ///
    /// Picks a random seed cell, then iteratively selects the next cell
    /// with probability ∝ exp(-d_min² / 2σ²), where d_min is the Euclidean
    /// distance to the nearest already-stressed cell.
    /// σ controls spread: small σ = tight clusters, large σ = more diffuse.
    /// At σ=0, falls back to uniform random selection.
    fn compute_contagion_order(w: usize, h: usize, sigma: f64, rng: &mut ChaCha12Rng) -> Vec<usize> {
        let n = w * h;
        if n == 0 { return vec![]; }

        // σ=0: fall back to shuffled random order (no spatial autocorrelation)
        if sigma <= 0.0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(rng);
            return order;
        }

        let two_sigma_sq = 2.0 * sigma * sigma;

        // Cell coordinates
        let coords: Vec<(f64, f64)> = (0..n)
            .map(|i| ((i % w) as f64, (i / w) as f64))
            .collect();

        let mut order = Vec::with_capacity(n);
        let mut stressed = vec![false; n];
        let mut min_dist_sq = vec![f64::INFINITY; n]; // distance² to nearest stressed cell

        // Pick random seed
        let seed = rng.gen_range(0..n);
        order.push(seed);
        stressed[seed] = true;

        // Update min_dist_sq from seed
        let (sx, sy) = coords[seed];
        for i in 0..n {
            if !stressed[i] {
                let dx = coords[i].0 - sx;
                let dy = coords[i].1 - sy;
                let dsq = dx * dx + dy * dy;
                if dsq < min_dist_sq[i] {
                    min_dist_sq[i] = dsq;
                }
            }
        }

        // Iteratively select remaining cells
        for _ in 1..n {
            // Compute weights for unstressed cells
            let mut weights: Vec<(usize, f64)> = Vec::new();
            let mut total_weight = 0.0;
            for i in 0..n {
                if !stressed[i] {
                    let w = (-min_dist_sq[i] / two_sigma_sq).exp();
                    weights.push((i, w));
                    total_weight += w;
                }
            }

            // Weighted random selection
            let mut r = rng.gen::<f64>() * total_weight;
            let mut chosen = weights[0].0;
            for &(idx, w) in &weights {
                r -= w;
                if r <= 0.0 {
                    chosen = idx;
                    break;
                }
            }

            order.push(chosen);
            stressed[chosen] = true;

            // Update min_dist_sq from newly stressed cell
            let (cx, cy) = coords[chosen];
            for i in 0..n {
                if !stressed[i] {
                    let dx = coords[i].0 - cx;
                    let dy = coords[i].1 - cy;
                    let dsq = dx * dx + dy * dy;
                    if dsq < min_dist_sq[i] {
                        min_dist_sq[i] = dsq;
                    }
                }
            }
        }

        order
    }

    /// Export current grid state for persistence.
    pub fn to_spatial_state(&self) -> SpatialState {
        let patch_states: Vec<PatchState> = self.patches.iter().map(|p| {
            PatchState {
                species: p.species.clone(),
                rng: p.rng.clone(),
                seed: p.config.seed,
            }
        }).collect();
        SpatialState {
            generation: self.generation,
            grid_w: self.grid_w,
            grid_h: self.grid_h,
            p_move: self.p_move,
            l: self.patches[0].config.l,
            j_dense: (*self.patches[0].j_engine.j_dense).clone(),
            patch_states,
            migration_rng: self.migration_rng.clone(),
        }
    }

    /// Get 4-connected neighbours (periodic boundary) for patch at index.
    fn neighbours(&self, idx: usize) -> [usize; 4] {
        let x = idx % self.grid_w;
        let y = idx / self.grid_w;
        let w = self.grid_w;
        let h = self.grid_h;
        [
            ((y + h - 1) % h) * w + x, // north
            ((y + 1) % h) * w + x,       // south
            y * w + (x + w - 1) % w,     // west
            y * w + (x + 1) % w,         // east
        ]
    }

    /// Collect all migration events using geometric skip sampling.
    /// Instead of Bernoulli(p_move) for each individual, uses geometric
    /// distribution to skip directly to the next migrant. O(n_migrants)
    /// instead of O(n_total).
    fn collect_migrations(&mut self) -> Vec<Migration> {
        if self.p_move <= 0.0 {
            return Vec::new();
        }

        let mut migrations = Vec::new();
        let n_patches = self.patches.len();
        let log_1mp = (1.0 - self.p_move).ln();

        for idx in 0..n_patches {
            let nbrs = self.neighbours(idx);
            let species_snapshot: Vec<(u64, u64)> = self.patches[idx]
                .species
                .iter()
                .map(|(&g, &c)| (g, c))
                .collect();

            for (genome, count) in species_snapshot {
                if count == 0 { continue; }

                // Geometric skip: how many of `count` migrate?
                // For small p_move, this is much faster than count Bernoulli draws
                let mut n_mig: u64 = 0;
                if count <= 20 {
                    // For small counts, direct Bernoulli is fine
                    for _ in 0..count {
                        if self.migration_rng.gen::<f64>() < self.p_move {
                            n_mig += 1;
                        }
                    }
                } else {
                    // Geometric skip sampling
                    let mut remaining = count as i64;
                    loop {
                        let u: f64 = self.migration_rng.gen();
                        let skip = (u.ln() / log_1mp).floor() as i64;
                        remaining -= skip + 1;
                        if remaining < 0 { break; }
                        n_mig += 1;
                    }
                }

                if n_mig == 0 { continue; }

                // Distribute migrants to random neighbours (batched by dest)
                let mut dest_counts = [0u64; 4];
                for _ in 0..n_mig {
                    dest_counts[self.migration_rng.gen_range(0..4)] += 1;
                }
                for (d, &c) in dest_counts.iter().enumerate() {
                    if c > 0 {
                        migrations.push(Migration {
                            from: idx,
                            to: nbrs[d],
                            genome,
                            count: c,
                        });
                    }
                }
            }
        }

        migrations
    }

    /// Apply collected migration events (batched by count).
    fn apply_migrations(&mut self, migrations: &[Migration]) {
        for m in migrations {
            let available = self.patches[m.from].species.get(&m.genome).copied().unwrap_or(0);
            let to_move = std::cmp::min(m.count, available);
            if to_move == 0 { continue; }

            for _ in 0..to_move {
                self.patches[m.from].remove_individual(m.genome);
                self.patches[m.to].add_individual(m.genome);
            }
        }
    }

    /// Run the full spatial simulation. Returns true if qESS reached.
    pub fn run(&mut self) -> bool {
        let start = Instant::now();
        let n_patches = self.patches.len();
        let start_gen = self.generation;
        let target_gen = self.generation + self.max_gen;

        eprintln!(
            "Spatial TNM | {}×{} grid ({} patches) | p_move={} | gen {}->{}",
            self.grid_w, self.grid_h, n_patches, self.p_move, start_gen, target_gen
        );

        // Report initial state
        let (total_n, total_s, gamma) = self.aggregate_stats();
        eprintln!(
            "  Initial: total_N={}, mean_N={:.0}, mean_S={:.1}, γ_diversity={}",
            total_n,
            total_n as f64 / n_patches as f64,
            total_s as f64 / n_patches as f64,
            gamma
        );

        // Report ramp timeline if active
        if self.stress_ramp > 0 && self.stress_r > 0.0 {
            let ramp_duration = n_patches * self.stress_ramp;
            let ramp_start = start_gen + self.stress_ramp_before;
            let ramp_end = ramp_start + ramp_duration;
            eprintln!(
                "  RAMP PROTOCOL: baseline gens {}-{} | ramp {}-{} (R {:.1} → {:.2}, {} cells over {} gens) | hold {}-{}",
                start_gen, ramp_start,
                ramp_start, ramp_end,
                self.original_r, self.stress_r, n_patches, ramp_duration,
                ramp_end, target_gen
            );
        }

        // Prepare output file
        let mut out = BufWriter::new(File::create(&self.out_file).expect("Failed to create out_file"));

        // Emit initial snapshot
        self.emit_snapshot(self.generation, &mut out);

        let mut qess_reached = false;

        // Apply stress at the start if configured
        if self.stress_duration > 0 && self.stress_r > 0.0 && !self.stress_applied {
            eprintln!(
                "  STRESS: applying R={:.1} to {} central cells for {} gens (cells: {:?})",
                self.stress_r, self.stress_cells.len(), self.stress_duration, self.stress_cells
            );
            for &idx in &self.stress_cells {
                self.patches[idx].config.r = self.stress_r;
            }
            self.stress_applied = true;
        }

        while self.generation < target_gen {
            self.generation += 1;
            let gen = self.generation;
            let gens_elapsed = gen - start_gen;

            // Phase 1: Parallel patch stepping (rayon)
            self.patches.par_iter_mut().for_each(|patch| {
                patch.step_one_generation();
            });

            // Phase 2: Migration (sequential — requires cross-patch access)
            let migrations = self.collect_migrations();
            self.apply_migrations(&migrations);

            // Phase 2.5: Stress release check (press mode)
            if self.stress_applied && self.stress_duration > 0 && gens_elapsed == self.stress_duration {
                eprintln!(
                    "  STRESS RELEASED at gen {} (after {} gens of stress). Restoring R={:.1}",
                    gen, self.stress_duration, self.original_r
                );
                for &idx in &self.stress_cells {
                    self.patches[idx].config.r = self.original_r;
                }
            }

            // Phase 2.6: Ramp stress — stress 1% of cells every stress_ramp gens
            //           Waits stress_ramp_before gens before starting
            if self.stress_ramp > 0 && self.stress_r > 0.0
                && gens_elapsed >= self.stress_ramp_before
                && (gens_elapsed - self.stress_ramp_before) % self.stress_ramp == 0
                && self.stress_ramp_count < self.stress_ramp_order.len()
            {
                let n_total = self.stress_ramp_order.len();
                let cells_per_step = std::cmp::max(1, n_total / 100);
                let mut last_cell = 0;
                for _ in 0..cells_per_step {
                    if self.stress_ramp_count >= n_total { break; }
                    let cell_idx = self.stress_ramp_order[self.stress_ramp_count];
                    self.patches[cell_idx].config.r = self.stress_r;
                    self.stress_ramp_count += 1;
                    last_cell = cell_idx;
                }
                let pct = (self.stress_ramp_count as f64 / n_total as f64 * 100.0) as usize;
                if pct % 10 == 0 || self.stress_ramp_count == n_total {
                    eprintln!(
                        "  RAMP: {}/{} cells stressed ({}%) at gen {} (latest: cell {})",
                        self.stress_ramp_count, n_total, pct, gen, last_cell
                    );
                }
            }

            // Phase 3: qESS detection on total N AND γ-diversity
            let total_n: u64 = self.patches.iter().map(|p| p.n).sum();
            let mut all_genomes = std::collections::HashSet::new();
            for p in &self.patches {
                for &g in p.species.keys() {
                    all_genomes.insert(g);
                }
            }
            let gamma = all_genomes.len() as u64;

            let cv_n = self.qess_n.push(total_n);
            let cv_g = self.qess_gamma.push(gamma);

            if let (Some(cn), Some(cg)) = (cv_n, cv_g) {
                if self.qess_n.is_qess(cn) && self.qess_gamma.is_qess(cg) && gens_elapsed > 10000 {
                    eprintln!("  qESS detected at gen {} (CV_N={:.4}, CV_γ={:.4})", gen, cn, cg);
                    qess_reached = true;
                    self.emit_snapshot(gen, &mut out);
                    break;
                }
            }

            // Phase 4: Output
            if self.output_interval > 0 && gens_elapsed % self.output_interval == 0 {
                self.emit_snapshot(gen, &mut out);
            }

            // Progress logging every 500 gen
            if gens_elapsed % 500 == 0 && gens_elapsed > 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let (total_n, total_s, gamma) = self.aggregate_stats();
                let gen_per_sec = gens_elapsed as f64 / elapsed;
                let remaining_gens = target_gen - gen;
                let remaining_s = remaining_gens as f64 / gen_per_sec;
                eprintln!(
                    "  gen {:6} | N={:6} S_mean={:.1} γ={:4} | {:.1} gen/s | ETA {:.0}s",
                    gen,
                    total_n,
                    total_s as f64 / n_patches as f64,
                    gamma,
                    gen_per_sec,
                    remaining_s
                );
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let gens_run = self.generation - start_gen;
        if qess_reached {
            eprintln!("  qESS reached after {} generations in {:.1}s", gens_run, elapsed);
        } else {
            eprintln!("  Completed {} generations in {:.1}s (no qESS)", gens_run, elapsed);
        }

        // Emit final summary
        self.emit_final(&mut out);
        out.flush().expect("Failed to flush output");

        // Save state if requested
        if let Some(ref path) = self.state_out {
            let state = self.to_spatial_state();
            match state.save(path) {
                Ok(_) => eprintln!("  State saved to {}", path),
                Err(e) => eprintln!("  Error saving state: {}", e),
            }
        }

        // Run Python visualization (skip if --no-viz)
        if !self.no_viz {
        eprintln!("  Generating visualizations...");
        let out = &self.out_file;
        for script in &["spatial_viz.py", "spatial_viz2.py"] {
            let status = Command::new("python3")
                .arg(script)
                .arg(out)
                .status();
            match status {
                Ok(s) if s.success() => {},
                Ok(_) | Err(_) => eprintln!("  Warning: Failed to run {}", script),
            }
        }
        eprintln!("  Visualization complete!");
        } // end if !self.no_viz

        qess_reached
    }

    /// Aggregate stats across all patches.
    fn aggregate_stats(&self) -> (u64, u64, usize) {
        let total_n: u64 = self.patches.iter().map(|p| p.n).sum();
        let total_s: u64 = self.patches.iter().map(|p| p.species.len() as u64).sum();
        // Gamma diversity: unique genomes across all patches
        let mut all_genomes = std::collections::HashSet::new();
        for p in &self.patches {
            for &g in p.species.keys() {
                all_genomes.insert(g);
            }
        }
        (total_n, total_s, all_genomes.len())
    }

    /// Emit a JSON snapshot of the grid state.
    fn emit_snapshot(&self, gen: usize, out: &mut impl Write) {
        let n_patches = self.patches.len();
        let (total_n, total_s, gamma) = self.aggregate_stats();

        // Per-patch summary (includes species composition for SAD/heatmap viz)
        let patch_data: Vec<serde_json::Value> = self
            .patches
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let species: std::collections::HashMap<String, u64> = p
                    .species
                    .iter()
                    .map(|(&g, &c)| (g.to_string(), c))
                    .collect();
                serde_json::json!({
                    "x": i % self.grid_w,
                    "y": i / self.grid_w,
                    "n": p.n,
                    "s": p.species.len(),
                    "species": species,
                })
            })
            .collect();

        let snapshot = serde_json::json!({
            "type": "spatial_snapshot",
            "gen": gen,
            "l": self.patches[0].config.l,
            "total_n": total_n,
            "mean_n": total_n as f64 / n_patches as f64,
            "total_s": total_s,
            "mean_s": total_s as f64 / n_patches as f64,
            "gamma_s": gamma,
            "patches": patch_data,
        });

        writeln!(out, "{}", snapshot).expect("Failed to write snapshot");
    }

    /// Emit final summary.
    fn emit_final(&self, out: &mut impl Write) {
        let (total_n, total_s, gamma) = self.aggregate_stats();
        let n_patches = self.patches.len();

        // Population distribution
        let ns: Vec<u64> = self.patches.iter().map(|p| p.n).collect();
        let ss: Vec<usize> = self.patches.iter().map(|p| p.species.len()).collect();

        let snapshot = serde_json::json!({
            "type": "spatial_final",
            "total_n": total_n,
            "mean_n": total_n as f64 / n_patches as f64,
            "total_s": total_s,
            "mean_s": total_s as f64 / n_patches as f64,
            "gamma_s": gamma,
            "patch_n": ns,
            "patch_s": ss,
        });

        writeln!(out, "{}", snapshot).expect("Failed to write final");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn make_grid(grid_size: usize, r: f64) -> SpatialGrid {
        let mut c = Config::default();
        c.l = 5; // small genome for fast tests
        c.r = r;
        c.grid_size = grid_size;
        c.p_move = 0.01;
        c.seed = 42;
        c.max_gen = 100;
        c.output_interval = 0;
        c.qess_window = 5000;
        c.qess_threshold = 0.0; // disable qESS
        c.out_file = "/tmp/test_spatial.jsonl".to_string();
        c.state_out = None;
        SpatialGrid::new(grid_size, grid_size, 0.01, &c)
    }

    #[test]
    fn test_contagion_order_spatial_autocorrelation() {
        // Contagion order with small sigma should produce spatially clustered sequences
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let order = SpatialGrid::compute_contagion_order(10, 10, 2.0, &mut rng);

        assert_eq!(order.len(), 100);
        // Check all cells present exactly once
        let mut seen = vec![false; 100];
        for &idx in &order {
            assert!(!seen[idx], "Cell {} appears twice", idx);
            seen[idx] = true;
        }

        // Spatial autocorrelation check: average distance between consecutive
        // cells should be small (< grid diagonal / 2)
        let mut total_dist = 0.0;
        for i in 1..order.len() {
            let (x1, y1) = (order[i-1] % 10, order[i-1] / 10);
            let (x2, y2) = (order[i] % 10, order[i] / 10);
            let dx = (x1 as f64 - x2 as f64).abs();
            let dy = (y1 as f64 - y2 as f64).abs();
            total_dist += (dx * dx + dy * dy).sqrt();
        }
        let avg_dist = total_dist / 99.0;
        // With sigma=2, consecutive cells should be closer than uniform random
        // Random avg on 10x10 ≈ 6.6; contagion should be noticeably less
        assert!(avg_dist < 5.5, "Avg consecutive distance {} too large for sigma=2", avg_dist);
    }

    #[test]
    fn test_contagion_order_sigma_zero_random() {
        // sigma=0 should fall back to uniform random (no spatial autocorrelation)
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let order = SpatialGrid::compute_contagion_order(10, 10, 0.0, &mut rng);
        assert_eq!(order.len(), 100);

        // All cells present
        let mut seen = vec![false; 100];
        for &idx in &order {
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_contagion_order_small_grid() {
        // Should work on 2x2 grid
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let order = SpatialGrid::compute_contagion_order(2, 2, 2.0, &mut rng);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_ramp_scales_with_grid_size() {
        // On a 20x20 grid (400 cells), 1% = 4 cells per step
        let mut grid = make_grid(20, 5.0);
        grid.stress_ramp = 100;
        grid.stress_r = 1.0;

        let n_total = grid.stress_ramp_order.len();
        let cells_per_step = std::cmp::max(1, n_total / 100);
        assert_eq!(cells_per_step, 4, "20x20 grid should stress 4 cells per step");
    }

    #[test]
    fn test_ramp_scales_10x10() {
        // On a 10x10 grid (100 cells), 1% = 1 cell per step
        let n_total = 100;
        let cells_per_step = std::cmp::max(1, n_total / 100);
        assert_eq!(cells_per_step, 1);
    }

    #[test]
    fn test_ramp_scales_100x100() {
        // On a 100x100 grid (10000 cells), 1% = 100 cells per step
        let n_total = 10000;
        let cells_per_step = std::cmp::max(1, n_total / 100);
        assert_eq!(cells_per_step, 100);
    }

    #[test]
    fn test_population_conservation_during_migration() {
        // Migration should conserve total population
        let mut grid = make_grid(5, 5.0);

        // Run a few generations to build up population
        for _ in 0..50 {
            grid.generation += 1;
            grid.patches.par_iter_mut().for_each(|p| { p.step_one_generation(); });

            let n_before: u64 = grid.patches.iter().map(|p| p.n).sum();
            let migrations = grid.collect_migrations();
            grid.apply_migrations(&migrations);
            let n_after: u64 = grid.patches.iter().map(|p| p.n).sum();

            assert_eq!(n_before, n_after,
                "Migration changed total N from {} to {} at gen {}",
                n_before, n_after, grid.generation);
        }
    }

    #[test]
    fn test_neighbours_toroidal() {
        let grid = make_grid(5, 5.0);

        // Corner cell (0,0) = index 0
        // Order: [north, south, west, east]
        let n = grid.neighbours(0);
        assert_eq!(n[0], 20);  // north wraps to row 4
        assert_eq!(n[1], 5);   // south = row 1
        assert_eq!(n[2], 4);   // west wraps to col 4
        assert_eq!(n[3], 1);   // east = col 1

        // Center cell (2,2) = index 12
        let n = grid.neighbours(12);
        assert_eq!(n[0], 7);   // north
        assert_eq!(n[1], 17);  // south
        assert_eq!(n[2], 11);  // west
        assert_eq!(n[3], 13);  // east
    }

    #[test]
    fn test_contagion_deterministic() {
        // Same seed should produce same order
        let mut rng1 = ChaCha12Rng::seed_from_u64(123);
        let mut rng2 = ChaCha12Rng::seed_from_u64(123);
        let order1 = SpatialGrid::compute_contagion_order(10, 10, 2.0, &mut rng1);
        let order2 = SpatialGrid::compute_contagion_order(10, 10, 2.0, &mut rng2);
        assert_eq!(order1, order2);
    }

    #[test]
    fn test_shared_init_identical_patches() {
        // Without independent_init, all patches should start with identical species
        let mut c = Config::default();
        c.l = 5;
        c.r = 5.0;
        c.grid_size = 4;
        c.p_move = 0.01;
        c.seed = 42;
        c.independent_init = false;
        c.max_gen = 100;
        c.output_interval = 0;
        c.qess_window = 5000;
        c.qess_threshold = 0.0;
        c.out_file = "/tmp/test_shared_init.jsonl".to_string();
        c.state_out = None;

        let grid = SpatialGrid::new(4, 4, 0.01, &c);

        // All patches should have the same species set (genomes)
        let species_0: std::collections::BTreeSet<u64> =
            grid.patches[0].species.keys().cloned().collect();
        for (i, patch) in grid.patches.iter().enumerate() {
            let species_i: std::collections::BTreeSet<u64> =
                patch.species.keys().cloned().collect();
            assert_eq!(species_0, species_i,
                "Patch {} has different species than patch 0 with shared init", i);
        }
    }

    #[test]
    fn test_independent_init_diverse_patches() {
        // With independent_init, patches should start with different species
        let mut c = Config::default();
        c.l = 10;  // larger genome space for diversity
        c.r = 5.0;
        c.grid_size = 4;
        c.p_move = 0.01;
        c.seed = 42;
        c.independent_init = true;
        c.max_gen = 100;
        c.output_interval = 0;
        c.qess_window = 5000;
        c.qess_threshold = 0.0;
        c.out_file = "/tmp/test_indep_init.jsonl".to_string();
        c.state_out = None;

        let grid = SpatialGrid::new(4, 4, 0.01, &c);

        // At least some patches should have different species sets
        let species_0: std::collections::BTreeSet<u64> =
            grid.patches[0].species.keys().cloned().collect();
        let mut any_different = false;
        for patch in &grid.patches[1..] {
            let species_i: std::collections::BTreeSet<u64> =
                patch.species.keys().cloned().collect();
            if species_0 != species_i {
                any_different = true;
                break;
            }
        }
        assert!(any_different,
            "With independent_init, at least some patches should have different species");
    }

    #[test]
    fn test_independent_init_shared_j_engine() {
        // Both init modes should share the same JEngine across all patches
        let mut c = Config::default();
        c.l = 5;
        c.r = 5.0;
        c.grid_size = 3;
        c.p_move = 0.01;
        c.seed = 42;
        c.max_gen = 100;
        c.output_interval = 0;
        c.qess_window = 5000;
        c.qess_threshold = 0.0;
        c.out_file = "/tmp/test_j_engine.jsonl".to_string();
        c.state_out = None;

        // Test with independent_init
        c.independent_init = true;
        let grid_indep = SpatialGrid::new(3, 3, 0.01, &c);

        // All patches should have the same J values for test genomes
        let test_genomes: Vec<u64> = vec![0, 1, 5, 15, 31];
        for g1 in &test_genomes {
            for g2 in &test_genomes {
                let j_ref = grid_indep.patches[0].j_engine.get_j(*g1, *g2);
                for (i, patch) in grid_indep.patches.iter().enumerate() {
                    let j_i = patch.j_engine.get_j(*g1, *g2);
                    assert!((j_ref - j_i).abs() < 1e-15,
                        "Patch {} JEngine differs for genomes ({}, {}): {} vs {}",
                        i, g1, g2, j_ref, j_i);
                }
            }
        }

        // Test with shared init — J should also be uniform
        c.independent_init = false;
        let grid_shared = SpatialGrid::new(3, 3, 0.01, &c);
        for g1 in &test_genomes {
            for g2 in &test_genomes {
                let j_ref = grid_shared.patches[0].j_engine.get_j(*g1, *g2);
                let j_indep = grid_indep.patches[0].j_engine.get_j(*g1, *g2);
                assert!((j_ref - j_indep).abs() < 1e-15,
                    "Shared vs independent JEngine differs for ({}, {})", g1, g2);
            }
        }
    }
}
