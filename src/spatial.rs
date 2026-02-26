// ---------------------------------------------------------------------------
// Spatial TNM: 2D grid of coupled TNM patches with migration.
// ---------------------------------------------------------------------------

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::time::Instant;

use rand::Rng;
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

        // Create one base simulation to generate the universal JEngine & initial species pool
        let base_sim = Simulation::new(base_config.clone());

        for i in 0..n_patches {
            let mut patch_sim = base_sim.clone();
            // Each patch gets its own RNG sequence for stochastic events (death, reproduction, mutation)
            patch_sim.rng = ChaCha12Rng::seed_from_u64(base_config.seed + i as u64 * 97);
            // Also give each patch a unique config.seed to match its RNG
            patch_sim.config.seed = base_config.seed + i as u64 * 97;
            patches.push(patch_sim);
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
        }
    }

    /// Reconstruct a SpatialGrid from a saved SpatialState.
    pub fn from_state(state: SpatialState, config: &Config) -> Self {
        let j_engine = JEngine::from_matrices(state.l, state.j_locus_matrices);
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
            j_locus_matrices: (*self.patches[0].j_engine.locus_matrices).clone(),
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

    /// Collect all migration events (binomial sampling from each species per patch).
    fn collect_migrations(&mut self) -> Vec<Migration> {
        if self.p_move <= 0.0 {
            return Vec::new();
        }

        let mut migrations = Vec::new();
        let n_patches = self.patches.len();

        for idx in 0..n_patches {
            let nbrs = self.neighbours(idx);
            // Iterate over species in this patch
            let species_snapshot: Vec<(u64, u64)> = self.patches[idx]
                .species
                .iter()
                .map(|(&g, &c)| (g, c))
                .collect();

            for (genome, count) in species_snapshot {
                if count == 0 {
                    continue;
                }
                // Binomial: how many of `count` individuals migrate?
                let mut n_mig: u64 = 0;
                for _ in 0..count {
                    if self.migration_rng.gen::<f64>() < self.p_move {
                        n_mig += 1;
                    }
                }
                if n_mig == 0 {
                    continue;
                }
                // Distribute migrants to random neighbours
                for _ in 0..n_mig {
                    let dest = nbrs[self.migration_rng.gen_range(0..4)];
                    migrations.push(Migration {
                        from: idx,
                        to: dest,
                        genome,
                        count: 1,
                    });
                }
            }
        }

        migrations
    }

    /// Apply collected migration events.
    fn apply_migrations(&mut self, migrations: &[Migration]) {
        for m in migrations {
            // Remove from source
            if let Some(count) = self.patches[m.from].species.get(&m.genome) {
                if *count > 0 {
                    self.patches[m.from].remove_individual(m.genome);
                    // Add to destination
                    self.patches[m.to].add_individual(m.genome);
                }
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

            // Phase 2.5: Stress release check
            if self.stress_applied && gens_elapsed == self.stress_duration {
                eprintln!(
                    "  STRESS RELEASED at gen {} (after {} gens of stress). Restoring R={:.1}",
                    gen, self.stress_duration, self.original_r
                );
                for &idx in &self.stress_cells {
                    self.patches[idx].config.r = self.original_r;
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

        // Run Python visualization
        eprintln!("  Generating visualizations with spatial_viz.py...");
        let status = Command::new("python3")
            .arg("spatial_viz.py")
            .arg(&self.out_file)
            .status();
        
        match status {
            Ok(s) if s.success() => eprintln!("  Visualization complete!"),
            Ok(_) | Err(_) => eprintln!("  Warning: Failed to run spatial_viz.py"),
        }

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

        // Per-patch summary
        let patch_data: Vec<serde_json::Value> = self
            .patches
            .iter()
            .enumerate()
            .map(|(i, p)| {
                serde_json::json!({
                    "x": i % self.grid_w,
                    "y": i / self.grid_w,
                    "n": p.n,
                    "s": p.species.len(),
                })
            })
            .collect();

        let snapshot = serde_json::json!({
            "type": "spatial_snapshot",
            "gen": gen,
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
