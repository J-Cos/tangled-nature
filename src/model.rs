use std::collections::{BTreeMap, HashMap, VecDeque};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::config::Config;
use crate::output;
use crate::state::SimState;

// ---------------------------------------------------------------------------
// JEngine: dense interaction matrix (matching Jensen et al. 2025 / reference)
// ---------------------------------------------------------------------------

use std::sync::Arc;

/// Dense J matrix: j_dense[a * n_genomes + b] = J(a,b).
/// For L=10, this is 1024×1024 × 8 bytes = 8 MB — trivial.
/// Each J[i][k] and J[k][i] are drawn independently from U(-1,1)
/// with probability Θ (zero otherwise), matching the reference exactly.
#[derive(Clone)]
pub struct JEngine {
    pub j_dense: Arc<Vec<f64>>,
    pub l: u32,
    pub n_genomes: usize,
}

impl JEngine {
    /// Create a new J matrix matching the reference code's `create_j()` exactly.
    /// For each pair (i,k) where i≠k, with probability Θ:
    ///   J[i][k] ~ U(-1,1) and J[k][i] ~ U(-1,1) independently.
    pub fn new(l: u32, theta: f64, rng: &mut impl Rng) -> Self {
        assert!(l <= 20, "L={} too large for dense J matrix (2^L = {})", l, 1u64 << l);
        let n_genomes = 1usize << l;
        let mut dense = vec![0.0f64; n_genomes * n_genomes];

        // Match reference: iterate i in 0..N, k in 0..i (lower triangle)
        for i in 0..n_genomes {
            for k in 0..i {
                if rng.gen::<f64>() < theta {
                    dense[i * n_genomes + k] = rng.gen_range(-1.0..1.0);
                    dense[k * n_genomes + i] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        JEngine {
            j_dense: Arc::new(dense),
            l,
            n_genomes,
        }
    }

    /// Reconstruct from a serialized dense matrix.
    pub fn from_dense(l: u32, dense: Vec<f64>) -> Self {
        let n_genomes = 1usize << l;
        assert_eq!(dense.len(), n_genomes * n_genomes);
        JEngine {
            j_dense: Arc::new(dense),
            l,
            n_genomes,
        }
    }

    /// O(1) J(a,b) lookup.
    #[inline(always)]
    pub fn get_or_compute(&mut self, a: u64, b: u64) -> f64 {
        self.get_j(a, b)
    }

    /// Immutable O(1) J(a,b) lookup.
    #[inline(always)]
    pub fn get_j(&self, a: u64, b: u64) -> f64 {
        unsafe { *self.j_dense.get_unchecked(a as usize * self.n_genomes + b as usize) }
    }

    /// Compute the full J submatrix for currently active species.
    pub fn compute_active_matrix(
        &mut self,
        species: &BTreeMap<u64, u64>,
    ) -> (Vec<u64>, Vec<Vec<f64>>) {
        let genomes: Vec<u64> = species.keys().cloned().collect();
        let s = genomes.len();
        let mut matrix = vec![vec![0.0f64; s]; s];
        for i in 0..s {
            for j in 0..s {
                matrix[i][j] = self.get_j(genomes[i], genomes[j]);
            }
        }
        (genomes, matrix)
    }
}

// ---------------------------------------------------------------------------
// QessDetector: rolling-window CV of N
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct QessDetector {
    window: VecDeque<u64>,
    window_size: usize,
    threshold: f64,
    sum: f64,
    sum_sq: f64,
}

impl QessDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        QessDetector {
            window: VecDeque::with_capacity(window_size + 1),
            window_size,
            threshold,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a new N observation; returns CV if window is full.
    pub fn push(&mut self, n: u64) -> Option<f64> {
        let x = n as f64;
        self.sum += x;
        self.sum_sq += x * x;
        self.window.push_back(n);

        if self.window.len() > self.window_size {
            let old = self.window.pop_front().unwrap() as f64;
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        if self.window.len() >= self.window_size {
            let len = self.window_size as f64;
            let mean = self.sum / len;
            if mean > 0.0 {
                let var = (self.sum_sq / len) - mean * mean;
                let cv = if var > 0.0 { var.sqrt() / mean } else { 0.0 };
                return Some(cv);
            }
        }
        None
    }

    pub fn is_qess(&self, cv: f64) -> bool {
        cv < self.threshold
    }
}

// ---------------------------------------------------------------------------
// Simulation: full TNM engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Simulation {
    pub config: Config,
    pub species: BTreeMap<u64, u64>,
    pub j_engine: JEngine,
    pub rng: ChaCha12Rng,
    pub generation: usize,
    pub n: u64,
    qess: QessDetector,
    /// Cached fitness contribution: h_cache[i] = Σ_j J(i,j) * n_j
    /// Updated incrementally on add/remove.
    h_cache: HashMap<u64, f64>,
    h_dirty: bool,
}

impl Simulation {
    /// Initialize a fresh simulation.
    pub fn new(config: Config) -> Self {
        let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
        let j_engine = JEngine::new(config.l, config.theta, &mut rng);

        let mut species = BTreeMap::new();
        let max_genome = if config.l >= 64 {
            u64::MAX
        } else {
            (1u64 << config.l) - 1
        };
        for _ in 0..config.n_init {
            let genome = rng.gen_range(0..=max_genome);
            *species.entry(genome).or_insert(0) += 1;
        }
        let n = config.n_init;
        let qess = QessDetector::new(config.qess_window, config.qess_threshold);

        let mut sim = Simulation {
            config,
            species,
            j_engine,
            rng,
            generation: 0,
            n,
            qess,
            h_cache: HashMap::new(),
            h_dirty: true,
        };
        sim.rebuild_h_cache();
        sim
    }

    /// Reconstruct simulation from persisted state.
    pub fn from_state(state: SimState, config: Config) -> Self {
        let qess = QessDetector::new(config.qess_window, config.qess_threshold);
        let n: u64 = state.species.values().sum();
        let mut sim = Simulation {
            config,
            species: state.species,
            j_engine: JEngine::from_dense(state.l, state.j_dense),
            rng: state.rng,
            generation: state.generation,
            n,
            qess,
            h_cache: HashMap::new(),
            h_dirty: true,
        };
        sim.rebuild_h_cache();
        sim
    }

    /// Reconstruct a patch simulation from a shared JEngine and patch-specific state.
    pub fn from_patch_state(
        config: Config,
        j_engine: JEngine,
        species: BTreeMap<u64, u64>,
        rng: ChaCha12Rng,
    ) -> Self {
        let n: u64 = species.values().sum();
        let qess = QessDetector::new(config.qess_window, config.qess_threshold);
        let mut sim = Simulation {
            config,
            species,
            j_engine,
            rng,
            generation: 0,
            n,
            qess,
            h_cache: HashMap::new(),
            h_dirty: true,
        };
        sim.rebuild_h_cache();
        sim
    }

    /// Export current state for persistence.
    pub fn to_state(&self) -> SimState {
        SimState {
            generation: self.generation,
            l: self.config.l,
            species: self.species.clone(),
            j_dense: (*self.j_engine.j_dense).clone(),
            rng: self.rng.clone(),
        }
    }

    /// Full rebuild of h_cache from scratch: H[i] = Σ_j J(i,j) * n_j
    pub fn rebuild_h_cache(&mut self) {
        self.h_cache.clear();
        let genomes: Vec<(u64, u64)> = self.species.iter().map(|(&g, &c)| (g, c)).collect();
        for &(gi, _) in &genomes {
            let mut h = 0.0;
            for &(gj, nj) in &genomes {
                h += self.j_engine.get_or_compute(gi, gj) * nj as f64;
            }
            self.h_cache.insert(gi, h);
        }
        self.h_dirty = false;
    }

    /// Update h_cache after +delta individuals of genome `g` were added/removed.
    fn update_h_cache_for_delta(&mut self, g: u64, delta: i64) {
        let is_new = delta > 0 && !self.h_cache.contains_key(&g);
        // If g is a NEW species being added, compute its H[g] from scratch first.
        // This must happen BEFORE the delta loop, which would create a partial
        // entry via or_insert(0.0). The scratch computation uses the current
        // species counts (which already include g's count=1 from add_individual).
        if is_new {
            let mut h = 0.0;
            for (&gj, &nj) in &self.species {
                h += self.j_engine.get_j(g, gj) * nj as f64;
            }
            self.h_cache.insert(g, h);
        }
        // For every existing species i: H[i] += J(i, g) * delta
        // Skip g if it was just computed from scratch (already includes its own count).
        let genomes: Vec<u64> = self.h_cache.keys().cloned().collect();
        for gi in &genomes {
            if is_new && *gi == g {
                continue; // Already correct from scratch computation above
            }
            let j_val = self.j_engine.get_j(*gi, g);
            *self.h_cache.entry(*gi).or_insert(0.0) += j_val * delta as f64;
        }
    }

    /// Sample a random species weighted by population.
    pub fn sample_species_weighted(&mut self) -> u64 {
        let target = self.rng.gen_range(0..self.n);
        let mut cumsum: u64 = 0;
        for (&genome, &count) in &self.species {
            cumsum += count;
            if cumsum > target {
                return genome;
            }
        }
        *self.species.keys().last().unwrap()
    }

    pub fn add_individual(&mut self, genome: u64) {
        let is_new = !self.species.contains_key(&genome);
        *self.species.entry(genome).or_insert(0) += 1;
        self.n += 1;
        if is_new {
            self.update_h_cache_for_delta(genome, 1);
        } else {
            // Pre-compute J values, then update h_cache (avoids borrow conflict)
            let updates: Vec<(u64, f64)> = self.h_cache.keys()
                .map(|&gi| (gi, self.j_engine.get_j(gi, genome)))
                .collect();
            for (gi, j_val) in updates {
                *self.h_cache.entry(gi).or_insert(0.0) += j_val;
            }
        }
    }

    pub fn remove_individual(&mut self, genome: u64) {
        let went_extinct;
        if let Some(count) = self.species.get_mut(&genome) {
            *count -= 1;
            went_extinct = *count == 0;
            if went_extinct {
                self.species.remove(&genome);
            }
        } else {
            went_extinct = false;
        }
        self.n -= 1;

        if went_extinct {
            self.h_cache.remove(&genome);
        }
        // Pre-compute J values, then update h_cache
        let updates: Vec<(u64, f64)> = self.h_cache.keys()
            .map(|&gi| (gi, self.j_engine.get_j(gi, genome)))
            .collect();
        for (gi, j_val) in updates {
            if let Some(h) = self.h_cache.get_mut(&gi) {
                *h -= j_val;
            }
        }
    }

    /// O(1) fitness lookup using cached H values.
    /// Fitness follows Arthur et al. (MNRAS 2024):
    ///   weight = W * Σ_j J(i,j)·n_j / N − N/R
    /// where R is the carrying capacity (stress applied by varying R).
    pub fn compute_p_off(&self, genome: u64) -> f64 {
        let interaction_sum = self.h_cache.get(&genome).copied().unwrap_or(0.0);
        let weight = self.config.w * interaction_sum / (self.n as f64)
            - (self.n as f64) / self.config.r;
        1.0 / (1.0 + (-weight).exp())
    }

    /// Mutate genome by flipping each bit with probability p_mut.
    pub fn mutate(&mut self, genome: u64) -> u64 {
        let mut mutant = genome;
        for i in 0..self.config.l {
            if self.rng.gen::<f64>() < self.config.p_mut {
                mutant ^= 1u64 << i;
            }
        }
        mutant
    }

    /// Run one generation (tau = N/p_kill micro-steps).
    /// Returns false if population went extinct.
    pub fn step_one_generation(&mut self) -> bool {
        if self.n == 0 {
            return false;
        }
        let tau = std::cmp::max((self.n as f64 / self.config.p_kill).round() as u64, 1);
        for _ in 0..tau {
            // Kill attempt
            if self.rng.gen::<f64>() < self.config.p_kill {
                let victim = self.sample_species_weighted();
                self.remove_individual(victim);
            }
            if self.n == 0 {
                return false;
            }
            // Reproduction attempt
            let parent = self.sample_species_weighted();
            let p_off = self.compute_p_off(parent);
            if self.rng.gen::<f64>() < p_off {
                let child1 = self.mutate(parent);
                let child2 = self.mutate(parent);
                self.add_individual(child1);
                self.add_individual(child2);
                self.remove_individual(parent);
            }
        }
        self.generation += 1;
        true
    }

    /// Run simulation until qESS or max_gen. Returns true if qESS reached.
    pub fn run(&mut self) -> Result<bool, String> {
        if self.n == 0 {
            output::emit_final(self.generation, 0, 0, false);
            return Err("Population extinct before start".into());
        }

        // Create/truncate output file if specified
        let out_file = self.config.out_file.clone();
        if !out_file.is_empty() {
            if let Ok(f) = std::fs::File::create(&out_file) {
                drop(f); // truncate and close
            }
        }

        let mut step: u64 = 0;
        let mut tau = std::cmp::max((self.n as f64 / self.config.p_kill).round() as u64, 1);
        let target_gen = self.generation + self.config.max_gen;
        let start_gen = self.generation; // track when this run began

        while self.generation < target_gen {
            // --- Kill attempt ---
            if self.rng.gen::<f64>() < self.config.p_kill {
                let victim = self.sample_species_weighted();
                self.remove_individual(victim);
            }
            if self.n == 0 {
                output::emit_final(self.generation, 0, 0, false);
                return Err("Extinction".into());
            }

            // --- Reproduction attempt ---
            let parent = self.sample_species_weighted();
            let p_off = self.compute_p_off(parent);
            if self.rng.gen::<f64>() < p_off {
                let child1 = self.mutate(parent);
                let child2 = self.mutate(parent);
                self.add_individual(child1);
                self.add_individual(child2);
                self.remove_individual(parent);
            }

            // --- Generation bookkeeping ---
            step += 1;
            if step >= tau {
                self.generation += 1;
                step = 0;
                tau = std::cmp::max((self.n as f64 / self.config.p_kill).round() as u64, 1);

                // Periodic snapshot
                if self.config.output_interval > 0
                    && self.generation % self.config.output_interval == 0
                {
                    output::emit_snapshot(self.generation, self.n, self.species.len());
                    // Write species data to file for visualization
                    if !out_file.is_empty() {
                        let species_vec: Vec<(u64, u64)> =
                            self.species.iter().map(|(&g, &c)| (g, c)).collect();
                        output::emit_snapshot_to_file(
                            &out_file, self.generation, self.n,
                            self.species.len(), &species_vec,
                        );
                    }
                }

                // qESS check — only after enough generations since THIS run started
                let gens_since_start = self.generation - start_gen;
                if gens_since_start >= self.config.qess_window {
                    if let Some(cv) = self.qess.push(self.n) {
                        if self.qess.is_qess(cv) {
                            self.emit_qess_data(cv);
                            if !out_file.is_empty() {
                                let species_vec: Vec<(u64, u64)> =
                                    self.species.iter().map(|(&g, &c)| (g, c)).collect();
                                output::emit_qess_to_file(
                                    &out_file, self.generation, self.n,
                                    self.species.len(), cv,
                                    &species_vec,
                                );
                                output::emit_final_to_file(
                                    &out_file, self.generation, self.n,
                                    self.species.len(), true,
                                );
                            }
                            output::emit_final(self.generation, self.n, self.species.len(), true);
                            return Ok(true);
                        }
                    }
                } else {
                    // Still filling the window — push but don't check
                    self.qess.push(self.n);
                }
            }
        }

        // Max generations reached without qESS — still emit data
        let final_cv = self.qess.push(self.n).unwrap_or(1.0);
        self.emit_qess_data(final_cv);
        if !out_file.is_empty() {
            let species_vec: Vec<(u64, u64)> =
                self.species.iter().map(|(&g, &c)| (g, c)).collect();
            output::emit_qess_to_file(
                &out_file, self.generation, self.n,
                self.species.len(), final_cv, &species_vec,
            );
            output::emit_final_to_file(
                &out_file, self.generation, self.n,
                self.species.len(), false,
            );
        }
        output::emit_final(self.generation, self.n, self.species.len(), false);
        Ok(false)
    }

    /// Emit species data (and optional J matrix) at a qESS checkpoint.
    fn emit_qess_data(&mut self, cv: f64) {
        let species_vec: Vec<(u64, u64)> =
            self.species.iter().map(|(&g, &c)| (g, c)).collect();
        if self.config.output_j {
            let (genomes, matrix) = self.j_engine.compute_active_matrix(&self.species);
            output::emit_qess_with_j(
                self.generation,
                self.n,
                self.species.len(),
                cv,
                &species_vec,
                &genomes,
                &matrix,
            );
        } else {
            output::emit_qess(
                self.generation,
                self.n,
                self.species.len(),
                cv,
                &species_vec,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j_determinism() {
        let mut rng1 = ChaCha12Rng::seed_from_u64(42);
        let j1 = JEngine::new(10, 0.25, &mut rng1);
        let mut rng2 = ChaCha12Rng::seed_from_u64(42);
        let j2 = JEngine::new(10, 0.25, &mut rng2);

        let a = 0b1010101010u64;
        let b = 0b0101010101u64;
        assert!((j1.get_j(a, b) - j2.get_j(a, b)).abs() < 1e-15);
    }

    #[test]
    fn test_j_finite_values() {
        let mut rng = ChaCha12Rng::seed_from_u64(99);
        let j = JEngine::new(10, 0.25, &mut rng);
        for a in [0u64, 1, 100, 500, 1023] {
            for b in [0u64, 1, 512, 1023] {
                let val = j.get_j(a, b);
                assert!(val.is_finite(), "J({},{}) = {} not finite", a, b, val);
            }
        }
    }

    #[test]
    fn test_j_asymmetric() {
        // J[i][k] and J[k][i] should be independently drawn (not equal in general)
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let j = JEngine::new(10, 0.25, &mut rng);
        let mut found_asymmetric = false;
        for a in 0..100u64 {
            for b in (a+1)..100u64 {
                let jab = j.get_j(a, b);
                let jba = j.get_j(b, a);
                if (jab - jba).abs() > 1e-15 && jab != 0.0 && jba != 0.0 {
                    found_asymmetric = true;
                    break;
                }
            }
            if found_asymmetric { break; }
        }
        assert!(found_asymmetric, "J should be asymmetric (J[i,k] != J[k,i] in general)");
    }

    #[test]
    fn test_j_dense_repeated_lookup() {
        // Repeated lookups must return identical values
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let j = JEngine::new(10, 0.25, &mut rng);
        let a = 100u64;
        let b = 200u64;
        let v1 = j.get_j(a, b);
        let v2 = j.get_j(a, b);
        assert!((v1 - v2).abs() < 1e-15);
    }

    #[test]
    fn test_remove_individual_extinction_h_cache() {
        // Removing last individual of a species should clean h_cache
        let config = Config {
            seed: 42,
            l: 10,
            n_init: 20,
            max_gen: 1,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);

        let new_genome = 888u64;
        sim.add_individual(new_genome);
        assert!(sim.h_cache.contains_key(&new_genome));

        sim.remove_individual(new_genome);
        assert!(
            !sim.h_cache.contains_key(&new_genome),
            "h_cache should not contain extinct species"
        );

        // Verify remaining h_cache is still valid
        let h_before: HashMap<u64, f64> = sim.h_cache.clone();
        sim.rebuild_h_cache();
        for (g, h_rebuilt) in &sim.h_cache {
            let h_inc = h_before.get(g).unwrap_or(&0.0);
            assert!(
                (h_inc - h_rebuilt).abs() < 1e-8,
                "h_cache mismatch after extinction for genome {}",
                g
            );
        }
    }

    #[test]
    fn test_qess_constant() {
        let mut det = QessDetector::new(100, 0.05);
        let mut last_cv = 1.0;
        for _ in 0..200 {
            if let Some(cv) = det.push(5000) {
                last_cv = cv;
            }
        }
        assert!(last_cv < 0.001, "CV for constant series: {}", last_cv);
    }

    #[test]
    fn test_qess_variable() {
        let mut det = QessDetector::new(10, 0.05);
        for i in 0..20 {
            let val = if i % 2 == 0 { 100 } else { 200 };
            det.push(val);
        }
        let cv = det.push(100).unwrap();
        assert!(cv > 0.1, "Variable series should have high CV: {}", cv);
    }

    #[test]
    fn test_weighted_sampling_distribution() {
        let config = Config {
            seed: 42,
            l: 10,
            n_init: 0,
            max_gen: 1,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);
        sim.species.clear();
        sim.species.insert(0, 900);
        sim.species.insert(1, 100);
        sim.n = 1000;

        let mut count_0 = 0u64;
        for _ in 0..10000 {
            if sim.sample_species_weighted() == 0 {
                count_0 += 1;
            }
        }
        let ratio = count_0 as f64 / 10000.0;
        assert!(
            (0.86..0.94).contains(&ratio),
            "Expected ~0.9, got {}",
            ratio
        );
    }

    #[test]
    fn test_mutation_range() {
        let config = Config {
            seed: 42,
            l: 10,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);
        let max_genome = (1u64 << 10) - 1;
        for genome in [0u64, max_genome, 512, 42] {
            for _ in 0..100 {
                let mutant = sim.mutate(genome);
                assert!(mutant <= max_genome, "mutant {} > max {}", mutant, max_genome);
            }
        }
    }

    #[test]
    fn test_state_roundtrip_h_cache() {
        let config = Config {
            seed: 42,
            l: 10,
            n_init: 100,
            max_gen: 1000,
            ..Config::default()
        };
        let mut sim = Simulation::new(config.clone());
        // Run 500 gens to get a settled state
        for _ in 0..500 {
            sim.step_one_generation();
        }

        // Snapshot h_cache before save
        let h_before: Vec<(u64, f64)> = {
            let mut v: Vec<_> = sim.h_cache.iter().map(|(&g, &h)| (g, h)).collect();
            v.sort_by_key(|&(g, _)| g);
            v
        };
        let species_before: Vec<(u64, u64)> = sim.species.iter().map(|(&g, &c)| (g, c)).collect();
        let n_before = sim.n;

        // Save and reload
        let state = sim.to_state();
        let json = serde_json::to_string(&state).unwrap();
        let state2: crate::state::SimState = serde_json::from_str(&json).unwrap();
        let sim2 = Simulation::from_state(state2, config);

        // Compare J values between original and reconstructed engine
        let mut j_mismatch = 0;
        for &(gi, _) in &species_before {
            for &(gj, _) in &species_before {
                let j_orig = sim.j_engine.get_j(gi, gj);
                let j_new = sim2.j_engine.get_j(gi, gj);
                if (j_orig - j_new).abs() > 1e-15 {
                    eprintln!("  J MISMATCH: J({},{}) orig={:.10} new={:.10}", gi, gj, j_orig, j_new);
                    j_mismatch += 1;
                }
            }
        }
        eprintln!("  J value mismatches: {}", j_mismatch);
        eprintln!("  Original engine n_genomes={}, L={}", sim.j_engine.n_genomes, sim.j_engine.l);
        eprintln!("  Rebuilt  engine n_genomes={}, L={}", sim2.j_engine.n_genomes, sim2.j_engine.l);

        // Manually compute H values to determine which cache is truth
        eprintln!("\n  Manual H verification:");
        for &(gi, _) in &species_before {
            let mut h_manual = 0.0;
            for &(gj, nj) in &species_before {
                h_manual += sim.j_engine.get_j(gi, gj) * nj as f64;
            }
            let h_inc = sim.h_cache.get(&gi).copied().unwrap_or(f64::NAN);
            let h_reb = sim2.h_cache.get(&gi).copied().unwrap_or(f64::NAN);
            let d_inc = (h_inc - h_manual).abs();
            let d_reb = (h_reb - h_manual).abs();
            if d_inc > 1e-6 || d_reb > 1e-6 {
                eprintln!("  genome={}: manual={:.6} incremental={:.6} (d={:.6}) rebuild={:.6} (d={:.6})",
                    gi, h_manual, h_inc, d_inc, h_reb, d_reb);
            }
        }

        // Compare species
        let species_after: Vec<(u64, u64)> = sim2.species.iter().map(|(&g, &c)| (g, c)).collect();
        assert_eq!(species_before, species_after, "Species mismatch after roundtrip");
        assert_eq!(n_before, sim2.n, "N mismatch after roundtrip");

        // Compare h_cache
        let h_after: Vec<(u64, f64)> = {
            let mut v: Vec<_> = sim2.h_cache.iter().map(|(&g, &h)| (g, h)).collect();
            v.sort_by_key(|&(g, _)| g);
            v
        };

        assert_eq!(h_before.len(), h_after.len(), "h_cache size mismatch: {} vs {}", h_before.len(), h_after.len());

        let mut max_diff = 0.0f64;
        let mut fail_count = 0;
        for ((g1, h1), (g2, h2)) in h_before.iter().zip(h_after.iter()) {
            assert_eq!(g1, g2, "h_cache key mismatch");
            let diff = (h1 - h2).abs();
            if diff > 1e-9 {
                let count = sim.species.get(g1).copied().unwrap_or(0);
                eprintln!("  MISMATCH genome={} count={}: incremental={:.6} vs rebuild={:.6} diff={:.6}", g1, count, h1, h2, diff);
                fail_count += 1;
            }
            max_diff = max_diff.max(diff);
        }
        eprintln!("  {} species, {} mismatches, max_diff={:.6}", h_before.len(), fail_count, max_diff);
        assert!(max_diff < 1e-9, "h_cache max diff = {} (should be < 1e-9)", max_diff);
    }
}
