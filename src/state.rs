use std::collections::BTreeMap;
use std::fs;

use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

/// Serializable simulation state for save/load between orchestrator calls.
#[derive(Serialize, Deserialize)]
pub struct SimState {
    pub generation: usize,
    pub l: u32,
    pub species: BTreeMap<u64, u64>,
    pub j_locus_matrices: Vec<[[f64; 2]; 2]>,
    pub rng: ChaCha12Rng,
}

impl SimState {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let state: SimState = serde_json::from_str(&json)?;
        Ok(state)
    }
}

/// Serializable spatial grid state for save/load (burn-in → fork workflows).
#[derive(Serialize, Deserialize)]
pub struct SpatialState {
    pub generation: usize,
    pub grid_w: usize,
    pub grid_h: usize,
    pub p_move: f64,
    pub l: u32,
    pub j_locus_matrices: Vec<[[f64; 2]; 2]>,
    pub patch_states: Vec<PatchState>,
    pub migration_rng: ChaCha12Rng,
}

/// Lightweight per-patch state (no J matrices — those are shared).
#[derive(Serialize, Deserialize)]
pub struct PatchState {
    pub species: BTreeMap<u64, u64>,
    pub rng: ChaCha12Rng,
    pub seed: u64,
}

impl SpatialState {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let state: SpatialState = serde_json::from_str(&json)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_state_round_trip() {
        let mut species = BTreeMap::new();
        species.insert(42u64, 100u64);
        species.insert(99u64, 50u64);

        let rng = ChaCha12Rng::seed_from_u64(12345);
        let locus = vec![
            [[0.1, -0.2], [0.3, 0.4]],
            [[0.5, -0.6], [0.7, 0.8]],
        ];

        let state = SimState {
            generation: 500,
            l: 2,
            species,
            j_locus_matrices: locus,
            rng,
        };

        let path = "/tmp/tnm_test_state.json";
        state.save(path).expect("save failed");
        let loaded = SimState::load(path).expect("load failed");

        assert_eq!(loaded.generation, 500);
        assert_eq!(loaded.l, 2);
        assert_eq!(loaded.species.len(), 2);
        assert_eq!(*loaded.species.get(&42).unwrap(), 100);
        assert_eq!(loaded.j_locus_matrices.len(), 2);
        assert!((loaded.j_locus_matrices[0][0][0] - 0.1).abs() < 1e-10);

        let _ = std::fs::remove_file(path);
    }
}
