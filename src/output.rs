use serde_json::json;

/// Emit a periodic progress snapshot (for monitoring).
pub fn emit_snapshot(gen: usize, n: u64, s: usize) {
    println!("{}", json!({"type": "snapshot", "gen": gen, "n": n, "s": s}));
}

/// Emit qESS checkpoint with species abundance data.
pub fn emit_qess(gen: usize, n: u64, s: usize, cv: f64, species: &[(u64, u64)]) {
    println!(
        "{}",
        json!({
            "type": "qess",
            "gen": gen,
            "n": n,
            "s": s,
            "cv": cv,
            "species": species,
        })
    );
}

/// Emit qESS checkpoint with species data AND active J submatrix.
pub fn emit_qess_with_j(
    gen: usize,
    n: u64,
    s: usize,
    cv: f64,
    species: &[(u64, u64)],
    genomes: &[u64],
    matrix: &[Vec<f64>],
) {
    println!(
        "{}",
        json!({
            "type": "qess",
            "gen": gen,
            "n": n,
            "s": s,
            "cv": cv,
            "species": species,
            "j_matrix": {
                "genomes": genomes,
                "matrix": matrix,
            }
        })
    );
}

/// Emit final status line (exit summary).
pub fn emit_final(gen: usize, n: u64, s: usize, qess: bool) {
    println!(
        "{}",
        json!({"type": "final", "gen": gen, "n": n, "s": s, "qess": qess})
    );
}
