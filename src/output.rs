use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;

/// Emit a periodic progress snapshot (for monitoring).
pub fn emit_snapshot(gen: usize, n: u64, s: usize) {
    println!("{}", json!({"type": "snapshot", "gen": gen, "n": n, "s": s}));
}

/// Emit a snapshot with full species data (for visualization).
pub fn emit_snapshot_with_species(gen: usize, n: u64, s: usize, species: &[(u64, u64)]) {
    println!(
        "{}",
        json!({
            "type": "snapshot",
            "gen": gen,
            "n": n,
            "s": s,
            "species": species,
        })
    );
}

/// Emit a snapshot to a file (appending).
pub fn emit_snapshot_to_file(path: &str, gen: usize, n: u64, s: usize, species: &[(u64, u64)]) {
    let line = json!({
        "type": "snapshot",
        "gen": gen,
        "n": n,
        "s": s,
        "species": species,
    });
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", line);
    }
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

/// Emit final status to a file.
pub fn emit_final_to_file(path: &str, gen: usize, n: u64, s: usize, qess: bool) {
    let line = json!({"type": "final", "gen": gen, "n": n, "s": s, "qess": qess});
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", line);
    }
}

/// Emit qESS checkpoint to a file.
pub fn emit_qess_to_file(
    path: &str,
    gen: usize,
    n: u64,
    s: usize,
    cv: f64,
    species: &[(u64, u64)],
) {
    let line = json!({
        "type": "qess",
        "gen": gen,
        "n": n,
        "s": s,
        "cv": cv,
        "species": species,
    });
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", line);
    }
}
