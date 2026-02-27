mod config;
mod model;
mod output;
mod spatial;
mod state;

use std::process;

fn main() {
    let config = match config::Config::from_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    if config.spatial {
        // ── Spatial mode ──────────────────────────────────────────────
        eprintln!(
            "Spatial TNM v0.1 | seed={} L={} W={:.1} R={:.1} p_kill={:.2} p_mut={} grid={}×{} p_move={}",
            config.seed, config.l, config.w, config.r, config.p_kill, config.p_mut,
            config.grid_size, config.grid_size, config.p_move
        );

        let mut grid = if let Some(ref path) = config.state_in {
            match state::SpatialState::load(path) {
                Ok(s) => {
                    eprintln!(
                        "Loaded spatial state: gen={} grid={}×{} patches={}",
                        s.generation, s.grid_w, s.grid_h, s.patch_states.len()
                    );
                    spatial::SpatialGrid::from_state(s, &config)
                }
                Err(e) => {
                    eprintln!("Error loading spatial state: {}", e);
                    process::exit(1);
                }
            }
        } else {
            spatial::SpatialGrid::new(
                config.grid_size,
                config.grid_size,
                config.p_move,
                &config,
            )
        };
        let qess = grid.run();
        if qess {
            eprintln!("qESS reached at gen {}", grid.generation);
        }
        process::exit(0);
    }

    // ── Standard (single-patch) mode ──────────────────────────────────
    eprintln!(
        "TNM v0.2 | seed={} L={} W={:.1} R={:.1} p_kill={:.2} p_mut={} max_gen={}",
        config.seed, config.l, config.w, config.r, config.p_kill, config.p_mut, config.max_gen
    );

    let out_file = config.out_file.clone();
    let no_viz = config.no_viz;

    let mut sim = if let Some(ref path) = config.state_in {
        match state::SimState::load(path) {
            Ok(s) => {
                eprintln!(
                    "Loaded state: gen={} N={} S={}",
                    s.generation,
                    s.species.values().sum::<u64>(),
                    s.species.len()
                );
                model::Simulation::from_state(s, config)
            }
            Err(e) => {
                eprintln!("Error loading state: {}", e);
                process::exit(1);
            }
        }
    } else {
        model::Simulation::new(config)
    };

    eprintln!("Initial: N={} S={}", sim.n, sim.species.len());

    match sim.run() {
        Ok(qess_reached) => {
            if let Some(ref path) = sim.config.state_out {
                let s = sim.to_state();
                if let Err(e) = s.save(path) {
                    eprintln!("Error saving state: {}", e);
                    process::exit(1);
                }
                eprintln!("State saved to {}", path);
            }

            // Run visualization (skip if --no-viz or no output file)
            if !no_viz && !out_file.is_empty() {
                use std::process::Command;
                eprintln!("  Generating visualizations...");
                let status = Command::new("python3")
                    .arg("classic_viz.py")
                    .arg(&out_file)
                    .status();
                match status {
                    Ok(s) if s.success() => {},
                    Ok(_) | Err(_) => eprintln!("  Warning: Failed to run classic_viz.py"),
                }
                eprintln!("  Visualization complete!");
            }

            if qess_reached {
                eprintln!("qESS reached at gen {}", sim.generation);
                process::exit(0);
            } else {
                eprintln!("Max gen reached without qESS");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            if let Some(ref path) = sim.config.state_out {
                let _ = sim.to_state().save(path);
            }
            process::exit(2);
        }
    }
}