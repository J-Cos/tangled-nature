use std::env;
use std::time::SystemTime;

/// All simulation parameters, configurable via CLI flags.
#[derive(Clone)]
pub struct Config {
    pub seed: u64,
    pub l: u32,
    pub n_init: u64,
    pub max_gen: usize,
    pub theta: f64,
    pub p_kill: f64,
    pub w: f64,
    pub r: f64,
    pub p_mut: f64,
    pub output_interval: usize,
    pub qess_window: usize,
    pub qess_threshold: f64,
    pub state_in: Option<String>,
    pub state_out: Option<String>,
    pub output_j: bool,
    // Spatial mode
    pub spatial: bool,
    pub grid_size: usize,
    pub p_move: f64,
    pub out_file: String,
    // Press stressor (spatial only)
    pub stress_r: f64,
    pub stress_duration: usize,
    pub stress_cells: Option<Vec<usize>>,  // explicit cell indices, or None for central 4
    pub stress_ramp: usize,               // ramp mode: stress 1 more cell every N gens (0=disabled)
    pub stress_sigma: f64,                // contagion spread parameter (grid units)
}

impl Config {
    pub fn default() -> Self {
        Config {
            seed: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            l: 10,
            n_init: 100,
            max_gen: 200_000,
            theta: 0.25,
            p_kill: 0.2,
            w: 33.0,
            r: 143.0,
            p_mut: 0.001,
            output_interval: 100,
            qess_window: 5000,
            qess_threshold: 0.05,
            state_in: None,
            state_out: None,
            output_j: false,
            spatial: false,
            grid_size: 10,
            p_move: 0.01,
            out_file: "spatial_output.jsonl".to_string(),
            stress_r: 0.0,
            stress_duration: 0,
            stress_cells: None,
            stress_ramp: 0,
            stress_sigma: 2.0,
        }
    }

    pub fn from_args() -> Result<Self, String> {
        let args: Vec<String> = env::args().collect();
        let mut config = Config::default();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seed" => config.seed = parse_next(&args, &mut i)?,
                "--l" => config.l = parse_next(&args, &mut i)?,
                "--n-init" => config.n_init = parse_next(&args, &mut i)?,
                "--max-gen" => config.max_gen = parse_next(&args, &mut i)?,
                "--theta" => config.theta = parse_next(&args, &mut i)?,
                "--p-kill" => config.p_kill = parse_next(&args, &mut i)?,
                "--mu" => { let _: f64 = parse_next(&args, &mut i)?; } // ignored, kept for backward compat
                "--w" => config.w = parse_next(&args, &mut i)?,
                "--r" => config.r = parse_next(&args, &mut i)?,
                "--p-mut" => config.p_mut = parse_next(&args, &mut i)?,
                "--output-interval" => config.output_interval = parse_next(&args, &mut i)?,
                "--qess-window" => config.qess_window = parse_next(&args, &mut i)?,
                "--qess-threshold" => config.qess_threshold = parse_next(&args, &mut i)?,
                "--state-in" => config.state_in = Some(parse_next_str(&args, &mut i)?),
                "--state-out" => config.state_out = Some(parse_next_str(&args, &mut i)?),
                "--output-j" => config.output_j = true,
                "--spatial" => config.spatial = true,
                "--grid-size" => config.grid_size = parse_next(&args, &mut i)?,
                "--p-move" => config.p_move = parse_next(&args, &mut i)?,
                "--out" => config.out_file = parse_next_str(&args, &mut i)?,
                "--stress-r" => config.stress_r = parse_next(&args, &mut i)?,
                "--stress-duration" => config.stress_duration = parse_next(&args, &mut i)?,
                "--stress-cells" => {
                    let s = parse_next_str(&args, &mut i)?;
                    config.stress_cells = Some(
                        s.split(',').map(|x| x.trim().parse::<usize>().expect("Invalid cell index")).collect()
                    );
                }
                "--stress-ramp" => config.stress_ramp = parse_next(&args, &mut i)?,
                "--stress-sigma" => config.stress_sigma = parse_next(&args, &mut i)?,
                other => return Err(format!("Unknown argument: {}", other)),
            }
            i += 1;
        }

        if config.l > 63 {
            return Err("L must be <= 63 (genome stored as u64)".into());
        }

        Ok(config)
    }
}

fn parse_next<T: std::str::FromStr>(args: &[String], i: &mut usize) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    *i += 1;
    if *i >= args.len() {
        return Err(format!("Missing value for {}", args[*i - 1]));
    }
    args[*i]
        .parse::<T>()
        .map_err(|e| format!("Invalid value for {}: {}", args[*i - 1], e))
}

fn parse_next_str(args: &[String], i: &mut usize) -> Result<String, String> {
    *i += 1;
    if *i >= args.len() {
        return Err(format!("Missing value for {}", args[*i - 1]));
    }
    Ok(args[*i].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.l, 10);
        assert!((config.p_kill - 0.2).abs() < 1e-10);
        assert!((config.w - 33.0).abs() < 1e-10);
        assert!((config.r - 143.0).abs() < 1.0);
        assert!(config.state_in.is_none());
        assert!(!config.output_j);
    }

    #[test]
    fn test_l_range_check() {
        // Simulate args with L=64 (too large)
        let result = {
            let mut c = Config::default();
            c.l = 64;
            if c.l > 63 {
                Err("L must be <= 63".to_string())
            } else {
                Ok(c)
            }
        };
        assert!(result.is_err());
    }
}
