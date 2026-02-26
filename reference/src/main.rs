//
// Created by Clem von Stengel on 13/08/2020
// email : clem@acsresearch.org
//

//imports from libraries needed
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::{Instant, SystemTime};
use std::env;

//all the parameters for the program. Make sure GENOMES is 2^L.
const L: u32 = 10;                  //length of genome
const GENOMES: u32 = 1024;          //total number of different species
const N_INIT: u32 = 100;            //initial total population size
const GENERATIONS: usize = 100005;   //simulation length
const THETA: f64 = 0.25;            //for j matrix initialisation
const P_KILL: f64 = 0.2;            //self explanatory
const R: f64 = 143.0;               //for calculating p_off
const W: f64 = 33.0;                //for calculating p_off

// const P_MUT: f64 = 0.001;        mutation probability per bit - currently calculated randomly every run

//type declaration for j matrix
type JMatrix = Vec<[f64; GENOMES as usize]>; //Vec of arrays is so it's stored on the heap

fn main() -> std::io::Result<()> {
    //get command line argument for seed, or set to time if none there
    let args: Vec<String> = env::args().collect();
    let seed: u64;
	let tsec: u64;
	let inp: u64;
	let p_mut: f64;
    if args.len() >= 2 { //first argument is the program itself
		tsec = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                    .expect("Duration since UNIX_EPOCH failed").as_secs();
        inp = args[1].parse().expect("seed must be an integer");
		seed = tsec*inp;
		p_mut = args[3].parse().expect("mutation rate must be a float");
    } else {
        seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                    .expect("Duration since UNIX_EPOCH failed").as_secs(); //shouldn't fail
		p_mut = 0.001;
    }
    //seed random number generator - functions for rng are defined at the bottom
    let mut rng = StdRng::seed_from_u64(seed);
    println!("seed: {}", seed);

    //initialise J matrix
    let mut j = create_j(&mut rng);
    if args.len() >= 3 { //overwrite with j from file if given
        let j_file = &args[2];
        let f = BufReader::new(File::open(j_file).expect("Could not find given file for J"));
        let j_arr: Vec<Vec<f64>> = f.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().expect("Some entry in J is not a number"))
                .collect())
            .collect();
        print!("{}\n", j_arr.len());
        for i in 1..GENOMES as usize {
            for k in 1..GENOMES as usize {
                j[i][k] = *j_arr.get(i).expect("J matrix given is too small")
                    .get(k).expect("J matrix given is too small");
            }
        }
    } else { //create j file with new matrix if none is given
        let mut j_file = File::create(format!("j_{}.txt", seed))?;
        print!("Created file j_{}.txt\n", seed);
        for i in 0..GENOMES as usize {
            for k in 0..(GENOMES-1) as usize {
                j_file.write_all(format!("{}\t", j[i][k]).as_bytes())?;
            }
            j_file.write_all(format!("{}", j[i][(GENOMES-1) as usize]).as_bytes())?;
            j_file.write_all(b"\n")?;
        }
        print!("Wrote to file\n");
    }

    //get random mutation rate 
    //let p_mut = 0.001;//gen_p_mut(&mut rng);

    //create files
    //let mut quasi_z_file = File::create(format!("quasi_z_{}_{}.txt", seed, p_mut))?;
    //let mut total_population_file = File::create(format!("total_population_{}_{}.txt", seed, p_mut))?;
    let mut species_genome_file = File::create(format!("simlog/species_genome_{}_{}.txt", seed, p_mut))?;
    let mut species_population_file = File::create(format!("simlog/species_population_{}_{}.txt", seed, p_mut))?;

    //initilaise other variables
    let mut n = N_INIT;//total population size

    let (mut existent, mut species) = get_initial_population(&mut rng);
    //species is a BTreeMap - which is basically a sortable HashMap with genomes as keys and population as values,
    //existent is a Vec which contains all individuals, with genomes as values
    //both contain all the data, but having the two formats allows for a more efficient program
    
    //get time at start to keep track of time taken to simulate
    let now = Instant::now();

    //initialise counter variables to keep track of generation
    let mut generation = 0;
    let mut step = 0;
    let mut tau = ((n as f64)/P_KILL).round() as u32; //how many steps per generation

    while generation < GENERATIONS {
        //attempt to kill a random individual with probability P_KILL
        kill(&mut species, &mut existent, &mut n, &mut rng);
        
        //choose individual, calculate p_off, and attempt to reproduce with p_off
        //  (also handles mutations)
        baby(&mut species, &mut existent, &mut n, &j, p_mut, &mut rng);

        step += 1;
        //check whether to advance to next generation
        if step == tau {
            generation += 1;
            step = 0;
            tau = ((n as f64)/P_KILL).round() as u32;

            //let quasi_z = compute_quasi_z(&species);
            
            //save data on file
            //quasi_z_file.write_all(format!("{}\n", quasi_z).as_bytes())?;
            //total_population_file.write_all(format!("{}\t{}\n", generation, n).as_bytes())?;
            

            //print progress on terminal as program runs
            if (generation % 1000 == 0)||(generation % 1000 == 1) {
				for (genome, population) in &species {
					species_genome_file.write_all(format!("{}\t", genome).as_bytes())?;
					species_population_file.write_all(format!("{}\t", population).as_bytes())?;
				}
				species_genome_file.write_all(b"\n")?;
				species_population_file.write_all(b"\n")?;
               println!("generation: {}, time elapsed: {} secs", generation, now.elapsed().as_secs());
            }
        }
    }

    //close files and exit
    Ok(())
}

// fn gen_p_mut(rng: &mut StdRng) -> f64 {
//     let mun = rng.choose_in_range(1, 11); //integer in range [0, 11)
//     (mun as f64)/200.0
// }

fn create_j(rng: &mut StdRng) -> JMatrix { 
    //create GENOMES*GENOMES matrix initilaised with 0.0 everywhere
    let mut j_matrix = vec![[0.0; GENOMES as usize]; GENOMES as usize];

    //for each i=/=k, with probability theta, create interaction link
    for i in 0..GENOMES as usize {
        for k in 0..i {
            if rng.with_probability(THETA) {
                j_matrix[i][k] = rng.uniform(-1.0, 1.0);
                j_matrix[k][i] = rng.uniform(-1.0, 1.0);
            }
        }
    }

    //return
    j_matrix
}

fn get_initial_population(rng: &mut StdRng) -> (Vec<u32>, BTreeMap<u32, u32>) {
    let mut species = BTreeMap::new();
    let mut existent = Vec::new();
    for _ in 0..N_INIT {
        //select random genome and increment number by 1 
        //  (adding genome as key to hashmap if first one)
        let genome = rng.choose_in_range(0, GENOMES);
        add_genome(genome, &mut species, &mut existent);
    }
    (existent, species)
}

fn kill(species: &mut BTreeMap<u32, u32>, existent: &mut Vec<u32>, n: &mut u32, rng: &mut StdRng) {
    if rng.with_probability(P_KILL) {
        let selected_index = rng.choose_in_range(0, existent.len());
        let selected_genome = existent[selected_index];

        *n -= 1;

        //get pointer to chosen population number
        let selected_population = species.get_mut(&selected_genome).expect("Kill selection out of bounds");

        *selected_population -= 1;
        if *selected_population == 0 {
            species.remove(&selected_genome);
        }

        existent.remove(selected_index);

        if *n == 0 {
            panic!("Extinction");
        }
    }
}

fn baby(species: &mut BTreeMap<u32, u32>, existent: &mut Vec<u32>, n: &mut u32, j: &JMatrix, p_mut: f64, rng: &mut StdRng) {
    let selected_index = rng.choose_in_range(0, existent.len());
    let selected_genome = existent[selected_index];

    //compute p_off
    let mut t1 = 0.0;
    for (genome, population) in species as &BTreeMap<u32, u32> { //"as &" to not move #just_rust_things
        t1 += j[selected_genome as usize][*genome as usize] * *population as f64;
    }
    let weight = W*t1/(*n as f64) - (*n as f64)/R;
    let p_off = 1.0 / (1.0 + (-weight).exp());

    if rng.with_probability(p_off) {
        *n += 1;

        //reproduce
        let child_genome_1 = mutate(selected_genome, p_mut, rng);
        let child_genome_2 = mutate(selected_genome, p_mut, rng);

        add_genome(child_genome_1, species, existent);
        add_genome(child_genome_2, species, existent);

        //kill parent
        let selected_population = species.get_mut(&selected_genome).expect("Baby selection out of bounds");
        *selected_population -= 1;
        if *selected_population == 0 {
            species.remove(&selected_genome);
        }
        existent.remove(selected_index);
    }
}

fn mutate(genome: u32, p_mut: f64, rng: &mut StdRng) -> u32 {
    let mut mutant = genome;
    for i in 0..L {
        if rng.with_probability(p_mut) {
            mutant ^= 2u32.pow(i); //flips bit at position i of genome in binary
        }
    }
    mutant
}

//quasi_z is the sum of exp(-1/population) for each species present
// fn compute_quasi_z(species: &BTreeMap<u32, u32>) -> f64 {
//     let mut quasi_z = 0.0;
//     for (_, population) in species {
//         quasi_z += (-1.0/(*population as f64)).exp();
//     }
//     quasi_z
// }

//add an individual with genome to species hashmap and existent vector
fn add_genome(genome: u32, species: &mut BTreeMap<u32, u32>, existent: &mut Vec<u32>) {
    existent.push(genome);
    let count = species.entry(genome).or_insert(0);
    *count += 1;
}

//random number stuff
trait RandomFuncs: Rng {
    fn with_probability(&mut self, theta: f64) -> bool {
        let random_float: f64 = self.gen(); //needed for type inference
        random_float < theta
    }

    fn uniform(&mut self, lower: f64, upper: f64) -> f64 {
        let random_float: f64 = self.gen(); //needed for type inference
        random_float * (upper - lower) + lower
    }

    fn choose_in_range<T>(&mut self, lower: T, upper: T) -> T 
    where T: rand::distributions::uniform::SampleUniform //so it behaves with gen_range
    {
        self.gen_range(lower, upper)
    }
}

impl RandomFuncs for StdRng {}
//to add these functions to a different random number generator, simply write:
//impl RandomFuncs for MyRng {}
//
//  (note MyRng {} must also implement the trait "Rng")