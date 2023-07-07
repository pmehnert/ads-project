use std::{error::Error, fs, io};

use clap::{Args, Parser, Subcommand, ValueEnum};
use input::{PredecessorInput, RangeMinimumInput};

mod input;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let args = Arguments::parse();
    match args.command {
        Commands::Generate(args) => run_gen(args),
        Commands::Check(args) => run_check(args),
    }
}

fn run_gen(args: GenerateArguments) -> Result<()> {
    let range = args.lower.unwrap_or(0)..=args.upper.unwrap_or(u64::MAX);
    let file = fs::OpenOptions::new().create(true).write(true).open(&args.path)?;

    match args.algo {
        Algorithm::Predecessor => {
            let input = PredecessorInput::new(args.num_values, args.num_queries, range);
            input.write(&mut io::BufWriter::new(file))?;
        },
        Algorithm::RangeMinimum => {
            let input = RangeMinimumInput::new(args.num_values, args.num_queries, range);
            input.write(&mut io::BufWriter::new(file))?;
        },
    }

    Ok(())
}

fn run_check(args: CheckArguments) -> Result<()> { Ok(()) }

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(name = "gen")]
    Generate(GenerateArguments),

    #[command(name = "check")]
    Check(CheckArguments),
}

#[derive(Args)]
pub struct GenerateArguments {
    #[arg(value_enum)]
    algo: Algorithm,
    path: String,
    num_values: usize,
    num_queries: usize,
    lower: Option<u64>,
    upper: Option<u64>,
}

#[derive(Args)]
pub struct CheckArguments {
    #[arg(value_enum)]
    algo: Algorithm,
    input: String,
    output: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Algorithm {
    #[value(name = "pd")]
    Predecessor,
    #[value(name = "rmq")]
    RangeMinimum,
}
