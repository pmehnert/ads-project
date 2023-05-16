pub mod bitvec;
pub mod int;
pub mod rmq;

use std::{
    error::Error,
    fmt, fs,
    io::{self, Write},
    path::PathBuf,
    process::{ExitCode, Termination},
    time::{Duration, Instant},
};

use crate::{
    int::IndexInt,
    rmq::{fits_index, Naive, RangeMinimum},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn div_ceil(lhs: usize, rhs: usize) -> usize { lhs.saturating_add(rhs - 1) / rhs }

pub fn main() -> Result<TestResults> {
    #[inline(never)]
    fn run_timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
        let before = Instant::now();
        let result = std::hint::black_box(f());
        let elapsed = before.elapsed();
        (result, elapsed)
    }

    fn run_pd(_input: PredecessorInput) -> (Vec<usize>, usize) { todo!() }

    fn run_rmq(input: RMQInput) -> (Vec<usize>, usize) {
        // todo do all three implementations need to be run here?
        match &input.values {
            // values if fits_index::<u8>(values) => run_rmq_with::<u8>(&input),
            values if fits_index::<u16>(values) => run_rmq_with::<u16>(&input),
            values if fits_index::<u32>(values) => run_rmq_with::<u32>(&input),
            // values if fits_index::<u64>(values) => run_rmq_with::<u64>(&input),
            _ => run_rmq_with::<usize>(&input),
        }
    }

    fn run_rmq_with<Idx: IndexInt>(input: &RMQInput) -> (Vec<usize>, usize) {
        let rmq = Naive::<Idx>::new(&input.values);
        let result = (input.queries.iter())
            .map(|(lower, upper)| rmq.range_min(*lower, *upper).unwrap())
            .collect();
        (result, rmq.size_bits())
    }

    let args = Arguments::parse()?;
    let input_file = fs::OpenOptions::new().read(true).open(args.in_path)?;
    let input_reader = io::BufReader::new(input_file);

    let ((output, space), time) = match args.algo {
        Algorithm::Predecessor => {
            let input = PredecessorInput::parse(input_reader)?;
            run_timed(|| run_pd(input))
        },
        Algorithm::RangeMinimumQuery => {
            let input = RMQInput::parse(input_reader)?;
            run_timed(|| run_rmq(input))
        },
    };

    let mut out_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        // todo .truncate(true) ?
        .open(&args.out_path)?;

    output.iter().try_for_each(|x| writeln!(out_file, "{}", x))?;

    Ok(TestResults { algo: args.algo, time, space })
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Predecessor,
    RangeMinimumQuery,
}

#[derive(Debug, Clone)]
pub struct Arguments {
    algo: Algorithm,
    in_path: PathBuf,
    out_path: PathBuf,
}

impl Arguments {
    pub fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);

        let algo = match args.next().as_deref() {
            Some("pd") => Ok(Algorithm::Predecessor),
            Some("rmq") => Ok(Algorithm::RangeMinimumQuery),
            Some(algo) => Err(format!("unknown algorithm '{}'", algo)),
            None => Err("missing algorithm parameter".to_string()),
        }?;

        let in_path = PathBuf::from(args.next().ok_or("missing input path")?);
        let out_path = PathBuf::from(args.next().ok_or("missing output path")?);

        match args.next() {
            Some(arg) => Err(format!("unexpected parameter '{}'", arg).into()),
            None => Ok(Self { algo, in_path, out_path }),
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct TestResults {
    algo: Algorithm,
    time: Duration,
    space: usize,
}

impl Termination for TestResults {
    fn report(self) -> ExitCode {
        // todo check output specification (equal sign after name)
        let _ = writeln!(
            std::io::stderr(),
            "RESULT algo={} name=pascal_mehnert time={} space={}",
            self.algo,
            self.time.as_millis(),
            self.space,
        );
        ExitCode::SUCCESS
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Algorithm::Predecessor => write!(f, "pd"),
            Algorithm::RangeMinimumQuery => write!(f, "rmq"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ParseError {
    MissingValue,
    NotARange,
    EmptyRange,
}

impl Error for ParseError {}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingValue => write!(f, "mssing a value in the input file"),
            ParseError::NotARange => write!(f, "expected a range in the input file"),
            ParseError::EmptyRange => write!(f, "found an empty range in the input file"),
        }
    }
}

pub struct PredecessorInput {
    values: Vec<u64>,
    queries: Vec<u64>,
}

impl PredecessorInput {
    pub fn parse(reader: impl io::BufRead) -> Result<Self> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            queries.push(line?.parse()?);
        }

        Ok(Self { values, queries })
    }
}

pub struct RMQInput {
    values: Vec<u64>,
    queries: Vec<(usize, usize)>,
}

impl RMQInput {
    pub fn parse(reader: impl io::BufRead) -> Result<Self> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            let line = line?;
            let (left, right) = line.split_once(',').ok_or(ParseError::NotARange)?;
            if left > right {
                return Err(ParseError::EmptyRange.into());
            }
            queries.push((left.parse()?, right.parse()?))
        }

        Ok(Self { values, queries })
    }
}

fn parse_values(lines: &mut io::Lines<impl io::BufRead>) -> Result<Vec<u64>> {
    let line = lines.next().ok_or(ParseError::MissingValue)??;
    let len = line.parse::<usize>()?;

    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let line = lines.next().ok_or(ParseError::MissingValue)??;
        values.push(line.parse()?);
    }

    Ok(values)
}
