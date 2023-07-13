pub mod bitvec;
pub mod cartesian;
pub mod int;
pub mod packed;
pub mod predecessor;
pub mod rmq;

use std::{
    error::Error,
    fmt, fs,
    io::{BufRead, BufReader, BufWriter, Lines, Write},
    path::{Path, PathBuf},
    process::{ExitCode, Termination},
    time::{Duration, Instant},
};

use crate::{
    int::{AsHalfSize, IndexInt},
    predecessor::{EliasFano, Predecessor},
    rmq::{fits_index, RangeMinimum},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn div_ceil(lhs: usize, rhs: usize) -> usize { lhs.saturating_add(rhs - 1) / rhs }

pub fn main() -> std::result::Result<TestResults, String> {
    #[inline(never)]
    fn run_timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
        let before = Instant::now();
        let result = std::hint::black_box(f());
        let elapsed = before.elapsed();
        (result, elapsed)
    }

    #[inline(never)]
    fn run_pd(input: PredecessorInput) -> (Vec<u64>, usize) {
        let pd = EliasFano::new(&input.values);

        // Don't need to keep values in memory
        std::mem::drop(input.values);

        let predecessor = |value| pd.predecessor(value).unwrap_or(u64::MAX);
        let results = input.queries.iter().copied().map(predecessor).collect();
        let space = 8 * pd.size_bytes() + 8 * std::mem::size_of::<EliasFano>();

        (results, space)
    }

    #[inline(never)]
    fn run_rmq(input: RMQInput) -> (Vec<usize>, usize) {
        match &input.values {
            values if fits_index::<u16>(values) => run_rmq_with::<u16>(input),
            values if fits_index::<u32>(values) => run_rmq_with::<u32>(input),
            _ => run_rmq_with::<usize>(input),
        }
    }

    fn run_rmq_with<Idx>(input: RMQInput) -> (Vec<usize>, usize)
    where
        Idx: IndexInt + AsHalfSize,
        Idx::HalfSize: IndexInt,
    {
        // let rmq = crate::rmq::Naive::<Idx>::new(&input.values);
        let rmq = crate::rmq::Sparse::<Idx, &[_]>::new(&input.values);
        // let rmq = crate::rmq::Cartesian::<Idx>::new(&input.values);

        let range_min = |(lower, upper)| rmq.range_min(lower, upper).unwrap();
        let results = input.queries.iter().copied().map(range_min).collect();
        let space = 8 * rmq.size_bytes() + 8 * std::mem::size_of_val(&rmq);

        (results, space)
    }

    fn write_results<T: fmt::Display>(out_path: &Path, results: &[T]) -> Result<()> {
        let out_file = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(out_path)?;
        let mut writer = BufWriter::new(out_file);
        results.iter().try_for_each(|x| writeln!(writer, "{x}"))?;
        Ok(())
    }

    fn run() -> Result<TestResults> {
        let args = Arguments::parse()?;
        let input_file = fs::OpenOptions::new().read(true).open(args.in_path)?;
        let input_reader = BufReader::new(input_file);

        let (space, time) = match args.algo {
            Algorithm::Predecessor => {
                let input = PredecessorInput::parse(input_reader)?;
                let ((results, space), time) = run_timed(|| run_pd(input));
                write_results(&args.out_path, &results)?;
                (space, time)
            },
            Algorithm::RangeMinimumQuery => {
                let input = RMQInput::parse(input_reader)?;
                let ((results, space), time) = run_timed(|| run_rmq(input));
                write_results(&args.out_path, &results)?;
                (space, time)
            },
        };
        Ok(TestResults { algo: args.algo, time, space })
    }

    run().map_err(|err| err.to_string())
}

/// A counterpart to [`std::mem::size_of`] for the dynamic size of types.
pub trait AllocationSize {
    /// Returns an estimate for the allocation size of `self` in bytes.
    fn size_bytes(&self) -> usize;
}

impl<T> AllocationSize for [T] {
    fn size_bytes(&self) -> usize { std::mem::size_of_val(self) }
}

impl<T> AllocationSize for Vec<T> {
    fn size_bytes(&self) -> usize { std::mem::size_of_val(self.as_slice()) }
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
        let _ = writeln!(
            std::io::stdout(),
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

// todo allow for empty lines in input
pub struct PredecessorInput {
    values: Vec<u64>,
    queries: Vec<u64>,
}

impl PredecessorInput {
    pub fn parse(reader: impl BufRead) -> Result<Self> {
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
    pub fn parse(reader: impl BufRead) -> Result<Self> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            let line = line?;
            let (left, right) = line.split_once(',').ok_or(ParseError::NotARange)?;
            let (lower, upper) = (left.parse()?, right.parse()?);
            if lower > upper {
                return Err(ParseError::EmptyRange.into());
            }
            queries.push((lower, upper))
        }

        Ok(Self { values, queries })
    }
}

fn parse_values(lines: &mut Lines<impl BufRead>) -> Result<Vec<u64>> {
    let line = lines.next().ok_or(ParseError::MissingValue)??;
    let len = line.parse::<usize>()?;

    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let line = lines.next().ok_or(ParseError::MissingValue)??;
        values.push(line.parse()?);
    }

    Ok(values)
}
