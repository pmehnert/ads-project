pub mod bitvec;
pub mod cartesian;
pub mod int;
pub mod packed;
pub mod predecessor;
pub mod rmq;

use std::{
    error::Error,
    fmt, fs, hint,
    io::{BufRead, BufReader, BufWriter, Lines, Write},
    path::{Path, PathBuf},
    process::{ExitCode, Termination},
    time::{Duration, Instant},
};

use crate::{
    int::{AsHalfSize, IndexInt},
    predecessor::{BinarySearch, EliasFano, Predecessor},
    rmq::{fits_index, Cartesian, Naive, RangeMinimum, Sparse},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn div_ceil(lhs: usize, rhs: usize) -> usize { lhs.saturating_add(rhs - 1) / rhs }

pub fn main() -> std::result::Result<TestResults, String> {
    #[inline(never)]
    fn run_timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
        let before = Instant::now();
        let result = hint::black_box(f());
        let elapsed = before.elapsed();
        (result, elapsed)
    }

    fn run_pd(
        out: &Path,
        input: PredecessorInput,
        algo: Algorithm,
    ) -> Result<Measurements> {
        match algo {
            Algorithm::EliasFano => run_pd_with_algo(out, &input, EliasFano::new),
            Algorithm::BinarySearch => run_pd_with_algo(out, &input, BinarySearch::new),
            algo => unreachable!("{algo}"),
        }
    }

    fn run_pd_with_algo<'a, Algo: Predecessor + AllocationSize>(
        out: &Path,
        input: &'a PredecessorInput,
        init: impl FnOnce(&'a [u64]) -> Algo,
    ) -> Result<Measurements> {
        let (pd, init_time) = run_timed(|| hint::black_box(init(&input.values)));

        let predecessor = |&query| pd.predecessor(query).unwrap_or(u64::MAX);
        let query_time = run_queries(out, &input.queries, predecessor)?;

        let (values, queries) = (input.values.len(), input.queries.len());
        let space = 8 * pd.size_bytes() + std::mem::size_of_val(&pd);

        Ok(Measurements { values, queries, init_time, query_time, space })
    }

    fn run_rmq(out: &Path, input: RMQInput, algo: Algorithm) -> Result<Measurements> {
        match &input.values {
            xs if fits_index::<u16>(xs) => run_rmq_with_index::<u16>(out, input, algo),
            xs if fits_index::<u32>(xs) => run_rmq_with_index::<u32>(out, input, algo),
            _ => run_rmq_with_index::<usize>(out, input, algo),
        }
    }

    fn run_rmq_with_index<Idx>(
        out: &Path,
        input: RMQInput,
        algo: Algorithm,
    ) -> Result<Measurements>
    where
        Idx: IndexInt + AsHalfSize,
        Idx::HalfSize: IndexInt,
    {
        match algo {
            Algorithm::Naive => run_rmq_with_algo(out, &input, Naive::<Idx>::new),
            Algorithm::Sparse => run_rmq_with_algo(out, &input, Sparse::<Idx, &[_]>::new),
            Algorithm::Cartesian => run_rmq_with_algo(out, &input, Cartesian::<Idx>::new),
            algo => unreachable!("{algo}"),
        }
    }

    fn run_rmq_with_algo<'a, Algo: RangeMinimum<Output = usize> + AllocationSize>(
        out: &Path,
        input: &'a RMQInput,
        init: impl FnOnce(&'a [u64]) -> Algo,
    ) -> Result<Measurements> {
        let (rmq, init_time) = run_timed(|| hint::black_box(init(&input.values)));
        let range_min = |&(lower, upper)| rmq.range_min(lower, upper).unwrap();
        let query_time = run_queries(out, &input.queries, range_min)?;

        let (values, queries) = (input.values.len(), input.queries.len());
        let space = 8 * rmq.size_bytes() + 8 * std::mem::size_of_val(&rmq);

        Ok(Measurements { values, queries, init_time, query_time, space })
    }


    fn run_queries<'a, T, R: fmt::Display>(
        #[allow(unused)] out: &Path,
        queries: &'a [T],
        f: impl Fn(&'a T) -> R,
    ) -> Result<Duration> {
        #[cfg(feature = "output")]
        let mut results = Vec::<R>::new();

        let (_, query_time) = run_timed(|| {
            for query in queries {
                #[cfg(feature = "output")]
                results.push(f(query));

                #[cfg(not(feature = "output"))]
                hint::black_box(f(query));
            }
        });

        #[cfg(feature = "output")]
        write_results(out, &results)?;

        Ok(query_time)
    }

    #[allow(dead_code)]
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
        let reader = BufReader::new(input_file);

        let measurements = match args.problem {
            Problem::Predecessor => {
                let input = PredecessorInput::parse(reader)?;
                run_pd(&args.out_path, input, args.algo)?
            },
            Problem::RangeMinimumQuery => {
                let input = RMQInput::parse(reader)?;
                run_rmq(&args.out_path, input, args.algo)?
            },
        };

        let Arguments { name, problem, algo, .. } = args;
        Ok(TestResults { name, problem, algo, measurements })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Problem {
    Predecessor,
    RangeMinimumQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    BinarySearch,
    EliasFano,
    Naive,
    Sparse,
    Cartesian,
}

#[derive(Debug, Clone)]
pub struct Arguments {
    name: String,
    problem: Problem,
    algo: Algorithm,
    in_path: PathBuf,
    #[allow(dead_code)]
    out_path: PathBuf,
}

impl Arguments {
    pub fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);

        let name = args.next().expect("missing name parameter");

        let problem = match args.next().as_deref() {
            Some("pd") => Ok(Problem::Predecessor),
            Some("rmq") => Ok(Problem::RangeMinimumQuery),
            Some(problem) => Err(format!("unknown problem '{problem}'")),
            None => Err("missing problem parameter".to_string()),
        }?;

        let algo = match args.next().as_deref() {
            Some("binary") => Ok(Algorithm::BinarySearch),
            Some("elias_fano") => Ok(Algorithm::EliasFano),
            Some("naive") => Ok(Algorithm::Naive),
            Some("sparse") => Ok(Algorithm::Sparse),
            Some("cartesian") => Ok(Algorithm::Cartesian),
            Some(algo) => Err(format!("unknown algorithm '{algo}'")),
            None => Err("missing algorithm parameter".to_string()),
        }?;

        let pd_algos = [Algorithm::BinarySearch, Algorithm::EliasFano];
        if (problem == Problem::Predecessor) != pd_algos.contains(&algo) {
            return Err(format!("illegal algorithm {algo} for problem {problem}").into());
        }

        let in_path = PathBuf::from(args.next().ok_or("missing input path")?);
        let out_path = PathBuf::from(args.next().ok_or("missing output path")?);

        match args.next() {
            Some(arg) => Err(format!("unexpected parameter '{arg}'").into()),
            None => Ok(Self { name, problem, algo, in_path, out_path }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestResults {
    name: String,
    problem: Problem,
    algo: Algorithm,
    measurements: Measurements,
}

#[derive(Debug, Clone, Copy)]
pub struct Measurements {
    values: usize,
    queries: usize,
    init_time: Duration,
    query_time: Duration,
    space: usize,
}

impl Termination for TestResults {
    fn report(self) -> ExitCode {
        let _ = writeln!(
            std::io::stdout(),
            "RESULT name={} problem={} algo={} values={} queries={} \
                init_time={} query_time={} space={}",
            self.name,
            self.problem,
            self.algo,
            self.measurements.values,
            self.measurements.queries,
            self.measurements.init_time.as_nanos(),
            self.measurements.query_time.as_nanos(),
            self.measurements.space,
        );
        ExitCode::SUCCESS
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Problem::Predecessor => write!(f, "pd"),
            Problem::RangeMinimumQuery => write!(f, "rmq"),
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Algorithm::BinarySearch => write!(f, "binary"),
            Algorithm::EliasFano => write!(f, "elias_fano"),
            Algorithm::Naive => write!(f, "naive"),
            Algorithm::Sparse => write!(f, "sparse"),
            Algorithm::Cartesian => write!(f, "cartesian"),
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
        lines
            .try_for_each::<_, Result<()>>(|line| {
                let line = line?;
                let (left, right) = line.split_once(',').ok_or(ParseError::NotARange)?;
                let (lower, upper) = (left.parse()?, right.parse()?);
                (lower <= upper)
                    .then(|| queries.push((lower, upper)))
                    .ok_or_else(|| ParseError::EmptyRange.into())
            })
            .map(|_| Self { values, queries })
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
