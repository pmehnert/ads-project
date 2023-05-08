use std::{
    error::Error,
    fmt, fs,
    io::{self, Write},
    ops::Range,
    process::{ExitCode, Termination},
    time::{Duration, Instant},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn main() -> Result<TestResults> {
    fn run_timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
        let before = Instant::now();
        let result = std::hint::black_box(f());
        let elapsed = before.elapsed();
        (result, elapsed)
    }

    fn run_pd(_input: PredecessorInput) -> (Vec<u64>, usize) { todo!() }

    fn run_rmq(_input: RMQInput) -> (Vec<u64>, usize) { todo!() }

    let mut args = std::env::args().skip(1);

    let algo = match args.next().as_deref() {
        Some("pd") => Ok(Algorithm::Predecessor),
        Some("rmq") => Ok(Algorithm::RangeMinimumQuery),
        Some(algo) => Err(format!("unknown algorithm '{}'", algo)),
        None => Err(format!("missing algorithm parameter")),
    }?;

    let in_path = args.next().ok_or_else(|| "missing input path")?;
    let out_path = args.next().ok_or_else(|| "missing output path")?;

    if let Some(arg) = args.next() {
        return Err(format!("unexpected parameter '{}'", arg).into());
    }

    let input_file = fs::OpenOptions::new().read(true).open(in_path)?;
    let input_reader = io::BufReader::new(input_file);

    // todo should parsing be part of the timed segment
    let ((output, space), time) = match algo {
        Algorithm::Predecessor => {
            let input = PredecessorInput::read(input_reader)?;
            run_timed(|| run_pd(input))
        },
        Algorithm::RangeMinimumQuery => {
            let input = RMQInput::read(input_reader)?;
            run_timed(|| run_rmq(input))
        },
    };

    let mut out_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        // todo .truncate(true) ?
        .open(&out_path)?;

    output.iter().try_for_each(|x| writeln!(out_file, "{}", x))?;

    Ok(TestResults { algo, time, space })
}

#[derive(Debug, Clone, Copy)]
pub struct TestResults {
    algo: Algorithm,
    time: Duration,
    space: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Predecessor,
    RangeMinimumQuery,
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
}

impl Error for ParseError {}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingValue => write!(f, "mssing a value in the input file"),
            ParseError::NotARange => write!(f, "expected a range in the input file"),
        }
    }
}

pub struct PredecessorInput {
    values: Vec<u64>,
    queries: Vec<u64>,
}

impl PredecessorInput {
    pub fn read(reader: impl io::BufRead) -> Result<Self> {
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
    queries: Vec<Range<u64>>,
}

impl RMQInput {
    pub fn read(reader: impl io::BufRead) -> Result<Self> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            let line = line?;
            let (left, right) = line.split_once(",").ok_or(ParseError::NotARange)?;
            let (start, end) = (left.parse()?, right.parse()?);
            queries.push(Range { start, end })
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
