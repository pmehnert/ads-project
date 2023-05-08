use std::{
    fmt, fs,
    io::{self, Write},
    process::{ExitCode, Termination},
    time::{Duration, Instant},
};

pub fn main() -> Result<TestResults, String> {
    fn run_timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
        let before = Instant::now();
        let result = std::hint::black_box(f());
        let elapsed = before.elapsed();
        (result, elapsed)
    }

    fn run_pd(_input: &[u8]) -> (Vec<u64>, usize) { todo!() }

    fn run_rmq(_input: &[u8]) -> (Vec<u64>, usize) { todo!() }

    fn write_output(output: &[u64], out_path: &str) -> io::Result<()> {
        let mut out_file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            // todo .truncate(true) ?
            .open(out_path)?;

        output.iter().try_for_each(|x| writeln!(out_file, "{}", x))
    }

    let mut args = std::env::args().skip(1);

    let algo = match args.next().as_deref() {
        Some("pd") => Ok(Algorithm::Predecessor),
        Some("rmq") => Ok(Algorithm::RangeMinimumQuery),
        Some(algo) => Err(format!("unknown algorithm '{}'", algo)),
        None => Err(format!("missing algorithm parameter")),
    }?;

    let in_path = args.next().ok_or_else(|| format!("missing input path"))?;
    let out_path = args.next().ok_or_else(|| format!("missing output path"))?;

    if let Some(arg) = args.next() {
        return Err(format!("unexpected parameter '{}'", arg));
    }

    // todo is reading and parsing part of output time?
    let in_file = fs::read(in_path).map_err(|e| e.to_string())?;

    let ((output, space), time) = match algo {
        Algorithm::Predecessor => run_timed(|| run_pd(&in_file)),
        Algorithm::RangeMinimumQuery => run_timed(|| run_rmq(&in_file)),
    };

    write_output(&output, &out_path).map_err(|e| e.to_string())?;

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
