use std::{cmp, fs, io, iter, ops::RangeInclusive};

use clap::Parser;
use rand::{distributions::Uniform, prelude::*};

#[allow(dead_code)]
pub fn main() -> io::Result<()> {
    let args = Arguments::parse();

    let range = args.lower.unwrap_or(0)..=args.upper.unwrap_or(u64::MAX);
    let input = RangeMinimumInput::new(args.num_values, args.num_queries, range);

    let ifile = fs::OpenOptions::new().create(true).write(true).open(&args.input)?;
    let ofile = fs::OpenOptions::new().create(true).write(true).open(&args.output)?;

    input.write_input(&mut io::BufWriter::new(ifile))?;
    input.write_output(&mut io::BufWriter::new(ofile))?;

    Ok(())
}

#[derive(Debug, Parser)]
pub struct Arguments {
    input: String,
    output: String,
    num_values: usize,
    num_queries: usize,
    lower: Option<u64>,
    upper: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RangeMinimumInput {
    values: Vec<u64>,
    queries: Vec<(usize, usize)>,
    expected: Vec<usize>,
}

impl RangeMinimumInput {
    pub fn new(
        num_values: usize,
        num_queries: usize,
        range: RangeInclusive<u64>,
    ) -> Self {
        assert!(num_values > 0);

        let mut rng = rand::thread_rng();

        let values = Uniform::from(range)
            .sample_iter(&mut rng)
            .take(num_values)
            .collect::<Vec<_>>();

        // todo does this way of choosing ranges make sense?
        let idx_dist = Uniform::new(0usize, values.len());
        let mut sample_idx = || idx_dist.sample(&mut rng);
        let queries = iter::from_fn(|| Some((sample_idx(), sample_idx())))
            .map(|(lhs, rhs)| (cmp::min(lhs, rhs), cmp::max(lhs, rhs)))
            .take(num_queries)
            .collect::<Vec<_>>();

        let expected = queries
            .iter()
            .map(|(lo, hi)| iter::zip(&values[*lo..=*hi], *lo..).min().unwrap().1)
            .collect::<Vec<_>>();

        assert_eq!(num_values, values.len());
        assert_eq!(num_queries, queries.len());
        assert_eq!(num_queries, expected.len());

        Self { values, queries, expected }
    }

    pub fn write_input(&self, writer: &mut impl io::Write) -> io::Result<()> {
        writeln!(writer, "{}", self.values.len())?;
        self.values.iter().try_for_each(|value| writeln!(writer, "{}", value))?;
        self.queries.iter().try_for_each(|(lo, hi)| writeln!(writer, "{}, {}", lo, hi))
    }

    pub fn write_output(&self, writer: &mut impl io::Write) -> io::Result<()> {
        self.expected.iter().try_for_each(|value| writeln!(writer, "{}", value))
    }
}
