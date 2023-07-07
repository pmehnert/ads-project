use std::{collections::HashSet, fs, io, ops::RangeInclusive};

use clap::Parser;
use rand::{distributions::Uniform, prelude::Distribution};

#[allow(dead_code)]
pub fn main() -> io::Result<()> {
    let args = Arguments::parse();

    let range = args.lower.unwrap_or(0)..=args.upper.unwrap_or(u64::MAX);
    let input = PredecessorInput::new(args.num_values, args.num_queries, range);

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
pub struct PredecessorInput {
    values: Vec<u64>,
    queries: Vec<u64>,
    output: Vec<u64>,
}

impl PredecessorInput {
    pub fn new(
        num_values: usize,
        num_queries: usize,
        range: RangeInclusive<u64>,
    ) -> Self {
        assert!(num_values > 0);

        let mut rng = rand::thread_rng();

        let mut value_set = HashSet::with_capacity(num_values);
        let value_dist = Uniform::from(range.clone());
        while value_set.len() < num_values {
            value_set.insert(value_dist.sample(&mut rng));
        }

        let mut values: Vec<_> = value_set.into_iter().collect();
        assert_eq!(num_values, values.len());
        values.sort();

        let query_dist = Uniform::new_inclusive(*values.first().unwrap(), *range.end());
        let queries: Vec<_> =
            query_dist.sample_iter(&mut rng).take(num_queries).collect();

        let output = queries
            .iter()
            .map(|query| match values.binary_search(query) {
                Ok(idx) => values[idx],
                Err(0) => unreachable!(),
                Err(idx) => values[idx - 1],
            })
            .collect();

        Self { values, queries, output }
    }

    pub fn write_input(&self, writer: &mut impl io::Write) -> io::Result<()> {
        writeln!(writer, "{}", self.values.len())?;
        self.values.iter().try_for_each(|value| writeln!(writer, "{}", value))?;
        self.queries.iter().try_for_each(|query| writeln!(writer, "{}", query))
    }

    pub fn write_output(&self, writer: &mut impl io::Write) -> io::Result<()> {
        self.output.iter().try_for_each(|value| writeln!(writer, "{}", value))
    }
}
