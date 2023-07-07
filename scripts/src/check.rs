use std::{error::Error, io, iter::zip, str::FromStr};

use crate::input::{PredecessorInput, RangeMinimumInput};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct Output<T> {
    pub results: Vec<T>,
}

impl Output<u64> {
    pub fn check_pd(&self, input: &PredecessorInput) {
        for (query, actual) in zip(&input.queries, &self.results) {
            let expected = match input.values.binary_search(&query) {
                Ok(idx) => input.values[idx],
                Err(0) => panic!("query {query} has no predecessor"),
                Err(idx) => input.values[idx - 1],
            };
            assert_eq!(expected, *actual);
        }
    }
}

impl Output<usize> {
    pub fn check_rmq(&self, input: &RangeMinimumInput) {
        for (&(lo, hi), &actual) in zip(&input.queries, &self.results) {
            let (_, expected) = zip(&input.values[lo..=hi], lo..).min().unwrap();
            assert_eq!(input.values[expected], input.values[actual]);
        }
    }
}

impl<T: FromStr> Output<T>
where
    T::Err: Error + 'static,
{
    pub fn parse(reader: impl io::BufRead) -> Result<Self> {
        let mut results = Vec::new();
        for line in reader.lines() {
            results.push(line?.parse()?);
        }
        Ok(Self { results })
    }
}
