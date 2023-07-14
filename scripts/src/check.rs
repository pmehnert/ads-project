use std::{error::Error, io, iter::zip, str::FromStr};

use crate::input::{PredecessorInput, RangeMinimumInput};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct Output<T> {
    pub results: Vec<T>,
}

impl Output<u64> {
    pub fn check_pd(&self, input: &PredecessorInput) {
        assert_eq!(input.queries.len(), self.results.len());
        for (query, actual) in zip(&input.queries, &self.results) {
            let expected = match input.values.binary_search(query) {
                Ok(idx) => input.values[idx],
                Err(0) => u64::MAX,
                Err(idx) => input.values[idx - 1],
            };

            if expected != *actual {
                panic!("query {query} failed, expected {expected} but got {actual}");
            }
        }
    }
}

impl Output<usize> {
    pub fn check_rmq(&self, input: &RangeMinimumInput) {
        assert_eq!(input.queries.len(), self.results.len());
        for (&(lo, hi), &actual) in zip(&input.queries, &self.results) {
            let (_, expected) = zip(&input.values[lo..=hi], lo..).min().unwrap();
            let expected_value = input.values[expected];
            let actual_value = input.values[actual];

            if expected_value != actual_value {
                panic!(
                    "query [{lo}, {hi}] failed, expected {expected_value} \
                    (e.g. at {expected}) but got {actual_value} (at {actual})"
                );
            }
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
