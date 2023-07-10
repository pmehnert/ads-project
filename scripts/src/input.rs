use std::{
    cmp,
    error::Error,
    io::{self, BufRead, Lines},
    iter,
    ops::RangeInclusive,
};

use rand::{distributions::Uniform, prelude::Distribution};

#[derive(Debug, Clone)]
pub struct PredecessorInput {
    pub values: Vec<u64>,
    pub queries: Vec<u64>,
}

impl PredecessorInput {
    pub fn new(
        num_values: usize,
        num_queries: usize,
        range: RangeInclusive<u64>,
    ) -> Self {
        assert!(num_values > 0);

        let mut rng = rand::thread_rng();
        let mut sampler = Uniform::from(range).sample_iter(&mut rng);

        let values: Vec<_> = sampler.by_ref().take(num_values).collect();
        let queries: Vec<_> = sampler.take(num_queries).collect();

        assert_eq!(num_values, values.len());
        assert_eq!(num_queries, queries.len());

        Self { values, queries }
    }

    pub fn parse(reader: impl BufRead) -> Result<Self, Box<dyn Error>> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            queries.push(line?.parse()?);
        }

        Ok(Self { values, queries })
    }

    pub fn write(&self, writer: &mut impl io::Write) -> io::Result<()> {
        writeln!(writer, "{}", self.values.len())?;
        self.values.iter().try_for_each(|value| writeln!(writer, "{value}"))?;
        self.queries.iter().try_for_each(|query| writeln!(writer, "{query}"))
    }
}

#[derive(Debug, Clone)]
pub struct RangeMinimumInput {
    pub values: Vec<u64>,
    pub queries: Vec<(usize, usize)>,
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

        assert_eq!(num_values, values.len());
        assert_eq!(num_queries, queries.len());

        Self { values, queries }
    }

    pub fn parse(reader: impl BufRead) -> Result<Self, Box<dyn Error>> {
        let mut lines = reader.lines();
        let values = parse_values(&mut lines)?;

        let mut queries = Vec::with_capacity(lines.size_hint().0);
        for line in lines {
            let line = line?;
            let (left, right) = line.split_once(',').ok_or("not a range")?;
            let (lower, upper) = (left.parse()?, right.parse()?);
            if lower > upper {
                return Err("range is empty".into());
            }
            queries.push((lower, upper))
        }

        Ok(Self { values, queries })
    }

    pub fn write(&self, writer: &mut impl io::Write) -> io::Result<()> {
        writeln!(writer, "{}", self.values.len())?;
        self.values.iter().try_for_each(|value| writeln!(writer, "{value}"))?;
        self.queries.iter().try_for_each(|(lo, hi)| writeln!(writer, "{lo},{hi}"))
    }
}

fn parse_values(lines: &mut Lines<impl BufRead>) -> Result<Vec<u64>, Box<dyn Error>> {
    let line = lines.next().ok_or("missing value")??;
    let len = line.parse::<usize>()?;

    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let line = lines.next().ok_or("missing value")??;
        values.push(line.parse()?);
    }

    Ok(values)
}
