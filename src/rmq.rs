use std::iter::zip;

/// Provides functionality for types that can be used to accelerate RMQ queries.
pub trait RMQ<'a> {
    // todo this probably doesn't belong in this trait
    /// Constructs the RMQ data structure for `values`.
    fn new(values: &'a [u64]) -> Self;

    /// Returns the size of the data structure in bits.
    fn size_bits(&self) -> usize;

    /// Returns `arg min {A[i] | lower <= i <= upper }`:
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize>;
}

/// Implements the naive approach for `O(1)` range minimum queries using `O(n²)`
/// space.
pub struct Naive<'a> {
    table: Vec<usize>,
    values: &'a [u64],
}

impl<'a> RMQ<'a> for Naive<'a> {
    /// Cosntructs the RMQ data structure using dynamic programming.
    /// Starting with ranges of length `1`, the minimum for all ranges with
    /// length `n+1` are trivially calculated using ranges of length `n`.
    /// Time complexity of the construction algorithm is in `O(n²)`
    fn new(values: &'a [u64]) -> Self {
        // The lookup table has length N + N-1 + ... + 1
        let mut table = vec![0; values.len() * (values.len() + 1) / 2];

        let (mut front, mut tail) = table.split_at_mut(values.len());
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i);

        for n in 1..values.len() {
            let iter = front.iter().zip(tail.iter_mut()).enumerate();
            for (i, (&min, dst)) in iter.take(values.len() - n) {
                *dst = arg_min(min, i + n, values);
            }

            (front, tail) = tail.split_at_mut(values.len() - n);
        }
        Self { table, values }
    }

    fn size_bits(&self) -> usize {
        // todo remember to make this generic
        8 * std::mem::size_of::<usize>() * self.table.len()
    }

    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.values.len() {
            let len = self.values.len();

            // Calculate N + (N-1) + ... + (N-(b-a)+1)
            // where N := self.len, a := lower, b := upper
            let from = len - (upper - lower) + 1;
            let offset = (len + 1 - from) * (from + len) / 2;

            Some(self.table[offset + lower])
        } else {
            None
        }
    }
}

pub struct Log<'a> {
    table: Vec<usize>,
    values: &'a [u64],
}

impl<'a> RMQ<'a> for Log<'a> {
    fn new(values: &'a [u64]) -> Self {
        if values.is_empty() {
            return Self { table: Vec::new(), values };
        }
        // todo would there be an efficient way of computing this?
        //   1       2       4           log n
        // (N-0) + (N-1) + (N-3) + ... + (N-?)
        // 0 + 1 + 3 + ... + log n

        let (len, log_len) = (values.len(), values.len().ilog2() as usize);
        let mut table = vec![0; len * (log_len + 1)];

        let (mut front, tail) = table.split_at_mut(len);
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i);

        for (k, chunk) in zip(1.., tail.chunks_exact_mut(len)) {
            let num = len - (1 << k) + 1;
            for (i, dst) in chunk[..num].iter_mut().enumerate() {
                let (a, b) = (front[i], front[i + (1 << k - 1)]);
                *dst = arg_min(a, b, values);
            }
            front = chunk;
        }
        Self { table, values }
    }

    fn size_bits(&self) -> usize {
        // todo remember to make this generic
        8 * std::mem::size_of::<usize>() * self.table.len()
    }

    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.values.len() {
            let log_len = usize::ilog2(upper - lower + 1) as usize;
            let offset = self.values.len() * log_len;
            let left = self.table[offset + lower];
            let right = self.table[offset + upper + 1 - (1 << log_len)];

            Some(arg_min(left, right, &self.values))
        } else {
            None
        }
    }
}

fn arg_min(rhs: usize, lhs: usize, values: &[u64]) -> usize {
    std::cmp::min_by_key(lhs, rhs, |x| values[*x])
}

#[cfg(test)]
mod test {
    use crate::rmq::{Log, Naive, RMQ};

    fn test_exhaustive<'a>(rmq: impl RMQ<'a>, values: &[u64]) {
        for a in 0..values.len() {
            for b in a..values.len() {
                let min = (a..=b).min_by_key(|i| values[*i]).unwrap();
                assert_eq!(min, rmq.range_min(a, b).unwrap());
            }
        }
    }

    #[test]
    fn test_empty() {
        let _rmq = Naive::new(&[]);
        let _rmq = Log::new(&[]);
    }

    #[test]
    fn test_one_element() {
        let numbers = &[42];
        test_exhaustive(Log::new(numbers), numbers);
        test_exhaustive(Naive::new(numbers), numbers);
    }

    #[test]
    fn test_spec_example() {
        let numbers = &[1, 0, 3, 7];
        test_exhaustive(Log::new(numbers), numbers);
        test_exhaustive(Naive::new(numbers), numbers);
    }

    #[test]
    fn test_simple_example() {
        let numbers = &[99, 32, 60, 90, 22, 26, 9];
        test_exhaustive(Log::new(numbers), numbers);
        test_exhaustive(Naive::new(numbers), numbers);
    }
}
