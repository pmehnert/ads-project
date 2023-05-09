pub trait RMQ {
    /// Constructs the RMQ data structure for `values`.
    fn new(values: &[u64]) -> Self;

    /// Returns the size of the data structure in bits.
    fn size_bits(&self) -> usize;

    /// Returns `arg min {A[i] | lower <= i <= upper }`:
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize>;
}

/// Implements the naive approach for `O(1)` range minimum queries using `O(n²)`
/// space.
pub struct Naive {
    table: Vec<usize>,
    len: usize,
}

impl RMQ for Naive {
    /// Cosntructs the RMQ data structure using dynamic programming.
    /// Starting with ranges of length `1`, the minimum for all ranges with
    /// length `n+1` are trivially calculated using ranges of length `n`.
    /// Time complexity of the construction algorithm is in `O(n²)`
    fn new(values: &[u64]) -> Self {
        // The lookup table has length N + N-1 + ... + 1
        let mut table = vec![0; values.len() * (values.len() + 1) / 2];

        let (mut front, mut tail) = table.split_at_mut(values.len());

        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i);

        for n in 1..values.len() {
            let iter = front.iter().zip(tail.iter_mut()).enumerate();
            for (i, (&min, dst)) in iter.take(values.len() - n) {
                *dst = std::cmp::min_by_key(min, i + n, |x| values[*x]);
            }

            (front, tail) = tail.split_at_mut(values.len() - n);
        }
        Self { table, len: values.len() }
    }

    fn size_bits(&self) -> usize {
        // todo remember to make this generic
        8 * std::mem::size_of::<usize>() * self.table.len()
    }

    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.len {
            // Calculate N + (N-1) + ... + (N-(b-a)+1)
            // where N := self.len, a := lower, b := upper
            let from = self.len - (upper - lower) + 1;
            let offset = (self.len + 1 - from) * (from + self.len) / 2;

            Some(self.table[offset + lower])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::rmq::{Naive, RMQ};

    #[test]
    fn test_example() {
        let rmq = Naive::new(&[1, 0, 3, 7]);

        assert_eq!(1, rmq.range_min(0, 3).unwrap());

        assert_eq!(1, rmq.range_min(0, 2).unwrap());
        assert_eq!(1, rmq.range_min(1, 3).unwrap());

        assert_eq!(1, rmq.range_min(0, 1).unwrap());
        assert_eq!(1, rmq.range_min(1, 2).unwrap());
        assert_eq!(2, rmq.range_min(2, 3).unwrap());

        assert_eq!(0, rmq.range_min(0, 0).unwrap());
        assert_eq!(1, rmq.range_min(1, 1).unwrap());
        assert_eq!(2, rmq.range_min(2, 2).unwrap());
        assert_eq!(3, rmq.range_min(3, 3).unwrap());
    }
}
