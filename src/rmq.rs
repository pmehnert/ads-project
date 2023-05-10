//! Data structures used to accelerate Range Minimum Queries (RMQ).
//!
//! # Range Minimum Queries
//!
//! TODO

use std::iter::zip;

use crate::int::{AsIndex, IndexInt};

/// Returns true iff `values` can be indexed using `Idx`.
pub fn fits_index<Idx: IndexInt>(values: &[u64]) -> bool {
    Idx::try_from(values.len().saturating_sub(1)).is_ok()
}

/// A trait for types that can be used to answer RMQs.
pub trait RangeMinimum {
    // todo size  bits does not belong here. move to global trait
    /// Returns the size of the data structure in bits.
    fn size_bits(&self) -> usize;

    /// Returns `RMQ(lower, upper)` or [`None`] if the range is empty or out of bounds.
    ///
    /// See the module level [documentation] for more information.
    ///
    /// [documentation]: crate::rmq#range-minimum-queries
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize>;
}

/// The naive approach for answering RMQs in `O(1)` time.
///
/// Stores the answer for every possible query in `table` using `O(n²)` space.
/// The answers are stored in `n` conceptual segments of size `n`, `n-1`, ..., `1`.
/// The segments store answeres for ranges of length `1`, `2`, ..., `n` respectively.
#[derive(Debug, Clone)]
pub struct Naive<'a, Idx: IndexInt> {
    table: Vec<Idx>,
    values: &'a [u64],
}

impl<'a, Idx: IndexInt> Naive<'a, Idx> {
    /// Cosntructs the RMQ data structure using dynamic programming.
    ///
    /// Starting with ranges of length `1`, the RMQ for each ranges of length
    /// `n+1` is calculated from a range of length `n` and a single additional
    /// position. Time complexity of the construction algorithm is in `O(n²)`
    ///
    /// # Panics
    ///
    /// Panics if `values` cannot be index with `Idx` (see also [`fits_index`]).
    pub fn new(values: &'a [u64]) -> Self {
        if !fits_index::<Idx>(values) {
            index_too_small_fail::<Idx>(values.len())
        }

        // The lookup table has length N + (N-1) + ... + 1
        let mut table = vec![Idx::ZERO; values.len() * (values.len() + 1) / 2];

        let (mut front, mut tail) = table.split_at_mut(values.len());
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i.as_index());

        for n in 1..values.len() {
            let iter = front.iter().zip(tail.iter_mut()).enumerate();
            for (i, (&min, dst)) in iter.take(values.len() - n) {
                *dst = arg_min(min.as_usize(), i + n, values).as_index();
            }

            (front, tail) = tail.split_at_mut(values.len() - n);
        }
        Self { table, values }
    }
}

impl<'a, Idx: IndexInt> RangeMinimum for Naive<'a, Idx> {
    fn size_bits(&self) -> usize { 8 * std::mem::size_of::<Idx>() * self.table.len() }

    /// Retrieves `RMQ(lower, upper)` from the lookup table.
    ///
    /// First determines the index of the segment containing RMQ values for
    /// ranges of length `upper - lower + 1`. Then retrieves the RMQ value from
    /// the lookup table.
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.values.len() {
            let len = self.values.len();

            // Calculate N + (N-1) + ... + (N-(b-a)+1)
            // where N := self.len, a := lower, b := upper
            let from = len - (upper - lower) + 1;
            let offset = (len + 1 - from) * (from + len) / 2;

            Some(self.table[offset + lower].as_usize())
        } else {
            None
        }
    }
}

/// The sparse table approach for answering RMQs in `O(1)` time.
///
/// Stores the answeres for every possible query whose length is a power of two
/// using `O(n log n)` space.
///
/// # References
///
/// \[1\] Michael A. Bender et al. _Lowest Common Ancestors in Trees and Directed
/// Acyclic Graphs_. DOI: [10.5555/1120060.1712350]
///
/// [10.5555/1120060.1712350]: https://dl.acm.org/doi/10.5555/1120060.1712350
pub struct Sparse<'a, Idx: IndexInt> {
    table: Vec<Idx>,
    values: &'a [u64],
}

impl<'a, Idx: IndexInt> Sparse<'a, Idx> {
    /// Constructs the sparse table data structure using dynamic programming.
    ///
    /// Starting with ranges of length `1`, the RMQ for each range of length
    /// `2^(k+1)` is computed from two consecutive ranges of length `2^k`.
    /// Time complexity of the construction algorithm is in `O(n log n)`.
    ///
    /// # Panics
    ///
    /// Panics if `values` cannot be index with `Idx` (see also [`fits_index`]).
    pub fn new(values: &'a [u64]) -> Self {
        if !fits_index::<Idx>(values) {
            index_too_small_fail::<Idx>(values.len());
        }
        if values.is_empty() {
            return Self { table: Vec::new(), values };
        }

        let (len, log_len) = (values.len(), values.len().ilog2() as usize);
        let mut table = vec![Idx::ZERO; len * (log_len + 1)];

        let (mut front, tail) = table.split_at_mut(len);
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i.as_index());

        for (k, chunk) in zip(1.., tail.chunks_exact_mut(len)) {
            let num = len - (1 << k) + 1;
            for (i, dst) in chunk[..num].iter_mut().enumerate() {
                let (a, b) = (front[i], front[i + (1 << k - 1)]);
                *dst = arg_min(a.as_usize(), b.as_usize(), values).as_index();
            }
            front = chunk;
        }
        Self { table, values }
    }
}

impl<'a, Idx: IndexInt> RangeMinimum for Sparse<'a, Idx> {
    fn size_bits(&self) -> usize { 8 * std::mem::size_of::<Idx>() * self.table.len() }

    /// Calculates `RMQ(lower, upper)` using two overlapping ranges from the lookup table.
    ///
    /// Both ranges have size `2^k` where `k` is maximal and `2^k` still fits
    /// into the original range. The first range starts at `lower` and the
    /// second ends at `upper`.
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.values.len() {
            let log_len = usize::ilog2(upper - lower + 1) as usize;
            let offset = self.values.len() * log_len;
            let left = self.table[offset + lower];
            let right = self.table[offset + upper + 1 - (1 << log_len)];

            Some(arg_min(left.as_usize(), right.as_usize(), &self.values))
        } else {
            None
        }
    }
}

#[doc(hidden)]
fn arg_min(rhs: usize, lhs: usize, values: &[u64]) -> usize {
    std::cmp::min_by_key(lhs, rhs, |x| values[*x])
}

#[doc(hidden)]
#[cold]
#[inline(never)]
fn index_too_small_fail<Idx: IndexInt>(len: usize) -> ! {
    panic!("index type too small: the maximum is {} but the length is {}", Idx::MAX, len);
}

#[cfg(test)]
mod test {
    use crate::rmq::{Naive, RangeMinimum, Sparse};

    #[test]
    fn test_empty() {
        let _rmq = Naive::<usize>::new(&[]);
        let _rmq = Sparse::<usize>::new(&[]);
    }

    #[test]
    fn test_one_element() {
        let numbers = &[42];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_spec_example() {
        let numbers = &[1, 0, 3, 7];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_simple_example() {
        let numbers = &[99, 32, 60, 90, 22, 26, 9];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_index_type_limits() {
        let numbers = (0..=255).collect::<Vec<_>>();
        test_exhaustive(Naive::<u8>::new(&numbers), &numbers);
        test_exhaustive(Sparse::<u8>::new(&numbers), &numbers);
    }

    #[should_panic]
    #[test]
    fn test_index_too_small_naive() {
        let _ = Naive::<u8>::new(&(0..=256).collect::<Vec<_>>());
    }

    #[should_panic]
    #[test]
    fn test_index_too_small_sparse() {
        let _ = Sparse::<u8>::new(&(0..=256).collect::<Vec<_>>());
    }

    fn test_exhaustive(rmq: impl RangeMinimum, values: &[u64]) {
        for a in 0..values.len() {
            for b in a..values.len() {
                let min = (a..=b).min_by_key(|i| values[*i]).unwrap();
                assert_eq!(min, rmq.range_min(a, b).unwrap());
            }
        }
    }
}
