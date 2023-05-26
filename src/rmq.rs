//! Data structures used to accelerate Range Minimum Queries (RMQ).
//!
//! # Range Minimum Queries
//!
//! TODO

// todo rename this range_minimum?

use std::{borrow::Borrow, iter::zip, num::NonZeroUsize};

use crate::{cartesian::*, int::*, AllocationSize};

/// Returns true iff `values` can be indexed using `Idx`.
pub fn fits_index<Idx: IndexInt>(values: &[u64]) -> bool {
    Idx::try_from(values.len().saturating_sub(1)).is_ok()
}

/// A trait for types that can be used to answer RMQs.
pub trait RangeMinimum {
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
pub struct Naive<Idx> {
    table: Vec<Idx>,
    len: usize,
}

impl<Idx: IndexInt> Naive<Idx> {
    /// Cosntructs the RMQ data structure using dynamic programming.
    ///
    /// Starting with ranges of length `1`, the RMQ for each ranges of length
    /// `n+1` is calculated from a range of length `n` and a single additional
    /// position. Time complexity of the construction algorithm is in `O(n²)`.
    ///
    /// # Panics
    ///
    /// Panics if `values` cannot be index with `Idx` (see also [`fits_index`]).
    pub fn new(values: &[u64]) -> Self {
        if !fits_index::<Idx>(values) {
            index_too_small_fail::<Idx>(values.len())
        }

        // The lookup table has length N + (N-1) + ... + 1.
        let mut table = vec![Idx::ZERO; values.len() * (values.len() + 1) / 2];

        let (mut front, mut tail) = table.split_at_mut(values.len());
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i.to_index());

        for n in 1..values.len() {
            let iter = front.iter().zip(tail.iter_mut()).enumerate();
            for (i, (&min, dst)) in iter.take(values.len() - n) {
                *dst = arg_min(min.to_usize(), i + n, values).to_index();
            }

            (front, tail) = tail.split_at_mut(values.len() - n);
        }
        Self { table, len: values.len() }
    }
}

impl<Idx> Default for Naive<Idx> {
    fn default() -> Self { Self { table: Vec::new(), len: 0 } }
}

impl<Idx> AllocationSize for Naive<Idx> {
    fn size_bytes(&self) -> usize { self.table.size_bytes() }
}

impl<Idx: IndexInt> RangeMinimum for Naive<Idx> {
    /// Retrieves `RMQ(lower, upper)` from the lookup table.
    ///
    /// First determines the index of the segment containing RMQ values for
    /// ranges of length `upper - lower + 1`. Then retrieves the RMQ value from
    /// the lookup table.
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.len {
            // Calculate N + (N-1) + ... + (N-(b-a)+1)
            // where N := self.len, a := lower, b := upper
            let from = self.len - (upper - lower) + 1;
            let offset = (self.len + 1 - from) * (from + self.len) / 2;

            Some(self.table[offset + lower].to_usize())
        } else {
            None
        }
    }
}

/// The sparse table \[1\] approach for answering RMQs in `O(1)` time.
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
#[derive(Debug, Clone)]
pub struct Sparse<Idx, Values> {
    table: Vec<Idx>,
    values: Values,
}

// todo sparse table needs to linearized
impl<Idx: IndexInt, Values: Borrow<[u64]>> Sparse<Idx, Values> {
    /// Constructs the sparse table data structure using dynamic programming.
    ///
    /// Starting with ranges of length `1`, the RMQ for each range of length
    /// `2^(k+1)` is computed from two consecutive ranges of length `2^k`.
    /// Time complexity of the construction algorithm is in `O(n log n)`.
    ///
    /// # Panics
    ///
    /// Panics if `values` cannot be index with `Idx` (see also [`fits_index`]).
    pub fn new(values: Values) -> Self {
        let values_ref = values.borrow();
        if !fits_index::<Idx>(values_ref) {
            index_too_small_fail::<Idx>(values_ref.len());
        }
        if values_ref.is_empty() {
            return Self { table: Vec::new(), values };
        }

        let (len, log_len) = (values_ref.len(), values_ref.len().ilog2() as usize);
        let mut table = vec![Idx::ZERO; len * (log_len + 1)];

        let (mut front, tail) = table.split_at_mut(len);
        front.iter_mut().enumerate().for_each(|(i, dst)| *dst = i.to_index());

        for (k, chunk) in zip(1.., tail.chunks_exact_mut(len)) {
            let num = len - (1 << k) + 1;
            for (i, dst) in chunk[..num].iter_mut().enumerate() {
                let (a, b) = (front[i], front[i + (1 << (k - 1))]);
                *dst = arg_min(a.to_usize(), b.to_usize(), values_ref).to_index();
            }
            front = chunk;
        }
        Self { table, values }
    }
}

impl<Idx, Values: Default> Default for Sparse<Idx, Values> {
    fn default() -> Self { Self { table: Vec::new(), values: Default::default() } }
}

impl<Idx, Values: AllocationSize> AllocationSize for Sparse<Idx, Values> {
    fn size_bytes(&self) -> usize { self.table.size_bytes() + self.values.size_bytes() }
}

impl<Idx: IndexInt, Values: Borrow<[u64]>> RangeMinimum for Sparse<Idx, Values> {
    /// Calculates `RMQ(lower, upper)` using two overlapping ranges from the lookup table.
    ///
    /// Both ranges have size `2^k` where `k` is maximal and `2^k` still fits
    /// into the original range. The first range starts at `lower` and the
    /// second ends at `upper`.
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        let values = self.values.borrow();
        if lower <= upper && upper < values.len() {
            let log_len = usize::ilog2(upper - lower + 1) as usize;
            let offset = values.len() * log_len;
            let left = self.table[offset + lower];
            let right = self.table[offset + upper + 1 - (1 << log_len)];

            Some(arg_min(left.to_usize(), right.to_usize(), values))
        } else {
            None
        }
    }
}

/// A data structure using cartesian trees to answer RMQs in `O(1)` time \[1\], \[2\].
///
/// # References
///
/// \[1\] Johannes Fischer and Volker Heun. _Theoretical and Practical Improvements
/// on the RMQ-Problem, with Applications to LCA and LCE._ DOI: [10.1007/11780441_5]
/// \[2\] Erik D. Demaine, et al. _On Cartesian Trees and Range Minimum Queries._
/// DOI: [10.1007/s00453-012-9683-x]
///
/// [10.1007/11780441_5]: https://doi.org/10.1007/11780441_5
/// [10.1007/s00453-012-9683-x]: https://doi.org/10.1007/s00453-012-9683-x
#[derive(Debug, Clone)]
pub struct Cartesian<'a, Idx>
where
    Idx: IndexInt + AsHalfSize,
    Idx::HalfSize: IndexInt,
{
    reprs: Representatives<Idx>,
    table: Table<Idx::HalfSize>,
    types: Vec<Idx::HalfSize>,
    values: &'a [u64],
}
// todo Default implementation

impl<'a, Idx> Cartesian<'a, Idx>
where
    Idx: IndexInt + AsHalfSize,
    Idx::HalfSize: IndexInt,
{
    pub fn new(values: &'a [u64]) -> Self {
        if !fits_index::<Idx>(values) {
            index_too_small_fail::<Idx>(values.len());
        }

        let reprs = Representatives::<Idx>::new(values);
        let block_size = reprs.block_size();
        let table = Table::<Idx::HalfSize>::new(block_size);
        let mut builder = Builder::<Idx::HalfSize>::new(block_size);

        // todo use chunks exact?
        let block_type = |block| builder.build(block).get();
        let types = values.chunks(block_size).map(block_type).collect();

        Self { reprs, table, types, values }
    }
}

impl<'a, Idx> AllocationSize for Cartesian<'a, Idx>
where
    Idx: IndexInt + AsHalfSize,
    Idx::HalfSize: IndexInt,
{
    fn size_bytes(&self) -> usize {
        self.reprs.size_bytes()
            + self.table.size_bytes()
            + self.types.size_bytes()
            + self.values.size_bytes()
    }
}

impl<'a, Idx> RangeMinimum for Cartesian<'a, Idx>
where
    Idx: IndexInt + AsHalfSize,
    Idx::HalfSize: IndexInt,
{
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        // todo use magic numbers for division
        if lower <= upper && upper < self.values.len() {
            let size = self.reprs.block_size();
            let (mut lower_block, lower_offset) = crate::div_mod(lower, size);
            let (mut upper_block, upper_offset) = crate::div_mod(upper, size);

            if lower_block == upper_block {
                let tree = self.types[lower_block.to_usize()];
                let idx = self.table.range_min(tree, lower_offset, upper_offset);
                return Some(lower_block * size + usize::from(idx));
            }

            let lower_min = (lower_offset > 0).then(|| {
                let tree = self.types[lower_block.to_usize()];
                let idx = self.table.range_min(tree, lower_offset, size - 1);
                let min = lower_block * size + usize::from(idx);
                lower_block += 1;
                min
            });
            let upper_min = (upper_offset < size - 1).then(|| {
                let tree = self.types[upper_block.to_usize()];
                let idx = self.table.range_min(tree, 0, upper_offset);
                let min = upper_block * size + usize::from(idx);
                upper_block -= 1;
                min
            });
            let middle_min = (self.reprs)
                .range_min(lower_block, upper_block)
                .map(|(x, i)| x * size + usize::from(i));

            let min = |lhs, rhs| arg_min(rhs, lhs, &self.values);
            [lower_min, upper_min, middle_min].iter().flatten().copied().reduce(min)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Representatives<Idx> {
    block_size: NonZeroUsize,
    blocks: Sparse<Idx, Vec<u64>>,
    offsets: Vec<u8>,
}

impl<Idx: IndexInt> Representatives<Idx> {
    // todo the codegen here kinda sucks
    pub fn new(values: &[u64]) -> Self {
        let block_size = values.len().next_power_of_two().ilog2() / 4;
        let block_size = u32::max(1, block_size) as usize;

        fn chunk_min(block: &[u64]) -> Option<(u8, u64)> {
            zip(0.., block.iter().copied()).min_by_key(|(_, x)| *x)
        }

        let len = crate::div_ceil(values.len(), block_size);
        let (mut offsets, mut blocks) = (vec![0; len], vec![0; len]);

        let mut chunks = values.chunks_exact(block_size);
        for (block, dst) in zip(&mut chunks, zip(&mut offsets, &mut blocks)) {
            (*dst.0, *dst.1) = chunk_min(block).unwrap();
        }
        if let Some((idx, min)) = chunk_min(chunks.remainder()) {
            *offsets.last_mut().unwrap() = idx;
            *blocks.last_mut().unwrap() = min;
        }

        let block_size = block_size.try_into().unwrap();
        Self { block_size, blocks: Sparse::new(blocks), offsets }
    }

    pub fn block_size(&self) -> usize { self.block_size.get() }

    pub fn range_min(&self, lower: usize, upper: usize) -> Option<(usize, u8)> {
        let block = self.blocks.range_min(lower, upper)?;
        let offset = self.offsets[block];
        Some((block, offset))
    }
}

impl<Idx> Default for Representatives<Idx> {
    fn default() -> Self {
        let block_size = NonZeroUsize::new(1).unwrap();
        Self { block_size, blocks: Default::default(), offsets: Vec::new() }
    }
}

impl<Idx> AllocationSize for Representatives<Idx> {
    fn size_bytes(&self) -> usize { self.blocks.size_bytes() + self.offsets.size_bytes() }
}

#[doc(hidden)]
fn arg_min(lhs: usize, rhs: usize, values: &[u64]) -> usize {
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
    use super::*;

    #[test]
    fn test_empty() {
        let _ = Naive::<usize>::new(&[]);
        let _ = Sparse::<usize, _>::new(&[][..]);
        let _ = Cartesian::<usize>::new(&[]);
    }

    #[test]
    fn test_one_element() {
        let numbers = &[42][..];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize, _>::new(numbers), numbers);
        test_exhaustive(Cartesian::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_spec_example() {
        let numbers = &[1, 0, 3, 7][..];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize, _>::new(numbers), numbers);
        test_exhaustive(Cartesian::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_simple_example() {
        let numbers = &[99, 32, 60, 90, 22, 26, 9][..];
        test_exhaustive(Naive::<usize>::new(numbers), numbers);
        test_exhaustive(Sparse::<usize, _>::new(numbers), numbers);
        test_exhaustive(Cartesian::<usize>::new(numbers), numbers);
    }

    #[test]
    fn test_index_type_limits() {
        let numbers = (0..=255).collect::<Vec<_>>();
        test_exhaustive(Naive::<u8>::new(&numbers), &numbers);
        test_exhaustive(Sparse::<u8, _>::new(&*numbers), &numbers);
        test_exhaustive(Cartesian::<u8>::new(&numbers), &numbers);
    }

    #[should_panic]
    #[test]
    fn test_index_too_small_naive() {
        let _ = Naive::<u8>::new(&(0..=256).collect::<Vec<_>>());
    }

    #[should_panic]
    #[test]
    fn test_index_too_small_sparse() {
        let _ = Sparse::<u8, _>::new((0..=256).collect::<Vec<_>>());
    }

    #[should_panic]
    #[test]
    fn test_index_too_small_cartesian() {
        let _ = Cartesian::<u8>::new(&(0..=256).collect::<Vec<_>>());
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
