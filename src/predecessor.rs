//! Data structures for accelerating predecessor queries.
//!
//! # Predecessor Problem
//!
//! Given a _universe_ `U = 0..u` and a sorted sequence `A` of `n` integers from
//! `U`, for any `x ∈ U` the _predecessor_ of `x` in `A` is defined as
//!
//! - `pred(x, A) = max {y ∈ A | y <= x}`.

use std::{iter, iter::FusedIterator, slice};

use crate::{
    bitvec::{BitVec, RankSelect},
    packed::PackedArray,
    AllocationSize,
};

/// A trait for types that can be used to answer predecessor queries.
pub trait Predecessor {
    /// Returns the predecessor of `value` in `self`.
    fn predecessor(&self, value: u64) -> Option<u64>;
}

/// An implementation for predecessor queries using binary search.
#[derive(Debug, Default, Clone)]
pub struct BinarySearch<'a> {
    values: &'a [u64],
}

impl<'a> BinarySearch<'a> {
    pub fn new(values: &'a [u64]) -> Self { Self { values } }
}

impl<'a> AllocationSize for BinarySearch<'a> {
    fn size_bytes(&self) -> usize { 0 }
}

impl<'a> Predecessor for BinarySearch<'a> {
    fn predecessor(&self, value: u64) -> Option<u64> {
        match self.values.binary_search(&value) {
            Ok(idx) => Some(self.values[idx]),
            Err(0) => None,
            Err(idx) => Some(self.values[idx - 1]),
        }
    }
}

/// An implementation of the quasi-succinct Elias-Fano coding for answering
/// predecessor queries.
///
/// Each value is split into an upper and a lower half. The upper halves are
/// stored in a bit vector using a kind of unary encoding, the lower halves are
/// stored in a byte packed array.
#[derive(Debug, Default, Clone)]
pub struct EliasFano {
    upper_half: RankSelect<BitVec>,
    lower_half: PackedArray,
    minimum: u64,
    maximum: Option<u64>,
}

impl EliasFano {
    /// Constructs the Elias-Fano coding for the given sequence of integers.
    ///
    /// The elements of `values` must be sorted in ascending order. If they are
    /// not, the program may panic or produce unexpected results, but will not
    /// result in undefined behaviour.
    pub fn new(values: &[u64]) -> Self {
        let minimum = values.first().copied().unwrap_or(u64::MAX);
        let maximum = match values.last() {
            Some(last) => *last,
            None => return Default::default(),
        };

        let unused_bits = maximum.leading_zeros();
        let upper_bits = values.len().next_power_of_two().ilog2();
        let lower_bits = 64u32.saturating_sub(upper_bits + unused_bits).max(1);

        let upper_iter = EliasFanoUpperIter::new(values, lower_bits);
        let upper_half = RankSelect::new(upper_iter.collect());
        let lower_half = PackedArray::new(lower_bits, values.iter().copied());

        Self { upper_half, lower_half, minimum, maximum: Some(maximum) }
    }
}

impl AllocationSize for EliasFano {
    fn size_bytes(&self) -> usize {
        self.upper_half.size_bytes()
            + self.upper_half.bitvec().size_bytes()
            + self.lower_half.size_bytes()
    }
}

impl Predecessor for EliasFano {
    /// Returns the [predecessor](crate::predecessor#predecessor-problem) of
    /// `value` in the integer array encoded by this instance.
    fn predecessor(&self, value: u64) -> Option<u64> {
        if value < self.minimum {
            return None;
        }

        let value_upper = value >> self.lower_half.int_bits();
        let value_lower = value & self.lower_half.mask();
        if value_upper >= self.upper_half.count_zeros() {
            return self.maximum;
        }

        // Find index of values in `lower_half` with equal upper half as `value`
        let bitvec_index = self.upper_half.select0(value_upper);
        let lower_index = bitvec_index as u64 - value_upper;
        if lower_index == 0 {
            return None;
        }

        // Scan through consecutive `1`-bits in `upper_half` bit vector
        let mut bitvec_indexes = (1..=bitvec_index).rev();
        let pred = bitvec_indexes
            .by_ref()
            .take_while(|i| self.upper_half.bitvec().index(*i - 1))
            .zip(self.lower_half.iter().take(lower_index as usize).rev())
            .find(|(_, pred_lower)| *pred_lower <= value_lower);

        match pred {
            Some((_, pred_lower)) => Some((value & !self.lower_half.mask()) | pred_lower),
            None => {
                // Find nearest value with a smaller upper half
                let rank = bitvec_indexes.next()? as u64 - value_upper;
                let bitvec_index = self.upper_half.select1(rank);
                let pred_upper = bitvec_index as u64 - rank;
                let pred_lower = self.lower_half.index(rank as usize);

                Some((pred_upper << self.lower_half.int_bits()) | pred_lower)
            },
        }
    }
}

/// An iterator over the upper half bits for the Elias-Fano coding of a sequence of integers.
#[derive(Debug, Clone)]
struct EliasFanoUpperIter<'a> {
    values: iter::Peekable<slice::Iter<'a, u64>>,
    shift: u32,
    current: u64,
    remaining: usize,
}

impl<'a> EliasFanoUpperIter<'a> {
    fn new(values: &'a [u64], shift: u32) -> Self {
        let maximum = values.last().copied().unwrap_or(0);
        let remaining = values.len() + (maximum >> shift) as usize + 1;
        let values = values.iter().peekable();
        Self { values, shift, current: 0, remaining }
    }
}

impl<'a> Iterator for EliasFanoUpperIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;
            match self.values.next_if(|v| **v >> self.shift == self.current) {
                Some(_) => Some(true),
                None => {
                    self.current += 1;
                    Some(false)
                },
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for EliasFanoUpperIter<'a> {}

impl<'a> FusedIterator for EliasFanoUpperIter<'a> {}

#[cfg(test)]
mod test {
    use std::{iter::repeat, iter::zip, mem::replace};

    use super::{BinarySearch, EliasFano, Predecessor};

    const SLIDES_EXAMPLE: [u64; 10] = [0, 1, 2, 4, 7, 10, 20, 21, 22, 32];

    #[test]
    fn test_empty() {
        let binary = BinarySearch::new(&[]);
        let elias = EliasFano::new(&[]);
        for i in 0..10 {
            assert_eq!(None, binary.predecessor(i));
            assert_eq!(None, elias.predecessor(i));
        }
    }

    #[test]
    fn test_single_element() {
        let pd = EliasFano::new(&[0]);
        (0..10).for_each(|i| assert_eq!(Some(0), pd.predecessor(i)));

        let pd = BinarySearch::new(&[0]);
        (0..10).for_each(|i| assert_eq!(Some(0), pd.predecessor(i)));

        let pd = EliasFano::new(&[10]);
        (0..10).for_each(|i| assert_eq!(None, pd.predecessor(i)));
        (10..20).for_each(|i| assert_eq!(Some(10), pd.predecessor(i)));

        let pd = BinarySearch::new(&[10]);
        (0..10).for_each(|i| assert_eq!(None, pd.predecessor(i)));
        (10..20).for_each(|i| assert_eq!(Some(10), pd.predecessor(i)));
    }

    #[test]
    fn test_slides_example() {
        let binary = BinarySearch::new(&SLIDES_EXAMPLE);
        let elias = EliasFano::new(&SLIDES_EXAMPLE);
        for (i, expected) in zip(0.., predecessors(&SLIDES_EXAMPLE).take(100)) {
            assert_eq!(expected, binary.predecessor(i));
            assert_eq!(expected, elias.predecessor(i));
        }
    }

    #[test]
    fn test_example_2() {
        let values = [1094, 1409, 2494, 3952, 6036, 7133, 7911, 8233, 8478, 9168];
        let binary = BinarySearch::new(&values);
        let elias = EliasFano::new(&values);
        for (i, expected) in zip(0.., predecessors(&values).take(10_000)) {
            assert_eq!(expected, binary.predecessor(i));
            assert_eq!(expected, elias.predecessor(i));
        }
    }

    #[test]
    fn test_example_blackboard() {
        const U32_MAX: u64 = 2u64.pow(32) - 1;
        let values = [1, 2, 3, 4, 5, 6, 7, 8, U32_MAX];
        let elias = EliasFano::new(&values);
        for (i, expected) in zip(0.., predecessors(&values).take(100)) {
            assert_eq!(expected, elias.predecessor(i));
        }
        assert_eq!(Some(U32_MAX), elias.predecessor(U32_MAX));
        assert_eq!(Some(U32_MAX), elias.predecessor(U32_MAX + 1));
    }

    #[test]
    fn test_repeated() {
        let values = [1094, 1409, 2494, 2494, 2494, 7911, 7911, 8233, 8478, 9168, 9168];
        let binary = BinarySearch::new(&values);
        let elias = EliasFano::new(&values);
        for (i, expected) in zip(0.., predecessors(&values).take(10_000)) {
            assert_eq!(expected, binary.predecessor(i));
            assert_eq!(expected, elias.predecessor(i));
        }
    }

    #[test]
    fn test_large_universe() {
        #[rustfmt::skip]
        let values = [
            2942875257683822081, 3481564381991444538, 6246157897464183172,
            6382946403627694172, 8493766260127173971, 13053696486080051115,
            14382533531801160911, 14777858983666334667, 15089910148873339419,
            18310810712268472099,
        ];

        let binary = BinarySearch::new(&values);
        let elias = EliasFano::new(&values);

        assert_eq!(None, binary.predecessor(0));
        assert_eq!(values.last().copied(), binary.predecessor(u64::MAX));

        assert_eq!(None, elias.predecessor(0));
        assert_eq!(values.last().copied(), elias.predecessor(u64::MAX));

        for value in values {
            for offset in (0..64).map(|i| i * 1_000_000_000) {
                assert_eq!(Some(value), binary.predecessor(value + offset));
                assert_eq!(Some(value), elias.predecessor(value + offset));
            }
        }
    }

    fn predecessors(values: &[u64]) -> impl Iterator<Item = Option<u64>> + '_ {
        let mut last = None;
        values
            .iter()
            .flat_map(move |val| {
                let delta = (*val - last.unwrap_or(0)) as usize;
                repeat(replace(&mut last, Some(*val))).take(delta)
            })
            .chain(repeat(values.last().copied()))
    }
}
