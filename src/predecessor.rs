//! Data structures for accelerating predecessor queries.
//!
//! # Predecessor Problem
//!
//! Given a _universe_ `U = 0..u` and a sorted sequence `A` of `n` integers from
//! `U`, for any `x ∈ U` the _predecessor_ of `x` in `A` is defined as
//!
//! > `pred(x, A) = max {y ∈ A | y <= x}`.

use std::{iter, iter::FusedIterator, slice};

use crate::{
    bitvec::{flat_popcount::FlatPopcount, BitVec},
    packed::PackedArray,
    AllocationSize,
};

/// An implementation of the quasi-succinct Elias-Fano coding for answering
/// predecessor queries.
///
/// Each value is split into an upper and a lower half. The upper halves are
/// stored in a bit vector using a kind of unary encoding, the lower halves are
/// stored in a byte packed array.
#[derive(Debug, Default, Clone)]
pub struct EliasFano {
    upper_half: FlatPopcount<BitVec>,
    lower_half: PackedArray,
    maximum: Option<u64>,
}

impl EliasFano {
    /// Constructs the Elias-Fano coding for the given sequence of integers.
    ///
    /// The elements of `values` must be sorted in ascending order. If they are
    /// not, the program may panic or produce unexpected results, but will not
    /// result in undefined behaviour.
    ///
    /// **Note:** This should really accept an iterator not a slice.
    pub fn new(values: &[u64]) -> Self {
        // todo ignore equal MSBs of values to be able to partition input
        let maximum = match values.last() {
            Some(last) => *last,
            None => return Default::default(),
        };

        let unused_bits = maximum.leading_zeros();
        let upper_bits = values.len().next_power_of_two().ilog2();
        let lower_bits = 64u32.saturating_sub(upper_bits + unused_bits).max(1);

        let upper_iter = EliasFanoUpperIter::new(values, lower_bits);
        let upper_half = FlatPopcount::new(upper_iter.collect());
        let lower_half = PackedArray::new(lower_bits, values.iter().copied());

        Self { upper_half, lower_half, maximum: Some(maximum) }
    }

    /// Retuns the [predecessor](crate::predecessor#predecessor-problem) of
    /// `value` in the integer array encoded by this instance.
    pub fn predecessor(&self, value: u64) -> Option<u64> {
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

impl AllocationSize for EliasFano {
    fn size_bytes(&self) -> usize {
        self.upper_half.size_bytes()
            + self.upper_half.bitvec().size_bytes()
            + self.lower_half.size_bytes()
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

    use super::EliasFano;

    const SLIDES_EXAMPLE: [u64; 10] = [0, 1, 2, 4, 7, 10, 20, 21, 22, 32];

    #[test]
    fn test_empty() {
        let pd = EliasFano::new(&[]);
        for i in 0..10 {
            assert_eq!(None, pd.predecessor(i));
        }
    }

    #[test]
    fn test_single_element() {
        let pd = EliasFano::new(&[0]);
        (0..10).for_each(|i| assert_eq!(Some(0), pd.predecessor(i)));

        let pd = EliasFano::new(&[10]);
        (0..10).for_each(|i| assert_eq!(None, pd.predecessor(i)));
        (10..20).for_each(|i| assert_eq!(Some(10), pd.predecessor(i)));
    }

    #[test]
    fn test_slides_example() {
        let pd = EliasFano::new(&SLIDES_EXAMPLE);
        for (i, expected) in zip(0.., predecessors(&SLIDES_EXAMPLE).take(100)) {
            assert_eq!(expected, pd.predecessor(i));
        }
    }

    #[test]
    fn test_example_2() {
        let values = [1094, 1409, 2494, 3952, 6036, 7133, 7911, 8233, 8478, 9168];
        let pd = EliasFano::new(&values);
        for (i, expected) in zip(0.., predecessors(&values).take(10_000)) {
            assert_eq!(expected, pd.predecessor(i));
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
        let pd = EliasFano::new(&values);

        assert_eq!(None, pd.predecessor(0));
        assert_eq!(values.last().copied(), pd.predecessor(u64::MAX));

        for value in values {
            for offset in (0..64).map(|i| i * 1_000_000_000) {
                assert_eq!(Some(value), pd.predecessor(value + offset));
            }
        }
    }

    fn predecessors(values: &[u64]) -> impl Iterator<Item = Option<u64>> + '_ {
        let mut last = None;
        values
            .iter()
            .flat_map(move |val| {
                let delta = dbg!((*val - last.unwrap_or(0)) as usize);
                repeat(replace(&mut last, Some(*val))).take(delta)
            })
            .chain(repeat(values.last().copied()))
    }
}
