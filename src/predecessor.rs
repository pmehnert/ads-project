//! TODO predecessor queries

use std::{iter::Peekable, slice};

use crate::{
    bitvec::{flat_popcount::FlatPopcount, BitVec},
    packed::PackedArray,
};

#[derive(Debug, Default, Clone)]
pub struct EliasFano {
    upper_half: FlatPopcount<BitVec>,
    lower_half: PackedArray,
    lower_bits: u32,
    maximum: Option<u64>,
}

impl EliasFano {
    // todo change this to accept an iterator?
    pub fn new(values: &[u64]) -> Self {
        let maximum = match values.last() {
            Some(last) => *last,
            None => return Default::default(),
        };

        // todo ignore equal MSBs of values
        // let (minimum, maximum) = match values {
        //     [first, .., last] => (first, last),
        //     [single] => todo!(),
        //     [] => return Default::default(),
        // };

        let unused_bits = maximum.leading_zeros();
        let upper_len = values.len().next_power_of_two().ilog2();
        let upper_bits = unused_bits + upper_len;
        let lower_bits = 64 - upper_bits;

        let upper_half: BitVec = EliasFanoUpperIter::new(values, lower_bits).collect();
        let upper_half = FlatPopcount::new(upper_half);
        let lower_half = PackedArray::new(lower_bits, values.iter().copied());

        Self { upper_half, lower_half, lower_bits, maximum: Some(maximum) }
    }

    pub fn size_bits(&self) -> usize {
        // todo include size_of(Self) everyhwere
        self.upper_half.size_bits()
            + self.upper_half.bitvec().size_bits()
            + self.lower_half.size_bits()
    }

    pub fn predecessor(&self, value: u64) -> Option<u64> {
        let value_upper = value >> self.lower_bits;
        if value_upper >= self.upper_half.count_zeros() {
            return self.maximum;
        }

        // todo change bitvec and flat-popcount to use u64 instead of usize?

        let bitvec_index = self.upper_half.select0(value_upper);
        let lower_index = bitvec_index as u64 - value_upper;
        if lower_index == 0 {
            return None;
        }

        let mask = u64::MAX >> (64 - self.lower_bits);
        let value_lower = value & mask;

        let mut bitvec_indexes = (1..=bitvec_index).rev();
        let pred = bitvec_indexes
            .by_ref()
            .take_while(|i| self.upper_half.bitvec().index(*i - 1))
            .zip(self.lower_half.iter().take(lower_index as usize).rev())
            .find(|(_, pred_lower)| *pred_lower <= value_lower);

        if let Some((_, pred_lower)) = pred {
            Some((value & !mask) | pred_lower)
        } else if let Some(bitvec_index) = bitvec_indexes.next() {
            let rank = bitvec_index as u64 - value_upper;
            let bitvec_index = self.upper_half.select1(rank);
            let pred_upper = bitvec_index as u64 - rank;
            let pred_lower = self.lower_half.index(rank as usize);

            Some((pred_upper << self.lower_bits) | pred_lower)
        } else {
            None
        }
    }
}


#[derive(Debug, Clone)]
struct EliasFanoUpperIter<'a> {
    values: Peekable<slice::Iter<'a, u64>>,
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
