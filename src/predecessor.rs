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
}

impl EliasFano {
    // todo change this to accept an iterator?
    pub fn new(values: &[u64]) -> Self {
        let max = match values.last() {
            Some(last) => last,
            None => return Default::default(),
        };

        let unused_bits = max.leading_zeros();
        let upper_len = values.len().next_power_of_two().ilog2();
        let upper_bits = unused_bits + upper_len;
        let lower_bits = 64 - upper_bits;

        let upper_half: BitVec = EliasFanoUpperIter::new(values, lower_bits).collect();

        Self {
            upper_half: FlatPopcount::new(upper_half),
            lower_half: PackedArray::new(lower_bits, values.iter().copied()),
            lower_bits,
        }
    }

    pub fn size_bits(&self) -> usize {
        // todo include size_of(Self) everywhere
        self.upper_half.size_bits()
            + self.upper_half.bitvec().size_bits()
            + self.lower_half.size_bits()
    }

    pub fn predecessor(&self, value: u64) -> u64 {
        let upper = value >> self.lower_bits;
        let index = self.upper_half.select1(upper); // todo this will panic
        let start = self.upper_half.rank0(index);
        let lower_start = 0;

        let mask = u64::MAX >> (64 - self.lower_bits);
        let masked_value = value & mask;

        todo!()
    }
}


struct EliasFanoUpperIter<'a> {
    values: Peekable<slice::Iter<'a, u64>>,
    shift: u32,
    current: u64,
    remaining: usize,
}

impl<'a> EliasFanoUpperIter<'a> {
    fn new(values: &'a [u64], shift: u32) -> Self {
        let max = values.last().copied().unwrap_or(0);
        Self {
            values: values.iter().peekable(),
            shift,
            current: 0,
            remaining: values.len() + (max >> shift) as usize,
        }
    }
}

impl<'a> Iterator for EliasFanoUpperIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;
            match self.values.next_if(|v| **v >> self.shift == self.current) {
                Some(_) => Some(false),
                None => {
                    self.current += 1;
                    Some(true)
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
    use super::EliasFano;

    const SLIDES_EXAMPLE: &[u64] = &[0, 1, 2, 4, 7, 10, 20, 21, 22, 32];

    #[test]
    fn test_slides() { // EliasFano::new(SLIDES_EXAMPLE);
    }
}
