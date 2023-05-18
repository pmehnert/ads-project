use std::{
    iter::{zip, Peekable},
    slice,
};

use crate::{
    bitvec::{flat_popcount::FlatPopcount, BitVec},
    div_ceil,
};

#[derive(Debug, Default, Clone)]
pub struct EliasFano {
    upper_half: FlatPopcount<BitVec>,
    lower_half: Vec<u8>,
    lower_bits: u32,
    lower_bytes: usize,
}

impl EliasFano {
    pub fn new(values: &[u64]) -> Self {
        let max = match values.last() {
            Some(last) => last,
            None => return Default::default(),
        };

        let unused_bits = max.leading_zeros();
        let upper_len = values.len().next_power_of_two().ilog2();
        let upper_bits = unused_bits + upper_len;
        let lower_bits = 64 - upper_bits;
        let lower_bytes = div_ceil(lower_bits as usize, 64);

        let upper_half: BitVec = EliasFanoUpperIter::new(values, 2).collect();
        let upper_half = FlatPopcount::new(upper_half);

        let mut lower_half = vec![0u8; lower_bytes * values.len()];

        let mask = !(u64::MAX << lower_bits);
        for (value, dst) in zip(values, lower_half.chunks_exact_mut(lower_bytes)) {
            let masked_value = value & mask;
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(&masked_value.to_be_bytes()[8 - lower_bytes..]);
            } else {
                dst.copy_from_slice(&masked_value.to_le_bytes()[..lower_bytes]);
            }
        }

        Self { upper_half, lower_half, lower_bits, lower_bytes }
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
        Self {
            values: values.iter().peekable(),
            shift,
            current: 0,
            // todo is 2 * n + 1 correct?
            remaining: 2 * values.len() + 1,
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

    #[test]
    fn test_slides() { EliasFano::new(&[0, 1, 2, 4, 7, 10, 20, 21, 22, 32]); }
}
