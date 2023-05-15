//! The individual blocks of a bit vector.

use std::{iter::FusedIterator, ops::Range};

/// An integer in the range of `0..63`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BitIndex(#[doc(hidden)] pub(super) u8);

impl From<usize> for BitIndex {
    fn from(value: usize) -> Self { Self::new(value) }
}

impl BitIndex {
    pub const MAX: Self = Self(Block::BITS as u8 - 1);
    pub const MIN: Self = Self(0);

    pub fn new(index: usize) -> BitIndex {
        BitIndex((index % Block::BITS as usize) as u8)
    }

    /// Returns a mask with the bit at position `self.0` set to `1`.
    pub fn mask_bit(self) -> u64 { 1 << self.0 }

    /// Returns a mask with the bit at position `self.0` set to `value`.
    pub fn mask_value(self, value: bool) -> u64 { (value as u64) << self.0 }
}

/// A block of bits in a bit vector.
///
/// The block is in least significant bit first order.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Block(pub u64);

impl Block {
    /// A block with all bits set to `0`.
    pub const ALL_ONES: Self = Self(u64::MAX);
    /// A block with all bits set to `1`.
    pub const ALL_ZEROS: Self = Self(0);
    /// The number of bits in a block.
    pub const BITS: u32 = u64::BITS;

    // todo name?
    pub fn div_index(index: usize) -> usize { index / Self::BITS as usize }

    /// Returns the bit as `index` in the block.
    pub fn get(&self, index: BitIndex) -> bool { self.0 & index.mask_bit() != 0 }

    /// Returns an iterator over the bits of this block.
    pub fn iter(&self) -> Iter<'_> { Iter { block: self, range: 0..Self::BITS as u8 } }

    /// Returns the number of `0`-bits in the block.
    pub fn count_zeroes(self) -> u32 { self.0.count_zeros() }

    /// Returns the number of `1`-bits in the block.
    pub fn count_ones(self) -> u32 { self.0.count_ones() }

    /// Returns the number of `1`-bits up to, and including position `index`.
    pub fn rank1_inclusive(self, index: BitIndex) -> u32 {
        (self.0 << (Self::BITS as u8 - 1 - index.0)).count_ones()
    }

    /// Returns the number of `0`-bits up to, and including position `index`.
    pub fn rank0_inclusive(self, index: BitIndex) -> u32 {
        (!self.0 << (Self::BITS as u8 - 1 - index.0)).count_ones()
    }

    /// Returns the index of the first `1`-bit with the given `1`-rank.
    pub fn select1(self, rank: u32) -> u32 { self.select_impl(rank) }

    /// Returns the index of the first `0`-bit with the given `0`-rank.
    pub fn select0(self, rank: u32) -> u32 { Block(!self.0).select_impl(rank) }

    /// Returns `select1(self, rank)`, or `None` if there is no `1`-bit with the given `1`-rank.
    pub fn checked_select1(self, rank: u32) -> Option<u32> {
        Some(self.select1(rank)).filter(|i| *i < Self::BITS)
    }

    /// Returns `select0(self, rank)`, or `None` if there is no `0`-bit with the given `0`-rank.
    pub fn checked_select0(self, rank: u32) -> Option<u32> {
        Some(self.select0(rank)).filter(|i| *i < Self::BITS)
    }

    #[doc(hidden)]
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    fn select_impl(self, rank: u32) -> u32 {
        use std::iter::zip;
        let mut acc = rank + 1;
        for (i, bit) in zip(0.., &self) {
            acc -= u32::from(bit);
            if acc == 0 {
                return i;
            }
        }
        Self::BITS
    }

    #[doc(hidden)]
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    fn select_impl(self, rank: u32) -> u32 {
        use std::arch::x86_64::*;

        // 'deposit' a `1` into the `rank`-th 1-bit of `self`
        let i = 1_u64 << u64::from(rank);
        let d = unsafe { _pdep_u64(i, self.0) };
        d.trailing_zeros()
    }
}

impl<'a> IntoIterator for &'a Block {
    type IntoIter = Iter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

/// An iterator over the bits in a block.
pub struct Iter<'a> {
    block: &'a Block,
    range: Range<u8>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(BitIndex).map(|i| self.block.get(i))
    }

    fn size_hint(&self) -> (usize, Option<usize>) { self.range.size_hint() }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(BitIndex).map(|i| self.block.get(i))
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(BitIndex).map(|i| self.block.get(i))
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth_back(n).map(BitIndex).map(|i| self.block.get(i))
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rank1() {
        let block = Block(0b0110_0101_0001_1101_0000_1101_0011_0101);

        assert_eq!(1, block.rank1_inclusive(BitIndex::new(0)));
        assert_eq!(5, block.rank1_inclusive(BitIndex::new(9)));
        assert_eq!(15, block.rank1_inclusive(BitIndex::new(31)));
        assert_eq!(15, block.rank1_inclusive(BitIndex::new(63)));
    }

    #[test]
    fn test_rank0() {
        let block = Block(0b0110_0101_0001_1101_0000_1101_0011_0101);

        assert_eq!(0, block.rank0_inclusive(BitIndex::new(0)));
        assert_eq!(5, block.rank0_inclusive(BitIndex::new(9)));
        assert_eq!(17, block.rank0_inclusive(BitIndex::new(31)));
        assert_eq!(49, block.rank0_inclusive(BitIndex::new(63)));
    }

    #[test]
    fn test_select1() {
        let block = Block(0b0110_0101_0001_1101_0000_1101_0011_0101);

        assert_eq!(0, block.select1(0));
        assert_eq!(2, block.select1(1));
        assert_eq!(5, block.select1(3));
        assert_eq!(30, block.select1(14));
        assert_eq!(64, block.select1(15));
        assert_eq!(64, block.select1(63));

        assert_eq!(0, Block::ALL_ONES.select1(0));
        assert_eq!(63, Block::ALL_ONES.select1(63));
        assert_eq!(64, Block::ALL_ZEROS.select1(0));
    }

    #[test]
    fn test_select0() {
        let block = Block(0b0110_0101_0001_1101_0000_1101_0011_0101);

        assert_eq!(1, block.select0(0));
        assert_eq!(3, block.select0(1));
        assert_eq!(7, block.select0(3));
        assert_eq!(63, block.select0(48));
        assert_eq!(64, block.select0(49));
        assert_eq!(64, block.select0(63));

        assert_eq!(0, Block::ALL_ZEROS.select0(0));
        assert_eq!(63, Block::ALL_ZEROS.select0(63));
        assert_eq!(64, Block::ALL_ONES.select0(0));
    }
}
