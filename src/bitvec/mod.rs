//! Compact arrays of bits.

use std::{fmt, iter, iter::zip, slice};

pub use block::Block;
use block::{AlignedBlock, BitIndex};

use crate::{div_ceil, AllocationSize};

pub mod block;
pub mod flat_popcount;

/// An acceleration data structure for rank and select queries on bit vectors.
pub type RankSelect<Bits> = flat_popcount::FlatPopcount<Bits>;

/// An iterator over the bits of a bit vector.
pub type Iter<'a> = iter::Take<iter::Flatten<slice::Iter<'a, Block>>>;

/// A contiguous, cache aligned, and compact array of bits, short for 'bit vector'.
///
/// # Invariants
///
/// Bit vectors always uphold the following invariants.
///
/// - The length of `self.blocks` is the minimum required to store `self.len` bits.
/// - Any unused position of a block in `self.blocks` is set to `0`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct BitVec {
    blocks: Vec<AlignedBlock>,
    len: usize,
}

impl BitVec {
    /// Returns a new, empty bit vector.
    pub fn new() -> Self { Default::default() }

    /// Returns a new, empty bit vector with at least the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { blocks: Vec::with_capacity(div_ceil(cap, Block::BITS)), len: 0 }
    }

    /// Returns the total number of bits the bit vector can hold without reallocating.
    pub fn capacity(&self) -> usize { self.blocks.capacity() * Block::BITS }

    /// Returns the number of bits in the bit vector.
    pub fn len(&self) -> usize { self.len }

    /// Returns `true` if the bit vector contains no bits.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns an (inefficient) iterator over the bits of the bit vector.
    pub fn iter(&self) -> Iter<'_> { self.blocks().iter().flatten().take(self.len) }

    /// Returns the bit at the geven index in the bit vector.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn index(&self, index: usize) -> bool {
        if index < self.len {
            let block = self.blocks()[index / Block::BITS];
            block.0 & BitIndex::new(index).mask_bit() != 0
        } else {
            index_out_of_bounds_fail(index, self.len)
        }
    }

    /// Returns a slice containing the underlying cache aligned blocks of the bit vector.
    ///
    /// The last element may contain unused bits, which are guranteed to be set to `0`.
    pub fn aligned_blocks(&self) -> &[AlignedBlock] { &self.blocks }

    /// Returns a slice containing the underlying blocks of the bit vector.
    ///
    /// The last [`AlignedBlock::BLOCKS`] elements may contain unused bits, which are
    /// guranteed to be set to `0`.
    pub fn blocks(&self) -> &[Block] {
        let aligned = self.aligned_blocks();

        // Safety: The multiplication can't overflow, because `aligned` is
        // already in the address space.
        let len =
            unsafe { aligned.len().checked_mul(AlignedBlock::BLOCKS).unwrap_unchecked() };

        // Safety: A pointer to `AlignedBlock` may be safely reinterpreted as
        // a pointer to `Block`.
        unsafe { std::slice::from_raw_parts(aligned.as_ptr() as *const Block, len) }
    }
}

impl AllocationSize for BitVec {
    fn size_bytes(&self) -> usize { self.blocks().size_bytes() }
}

impl<'a> IntoIterator for &'a BitVec {
    type IntoIter = Iter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl FromIterator<bool> for BitVec {
    fn from_iter<Iter: IntoIterator<Item = bool>>(iter: Iter) -> Self {
        let mut iter = iter.into_iter().fuse().peekable();
        let cap = crate::div_ceil(iter.size_hint().0, Block::BITS);
        let (mut blocks, mut len) = (Vec::with_capacity(cap), 0);

        while iter.peek().is_some() {
            let mut aligned = AlignedBlock([Block::ALL_ZEROS; 8]);

            for block in &mut aligned.0 {
                len += zip(0..Block::BITS as u8, iter.by_ref())
                    .map(|(i, bit)| block.0 |= BitIndex(i).mask_value(bit))
                    .count();
            }
            blocks.push(aligned);
        }
        Self { blocks, len }
    }
}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter().map(u8::from)).finish()
    }
}

#[doc(hidden)]
#[cold]
#[inline(never)]
#[track_caller]
fn index_out_of_bounds_fail(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the length is {} but the index is {}", len, index)
}

#[cfg(test)]
mod test {
    use std::iter::{empty, once, repeat};

    use super::BitVec;

    #[test]
    fn test_len_and_is_empty() {
        assert_eq!(0, BitVec::new().len());
        assert!(BitVec::new().is_empty());

        assert_eq!(1, BitVec::from_iter(Some(true).into_iter()).len());
        assert!(!BitVec::from_iter(Some(true).into_iter()).is_empty());

        assert_eq!(1000, BitVec::from_iter((0..1000).map(|_| true)).len());
        assert!(!BitVec::from_iter((0..1000).map(|_| true)).is_empty());
    }

    #[test]
    fn test_iter() {
        assert!(empty::<bool>().eq(&BitVec::new()));

        let iter = (0..1000).map(|i| i % 4 == 0 || i % 7 == 0);
        assert!(iter.clone().eq(&BitVec::from_iter(iter)));
    }

    #[test]
    fn test_eq() {
        assert_eq!(BitVec::new(), BitVec::new());

        let iter = (0..1000).map(|i| i % 4 == 0 || i % 7 == 0);
        let vec999: BitVec = iter.clone().take(999).collect();
        let vec1000: BitVec = iter.collect();
        assert_eq!(vec1000, vec1000);
        assert_ne!(vec999, vec1000);

        let vec1: BitVec = once(false).collect();
        let vec64: BitVec = repeat(false).take(64).collect();
        assert_ne!(vec1, vec64);
    }

    #[test]
    fn test_index() {
        let pred = |i| i % 4 == 0 || i % 7 == 0;
        let bitvec: BitVec = (0..1000).map(pred).collect();
        for i in 0..1000 {
            assert_eq!(pred(i), bitvec.index(i));
        }
    }
}
