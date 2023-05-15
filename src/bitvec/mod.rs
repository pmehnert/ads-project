use std::{
    fmt,
    iter::{zip, Flatten, Take},
    slice,
};

pub use block::BitIndex;
pub use block::Block;

pub mod block;
pub mod select;

/// An iterator over the bits of a bit vector.
pub type Iter<'a> = Take<Flatten<slice::Iter<'a, Block>>>;

/// A contiguous and compact array of bits, short for 'bit vector'.
///
/// # Invariants
///
/// Bit vectors always uphold the following invariants.
///
/// - All but the last block in `self.blocks` are filled completely.
/// - The last block in `self.blocks` contains at least one bit.
/// - All unused positions of the last block in `self.blocks` are set to zero.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct BitVec {
    blocks: Vec<Block>,
    len: usize,
}

impl BitVec {
    /// Returns a new, empty bit vector.
    pub fn new() -> Self { Default::default() }

    /// Returns a new, empty bit vector with at least the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { blocks: Vec::with_capacity(cap / Block::BITS as usize), len: 0 }
    }

    /// Returns the total number of bits the bit vector can hold without reallocating.
    pub fn capacity(&self) -> usize { self.blocks.capacity() * Block::BITS as usize }

    /// Returns the number of bits in the bit vector.
    pub fn len(&self) -> usize { self.len }

    /// Returns `true` if the bit vector contains no bits.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns an (inefficient) iterator over the bits of the bit vector.
    pub fn iter(&self) -> Iter<'_> { self.blocks.iter().flatten().take(self.len) }

    /// Returns a slice containing the underlying blocks of the bit vector.
    pub fn blocks(&self) -> &[Block] { &self.blocks }

    /// Returns a mutable slice containing the underlying blocks of the bit vector.
    pub fn blocks_mut(&mut self) -> &mut [Block] { &mut self.blocks }
}

impl<'a> IntoIterator for &'a BitVec {
    type IntoIter = Iter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl FromIterator<bool> for BitVec {
    fn from_iter<Iter: IntoIterator<Item = bool>>(iter: Iter) -> Self {
        let mut iter = iter.into_iter().peekable();
        let cap = (iter.size_hint().0 + Block::BITS as usize - 1) / Block::BITS as usize;
        let (mut blocks, mut len) = (Vec::with_capacity(cap), 0);

        while iter.peek().is_some() {
            let mut block = Block::ALL_ZEROS;

            len += zip(0..Block::BITS as u8, iter.by_ref())
                .map(|(i, bit)| block.0 |= BitIndex(i).mask_value(bit))
                .count();
            blocks.push(block);
        }
        Self { blocks, len }
    }
}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter().map(u8::from)).finish()
    }
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
}
