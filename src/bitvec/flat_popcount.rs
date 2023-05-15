use std::{fmt, mem};

use super::BitVec;
use crate::bitvec::{block::BitIndex, AlignedBlock, Block};

#[allow(unused)]
mod config {
    use crate::bitvec::Block;

    pub const L1_SIZE_BITS: usize = 4096;
    pub const L2_SIZE_BITS: usize = 512;

    pub const L2_PER_L1: usize = L1_SIZE_BITS / L2_SIZE_BITS;
    pub const L1_SIZE_U64: usize = L1_SIZE_BITS / Block::BITS as usize;
    pub const L2_SIZE_U64: usize = L2_SIZE_BITS / Block::BITS as usize;
}

#[derive(Debug, Clone)]
pub struct FlatPopcount<'a> {
    bitvec: &'a BitVec,
    data: Vec<L1Data>,
}

/// # Layout
///
/// Using ZIGs wonderful integer types, the struct could be defined as follows:
///
/// TODO
/// ```zig
/// const L1Data = packed struct {
///     l1_ones: u44,
///     l2_ones: packed struct
///     l2_1: u12,
///     // l2_2, ..., l2_6
///     l2_7: u12,
/// }
/// ```
#[derive(Default, Clone, Copy)]
#[repr(C, align(16))]
pub struct L1Data(u128);

impl<'a> FlatPopcount<'a> {
    pub fn new(bitvec: &'a BitVec) -> Self {
        assert!(bitvec.len() <= (1 << 44));

        assert_eq!(AlignedBlock::BLOCKS, config::L2_SIZE_U64);
        let aligned_blocks = bitvec.aligned_blocks();
        let mut data = Vec::with_capacity(aligned_blocks.len());

        let (mut pre_l1_ones, mut l1_ones, mut l2_data) = (0u64, 0u32, [0u16; 7]);
        for (i, chunk) in aligned_blocks.iter().enumerate() {
            if i != 0 && i % config::L2_PER_L1 == 0 {
                data.push(L1Data::new(pre_l1_ones, mem::take(&mut l2_data)));
                pre_l1_ones += u64::from(mem::take(&mut l1_ones));
            }

            l1_ones += chunk.0.iter().copied().map(Block::count_ones).sum::<u32>();

            if i % config::L2_PER_L1 < config::L2_PER_L1 - 1 {
                l2_data[i % config::L2_PER_L1] = l1_ones as u16;
            }
        }

        if aligned_blocks.len() % config::L2_PER_L1 != 0 {
            l2_data[aligned_blocks.len() % config::L2_PER_L1..].fill(l1_ones as u16);
        }
        data.push(L1Data::new(pre_l1_ones, l2_data));

        Self { bitvec, data }
    }

    pub fn rank1(&self, index: usize) -> u64 {
        // todo handle out of bounds

        let l1_index = index / config::L1_SIZE_BITS;
        let sub_index = index % config::L1_SIZE_BITS;

        let l1_data = &self.data[l1_index];
        let mut rank = l1_data.l1_ones();
        if sub_index == 0 {
            return rank;
        }

        let l2_index = sub_index / config::L2_SIZE_BITS;
        let sub_index = sub_index % config::L2_SIZE_BITS;

        // todo maybe inline this manually
        rank += u64::from(l1_data.l2_ones(l2_index));
        if sub_index == 0 {
            return rank;
        }

        let aligned_index = index / config::L2_SIZE_BITS;
        let l2_block = &self.bitvec.aligned_blocks()[aligned_index].0;

        let block_index = sub_index / Block::BITS as usize;
        let sub_index = sub_index % Block::BITS as usize;

        for block in &l2_block[..block_index] {
            rank += u64::from(block.count_ones())
        }

        if sub_index != 0 {
            // todo use a rank1 instead
            let index_inclusive = BitIndex::new(sub_index - 1);
            rank += u64::from(l2_block[block_index].rank1_inclusive(index_inclusive));
        }

        rank
    }

    pub fn rank0(&self, index: usize) -> u64 { index as u64 - self.rank1(index) }
}

impl L1Data {
    const U12_MASK: u128 = 0b1111_1111_1111;
    const U44_MASK: u128 = 0xFFF_FFFF_FFFF;

    #[allow(clippy::erasing_op, clippy::identity_op)]
    pub fn new(l1_ones: u64, l2_ones: [u16; 7]) -> Self {
        let [l2_1, l2_2, l2_3, l2_4, l2_5, l2_6, l2_7] = l2_ones;
        Self(
            (Self::U44_MASK & u128::from(l1_ones))
                | (Self::U12_MASK & u128::from(l2_1)) << (44 + 0 * 12)
                | (Self::U12_MASK & u128::from(l2_2)) << (44 + 1 * 12)
                | (Self::U12_MASK & u128::from(l2_3)) << (44 + 2 * 12)
                | (Self::U12_MASK & u128::from(l2_4)) << (44 + 3 * 12)
                | (Self::U12_MASK & u128::from(l2_5)) << (44 + 4 * 12)
                | (Self::U12_MASK & u128::from(l2_6)) << (44 + 5 * 12)
                | (Self::U12_MASK & u128::from(l2_7)) << (44 + 6 * 12),
        )
    }

    pub fn l1_ones(self) -> u64 { (Self::U44_MASK & self.0) as u64 }

    // todo the general case here is terrible
    // todo what about out of bounds
    pub fn l2_ones(self, index: usize) -> u16 {
        if index > 0 {
            (Self::U12_MASK & (self.0 >> (index * 12 + 44 - 12))) as u16
        } else {
            0
        }
    }
}

impl fmt::Debug for L1Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l2_ones: [_; 8] = std::array::from_fn(|i| self.l2_ones(i));
        f.debug_struct("L1Data")
            .field("l1_ones", &self.l1_ones())
            .field("l2_ones", &l2_ones)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use std::iter::repeat;

    use super::{config::*, *};
    use crate::bitvec::BitVec;

    #[test]
    fn test_l1l2_layout() {
        assert_eq!(16, std::mem::size_of::<L1Data>());
        assert_eq!(16, std::mem::align_of::<L1Data>());
    }

    #[test]
    fn test_l1l2_construction() {
        let data = L1Data::new(42, [1, 2, 3, 4, 5, 6, 4095]);
        assert_eq!(42, data.l1_ones());
        assert_eq!(0, data.l2_ones(0));
        assert_eq!(1, data.l2_ones(1));
        assert_eq!(2, data.l2_ones(2));
        assert_eq!(3, data.l2_ones(3));
        assert_eq!(4, data.l2_ones(4));
        assert_eq!(5, data.l2_ones(5));
        assert_eq!(6, data.l2_ones(6));
        assert_eq!(4095, data.l2_ones(7));
    }

    #[test]
    fn test_rank_select_empty() {
        let bitvec = BitVec::new();
        let rank = FlatPopcount::new(&bitvec);
    }

    #[test]
    fn test_rank_select_all_ones() {
        let bitvec: BitVec = repeat(true).take(10_000).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(8192, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(9728, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(8512, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS as usize));
        assert_eq!(1527, rank.rank1(1527));
    }

    #[test]
    fn test_rank_select_all_zeros() {
        let bitvec: BitVec = repeat(false).take(10_000).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS as usize));
        assert_eq!(0, rank.rank1(1527));
    }

    #[test]
    fn test_rank_select_exact() {
        let bitvec: BitVec = repeat(true).take(config::L1_SIZE_BITS * 4).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(8192, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(9728, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(8512, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS as usize));
        assert_eq!(1527, rank.rank1(1527));
    }

    #[test]
    fn test_rank_select_non_trivial() {
        let bitvec: BitVec = (0..15_000).map(|i| i % 3 == 0).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(2731, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(3926, rank.rank1(2 * L1_SIZE_BITS + 7 * L2_SIZE_BITS));
        assert_eq!(2838, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS as usize));
        assert_eq!(509, rank.rank1(1527));
    }
}
