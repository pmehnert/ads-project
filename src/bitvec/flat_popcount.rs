//! An implementation of _flat-popcount_ for rank and select queries on bit vectors.

use std::{borrow::Borrow, fmt, iter::zip};

use crate::{
    bitvec::{block::BitIndex, AlignedBlock, BitVec, Block},
    div_ceil,
};

/// Various static haracteristic values of the flat-popcount data structure.
#[allow(unused)]
pub mod config {
    use crate::bitvec::Block;

    /// The size of the largest supported bit vector in bits.
    pub const MAX_SIZE_BITS: usize = 2_usize.pow(44);
    /// The size of each L1 block in bits.
    pub const L1_SIZE_BITS: usize = 4096;
    /// The size of each L2 block in bits.
    pub const L2_SIZE_BITS: usize = 512;
    /// The size of each L1 block in L2 blocks.
    pub const L2_PER_L1: usize = L1_SIZE_BITS / L2_SIZE_BITS;
    /// The size of each L1 block in bit vector [`Block`]s.
    pub const L1_SIZE_U64: usize = L1_SIZE_BITS / Block::BITS;
    /// The size of each L2 block in bit vector [`Block`]s.
    pub const L2_SIZE_U64: usize = L2_SIZE_BITS / Block::BITS;
    /// The frequency at which `1`-bits in the bit vector are sampled.
    pub const SELECT_SAMPLE_RATE: u64 = 8192;
}

/// An index into [`FlatPopcount::data`], hinting at the position of a `1`-bit.
///
/// The data structure supports up to `2^44` bits and each L1 blocks spans
/// `2^12` bits, which limits the number of L1 blocks to `2^32`. Therefore,
/// a `32`-bit integer always suffices as index.
pub type Hint = u32;

/// An implementation of _flat-popcount_ \[1\] for rank and select queries on bit vectors.
///
/// The bit vector is partitioned into two layers of blocks --- L2 blocks spanning
/// 512 bits and L1 blocks spanning 4096 bits respectively. For each block the number
/// of `1`-bits, from the start of the bit vector, or surrounding L1 block, is
/// precomputed and stored. The counts are stored in an interleaved, cache-friendly
/// format (see [`L1L2Data`]). Note that the L0 blocks described in \[1\] are omitted.
/// Thus, only bit vectors with up to `2^44` bits are supported.
///
/// To speed up select queries, the L1 block of every `8192`th `1`-bit is stored
/// explicitly.
///
/// # References
///
/// \[1\] Florian Kurpicz. _Engineering Compact Data Structures for Rank and
/// Select Queries on Bit Vectors_. DOI: [10.48550/arXiv.2206.01149]
///
/// [10.48550/arXiv.2206.01149]: https://doi.org/10.48550/arXiv.2206.01149
#[derive(Debug, Default, Clone)]
pub struct FlatPopcount<Bits: Borrow<BitVec>> {
    bitvec: Bits,
    data: Vec<L1L2Data>,
    one_hints: Vec<Hint>,
    total_ones: u64,
}

/// The space efficient, interleaved L1 and L2 indeces of flat-popcount.
///
/// Stores the number of ones up to the corresponding L1 block, as well as the
/// number of ones from the start of the L1 block up to each contained L2 block
/// (except for the first).
///
/// # Layout
///
/// The type is guaranteed to have a size and alignment of 128 bit. Using [Zig]s
/// wonderful integer types, the struct's members could be declared as follows:
///
/// ```
/// const L1Data = packed struct {
///     l1_ones: u44,
///     l2_ones: packed struct {
///         l2_1: u12, l2_2: u12, l2_3: u12, l2_4: u12,  
///         l2_5: u12, l2_6: u12, l2_7: u12,
///     }
/// }
/// ```
///
/// [Zig]: https://ziglang.org/
#[derive(Default, Clone, Copy)]
#[repr(C, align(16))]
pub struct L1L2Data(#[doc(hidden)] u128);

impl<Bits: Borrow<BitVec>> FlatPopcount<Bits> {
    /// Constructs the flat-popcount rank and select data structure.
    ///
    /// The construction has time complexity `O(n)` (ignoring gains made through
    /// native POPCNT and SIMD instructions) and broadly works in two phases.
    ///
    /// 1. Scan the bit vector's blocks and count the number of ones in each.
    /// Directly create [`L1L2Data`] for all L1 and L2 blocks.
    /// 2. Scan the just created L1 data and store the L1 index of every `8192`th
    /// `1`-bit.
    ///
    /// # Panics
    ///
    /// Panics if `bitvec` contains more than `2^44` bits.
    pub fn new(bitvec: Bits) -> Self {
        assert!(
            bitvec.borrow().len() <= config::MAX_SIZE_BITS,
            "only up to 2^44 bits supported"
        );

        #[doc(hidden)]
        fn count_l2_ones(l2_block: &AlignedBlock) -> u32 {
            l2_block.0.iter().copied().map(Block::count_ones).sum::<u32>()
        }

        assert_eq!(AlignedBlock::BLOCKS, config::L2_SIZE_U64);
        let aligned_blocks = bitvec.borrow().aligned_blocks();
        let l1_count = div_ceil(aligned_blocks.len(), config::L2_PER_L1);
        let mut data = Vec::with_capacity(l1_count);

        let mut l1_blocks = aligned_blocks.chunks_exact(config::L2_PER_L1);
        data.resize(l1_blocks.len(), Default::default());

        let mut total_ones = 0u64;

        // Write L1L2Data for all complete L1 blocks
        for (l1l2_data, l1_block) in zip(&mut data, l1_blocks.by_ref()) {
            let (mut l1_ones, mut l2_ones) = (0u16, [0u16; 8]);
            for (dst, l2_block) in zip(&mut l2_ones, l1_block.iter()) {
                l1_ones += count_l2_ones(l2_block) as u16;
                *dst = l1_ones;
            }

            let l2_ones = <&[_; 7]>::try_from(&l2_ones[0..7]).unwrap();
            *l1l2_data = L1L2Data::new(total_ones, l2_ones);
            total_ones += u64::from(l1_ones);
        }

        // Write L1L2Data for the partially filled L1 bock if one exists
        if !l1_blocks.remainder().is_empty() {
            let (mut l1_ones, mut l2_ones) = (0u16, [0u16; 8]);

            let mut dst_iter = l2_ones.iter_mut();
            for (dst, l2_block) in zip(dst_iter.by_ref(), l1_blocks.remainder()) {
                l1_ones += count_l2_ones(l2_block) as u16;
                *dst = l1_ones;
            }
            dst_iter.for_each(|dst| *dst = l1_ones);

            let l2_ones = <&[_; 7]>::try_from(&l2_ones[0..7]).unwrap();
            data.push(L1L2Data::new(total_ones, l2_ones));
            total_ones += u64::from(l1_ones);
        }

        // Derive number of required one hints and write them out
        let cap = div_ceil(total_ones as usize, config::SELECT_SAMPLE_RATE as usize);
        let mut one_hints = Vec::with_capacity(cap);
        let mut next_hint = 0;
        for (i, data) in zip(0.., &data) {
            if data.l1_ones() >= next_hint {
                next_hint += config::SELECT_SAMPLE_RATE;
                one_hints.push(i);
            }
        }

        Self { bitvec, data, one_hints, total_ones }
    }

    /// Estimates the allocation size of the flat-popcount data structure in bits.
    pub fn size_bits(&self) -> usize {
        self.data.len() * std::mem::size_of::<L1L2Data>() * 8
            + self.one_hints.len() * std::mem::size_of::<Hint>() * 8
    }

    /// Returns a reference to the underlying bit vector.
    pub fn bitvec(&self) -> &BitVec { self.bitvec.borrow() }

    /// Returns the number of `1`-bits in the bit vector.
    pub fn count_ones(&self) -> u64 { self.total_ones }

    /// Returns the number of `0`-bits in the bit vector.
    pub fn count_zeros(&self) -> u64 { self.bitvec().len() as u64 - self.count_ones() }

    /// Returns the number of `1`-bits up to and not including the given index.
    ///
    /// # Algorithm
    ///
    /// The query is answered using up to three subqueries, with short circuting
    /// if a query falls exactly between two blocks.
    ///
    /// 1. Look up the number of ones that occur up to the corresponding L1 block.
    /// 2. Look up the number of `1` that occour in the L1 block up to the
    /// corresponding L2 block.
    /// 3. Compute the rank in the L2 block using native POPCNT instructions.
    ///
    /// Because of the layout of both the bit vector and the interleaved L1/L2
    /// indices, this will cause at most two cache misses (assuming 512 bit
    /// cache lines).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds for `self.bitvec`.
    pub fn rank1(&self, index: usize) -> u64 {
        if index >= self.bitvec().len() {
            index_out_of_bounds_fail(index, self.bitvec().len());
        }

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
        let l2_block = &self.bitvec().aligned_blocks()[aligned_index].0;

        let block_index = sub_index / Block::BITS;
        let sub_index = sub_index % Block::BITS;

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

    /// Returns the number of `0`-bits up to and not including the given index.
    ///
    /// See [`rank1`](Self::rank1) for a detailed description of the algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds for `self.bitvec`.
    pub fn rank0(&self, index: usize) -> u64 { index as u64 - self.rank1(index) }

    /// Returns the index of the first `1`-bit with the given rank, i.e. the `rank`th `1`-bit.
    ///
    /// # Algorithm
    ///
    /// 1. Look up the closest preceeding sampled `1`-bit.
    /// 2. Starting at the hinted at position, scan forward through L1 blocks to
    /// find the L1 block that contains the desired `1`-bit.
    /// 3. Scan forward through the contained L2 blocks to find the desired L2 block.
    /// 4. Scan forward through bit vector blocks using POPCNT instructions.
    /// 5. Perform a local select on the bit vector block.
    ///
    /// # Panics
    ///
    /// Panics if the bit vector does not contain more than `rank` `1`-bits.
    pub fn select1(&self, rank: u64) -> usize {
        if rank >= self.total_ones {
            select_out_of_bounds_fail(rank, self.total_ones)
        }

        let hint_index = (rank / config::SELECT_SAMPLE_RATE) as usize;
        let hint = self.one_hints[hint_index] as usize;

        debug_assert!(self.data[hint].l1_ones() <= rank);
        let (l1_index, l1_data) = zip(hint.., &self.data[hint..])
            .take_while(|(_, l1)| l1.l1_ones() <= rank)
            .last()
            .unwrap();

        let sub_rank = (rank - l1_data.l1_ones()) as u32;
        debug_assert!(sub_rank <= u16::MAX.into());

        // todo make sure this is unrolled
        // todo binary search
        let (l2_index, l2_ones) = (0..config::L2_PER_L1)
            .map(|i| (i, u32::from(l1_data.l2_ones(i))))
            .take_while(|(_, l2)| *l2 <= sub_rank)
            .last()
            .unwrap();

        let mut sub_rank = sub_rank - l2_ones;
        debug_assert!(sub_rank < config::L2_SIZE_BITS as u32);

        let block_index = l1_index * config::L2_PER_L1 + l2_index;
        let l2_block = self.bitvec().aligned_blocks()[block_index];

        for (i, block) in l2_block.0.iter().enumerate() {
            let block_ones = block.count_ones();
            if sub_rank < block_ones {
                return block_index * config::L2_SIZE_BITS
                    + i * Block::BITS
                    + block.select1(sub_rank) as usize;
            }
            sub_rank -= block_ones;
        }
        unreachable!();
    }
}

impl L1L2Data {
    #[doc(hidden)]
    const U12_MASK: u128 = 0b1111_1111_1111;
    #[doc(hidden)]
    const U44_MASK: u128 = 0xFFF_FFFF_FFFF;

    /// TODO
    #[allow(clippy::erasing_op, clippy::identity_op)]
    pub fn new(l1_ones: u64, l2_ones: &[u16; 7]) -> Self {
        let [l2_1, l2_2, l2_3, l2_4, l2_5, l2_6, l2_7] = l2_ones;
        Self(
            (Self::U44_MASK & u128::from(l1_ones))
                | (Self::U12_MASK & u128::from(*l2_1)) << (44 + 0 * 12)
                | (Self::U12_MASK & u128::from(*l2_2)) << (44 + 1 * 12)
                | (Self::U12_MASK & u128::from(*l2_3)) << (44 + 2 * 12)
                | (Self::U12_MASK & u128::from(*l2_4)) << (44 + 3 * 12)
                | (Self::U12_MASK & u128::from(*l2_5)) << (44 + 4 * 12)
                | (Self::U12_MASK & u128::from(*l2_6)) << (44 + 5 * 12)
                | (Self::U12_MASK & u128::from(*l2_7)) << (44 + 6 * 12),
        )
    }

    /// Returns the number of `1`-bits up to the beginning of the L1-block.
    pub fn l1_ones(self) -> u64 { (Self::U44_MASK & self.0) as u64 }

    // todo the general case here is terrible
    // todo what about out of bounds
    /// Returns the number of `1`-bits from the beginning of the L1-block up to the given L2-block.
    pub fn l2_ones(self, index: usize) -> u16 {
        if index > 0 {
            (Self::U12_MASK & (self.0 >> (index * 12 + 44 - 12))) as u16
        } else {
            0
        }
    }
}

impl fmt::Debug for L1L2Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l2_ones: [_; 8] = std::array::from_fn(|i| self.l2_ones(i));
        f.debug_struct("L1Data")
            .field("l1_ones", &self.l1_ones())
            .field("l2_ones", &l2_ones)
            .finish()
    }
}

#[doc(hidden)]
#[cold]
#[inline(never)]
#[track_caller]
fn index_out_of_bounds_fail(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the length is {} but the index is {}", len, index)
}

#[doc(hidden)]
#[cold]
#[inline(never)]
#[track_caller]
fn select_out_of_bounds_fail(rank: u64, total_ones: u64) -> ! {
    panic!(
        "select out of bounds: the bit vector contains {} ones but the rank is {}",
        total_ones, rank
    )
}

#[cfg(test)]
mod test {
    use std::iter::repeat;

    use super::{config::*, *};
    use crate::bitvec::BitVec;

    #[test]
    fn test_l1l2_layout() {
        assert_eq!(16, std::mem::size_of::<L1L2Data>());
        assert_eq!(16, std::mem::align_of::<L1L2Data>());
    }

    #[test]
    fn test_l1l2_construction() {
        let data = L1L2Data::new(42, &[1, 2, 3, 4, 5, 6, 4095]);
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
        let _rank = FlatPopcount::new(&bitvec);
    }

    #[test]
    fn test_rank_select_all_ones() {
        let bitvec: BitVec = repeat(true).take(10_000).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(8192, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(9728, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(8512, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(1527, rank.rank1(1527));

        assert_eq!(0, rank.rank0(0));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(0, rank.rank0(1527));

        assert_eq!(0, rank.select1(0));
        assert_eq!(2 * config::L1_SIZE_BITS, rank.select1(8192));
        assert_eq!(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS, rank.select1(9728));
        assert_eq!(2 * L1_SIZE_BITS + 5 * Block::BITS, rank.select1(8512));
        assert_eq!(4620, rank.select1(4620));
    }

    #[test]
    fn test_rank_select_all_zeros() {
        let bitvec: BitVec = repeat(false).take(10_000).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(0, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(0, rank.rank1(1527));

        assert_eq!(0, rank.rank0(0));
        assert_eq!(8192, rank.rank0(2 * L1_SIZE_BITS));
        assert_eq!(9728, rank.rank0(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(8512, rank.rank0(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(1527, rank.rank0(1527));
    }

    #[test]
    fn test_rank_select_exact() {
        let bitvec: BitVec = repeat(true).take(config::L1_SIZE_BITS * 4).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(8192, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(9728, rank.rank1(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(8512, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(1527, rank.rank1(1527));

        assert_eq!(0, rank.rank0(0));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS));
        assert_eq!(0, rank.rank0(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(0, rank.rank0(1527));

        assert_eq!(0, rank.select1(0));
        assert_eq!(2 * config::L1_SIZE_BITS, rank.select1(8192));
        assert_eq!(2 * L1_SIZE_BITS + 3 * L2_SIZE_BITS, rank.select1(9728));
        assert_eq!(2 * L1_SIZE_BITS + 5 * Block::BITS, rank.select1(8512));
        assert_eq!(4620, rank.select1(4620));
    }

    #[test]
    fn test_rank_select_non_trivial() {
        let bitvec: BitVec = (0..15_000).map(|i| i % 3 == 0).collect();
        let rank = FlatPopcount::new(&bitvec);

        assert_eq!(0, rank.rank1(0));
        assert_eq!(2731, rank.rank1(2 * L1_SIZE_BITS));
        assert_eq!(3926, rank.rank1(2 * L1_SIZE_BITS + 7 * L2_SIZE_BITS));
        assert_eq!(2838, rank.rank1(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(509, rank.rank1(1527));

        assert_eq!(0, rank.rank0(0));
        assert_eq!(5461, rank.rank0(2 * L1_SIZE_BITS));
        assert_eq!(7850, rank.rank0(2 * L1_SIZE_BITS + 7 * L2_SIZE_BITS));
        assert_eq!(5674, rank.rank0(2 * L1_SIZE_BITS + 5 * Block::BITS));
        assert_eq!(1018, rank.rank0(1527));

        assert_eq!(0, rank.select1(0));
        assert_eq!(8193, rank.select1(2731));
        assert_eq!(11778, rank.select1(3926));
        assert_eq!(8514, rank.select1(2838));
        assert_eq!(4173, rank.select1(1391));
    }
}
