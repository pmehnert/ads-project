//! Functionality for dealing with cartesian trees of small integer sequences.
//!
//! # Cartesian Trees
//!
//! Given a sequence `A`, a _cartesian tree_ of `A` is a labeled binary tree where
//!
//! - the root `r` is labeled with the smallest element in `A`,
//! - the left and right subtrees of `r` are recursively defined as the cartesian
//! trees of the sequences to the left and right of the minimum.

use std::marker::PhantomData;

use crate::{int::IndexInt, rmq::RangeMinimum, AllocationSize};

/// A compact, unqiue representation of a (small) cartesian tree
///
/// Each cartesian tree has a unqiue _signature_, which is defined by "the number
/// of nodes removed from the rightmost path of the tree when inserting the `i`th
/// element" [\[2\]](crate::rmq::Cartesian#references).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CartesianTree<Idx> {
    tree: usize,
    size: usize,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> CartesianTree<Idx> {
    pub fn new(tree: usize, size: usize) -> Self {
        debug_assert!(2 * size <= Idx::BITS as usize);
        Self { tree, size, _phantom: PhantomData }
    }

    pub fn get(&self) -> Idx { Idx::from_usize(self.tree) }

    /// Returns `true` for every valid cartesian tree and **may** return `false`
    /// for invalid ones (e.g. too many`1`-bits).
    pub fn maybe_valid(&self) -> bool { (self.tree.count_ones() as usize) < self.size }
}

impl<Idx: IndexInt> RangeMinimum for CartesianTree<Idx> {
    type Output = usize;

    /// Returns `RMQ(lower, upper)` using the cartesian [\[2\]] or [`None`] if
    /// the range is empty or out of bounds.
    ///
    /// [\[2\]]: (crate::rmq::Cartesian#references)
    fn range_min(&self, lower: usize, upper: usize) -> Option<usize> {
        if lower <= upper && upper < self.size {
            let mut index = self.tree.count_ones() as usize + self.size - 1;
            if index >= usize::BITS as usize {
                return Some(lower);
            }

            let mut elem = 0usize;
            while elem < lower {
                index -= 1;
                elem += usize::from(self.tree & (1 << index) == 0);
            }

            let (mut min, mut depth) = (lower, 0u32);
            while elem < upper {
                index -= 1;
                elem += 1;

                let mut delta = 0u32;
                while self.tree & (1 << index) != 0 {
                    delta += 1;
                    index -= 1;
                }

                (min, depth) = match depth.checked_sub(delta) {
                    Some(depth) => (min, depth + 1),
                    None => (elem, 0),
                };
            }
            Some(min)
        } else {
            None
        }
    }
}

/// Functionality for building cartesian trees.
#[derive(Debug, Clone)]
pub struct CartesianBuilder<Idx> {
    size: usize,
    stack: Vec<u64>,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> CartesianBuilder<Idx> {
    pub fn new(size: usize) -> Self {
        Self { stack: Vec::with_capacity(size), size, _phantom: PhantomData }
    }

    /// Returns the [representation](CartesianTree) of the cartesian tree for
    /// the given slice of values.
    ///
    /// Scans over the slice, keeps track of the rightmost path of the tree and
    /// checks how many values are removed from the path each iteration.
    ///
    /// If `values` contains too few elements, it is padded with implicit,
    /// infinitely large values.
    pub fn build(&mut self, values: &[u64]) -> CartesianTree<Idx> {
        debug_assert!(values.len() <= self.size);

        self.stack.clear();
        self.stack.extend(values.first());

        let word = values.iter().skip(1).fold(usize::MAX, |word, value| {
            let pos = self.stack.iter().rposition(|x| x <= value).map_or(0, |i| i + 1);
            let shift = self.stack.len() - pos;
            self.stack.truncate(pos);
            self.stack.push(*value);

            (word << (shift + 1)) | 1
        });

        CartesianTree::new(!word << (self.size - values.len()), self.size)
    }
}

impl<Idx> AllocationSize for CartesianBuilder<Idx> {
    fn size_bytes(&self) -> usize { self.stack.size_bytes() }
}

/// A lookup table that stores the answer to every possible RMQ for every
/// cartesian tree with a given size.
#[derive(Debug, Clone)]
pub struct CartesianTable<Idx> {
    size: usize,
    table: Vec<u8>,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> CartesianTable<Idx> {
    /// Constructs the lookup table by calculating every RMQ for every cartesian
    /// tree with the given size.
    ///
    /// The function simply iterates over every bit string of length `2*size-1`
    /// and answers each RMQ using the algorithm from [\[2\]] (see also
    /// [`CartesianTree::range_min`]). Note that not every bit string is valid,
    /// which generates "garbage" that is never actually accessed.
    ///
    /// [\[2\]]: crate::rmq::Cartesian#references
    pub fn new(size: usize) -> Self {
        assert!(0 < size && 2 * size <= Idx::BITS as usize);

        let num_trees = 2usize.pow(2 * size as u32);

        let mut table = vec![0u8; size * size * num_trees];
        let chunks = table.chunks_exact_mut(size * size);

        for (tree, dst) in chunks.enumerate() {
            let tree = CartesianTree::<Idx>::new(tree, size);
            if tree.maybe_valid() {
                for i in 0..size {
                    for j in i..size {
                        dst[i * size + j] = tree.range_min(i, j).unwrap() as u8;
                    }
                }
            }
        }

        Self { size, table, _phantom: PhantomData }
    }

    /// Returns the precomputed results for `RMQ(lower, upper)` on the given
    /// cartesian tree.
    pub fn get(&self, tree: Idx, lower: usize, upper: usize) -> u8 {
        self.table[(self.size * tree.to_usize() + lower) * self.size + upper]
    }
}

impl<Idx> Default for CartesianTable<Idx> {
    fn default() -> Self { Self { table: Vec::new(), size: 0, _phantom: PhantomData } }
}

impl<Idx> AllocationSize for CartesianTable<Idx> {
    fn size_bytes(&self) -> usize { self.table.size_bytes() }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_builder() {
        let mut builder = CartesianBuilder::<u16>::new(1);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1]).get());

        let mut builder = CartesianBuilder::<u16>::new(2);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1, 2]).get());
        assert_eq!(0b010, builder.build(&[2, 1]).get());

        let mut builder = CartesianBuilder::<u16>::new(3);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1, 2]).get());
        assert_eq!(0b0100, builder.build(&[2, 1]).get());
        assert_eq!(0b01010, builder.build(&[3, 2, 1]).get());

        let mut builder = CartesianBuilder::<u16>::new(4);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0011010, builder.build(&[3, 4, 2, 1]).get());

        let mut builder = CartesianBuilder::<u16>::new(8);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b001011000100, builder.build(&[1, 7, 3, 0, 2, 6, 4, 5]).get());
    }

    #[test]
    fn test_cartesian_tree_len_3_unique() {
        let types = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 2, 1], [2, 3, 1]];

        let mut builder = CartesianBuilder::<u8>::new(3);
        let mut trees = HashSet::new();
        for t in types {
            assert!(trees.insert(builder.build(&t)));
        }
    }

    #[test]
    fn test_cartesian_tree_len_4_unique() {
        #[rustfmt::skip]
        let types  = [
            [1, 2, 3, 4], [1, 3, 4, 2], [1, 4, 3, 2],
            [1, 3, 2, 4], [2, 1, 3, 4], [2, 1, 4, 3],
            [4, 3, 2, 1], [2, 4, 3, 1], [2, 3, 4, 1],
            [4, 2, 3, 1], [4, 3, 1, 2], [3, 4, 1, 2],
        ];

        let mut builder = CartesianBuilder::<u8>::new(4);
        let mut trees = HashSet::new();
        for t in types {
            assert!(trees.insert(builder.build(&t)));
        }
    }

    #[test]
    fn test_cartesian_table_u8() {
        for n in 1..=4 {
            let _ = CartesianTable::<u8>::new(n);
        }
    }

    #[test]
    fn test_cartesian_table_u16() {
        for n in 1..=8 {
            let _ = CartesianTable::<u16>::new(n);
        }
    }

    #[test]
    fn test_tree_range_min() {
        fn go(size: usize, values: &mut [u64]) {
            let mut builder = CartesianBuilder::<usize>::new(size);
            while next_combination(values) {
                let tree = builder.build(values);
                for i in 0..values.len() {
                    for j in i..values.len() {
                        let min = values[i..=j].iter().min().unwrap();
                        assert_eq!(*min, values[tree.range_min(i, j).unwrap()]);
                    }
                }
            }
        }

        for n in 1..=6 {
            go(n, &mut [u64::MAX, 0, 0, 0, 0, 0][..n]);
        }
        for n in 1..=5 {
            go(8, &mut [u64::MAX, 0, 0, 0, 0, 0][..n]);
        }
    }

    #[must_use]
    fn next_combination(slice: &mut [u64]) -> bool {
        slice[0] = slice[0].wrapping_add(1);
        for i in 0..slice.len() - 1 {
            slice[i + 1] += slice[i] / slice.len() as u64;
            slice[i] %= slice.len() as u64;
        }
        slice.last().unwrap() / slice.len() as u64 == 0
    }
}
