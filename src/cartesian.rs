use std::{marker::PhantomData, num::Wrapping};

use crate::{int::IndexInt, AllocationSize};

/// A compact, unqiue representation of a (small) cartesian tree
///
/// Each cartesian tree has a unqiue _signature_, which is defined by "the number
/// of nodes removed from the rightmost path of the tree when inserting the `i`th
/// element" [\[2\]](crate::rmq::Cartesian#references).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tree<Idx> {
    tree: usize,
    size: usize,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> Tree<Idx> {
    pub fn new(tree: usize, size: usize) -> Self {
        debug_assert!(2 * size <= Idx::BITS as usize);
        Self { tree, size, _phantom: PhantomData }
    }

    pub fn get(&self) -> Idx { Idx::from_usize(self.tree) }

    pub fn maybe_valid(&self) -> bool { (self.tree.count_ones() as usize) < self.size }

    // todo check lower  <= uper < len
    pub fn range_min(&self, lower: usize, upper: usize) -> usize {
        let mut index = Wrapping(self.tree.count_ones() as usize + self.size - 1);
        let mut elem = 0;
        while elem <= lower {
            elem += usize::from(self.tree & (1 << index.0) == 0);
            index -= 1;
        }

        let (mut min, mut depth) = (lower, 0);
        while elem <= upper {
            let mut delta = 0;
            while self.tree & (1 << index.0) != 0 {
                delta += 1;
                index -= 1;
            }

            if delta >= depth + 1 {
                (min, depth) = (elem, 0);
            } else {
                depth = depth + 1 - delta;
            }

            elem += 1;
            index -= 1;
        }
        min
    }
}

#[derive(Debug, Clone)]
pub struct Builder<Idx> {
    size: usize,
    stack: Vec<u64>,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> Builder<Idx> {
    pub fn new(size: usize) -> Self {
        Self { stack: Vec::with_capacity(size), size, _phantom: PhantomData }
    }

    pub fn build(&mut self, values: &[u64]) -> Tree<Idx> {
        debug_assert!(values.len() <= self.size);

        let mut iter = values.iter();
        self.stack.clear();
        self.stack.extend(iter.next());

        let word = iter.fold(usize::MAX, |word, value| {
            let pos = self.stack.iter().rposition(|x| x <= value).map_or(0, |i| i + 1);
            let shift = self.stack.len() - pos;
            self.stack.truncate(pos);
            self.stack.push(*value);

            (word << shift + 1) | 1
        });

        Tree::new(!word << (self.size - values.len()), self.size)
    }
}

impl<Idx> AllocationSize for Builder<Idx> {
    fn size_bytes(&self) -> usize { self.stack.size_bytes() }
}

#[derive(Debug, Clone)]
pub struct Table<Idx> {
    size: usize,
    table: Vec<u8>,
    _phantom: PhantomData<Idx>,
}

impl<Idx: IndexInt> Table<Idx> {
    pub fn new(size: usize) -> Self {
        assert!(0 < size && 2 * size <= Idx::BITS as usize);

        let num_trees = 2usize.checked_pow(2 * size as u32).unwrap();

        let mut table = vec![0u8; size * size * num_trees];
        let chunks = table.chunks_exact_mut(size * size);


        for (tree, dst) in chunks.enumerate() {
            let tree = Tree::<Idx>::new(tree, size);
            if tree.maybe_valid() {
                for i in 0..size {
                    for j in i..size {
                        dst[i * size + j] = tree.range_min(i, j) as u8;
                    }
                }
            }
        }
        Self { size, table, _phantom: PhantomData }
    }

    pub fn range_min(&self, tree: Idx, lower: usize, upper: usize) -> u8 {
        self.table[(self.size * tree.to_usize() + lower) * self.size + upper]
    }
}

impl<Idx> Default for Table<Idx> {
    fn default() -> Self { Self { table: Vec::new(), size: 0, _phantom: PhantomData } }
}

impl<Idx> AllocationSize for Table<Idx> {
    fn size_bytes(&self) -> usize { self.table.size_bytes() }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_builder() {
        let mut builder = Builder::<u16>::new(1);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1]).get());

        let mut builder = Builder::<u16>::new(2);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1, 2]).get());
        assert_eq!(0b010, builder.build(&[2, 1]).get());

        let mut builder = Builder::<u16>::new(3);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0, builder.build(&[1, 2]).get());
        assert_eq!(0b0100, builder.build(&[2, 1]).get());
        assert_eq!(0b01010, builder.build(&[3, 2, 1]).get());

        let mut builder = Builder::<u16>::new(4);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b0011010, builder.build(&[3, 4, 2, 1]).get());

        let mut builder = Builder::<u16>::new(8);
        assert_eq!(0b0, builder.build(&[]).get());
        assert_eq!(0b001011000100, builder.build(&[1, 7, 3, 0, 2, 6, 4, 5]).get());
    }

    #[test]
    fn test_cartesian_tree_len_3_unique() {
        let types = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 2, 1], [2, 3, 1]];

        let mut builder = Builder::<u8>::new(3);
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

        let mut builder = Builder::<u8>::new(4);
        let mut trees = HashSet::new();
        for t in types {
            assert!(trees.insert(builder.build(&t)));
        }
    }

    #[test]
    fn test_cartesian_table_u8() {
        for n in 1..=4 {
            let _ = Table::<u8>::new(n);
        }
    }

    #[test]
    fn test_cartesian_table_u16() {
        for n in 1..=8 {
            let _ = Table::<u16>::new(n);
        }
    }

    #[test]
    fn test_tree_range_min() {
        fn go(size: usize, values: &mut [u64]) {
            let mut builder = Builder::<usize>::new(size);
            while next_combination(values) {
                let tree = builder.build(values);
                for i in 0..values.len() {
                    for j in i..values.len() {
                        let min = values[i..=j].iter().min().unwrap();
                        assert_eq!(*min, values[tree.range_min(i, j)]);
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
