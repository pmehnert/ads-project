//! A byte packed array of equally sized integers.

use std::{fmt, iter::FusedIterator, iter::StepBy, slice::Windows};

/// A byte packed array of equally sized integers.
///
/// Each integer fits into `size_bits` bits and occupies `size_bytes` bytes in
/// the array. Bytes are stored in native endian order so that they can be
/// accessed with basic instructions.
///
/// Note that this container makes liberal use of unaligned reads and writes.
#[derive(Clone)]
pub struct PackedArray {
    bytes: Vec<u8>,
    size_bytes: usize,
    size_bits: u32,
}

impl PackedArray {
    /// Constructs an array over the `size_bits` least significant bits of the
    /// integers of `values`.
    ///
    /// # Panics
    ///
    /// Panics if `size_bits` is not in `1..=64`.
    #[inline(never)]
    pub fn new<Values>(size_bits: u32, values: Values) -> Self
    where
        Values: IntoIterator<Item = u64>,
        Values::IntoIter: DoubleEndedIterator + ExactSizeIterator,
    {
        assert!(
            (1..=64).contains(&size_bits),
            "illegal number of bits: must be in 1..=64 but is {}",
            size_bits,
        );

        let values = values.into_iter();
        let size_bytes = crate::div_ceil(size_bits as usize, 8);
        let excess_bytes = 8 - size_bytes;
        assert!((1..=8).contains(&size_bytes));

        // Allocate enough bytes so that every value can be accessed as the
        // `size_bytes` least significant bytes of a 64-bit integer.
        let mut bytes = vec![0u8; size_bytes * values.len() + excess_bytes];

        let write_ne_bytes = |(i, ne_bytes)| {
            let idx = i * size_bytes;

            // Using `get_unchecked` allows the compiler to vectorize this
            let slice = unsafe { bytes.get_unchecked_mut(idx..idx + 8) };

            let dst = <&mut [_; 8]>::try_from(slice).unwrap();
            *dst = ne_bytes;
        };

        let mask = u64::MAX >> (64 - size_bits);
        let iter = values.into_iter().map(|value| u64::to_ne_bytes(value & mask));

        // Write back to front for big endian systems, as to not overwrite
        // previously written bytes
        if cfg!(target_endian = "big") {
            iter.enumerate().rev().for_each(write_ne_bytes);
        } else {
            iter.enumerate().for_each(write_ne_bytes);
        }

        Self { bytes, size_bytes, size_bits }
    }

    /// Estimates the array's allocation size in bits.
    pub fn size_bits(&self) -> usize { self.bytes.len() * 8 }

    // Returns the number of integers in the array.
    pub fn len(&self) -> usize {
        (self.bytes.len().saturating_sub(8 - self.size_bytes)) / self.size_bytes
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Returns an iterator over the elements of the array.
    pub fn iter(&self) -> Iter<'_> { Iter::new(self) }

    pub fn index(&self, index: usize) -> u64 {
        // todo bounds checking??
        let start = index * self.size_bytes;
        let slice = &self.bytes[start..start + 8];
        let ne_bytes = <&[_; 8]>::try_from(slice).unwrap();
        let value = u64::from_ne_bytes(*ne_bytes);
        value & (u64::MAX >> (64 - self.size_bits))
    }
}

impl Default for PackedArray {
    fn default() -> Self { Self { bytes: Vec::new(), size_bytes: 1, size_bits: 1 } }
}

impl<'a> IntoIterator for &'a PackedArray {
    type IntoIter = Iter<'a>;
    type Item = u64;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl fmt::Debug for PackedArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// An iterator over the integers in a packed array.
pub struct Iter<'a> {
    inner: StepBy<Windows<'a, u8>>, // todo array_windows?!
    mask: u64,
}

#[doc(hidden)]
impl<'a> Iter<'a> {
    fn new(array: &'a PackedArray) -> Self {
        Self {
            inner: array.bytes.windows(8).step_by(array.size_bytes),
            mask: u64::MAX >> (64 - array.size_bits),
        }
    }

    fn map_fn(&self) -> impl '_ + FnOnce(&[u8]) -> u64 {
        move |slice| u64::from_ne_bytes(slice.try_into().unwrap()) & self.mask
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> { self.inner.next().map(self.map_fn()) }

    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(self.map_fn())
    }
}


impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(self.map_fn())
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth_back(n).map(self.map_fn())
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}


#[cfg(test)]
mod test {
    use std::{
        assert_eq,
        iter::{empty, zip},
    };

    use super::PackedArray;

    const SLIDES_EXAMPLE: [u64; 10] = [0, 1, 2, 4, 7, 10, 20, 21, 22, 32];

    #[test]
    fn test_slides_example() {
        let expected = [
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 2, 0, 3, 2, 0, 1, 2, 0],
            [0, 1, 2, 4, 7, 2, 4, 5, 6, 0],
            [0, 1, 2, 4, 7, 10, 4, 5, 6, 0],
            [0, 1, 2, 4, 7, 10, 20, 21, 22, 0],
            [0, 1, 2, 4, 7, 10, 20, 21, 22, 32],
        ];

        for (n, expected) in zip(1.., expected) {
            let actual = PackedArray::new(n, SLIDES_EXAMPLE);
            assert!(expected.into_iter().eq(&actual))
        }

        for n in 7..=64 {
            let actual = PackedArray::new(n, SLIDES_EXAMPLE);
            assert!(SLIDES_EXAMPLE.into_iter().eq(&actual));
        }
    }

    #[should_panic]
    #[test]
    fn test_zero_bits() { PackedArray::new(0, SLIDES_EXAMPLE); }

    #[should_panic]
    #[test]
    fn test_too_many_bits() { PackedArray::new(65, SLIDES_EXAMPLE); }

    #[test]
    fn test_len_is_empty() {
        assert!(PackedArray::new(2, empty()).is_empty());
        assert!(!PackedArray::new(2, SLIDES_EXAMPLE).is_empty());

        assert_eq!(0, PackedArray::new(3, empty()).len());
        assert_eq!(10, PackedArray::new(2, SLIDES_EXAMPLE).len());
        assert_eq!(10, PackedArray::new(64, SLIDES_EXAMPLE).len());
    }

    #[test]
    fn test_index() {
        let array = PackedArray::new(2, SLIDES_EXAMPLE);

        let expected = [0, 1, 2, 0, 3, 2, 0, 1, 2, 0];
        for (i, expected) in expected.into_iter().enumerate() {
            assert_eq!(expected, array.index(i));
        }
    }
}
