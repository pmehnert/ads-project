//! Basic traits for generic integers.

use std::fmt;

/// Integers that may be used to store indices.
///
/// Types implementing this trait are guaranteed to be unsigned integers with
/// values ranges of [`ZERO`] ..= [`MAX`].
///
/// This traits is _sealed_ and can therefore not be implemented on foreign types.
///
/// [`ZERO`]: Self::ZERO
/// [`MAX`]: Self::MAX
pub trait IndexInt:
    'static + Sized + Copy + TryFrom<usize> + fmt::Display + sealed::Sealed
{
    /// Representation of `0` for `Self`.
    const ZERO: Self;

    /// The largest values that `Self` can represent.
    const MAX: Self;

    /// Converts `value` to an instance of `Self` using a primitive cast.
    fn from_usize(value: usize) -> Self;

    /// Converts `self` to a [`usize`] using a primitive cast.
    fn to_usize(self) -> usize;
}

/// Primitive conversion of integers to [`IndexInt`]s.
pub trait AsIndex: Sized + sealed::Sealed {
    /// Converts `self` to an instance of `Idx` using a primitve cast.
    fn to_index<Idx: IndexInt>(self) -> Idx;
}

impl AsIndex for usize {
    fn to_index<Idx: IndexInt>(self) -> Idx { Idx::from_usize(self) }
}

#[doc(hidden)]
macro_rules! impl_index_int {
    ($($type:ty),* $(,)?) => {
        $(
            impl IndexInt for $type {
                const ZERO: Self = 0;
                const MAX: Self = <$type>::MAX;

                fn from_usize(value: usize) -> Self {
                    debug_assert!(Self::try_from(value).is_ok());
                    value as Self
                }

                fn to_usize(self) -> usize { self as usize }
            }
        )*
    };
}

impl_index_int!(usize, u64, u32, u16, u8);

#[doc(hidden)]
mod sealed {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for u64 {}
    impl Sealed for u32 {}
    impl Sealed for u16 {}
    impl Sealed for u8 {}
}
