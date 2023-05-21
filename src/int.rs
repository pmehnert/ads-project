//! Basic traits for generic integers.

use std::fmt;

/// Integers that may be used to store indices.
///
/// Types implementing this trait are guaranteed to be unsigned integers with
/// value ranges of [`ZERO`](Self::ZERO) ..= [`MAX`](Self::MAX).
///
/// This traits is _sealed_ and can therefore not be implemented on foreign types.
pub trait IndexInt:
    'static + Sized + Copy + TryFrom<usize> + fmt::Display + sealed::Sealed
{
    /// The size of `Self` in bits.
    const BITS: u32;

    /// Representation of `0` for `Self`.
    const ZERO: Self;

    /// The largest values that `Self` can represent.
    const MAX: Self;

    /// Converts `value` to an instance of `Self` using a primitive cast.
    fn from_usize(value: usize) -> Self;

    /// Converts `self` to a [`usize`] using a primitive cast.
    fn to_usize(self) -> usize;
}

/// Integers with a corresponding smaller integer type.
///
/// This traits is _sealed_ and can therefore not be implemented on foreign types.
pub trait AsHalfSize: sealed::Sealed {
    /// An integer type with at least half as many bits as `Self`.
    type HalfSize;
}

/// Primitive conversion of integers to [`IndexInt`]s.
pub trait AsIndex: Sized + sealed::Sealed {
    /// Converts `self` to an instance of `Idx` using a primitve cast.
    fn to_index<Idx: IndexInt>(self) -> Idx;
}

#[doc(hidden)]
macro_rules! impl_integer_traits {
    ($($type:ty => $half_size:ty),* $(,)?) => {
        $(
            impl sealed::Sealed for $type {}
            impl IndexInt for $type {
                const BITS: u32 = <$type>::BITS;
                const ZERO: Self = 0;
                const MAX: Self = <$type>::MAX;

                fn from_usize(value: usize) -> Self {
                    debug_assert!(Self::try_from(value).is_ok());
                    value as Self
                }

                fn to_usize(self) -> usize { self as usize }
            }
            impl AsHalfSize for $type {
                type HalfSize = $half_size;
            }
        )*
    };
}

impl_integer_traits!(u64 => u32, u32 => u16, u16 => u8, u8 => u8);

#[cfg(target_pointer_width = "64")]
impl_integer_traits!(usize => u32);
#[cfg(target_pointer_width = "32")]
impl_integer_traits!(usize => u16);
#[cfg(target_pointer_width = "16")]
impl_integer_traits!(usize => u8);

impl AsIndex for usize {
    fn to_index<Idx: IndexInt>(self) -> Idx { Idx::from_usize(self) }
}

#[doc(hidden)]
mod sealed {
    pub trait Sealed {}
}
