use super::bit::Bit;
use num_traits::{
    AsPrimitive, One, WrappingAdd, WrappingMul, WrappingNeg, WrappingShl, WrappingShr, WrappingSub,
    Zero,
};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Neg, Not, Rem},
};

pub trait IntRing2k:
    std::fmt::Display
    + Serialize
    + for<'a> Deserialize<'a>
    + Default
    + WrappingAdd
    + WrappingSub
    + WrappingMul
    + WrappingNeg
    + WrappingShl
    + WrappingShr
    + Not<Output = Self>
    + BitXor<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXorAssign
    + BitAndAssign
    + BitOrAssign
    + PartialEq
    + From<bool>
    + Into<u128>
    + Copy
    + Debug
    + Zero
    + One
    + Sized
    + Send
    + Sync
    + Rem<Output = Self>
    + 'static
    + bytemuck::NoUninit
    + bytemuck::AnyBitPattern
{
    type Signed: Neg<Output = Self::Signed> + From<bool> + AsPrimitive<Self>;
    const K: usize;
    const BYTES: usize;

    /// a += b
    #[inline(always)]
    fn wrapping_add_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_add(rhs);
    }

    /// a -= b
    #[inline(always)]
    fn wrapping_sub_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_sub(rhs);
    }

    /// a = -a
    #[inline(always)]
    fn wrapping_neg_inplace(&mut self) {
        *self = self.wrapping_neg();
    }

    /// a*= b
    #[inline(always)]
    fn wrapping_mul_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_mul(rhs);
    }

    /// a <<= b
    #[inline(always)]
    fn wrapping_shl_assign(&mut self, rhs: u32) {
        *self = self.wrapping_shl(rhs);
    }

    /// a >>= b
    #[inline(always)]
    fn wrapping_shr_assign(&mut self, rhs: u32) {
        *self = self.wrapping_shr(rhs);
    }
}

impl IntRing2k for Bit {
    type Signed = Bit;
    const K: usize = 1;
    const BYTES: usize = 1;
}

impl IntRing2k for u8 {
    type Signed = i8;
    const K: usize = Self::BITS as usize;
    const BYTES: usize = Self::K / 8;
}

impl IntRing2k for u16 {
    type Signed = i16;
    const K: usize = Self::BITS as usize;
    const BYTES: usize = Self::K / 8;
}

impl IntRing2k for u32 {
    type Signed = i32;
    const K: usize = Self::BITS as usize;
    const BYTES: usize = Self::K / 8;
}

impl IntRing2k for u64 {
    type Signed = i64;
    const K: usize = Self::BITS as usize;
    const BYTES: usize = Self::K / 8;
}

impl IntRing2k for u128 {
    type Signed = i128;
    const K: usize = Self::BITS as usize;
    const BYTES: usize = Self::K / 8;
}
