use super::{bit::Bit, int_ring::IntRing2k};
use num_traits::{One, Zero};
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Neg, Not, Rem, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, PartialOrd, Eq, Ord, Hash,
)]
#[serde(bound = "")]
#[repr(transparent)]
pub struct RingElement<T: IntRing2k + std::fmt::Display>(pub T);

pub struct BitIter<'a, T: IntRing2k> {
    bits:    &'a RingElement<T>,
    index:   usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: IntRing2k> Iterator for BitIter<'_, T> {
    type Item = Bit;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= T::K {
            None
        } else {
            let bit = ((self.bits.0 >> self.index) & T::one()) == T::one();
            self.index += 1;
            Some(Bit(bit as u8))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = T::K - self.index;
        (len, Some(len))
    }
}

impl<T: IntRing2k> RingElement<T> {
    /// Safe because RingElement has repr(transparent)
    pub fn convert_slice(vec: &[Self]) -> &[T] {
        // SAFETY: RingElement has repr(transparent)
        unsafe { &*(vec as *const [Self] as *const [T]) }
    }

    /// Safe because RingElement has repr(transparent)
    pub fn convert_vec(vec: Vec<Self>) -> Vec<T> {
        let me = ManuallyDrop::new(vec);
        // SAFETY: RingElement has repr(transparent)
        unsafe { Vec::from_raw_parts(me.as_ptr() as *mut T, me.len(), me.capacity()) }
    }

    /// Safe because RingElement has repr(transparent)
    pub fn convert_slice_rev(vec: &[T]) -> &[Self] {
        // SAFETY: RingElement has repr(transparent)
        unsafe { &*(vec as *const [T] as *const [Self]) }
    }

    /// Safe because RingElement has repr(transparent)
    pub fn convert_vec_rev(vec: Vec<T>) -> Vec<Self> {
        let me = ManuallyDrop::new(vec);
        // SAFETY: RingElement has repr(transparent)
        unsafe { Vec::from_raw_parts(me.as_ptr() as *mut Self, me.len(), me.capacity()) }
    }

    pub fn convert(self) -> T {
        self.0
    }

    pub(crate) fn bit_iter(&self) -> BitIter<'_, T> {
        BitIter {
            bits:    self,
            index:   0,
            _marker: PhantomData,
        }
    }

    pub fn get_bit(&self, index: usize) -> Self {
        RingElement((self.0 >> index) & T::one())
    }

    pub(crate) fn get_bit_as_bit(&self, index: usize) -> RingElement<Bit> {
        let bit = ((self.0 >> index) & T::one()) == T::one();
        RingElement(Bit(bit as u8))
    }

    pub fn upgrade_to_128(self) -> RingElement<u128> {
        RingElement(self.0.into())
    }
}

impl<T: IntRing2k + std::fmt::Display> std::fmt::Display for RingElement<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl<T: IntRing2k> Add for RingElement<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_add(&rhs.0))
    }
}

impl<T: IntRing2k> Add<&Self> for RingElement<T> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        Self(self.0.wrapping_add(&rhs.0))
    }
}

impl<T: IntRing2k> AddAssign for RingElement<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0.wrapping_add_assign(&rhs.0)
    }
}

impl<T: IntRing2k> AddAssign<&Self> for RingElement<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.0.wrapping_add_assign(&rhs.0)
    }
}

impl<T: IntRing2k> Sub for RingElement<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_sub(&rhs.0))
    }
}

impl<T: IntRing2k> Sub<&Self> for RingElement<T> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self(self.0.wrapping_sub(&rhs.0))
    }
}

impl<T: IntRing2k> SubAssign for RingElement<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.wrapping_sub_assign(&rhs.0)
    }
}

impl<T: IntRing2k> SubAssign<&Self> for RingElement<T> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.0.wrapping_sub_assign(&rhs.0)
    }
}

impl<T: IntRing2k> Mul<T> for RingElement<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.wrapping_mul(&rhs))
    }
}

impl<T: IntRing2k> Mul<&T> for RingElement<T> {
    type Output = Self;

    fn mul(self, rhs: &T) -> Self::Output {
        Self(self.0.wrapping_mul(rhs))
    }
}

impl<T: IntRing2k> Mul for RingElement<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_mul(&rhs.0))
    }
}

impl<T: IntRing2k> Mul<&Self> for RingElement<T> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        Self(self.0.wrapping_mul(&rhs.0))
    }
}

impl<T: IntRing2k> MulAssign for RingElement<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.wrapping_mul_assign(&rhs.0)
    }
}

impl<T: IntRing2k> MulAssign<&Self> for RingElement<T> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0.wrapping_mul_assign(&rhs.0)
    }
}

impl<T: IntRing2k> MulAssign<T> for RingElement<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.0.wrapping_mul_assign(&rhs)
    }
}

impl<T: IntRing2k> MulAssign<&T> for RingElement<T> {
    fn mul_assign(&mut self, rhs: &T) {
        self.0.wrapping_mul_assign(rhs)
    }
}

impl<T: IntRing2k> Zero for RingElement<T> {
    fn zero() -> Self {
        Self(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T: IntRing2k> One for RingElement<T> {
    fn one() -> Self {
        Self(T::one())
    }
}

impl<T: IntRing2k> Neg for RingElement<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.wrapping_neg())
    }
}

impl<T: IntRing2k> Distribution<RingElement<T>> for Standard
where
    Standard: Distribution<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> RingElement<T> {
        RingElement(rng.gen())
    }
}

impl<T: IntRing2k> Not for RingElement<T> {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl<T: IntRing2k> BitXor for RingElement<T> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        RingElement(self.0 ^ rhs.0)
    }
}

impl<T: IntRing2k> BitXor<&Self> for RingElement<T> {
    type Output = Self;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        RingElement(self.0 ^ rhs.0)
    }
}

impl<T: IntRing2k> BitXorAssign for RingElement<T> {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl<T: IntRing2k> BitXorAssign<&Self> for RingElement<T> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl<T: IntRing2k> BitOr for RingElement<T> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        RingElement(self.0 | rhs.0)
    }
}

impl<T: IntRing2k> BitOr<&Self> for RingElement<T> {
    type Output = Self;

    fn bitor(self, rhs: &Self) -> Self::Output {
        RingElement(self.0 | rhs.0)
    }
}

impl<T: IntRing2k> BitOrAssign for RingElement<T> {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl<T: IntRing2k> BitOrAssign<&Self> for RingElement<T> {
    fn bitor_assign(&mut self, rhs: &Self) {
        self.0 |= rhs.0;
    }
}

impl<T: IntRing2k> BitAnd for RingElement<T> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        RingElement(self.0 & rhs.0)
    }
}

impl<T: IntRing2k> BitAnd<T> for RingElement<T> {
    type Output = Self;

    fn bitand(self, rhs: T) -> Self::Output {
        RingElement(self.0 & rhs)
    }
}

impl<T: IntRing2k> BitAnd<&Self> for RingElement<T> {
    type Output = Self;

    fn bitand(self, rhs: &Self) -> Self::Output {
        RingElement(self.0 & rhs.0)
    }
}

impl<T: IntRing2k> BitAndAssign for RingElement<T> {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl<T: IntRing2k> BitAndAssign<&Self> for RingElement<T> {
    fn bitand_assign(&mut self, rhs: &Self) {
        self.0 &= rhs.0;
    }
}

impl<T: IntRing2k> Shl<u32> for RingElement<T> {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self::Output {
        RingElement(self.0.wrapping_shl(rhs))
    }
}

impl<T: IntRing2k> ShlAssign<u32> for RingElement<T> {
    fn shl_assign(&mut self, rhs: u32) {
        self.0.wrapping_shl_assign(rhs)
    }
}

impl<T: IntRing2k> Shr<u32> for RingElement<T> {
    type Output = Self;

    fn shr(self, rhs: u32) -> Self::Output {
        RingElement(self.0.wrapping_shr(rhs))
    }
}

impl<T: IntRing2k> ShrAssign<u32> for RingElement<T> {
    fn shr_assign(&mut self, rhs: u32) {
        self.0.wrapping_shr_assign(rhs)
    }
}

impl<T: IntRing2k> Rem<T> for RingElement<T> {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        RingElement(self.0 % rhs)
    }
}

#[cfg(test)]
mod unsafe_test {
    use super::*;
    use aes_prng::AesRng;
    use rand::{Rng, SeedableRng};

    const ELEMENTS: usize = 100;

    fn conversion_test<T: IntRing2k>()
    where
        Standard: Distribution<T>,
    {
        let mut rng = AesRng::from_entropy();
        let t_vec: Vec<T> = (0..ELEMENTS).map(|_| rng.gen()).collect();
        let rt_vec: Vec<RingElement<T>> =
            (0..ELEMENTS).map(|_| rng.gen::<RingElement<T>>()).collect();

        // Convert vec<T> to vec<R<T>>
        let t_conv = RingElement::convert_vec_rev(t_vec.to_owned());
        assert_eq!(t_conv.len(), t_vec.len());
        for (a, b) in t_conv.iter().zip(t_vec.iter()) {
            assert_eq!(a.0, *b)
        }

        // Convert slice vec<T> to vec<R<T>>
        let t_conv = RingElement::convert_slice_rev(&t_vec);
        assert_eq!(t_conv.len(), t_vec.len());
        for (a, b) in t_conv.iter().zip(t_vec.iter()) {
            assert_eq!(a.0, *b)
        }

        // Convert vec<R<T>> to vec<T>
        let rt_conv = RingElement::convert_vec(rt_vec.to_owned());
        assert_eq!(rt_conv.len(), rt_vec.len());
        for (a, b) in rt_conv.iter().zip(rt_vec.iter()) {
            assert_eq!(*a, b.0)
        }

        // Convert slice vec<R<T>> to vec<T>
        let rt_conv = RingElement::convert_slice(&rt_vec);
        assert_eq!(rt_conv.len(), rt_vec.len());
        for (a, b) in rt_conv.iter().zip(rt_vec.iter()) {
            assert_eq!(*a, b.0)
        }
    }

    macro_rules! test_impl {
        ($([$ty:ty,$fn:ident]),*) => ($(
            #[test]
            fn $fn() {
                conversion_test::<$ty>();
            }
        )*)
    }

    test_impl! {
        [Bit, bit_test],
        [u8, u8_test],
        [u16, u16_test],
        [u32, u32_test],
        [u64, u64_test],
        [u128, u128_test]
    }
}
