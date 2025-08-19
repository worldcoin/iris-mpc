use eyre::{eyre, Error};
use num_traits::{
    AsPrimitive, One, WrappingAdd, WrappingMul, WrappingNeg, WrappingShl, WrappingShr, WrappingSub,
    Zero,
};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use serde::{Deserialize, Serialize};
use std::ops::{
    Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Neg, Not, Rem, Shl,
    Shr, Sub,
};

/// Bit is a sharable wrapper for a boolean value
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    bytemuck::NoUninit,
    bytemuck::AnyBitPattern,
)]
#[repr(transparent)]
/// This transparent is important due to some typecasts!
pub struct Bit(u8);

impl AsPrimitive<Self> for Bit {
    fn as_(self) -> Self {
        self
    }
}

impl std::fmt::Display for Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            1 => write!(f, "1"),
            0 => write!(f, "0"),
            _ => unreachable!(),
        }
    }
}

impl Bit {
    pub fn new(value: bool) -> Self {
        Self(value as u8)
    }

    pub fn convert(self) -> bool {
        debug_assert!(self.0 == 0 || self.0 == 1);
        self.0 == 1
    }
}

impl TryFrom<u8> for Bit {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Bit(0)),
            1 => Ok(Bit(1)),
            _ => Err(eyre!("Conversion Error Bit From u8")),
        }
    }
}

impl TryFrom<usize> for Bit {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Bit(0)),
            1 => Ok(Bit(1)),
            _ => Err(eyre!("Conversion error Bit from usize")),
        }
    }
}

impl Add for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self ^ rhs
    }
}

impl Add<&Bit> for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, rhs: &Self) -> Self::Output {
        self ^ rhs
    }
}

impl WrappingAdd for Bit {
    #[inline(always)]
    fn wrapping_add(&self, rhs: &Self) -> Self::Output {
        *self ^ *rhs
    }
}

impl Sub for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self ^ rhs
    }
}

impl Sub<&Bit> for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self::Output {
        self ^ rhs
    }
}

impl WrappingSub for Bit {
    #[inline(always)]
    fn wrapping_sub(&self, rhs: &Self) -> Self::Output {
        *self ^ *rhs
    }
}

impl Neg for Bit {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        self
    }
}

impl WrappingNeg for Bit {
    #[inline(always)]
    fn wrapping_neg(&self) -> Self {
        -*self
    }
}

impl BitXor for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXor<&Bit> for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: &Self) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bit {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl BitXorAssign<&Bit> for Bit {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl BitOr for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOr<&Bit> for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: &Self) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bit {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<&Bit> for Bit {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: &Self) {
        self.0 |= rhs.0;
    }
}

impl Not for Bit {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        Self(self.0 ^ 1)
    }
}

impl BitAnd for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAnd<&Bit> for Bit {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: &Self) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bit {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitAndAssign<&Bit> for Bit {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: &Self) {
        self.0 &= rhs.0;
    }
}

impl Mul for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self & rhs
    }
}

impl Mul<&Bit> for Bit {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        self & rhs
    }
}

impl WrappingMul for Bit {
    #[inline(always)]
    fn wrapping_mul(&self, rhs: &Self) -> Self::Output {
        *self & *rhs
    }
}

impl Zero for Bit {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Bit {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
}

impl From<Bit> for u8 {
    #[inline(always)]
    fn from(other: Bit) -> Self {
        other.0
    }
}

impl From<bool> for Bit {
    #[inline(always)]
    fn from(other: bool) -> Self {
        Bit(other as u8)
    }
}

impl From<Bit> for bool {
    #[inline(always)]
    fn from(other: Bit) -> Self {
        other.0 == 1
    }
}

impl Shl<usize> for Bit {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self {
        if rhs == 0 {
            self
        } else {
            Self(0)
        }
    }
}

impl WrappingShl for Bit {
    #[inline(always)]
    fn wrapping_shl(&self, rhs: u32) -> Self {
        *self << rhs as usize
    }
}

impl Shr<usize> for Bit {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self {
        if rhs == 0 {
            self
        } else {
            Self(0)
        }
    }
}

impl WrappingShr for Bit {
    #[inline(always)]
    fn wrapping_shr(&self, rhs: u32) -> Self {
        *self >> rhs as usize
    }
}

impl Distribution<Bit> for Standard {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bit {
        Bit(rng.gen::<bool>() as u8)
    }
}

impl AsRef<Bit> for Bit {
    fn as_ref(&self) -> &Bit {
        self
    }
}

impl From<Bit> for u128 {
    fn from(val: Bit) -> Self {
        u128::from(val.0)
    }
}

impl Rem for Bit {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        match rhs {
            Bit(0) => panic!("Division by zero"),
            Bit(1) => Bit(0),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use rand::Rng;

    #[test]
    fn test_aritmetic() {
        let mut rng = rand::thread_rng();
        let a: bool = rng.gen();
        let b: bool = rng.gen();
        let bit_a = Bit::new(a);
        let bit_b = Bit::new(b);
        let bit_b_ref = &bit_b;

        // Addition and subtraction
        let expected_xor = Bit::new(a ^ b);
        assert_eq!(bit_a + bit_b, expected_xor);
        assert_eq!(bit_a + bit_b_ref, expected_xor);
        assert_eq!(bit_a.wrapping_add(&bit_b), expected_xor);
        assert_eq!(bit_a - bit_b, expected_xor);
        assert_eq!(bit_a - bit_b.as_ref(), expected_xor);
        assert_eq!(bit_a.wrapping_sub(&bit_b), expected_xor);
        assert_eq!(bit_a ^ bit_b, expected_xor);
        assert_eq!(bit_a ^ bit_b_ref, expected_xor);
        let mut bit_c = bit_a;
        bit_c ^= bit_b;
        assert_eq!(bit_c, expected_xor);
        let mut bit_c = bit_a;
        bit_c ^= bit_b_ref;
        assert_eq!(bit_c, expected_xor);

        // Multiplication
        let expected_and = Bit::new(a & b);
        assert_eq!(bit_a * bit_b, expected_and);
        assert_eq!(bit_a * bit_b_ref, expected_and);
        assert_eq!(bit_a.wrapping_mul(&bit_b), expected_and);
        let mut bit_c = bit_a;
        bit_c &= bit_b;
        assert_eq!(bit_c, expected_and);
        let mut bit_c = bit_a;
        bit_c &= bit_b_ref;
        assert_eq!(bit_c, expected_and);

        // OR
        let expected_or = Bit::new(a | b);
        assert_eq!(bit_a | bit_b, expected_or);
        assert_eq!(bit_a | bit_b_ref, expected_or);
        let mut c = bit_a;
        c |= bit_b;
        assert_eq!(c, expected_or);
        let mut c = bit_a;
        c |= &bit_b;
        assert_eq!(c, expected_or);

        // Modulo operation
        assert_eq!(bit_a % Bit::new(true), Bit::new(false));
        let is_err = std::panic::catch_unwind(|| {
            let _ = bit_a % Bit::new(false);
        });
        assert!(is_err.is_err());

        // Negation
        assert_eq!(-bit_a, bit_a);
        assert_eq!(bit_a.wrapping_neg(), bit_a);

        // Not
        assert_eq!(!bit_a, Bit::new(!a));

        // Identities
        assert_eq!(Bit::zero(), Bit::new(false));
        assert!(Bit::new(false).is_zero());
        assert_eq!(Bit::one(), Bit::new(true));

        // Shifting
        let bit_a = rng.gen::<Bit>();
        assert_eq!(bit_a << 1, Bit::new(false));
        assert_eq!(bit_a >> 1, Bit::new(false));
        assert_eq!(bit_a.wrapping_shl(1), Bit::new(false));
        assert_eq!(bit_a.wrapping_shr(1), Bit::new(false));
        assert_eq!(bit_a << 0, bit_a);
        assert_eq!(bit_a >> 0, bit_a);
        assert_eq!(bit_a.wrapping_shl(0), bit_a);
        assert_eq!(bit_a.wrapping_shr(0), bit_a);
    }

    #[test]
    fn test_conversion() -> Result<()> {
        let expected = vec![Bit::new(false), Bit::new(true)];

        let a: Vec<Bit> = [0_u8, 1_u8]
            .into_iter()
            .map(Bit::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(a, expected);

        let a: Vec<Bit> = [0_usize, 1_usize]
            .into_iter()
            .map(Bit::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(a, expected);

        assert!(Bit::try_from(2_u8).is_err());
        assert!(Bit::try_from(2_usize).is_err());

        let a = expected.iter().map(|x| x.as_()).collect::<Vec<_>>();
        assert_eq!(a, expected);

        let a = expected.iter().map(|x| u8::from(*x)).collect::<Vec<u8>>();
        assert_eq!(a, vec![0, 1]);

        let a = expected
            .iter()
            .map(|x| u128::from(*x))
            .collect::<Vec<u128>>();
        assert_eq!(a, vec![0, 1]);

        let a = expected
            .iter()
            .map(|x| bool::from(*x))
            .collect::<Vec<bool>>();
        assert_eq!(a, vec![false, true]);

        let a = [false, true]
            .into_iter()
            .map(Bit::from)
            .collect::<Vec<Bit>>();
        assert_eq!(a, expected);

        Ok(())
    }

    #[test]
    fn test_display() {
        let a = Bit::new(false);
        let b = Bit::new(true);
        assert_eq!(format!("{}", a), "0");
        assert_eq!(format!("{}", b), "1");
    }
}
