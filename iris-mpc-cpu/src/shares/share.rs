use super::{int_ring::IntRing2k, ring_impl::RingElement};
use crate::execution::player::Role;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{
    Add, AddAssign, BitAnd, BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Shl, Shr, Sub,
    SubAssign,
};

#[derive(Clone, Debug, PartialEq, Default, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
#[serde(bound = "")]
pub struct Share<T: IntRing2k> {
    pub a: RingElement<T>,
    pub b: RingElement<T>,
}

impl<T: IntRing2k> Share<T> {
    pub fn new(a: RingElement<T>, b: RingElement<T>) -> Self {
        Self { a, b }
    }

    pub fn from_const(value: T, role: Role) -> Self {
        let mut res = Self::zero();
        res.add_assign_const_role(value, role);
        res
    }

    pub fn add_assign_const_role(&mut self, other: T, role: Role) {
        match role.index() {
            0 => self.a += RingElement(other),
            1 => self.b += RingElement(other),
            2 => {}
            _ => unimplemented!(),
        }
    }

    pub fn get_a(self) -> RingElement<T> {
        self.a
    }

    pub fn get_b(self) -> RingElement<T> {
        self.b
    }

    pub fn get_ab(self) -> (RingElement<T>, RingElement<T>) {
        (self.a, self.b)
    }

    pub fn get_ab_ref(&self) -> (RingElement<T>, RingElement<T>) {
        (self.a, self.b)
    }
}

impl<T: IntRing2k> Add<&Self> for Share<T> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        Share {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<T: IntRing2k> Add<Self> for Share<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Share {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<T: IntRing2k> Sub<Self> for Share<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Share {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T: IntRing2k> Sub<&Self> for Share<T> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Share {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T: IntRing2k> AddAssign<Self> for Share<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<T: IntRing2k> AddAssign<&Self> for Share<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<T: IntRing2k> SubAssign for Share<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<T: IntRing2k> SubAssign<&Self> for Share<T> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<T: IntRing2k> Mul<RingElement<T>> for Share<T> {
    type Output = Self;

    fn mul(self, rhs: RingElement<T>) -> Self::Output {
        Share {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<T: IntRing2k> Mul<T> for Share<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self * RingElement(rhs)
    }
}

impl<T: IntRing2k> Mul<T> for &Share<T> {
    type Output = Share<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Share {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<T: IntRing2k> MulAssign<T> for Share<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.a *= rhs;
        self.b *= rhs;
    }
}

/// This is only the local part of the multiplication (so without randomness and
/// without communication)!
impl<T: IntRing2k> Mul<Self> for &Share<T> {
    type Output = RingElement<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.a * rhs.a + self.b * rhs.a + self.a * rhs.b
    }
}

impl<T: IntRing2k> BitXor<Self> for &Share<T> {
    type Output = Share<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Share {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
        }
    }
}

impl<T: IntRing2k> BitXor<Self> for Share<T> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Share {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
        }
    }
}

impl<T: IntRing2k> BitXor<&Self> for Share<T> {
    type Output = Self;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        Share {
            a: self.a ^ rhs.a,
            b: self.b ^ rhs.b,
        }
    }
}

impl<T: IntRing2k> BitXorAssign<&Self> for Share<T> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.a ^= rhs.a;
        self.b ^= rhs.b;
    }
}

impl<T: IntRing2k> BitXorAssign<Self> for Share<T> {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.a ^= rhs.a;
        self.b ^= rhs.b;
    }
}

/// This is only the local part of the AND (so without randomness and without
/// communication)!
impl<T: IntRing2k> BitAnd<Self> for &Share<T> {
    type Output = RingElement<T>;

    fn bitand(self, rhs: Self) -> Self::Output {
        (self.a & rhs.a) ^ (self.b & rhs.a) ^ (self.a & rhs.b)
    }
}

impl<T: IntRing2k> BitAnd<&RingElement<T>> for &Share<T> {
    type Output = Share<T>;

    fn bitand(self, rhs: &RingElement<T>) -> Self::Output {
        Share {
            a: self.a & rhs,
            b: self.b & rhs,
        }
    }
}

impl<T: IntRing2k> BitAnd<T> for Share<T> {
    type Output = Share<T>;

    fn bitand(self, rhs: T) -> Self::Output {
        Share {
            a: self.a & rhs,
            b: self.b & rhs,
        }
    }
}

impl<T: IntRing2k> Zero for Share<T> {
    fn zero() -> Self {
        Self {
            a: RingElement::zero(),
            b: RingElement::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

impl<T: IntRing2k> Neg for Share<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<T: IntRing2k> Neg for &Share<T> {
    type Output = Share<T>;

    fn neg(self) -> Self::Output {
        Share {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<T: IntRing2k> Not for &Share<T> {
    type Output = Share<T>;

    fn not(self) -> Self::Output {
        Share {
            a: !self.a,
            b: !self.b,
        }
    }
}

impl<T: IntRing2k> Shr<u32> for &Share<T> {
    type Output = Share<T>;

    fn shr(self, rhs: u32) -> Self::Output {
        Share {
            a: self.a >> rhs,
            b: self.b >> rhs,
        }
    }
}

impl<T: IntRing2k> Shl<u32> for Share<T> {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self::Output {
        Self {
            a: self.a << rhs,
            b: self.b << rhs,
        }
    }
}

/// Additive share of a relative Hamming distance.
/// The distance is represented as a pair of shares `(code_dot, mask_dot)`, where
/// - `code_dot` is the number of matching unmasked iris bits minus the number of non-matching unmasked iris bits,
/// - `mask_dot` is the number of common unmasked bits.
///
/// The greater the ratio `code_dot / mask_dot`, the more similar the irises are.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DistanceShare<T: IntRing2k> {
    pub code_dot: Share<T>,
    pub mask_dot: Share<T>,
}

impl<T> DistanceShare<T>
where
    T: IntRing2k,
{
    pub fn new(code_dot: Share<T>, mask_dot: Share<T>) -> Self {
        DistanceShare { code_dot, mask_dot }
    }
}

impl<T: IntRing2k> Add<&Self> for DistanceShare<T> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        DistanceShare {
            code_dot: self.code_dot + &rhs.code_dot,
            mask_dot: self.mask_dot + &rhs.mask_dot,
        }
    }
}

impl<T: IntRing2k> Add<Self> for DistanceShare<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DistanceShare {
            code_dot: self.code_dot + rhs.code_dot,
            mask_dot: self.mask_dot + rhs.mask_dot,
        }
    }
}

impl<T: IntRing2k> AddAssign<&Self> for DistanceShare<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.code_dot += &rhs.code_dot;
        self.mask_dot += &rhs.mask_dot;
    }
}

#[cfg(test)]
mod tests {
    use crate::shares::bit::Bit;

    use super::*;
    use aes_prng::AesRng;
    use itertools::izip;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Standard};

    fn get_shares<T: IntRing2k>(value: T, bitwise: bool) -> Vec<Share<T>>
    where
        Standard: Distribution<T>,
    {
        let mut rng = AesRng::from_entropy();
        let b = RingElement(rng.gen());
        let c = RingElement(rng.gen());
        let a = if bitwise {
            RingElement(value) ^ b ^ c
        } else {
            RingElement(value) - b - c
        };
        vec![Share::new(a, c), Share::new(b, a), Share::new(c, b)]
    }

    fn reconstruct_shares<T: IntRing2k>(shares: Vec<Share<T>>) -> RingElement<T> {
        shares[0].a + shares[1].a + shares[2].a
    }

    fn reconstruct_bit_shares<T: IntRing2k>(shares: Vec<Share<T>>) -> RingElement<T> {
        shares[0].a ^ shares[1].a ^ shares[2].a
    }

    fn reconstruct_mul_shares<T: IntRing2k>(shares: Vec<RingElement<T>>) -> RingElement<T> {
        shares[0] + shares[1] + shares[2]
    }

    fn reconstruct_mul_bit_shares<T: IntRing2k>(shares: Vec<RingElement<T>>) -> RingElement<T> {
        shares[0] ^ shares[1] ^ shares[2]
    }

    fn arithmetic_test<T: IntRing2k>()
    where
        Standard: Distribution<T>,
    {
        let mut rng = AesRng::from_entropy();
        let a_t: T = rng.gen();
        let b_t: T = rng.gen();

        let a = get_shares(a_t, false);
        let b = get_shares(b_t, false);

        // Addition
        let expected_add = RingElement(a_t.wrapping_add(&b_t));
        let mut c = izip!(a.clone(), b.clone())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_add);

        c = izip!(a.clone(), b.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_add);

        c = a.clone();
        c.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b);
        assert_eq!(reconstruct_shares(c), expected_add);

        c = a.clone();
        c.iter_mut()
            .zip(b.iter().cloned())
            .for_each(|(a, b)| *a += b);
        assert_eq!(reconstruct_shares(c), expected_add);

        // Addition with a constant
        c = a.clone();
        c.iter_mut()
            .enumerate()
            .for_each(|(i, a)| a.add_assign_const_role(T::one(), Role::new(i)));
        assert_eq!(
            reconstruct_shares(c),
            RingElement(a_t.wrapping_add(&T::one()))
        );

        // Subtraction
        let expected_sub = RingElement(a_t.wrapping_sub(&b_t));
        let mut c = izip!(a.clone(), b.clone())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_sub);

        c = izip!(a.clone(), b.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_sub);

        c = a.clone();
        c.iter_mut().zip(b.iter()).for_each(|(a, b)| *a -= b);
        assert_eq!(reconstruct_shares(c), expected_sub);

        c = a.clone();
        c.iter_mut()
            .zip(b.iter().cloned())
            .for_each(|(a, b)| *a -= b);
        assert_eq!(reconstruct_shares(c), expected_sub);

        // Multiplication
        let expected_mul = RingElement(a_t.wrapping_mul(&b_t));
        let c = izip!(a.iter(), b.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_mul_shares(c), expected_mul);

        // Multiplication with a constant
        let mut c = a.iter().map(|a| a * b_t).collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_mul);
        c = a.iter().cloned().map(|a| a * b_t).collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_mul);
        c = a
            .iter()
            .cloned()
            .map(|a| a * RingElement(b_t))
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_mul);
        c = a.clone();
        c.iter_mut().for_each(|a| *a *= b_t);
        assert_eq!(reconstruct_shares(c), expected_mul);

        // Negation
        let expected_neg = -RingElement(a_t);
        let mut c = a.iter().map(|a| -a).collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_neg);
        c = a.iter().cloned().map(|a| -a).collect::<Vec<_>>();
        assert_eq!(reconstruct_shares(c), expected_neg);

        let a = get_shares(a_t, true);
        let b = get_shares(b_t, true);

        // XOR
        let expected_xor = RingElement(a_t ^ b_t);
        let mut c = izip!(a.clone(), b.clone())
            .map(|(a, b)| a ^ b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_xor);

        c = izip!(a.clone(), b.iter())
            .map(|(a, b)| a ^ b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_xor);

        c = izip!(a.iter(), b.iter())
            .map(|(a, b)| a ^ b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_xor);

        c = a.clone();
        c.iter_mut().zip(b.iter()).for_each(|(a, b)| *a ^= b);
        assert_eq!(reconstruct_bit_shares(c), expected_xor);

        c = a.clone();
        c.iter_mut()
            .zip(b.iter().cloned())
            .for_each(|(a, b)| *a ^= b);
        assert_eq!(reconstruct_bit_shares(c), expected_xor);

        // AND
        let expected_and = RingElement(a_t & b_t);
        let c = izip!(a.iter(), b.iter())
            .map(|(a, b)| a & b)
            .collect::<Vec<_>>();
        assert_eq!(reconstruct_mul_bit_shares(c), expected_and);
        let mut c = a.iter().cloned().map(|a| a & b_t).collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_and);
        c = a.iter().map(|a| a & &RingElement(b_t)).collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_and);

        // NOT
        let expected_not = RingElement(!a_t);
        c = a.iter().map(|a| !a).collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_not);

        // Shift
        let expected_shl = RingElement(a_t << 1);
        let mut c = a.iter().cloned().map(|a| a << 1).collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_shl);
        let expected_shr = RingElement(a_t >> 1);
        c = a.iter().map(|a| a >> 1).collect::<Vec<_>>();
        assert_eq!(reconstruct_bit_shares(c), expected_shr);
    }

    fn identity_test<T: IntRing2k>() {
        let a: Share<T> = Share::zero();
        assert_eq!(a.a, RingElement::zero());
        assert_eq!(a.b, RingElement::zero());
        assert!(a.is_zero());
    }

    macro_rules! test_impl {
        ($([$ty:ty,$fn:ident]),*) => ($(
            #[test]
            fn $fn() {
                arithmetic_test::<$ty>();
                identity_test::<$ty>();
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
