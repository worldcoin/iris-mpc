use super::{int_ring::IntRing2k, ring_impl::RingElement};
use crate::execution::player::Role;
use iris_mpc_common::id::PartyID;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{
    Add, AddAssign, BitAnd, BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Shl, Shr, Sub,
    SubAssign,
};

#[derive(Clone, Debug, PartialEq, Default, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Share<T: IntRing2k> {
    pub a: RingElement<T>,
    pub b: RingElement<T>,
}

impl<T: IntRing2k> Share<T> {
    pub fn new(a: RingElement<T>, b: RingElement<T>) -> Self {
        Self { a, b }
    }

    pub(crate) fn sub_from_const(&self, other: T, id: PartyID) -> Self {
        let mut a = -self;
        a.add_assign_const(other, id);
        a
    }

    pub(crate) fn sub_from_const_role(&self, other: T, role: Role) -> Self {
        let mut a = -self;
        a.add_assign_const_role(other, role);
        a
    }

    pub fn add_assign_const(&mut self, other: T, id: PartyID) {
        match id {
            PartyID::ID0 => self.a += RingElement(other),
            PartyID::ID1 => self.b += RingElement(other),
            PartyID::ID2 => {}
        }
    }

    pub fn add_assign_const_role(&mut self, other: T, role: Role) {
        match role.zero_based() {
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
