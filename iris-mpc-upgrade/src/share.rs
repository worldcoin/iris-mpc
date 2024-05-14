use num_traits::{WrappingAdd, Zero};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, BitAnd, BitXor, BitXorAssign, Shl, Shr, ShrAssign};

use crate::PartyID;

#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct RepShare<T> {
    pub a: T,
    pub b: T,
}

impl<T> RepShare<T> {
    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }
}

impl<T> RepShare<T>
where
    RepShare<T>: Zero,
{
    pub fn a2b_pre(self, id: PartyID) -> (Self, Self, Self) {
        let mut x1 = RepShare::zero();
        let mut x2 = RepShare::zero();
        let mut x3 = RepShare::zero();

        match id {
            PartyID::ID0 => {
                x1.a = self.a;
                x3.b = self.b;
            }
            PartyID::ID1 => {
                x2.a = self.a;
                x1.b = self.b;
            }
            PartyID::ID2 => {
                x3.a = self.a;
                x2.b = self.b;
            }
        }
        (x1, x2, x3)
    }

    pub fn bitinject_pre(a: bool, b: bool, id: PartyID) -> (Self, Self, Self)
    where
        T: From<bool>,
    {
        let mut x1 = RepShare::zero();
        let mut x2 = RepShare::zero();
        let mut x3 = RepShare::zero();

        match id {
            PartyID::ID0 => {
                x1.a = T::from(a);
                x3.b = T::from(b);
            }
            PartyID::ID1 => {
                x2.a = T::from(a);
                x1.b = T::from(b);
            }
            PartyID::ID2 => {
                x3.a = T::from(a);
                x2.b = T::from(b);
            }
        }
        (x1, x2, x3)
    }
}

impl<T> AddAssign for RepShare<T>
where
    T: WrappingAdd,
{
    fn add_assign(&mut self, other: Self) {
        self.a = self.a.wrapping_add(&other.a);
        self.b = self.b.wrapping_add(&other.b);
    }
}

impl<T> Shr<usize> for &RepShare<T>
where
    for<'a> &'a T: Shr<usize, Output = T>,
{
    type Output = RepShare<T>;

    fn shr(self, rhs: usize) -> Self::Output {
        RepShare::new(&self.a >> rhs, &self.b >> rhs)
    }
}

impl<T> ShrAssign<usize> for RepShare<T>
where
    T: ShrAssign<usize>,
{
    fn shr_assign(&mut self, rhs: usize) {
        self.a >>= rhs;
        self.b >>= rhs;
    }
}

impl<T> Shl<usize> for RepShare<T>
where
    T: Shl<usize, Output = T>,
{
    type Output = RepShare<T>;

    fn shl(self, rhs: usize) -> Self::Output {
        RepShare::new(self.a << rhs, self.b << rhs)
    }
}

impl<T> BitXor<&Self> for RepShare<T>
where
    T: for<'a> BitXor<&'a T, Output = T>,
{
    type Output = RepShare<T>;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        RepShare::new(self.a ^ &rhs.a, self.b ^ &rhs.b)
    }
}

impl<T> BitXor for RepShare<T>
where
    T: BitXor<Output = T>,
{
    type Output = RepShare<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        RepShare::new(self.a ^ rhs.a, self.b ^ rhs.b)
    }
}

impl<T> BitXorAssign<&Self> for RepShare<T>
where
    T: for<'a> BitXorAssign<&'a T>,
{
    fn bitxor_assign(&mut self, other: &Self) {
        self.a ^= &other.a;
        self.b ^= &other.b;
    }
}

impl<T> BitXorAssign for RepShare<T>
where
    T: BitXorAssign,
{
    fn bitxor_assign(&mut self, other: Self) {
        self.a ^= other.a;
        self.b ^= other.b;
    }
}

impl<T> BitAnd<T> for RepShare<T>
where
    T: for<'a> BitAnd<&'a T, Output = T>,
    T: BitAnd<Output = T>,
{
    type Output = RepShare<T>;

    fn bitand(self, rhs: T) -> Self::Output {
        RepShare::new(self.a & &rhs, self.b & rhs)
    }
}

/// This is only the local part of the AND (so without randomness and without communication)!
impl<T> BitAnd<Self> for &RepShare<T>
where
    for<'a> &'a T: BitAnd<&'a T, Output = T>,
    T: BitXor<Output = T>,
{
    type Output = T;

    fn bitand(self, rhs: Self) -> Self::Output {
        (&self.a & &rhs.a) ^ (&self.b & &rhs.a) ^ (&self.a & &rhs.b)
    }
}

impl<T> Add for RepShare<T>
where
    T: WrappingAdd,
{
    type Output = RepShare<T>;

    fn add(self, rhs: Self) -> Self::Output {
        RepShare::new(self.a.wrapping_add(&rhs.a), self.b.wrapping_add(&rhs.b))
    }
}

impl<T> Zero for RepShare<T>
where
    T: Zero + WrappingAdd,
{
    fn zero() -> Self {
        RepShare::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

pub struct RepSharedBits<'a> {
    share: &'a [RepShare<u64>],
    current: RepShare<u64>,
    index: usize,
}

impl RepSharedBits<'_> {
    pub fn new(share: &[RepShare<u64>]) -> RepSharedBits<'_> {
        RepSharedBits {
            share,
            current: RepShare::new(0, 0),
            index: 0,
        }
    }
}

impl Iterator for RepSharedBits<'_> {
    type Item = (bool, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.share.len() * 64 {
            None
        } else {
            if self.index % 64 == 0 {
                self.current.a = self.share[self.index / 64].a;
                self.current.b = self.share[self.index / 64].b;
            }
            let res_a = self.current.a & 1 == 1;
            let res_b = self.current.b & 1 == 1;
            self.current >>= 1;
            self.index += 1;
            Some((res_a, res_b))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.share.len() * 64 - self.index,
            Some(self.share.len() * 64 - self.index),
        )
    }
}

impl ExactSizeIterator for RepSharedBits<'_> {}

pub fn share_p3<R: Rng + CryptoRng>(secret: u16, rng: &mut R) -> [u16; 3] {
    let mut shares = [0; 3];
    shares[0] = rng.gen();
    shares[1] = rng.gen();
    shares[2] = secret.wrapping_sub(shares[0].wrapping_add(&shares[1]));
    shares
}

pub fn to_repshares(shares: [u16; 3]) -> [RepShare<u16>; 3] {
    [
        RepShare::new(shares[0], shares[2]),
        RepShare::new(shares[1], shares[0]),
        RepShare::new(shares[2], shares[1]),
    ]
}

pub fn share_to_rep_shares<R: Rng + CryptoRng>(
    data: &[u16],
    rng: &mut R,
) -> [Vec<RepShare<u16>>; 3] {
    let len = data.len();
    let mut results = [
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    ];

    for code in data.iter() {
        let [share0, share1, share2] = to_repshares(share_p3(*code, rng));
        results[0].push(share0);
        results[1].push(share1);
        results[2].push(share2);
    }

    results
}
