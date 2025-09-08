use super::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement, share::Share};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, BitXor, BitXorAssign, Deref, DerefMut, Not, SubAssign};

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SliceShare<'a, T: IntRing2k> {
    shares: &'a [Share<T>],
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T: IntRing2k> SliceShare<'a, T> {
    pub fn split_at(&self, mid: usize) -> (SliceShare<T>, SliceShare<T>) {
        let (a, b) = self.shares.split_at(mid);
        (SliceShare { shares: a }, SliceShare { shares: b })
    }

    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = SliceShare<T>> + '_ {
        self.shares
            .chunks(chunk_size)
            .map(|x| SliceShare { shares: x })
    }

    pub fn len(&self) -> usize {
        self.shares.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Share<T>> {
        self.shares.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct SliceShareMut<'a, T: IntRing2k> {
    shares: &'a mut [Share<T>],
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T: IntRing2k> SliceShareMut<'a, T> {
    pub fn to_vec(&self) -> VecShare<T> {
        VecShare {
            shares: self.shares.to_vec(),
        }
    }

    pub fn to_slice(&self) -> SliceShare<T> {
        SliceShare {
            shares: self.shares,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "")]
#[repr(transparent)]
pub struct VecShare<T: IntRing2k> {
    pub(crate) shares: Vec<Share<T>>,
}

impl<T: IntRing2k> VecShare<T> {
    #[cfg(test)]
    pub fn new_share(share: Share<T>) -> Self {
        let shares = vec![share];
        Self { shares }
    }

    pub fn new_vec(shares: Vec<Share<T>>) -> Self {
        Self { shares }
    }

    pub fn inner(self) -> Vec<Share<T>> {
        self.shares
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let shares = Vec::with_capacity(capacity);
        Self { shares }
    }

    pub fn extend(&mut self, items: Self) {
        self.shares.extend(items.shares);
    }

    pub fn extend_from_slice(&mut self, items: SliceShare<T>) {
        self.shares.extend_from_slice(items.shares);
    }

    pub fn len(&self) -> usize {
        self.shares.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Share<T>> {
        self.shares.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Share<T>> {
        self.shares.iter_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    pub fn push(&mut self, el: Share<T>) {
        self.shares.push(el)
    }

    pub fn pop(&mut self) -> Option<Share<T>> {
        self.shares.pop()
    }

    pub fn sum(&self) -> Share<T> {
        self.shares.iter().fold(Share::zero(), |a, b| a + b)
    }

    pub fn not_inplace(&mut self) {
        for x in self.shares.iter_mut() {
            *x = !&*x;
        }
    }

    pub fn split_at(&self, mid: usize) -> (SliceShare<T>, SliceShare<T>) {
        let (a, b) = self.shares.split_at(mid);
        (SliceShare { shares: a }, SliceShare { shares: b })
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (SliceShareMut<T>, SliceShareMut<T>) {
        let (a, b) = self.shares.split_at_mut(mid);
        (SliceShareMut { shares: a }, SliceShareMut { shares: b })
    }

    pub fn get_at(&self, index: usize) -> Share<T> {
        self.shares[index].to_owned()
    }

    pub fn from_ab(a: Vec<RingElement<T>>, b: Vec<RingElement<T>>) -> Self {
        let shares = a
            .into_iter()
            .zip(b)
            .map(|(a_, b_)| Share::new(a_, b_))
            .collect();
        Self { shares }
    }

    pub fn from_iter_ab(
        a: impl Iterator<Item = RingElement<T>>,
        b: impl Iterator<Item = RingElement<T>>,
    ) -> Self {
        let shares = a.zip(b).map(|(a_, b_)| Share::new(a_, b_)).collect();
        Self { shares }
    }

    pub fn flatten(inp: Vec<Self>) -> impl Iterator<Item = Share<T>> {
        inp.into_iter().flat_map(|x| x.shares)
    }

    pub fn convert_to_bits(self) -> VecShare<Bit> {
        let mut res = VecShare::with_capacity(T::K * self.shares.len());
        for share in self.shares.into_iter() {
            let (a, b) = share.get_ab();
            for (a, b) in a.bit_iter().zip(b.bit_iter()) {
                res.push(Share::new(RingElement(a), RingElement(b)));
            }
        }
        res
    }

    pub fn truncate(&mut self, len: usize) {
        self.shares.truncate(len);
    }

    pub fn as_slice(&self) -> SliceShare<T> {
        SliceShare {
            shares: &self.shares,
        }
    }

    pub fn as_slice_mut(&mut self) -> SliceShareMut<T> {
        SliceShareMut {
            shares: &mut self.shares,
        }
    }
}

impl VecShare<Bit> {
    #[allow(clippy::manual_div_ceil)]
    pub fn pack<T: IntRing2k>(self) -> VecShare<T> {
        let outlen = (self.shares.len() + T::K - 1) / T::K;
        let mut out = VecShare::with_capacity(outlen);

        for a_ in self.shares.chunks(T::K) {
            let mut share_a = RingElement::<T>::zero();
            let mut share_b = RingElement::<T>::zero();
            for (i, bit) in a_.iter().enumerate() {
                let (bit_a, bit_b) = bit.to_owned().get_ab();
                share_a |= RingElement(T::from(bit_a.convert().convert()) << i);
                share_b |= RingElement(T::from(bit_b.convert().convert()) << i);
            }
            let share = Share::new(share_a, share_b);
            out.push(share);
        }

        out
    }

    pub fn from_share<T: IntRing2k>(share: Share<T>) -> Self {
        let (a, b) = share.get_ab();
        let mut res = VecShare::with_capacity(T::K);
        for (a, b) in a.bit_iter().zip(b.bit_iter()) {
            res.push(Share::new(RingElement(a), RingElement(b)));
        }
        res
    }
}

impl<T: IntRing2k> IntoIterator for VecShare<T> {
    type Item = Share<T>;
    type IntoIter = std::vec::IntoIter<Share<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.shares.into_iter()
    }
}

impl<T: IntRing2k> Not for SliceShare<'_, T> {
    type Output = VecShare<T>;

    fn not(self) -> Self::Output {
        let mut v = VecShare::with_capacity(self.shares.len());
        for x in self.shares.iter() {
            v.push(!x);
        }
        v
    }
}

impl<T: IntRing2k> BitXor<Self> for SliceShare<'_, T> {
    type Output = VecShare<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        let mut v = VecShare::with_capacity(self.shares.len());
        for (x1, x2) in self.shares.iter().zip(rhs.shares.iter()) {
            v.push(x1 ^ x2);
        }
        v
    }
}

impl<T: IntRing2k> BitXor<Self> for VecShare<T> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        let mut v = VecShare::with_capacity(self.shares.len());
        for (x1, x2) in self.shares.into_iter().zip(rhs.shares) {
            v.push(x1 ^ x2);
        }
        v
    }
}

impl<T: IntRing2k> BitXor<SliceShare<'_, T>> for VecShare<T> {
    type Output = Self;

    fn bitxor(self, rhs: SliceShare<'_, T>) -> Self::Output {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        let mut v = VecShare::with_capacity(self.shares.len());
        for (x1, x2) in self.shares.into_iter().zip(rhs.shares.iter()) {
            v.push(x1 ^ x2);
        }
        v
    }
}

impl<T: IntRing2k> AddAssign<SliceShare<'_, T>> for VecShare<T> {
    fn add_assign(&mut self, rhs: SliceShare<'_, T>) {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        for (x1, x2) in self.shares.iter_mut().zip(rhs.shares.iter()) {
            *x1 += x2;
        }
    }
}

impl<T: IntRing2k> SubAssign<SliceShare<'_, T>> for VecShare<T> {
    fn sub_assign(&mut self, rhs: SliceShare<'_, T>) {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        for (x1, x2) in self.shares.iter_mut().zip(rhs.shares.iter()) {
            *x1 -= x2;
        }
    }
}

impl<T: IntRing2k> SubAssign for VecShare<T> {
    fn sub_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        for (x1, x2) in self.shares.iter_mut().zip(rhs.shares.into_iter()) {
            *x1 -= x2;
        }
    }
}

impl<T: IntRing2k> BitXorAssign<SliceShare<'_, T>> for VecShare<T> {
    fn bitxor_assign(&mut self, rhs: SliceShare<'_, T>) {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        for (x1, x2) in self.shares.iter_mut().zip(rhs.shares.iter()) {
            *x1 ^= x2;
        }
    }
}

impl<T: IntRing2k> BitXorAssign<Self> for VecShare<T> {
    fn bitxor_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shares.len(), rhs.shares.len());
        for (x1, x2) in self.shares.iter_mut().zip(rhs.shares) {
            *x1 ^= x2;
        }
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T: IntRing2k> Deref for SliceShare<'a, T> {
    type Target = [Share<T>];

    fn deref(&self) -> &Self::Target {
        self.shares
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T: IntRing2k> Deref for SliceShareMut<'a, T> {
    type Target = [Share<T>];

    fn deref(&self) -> &Self::Target {
        self.shares
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T: IntRing2k> DerefMut for SliceShareMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.shares
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;
    use rand::Rng;
    use rand_distr::{Distribution, Standard};

    fn test_access<T: IntRing2k>()
    where
        Standard: Distribution<RingElement<T>>,
    {
        let mut rng = rand::thread_rng();
        let share1: Share<T> = Share::new(rng.gen(), rng.gen());

        let mut vec_share = VecShare::new_share(share1.clone());
        assert!(!vec_share.is_empty());
        let popped_share = vec_share.pop();
        assert!(popped_share.is_some());
        assert_eq!(popped_share.unwrap(), share1);
        assert!(vec_share.is_empty());

        let mut shares = vec![Share::zero()];
        let one_share = Share::new(RingElement::one(), RingElement::one());
        for i in 1..17 {
            shares.push(one_share.clone() + &shares[i - 1]);
        }
        vec_share = VecShare::new_vec(shares);
        let (slice1, slice2) = vec_share.split_at(3);
        assert!(!slice1.is_empty());
        assert_eq!(slice1.len(), 3);
        assert_eq!(slice2.len(), 14);
        for chunk in slice2.chunks(2) {
            assert_eq!(chunk.len(), 2);
            assert_eq!(one_share.clone() + &chunk[0], chunk[1]);
        }

        let mut vec_share = VecShare::new_share(Share::zero());
        vec_share.extend_from_slice(slice1);
        assert_eq!(vec_share.len(), 4);
        let slice_mut = vec_share.as_slice_mut();
        let vec_share_tmp = slice_mut.to_vec();
        let three_share = one_share.clone() + one_share.clone() + one_share.clone();
        assert_eq!(vec_share_tmp.sum(), three_share);
        let (slice21, slice22) = slice2.split_at(2);
        assert_eq!(slice21.len(), 2);
        assert_eq!(slice22.len(), 12);
        assert_eq!(slice21[0], three_share);
        assert_eq!(slice21[1], one_share.clone() + three_share);

        vec_share = VecShare::new_vec(vec![share1.clone(), !&share1]);
        let mut not_vec_share = vec_share.clone();
        not_vec_share.not_inplace();
        assert_eq!(vec_share.get_at(0), not_vec_share.get_at(1));
        assert_eq!(vec_share.get_at(1), not_vec_share.get_at(0));
    }

    fn test_arithmetic<T: IntRing2k>()
    where
        Standard: Distribution<RingElement<T>>,
    {
        let mut rng = rand::thread_rng();
        let share1: Share<T> = Share::new(rng.gen(), rng.gen());

        // NOT
        let vec_share = VecShare::new_vec(vec![share1.clone(), !&share1]);
        let mut not_vec_share = vec_share.clone();
        not_vec_share.not_inplace();
        assert_eq!(vec_share.get_at(0), not_vec_share.get_at(1));
        assert_eq!(vec_share.get_at(1), not_vec_share.get_at(0));
        let slice_share = vec_share.as_slice();
        assert_eq!(!slice_share, not_vec_share);

        // XOR
        let vec_zero = VecShare::new_vec(vec![Share::zero(); 2]);
        assert_eq!(vec_share.clone() ^ vec_share.clone(), vec_zero);
        assert_eq!(vec_share.clone() ^ slice_share, vec_zero);
        assert_eq!(
            (slice_share ^ vec_share.as_slice()).inner(),
            vec_zero.clone().inner()
        );

        let mut c = vec_zero.clone();
        c ^= vec_share.clone();
        assert_eq!(c, vec_share);
        c = vec_zero.clone();
        c ^= slice_share;
        assert_eq!(c, vec_share);

        // Addition
        let mut c = vec_zero.clone();
        c += slice_share;
        assert_eq!(c, vec_share);

        // Subtraction
        let mut c = vec_share.clone();
        c -= vec_share.clone();
        assert_eq!(c, vec_zero);
        c = vec_share.clone();
        c -= slice_share;
        assert_eq!(c, vec_zero);
    }

    fn test_bit<T: IntRing2k>()
    where
        Standard: Distribution<RingElement<T>> + Distribution<RingElement<Bit>>,
    {
        let mut rng = rand::thread_rng();
        let share: Share<T> = Share::new(rng.gen(), rng.gen());
        let bit_vec_share = VecShare::from_share(share.clone());
        assert_eq!(bit_vec_share.len(), T::K);
        for (i_bit, bit) in bit_vec_share.into_iter().enumerate() {
            assert_eq!(bit.a, share.a.get_bit_as_bit(i_bit));
            assert_eq!(bit.b, share.b.get_bit_as_bit(i_bit));
        }

        let mut bit_rng = rand::thread_rng();
        let bit_shares = VecShare::new_vec(
            (0..(T::K + 1))
                .map(|_| {
                    let share: Share<Bit> = Share::new(bit_rng.gen(), bit_rng.gen());
                    share
                })
                .collect::<Vec<_>>(),
        );
        let t_shares: VecShare<T> = bit_shares.clone().pack();
        for (i_bit, bit) in bit_shares.into_iter().enumerate() {
            let share_index = i_bit / T::K;
            let bit_index = i_bit % T::K;
            assert_eq!(
                bit.a,
                t_shares.get_at(share_index).a.get_bit_as_bit(bit_index)
            );
        }
    }

    macro_rules! test_impl {
        ($([$ty:ty,$fn:ident]),*) => ($(
            #[test]
            fn $fn() {
                test_access::<$ty>();
                test_arithmetic::<$ty>();
                test_bit::<$ty>();
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
