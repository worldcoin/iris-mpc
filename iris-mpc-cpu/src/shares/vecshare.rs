use super::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement, share::Share};
use bytes::{Buf, BytesMut};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    ops::{AddAssign, BitXor, BitXorAssign, Deref, DerefMut, Not, SubAssign},
};

#[repr(transparent)]
pub struct RingBytesIter<T: IntRing2k> {
    bytes:   BytesMut,
    _marker: std::marker::PhantomData<T>,
}

impl<T: IntRing2k> RingBytesIter<T> {
    pub fn new(bytes: BytesMut) -> Self {
        Self {
            bytes,
            _marker: PhantomData,
        }
    }
}

impl<T: IntRing2k> Iterator for RingBytesIter<T> {
    type Item = RingElement<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.remaining() == 0 {
            None
        } else {
            let res = bytemuck::pod_read_unaligned(&self.bytes.chunk()[..T::BYTES]);
            self.bytes.advance(T::BYTES);
            Some(RingElement(res))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bytes.remaining() / T::BYTES;
        (len, Some(len))
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SliceShare<'a, T: IntRing2k> {
    shares: &'a [Share<T>],
}

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

    pub fn get_at(self, index: usize) -> Share<T> {
        self.shares[index].to_owned()
    }

    pub fn from_avec_biter(a: Vec<RingElement<T>>, b: RingBytesIter<T>) -> Self {
        let shares = a
            .into_iter()
            .zip(b)
            .map(|(a_, b_)| Share::new(a_, b_))
            .collect();
        Self { shares }
    }

    pub fn from_ab(a: Vec<RingElement<T>>, b: Vec<RingElement<T>>) -> Self {
        let shares = a
            .into_iter()
            .zip(b)
            .map(|(a_, b_)| Share::new(a_, b_))
            .collect();
        Self { shares }
    }

    pub fn flatten(inp: Vec<Self>) -> Self {
        Self {
            shares: inp.into_iter().flat_map(|x| x.shares).collect(),
        }
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

impl<'a, T: IntRing2k> Deref for SliceShare<'a, T> {
    type Target = [Share<T>];

    fn deref(&self) -> &Self::Target {
        self.shares
    }
}

impl<'a, T: IntRing2k> Deref for SliceShareMut<'a, T> {
    type Target = [Share<T>];

    fn deref(&self) -> &Self::Target {
        self.shares
    }
}

impl<'a, T: IntRing2k> DerefMut for SliceShareMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.shares
    }
}
