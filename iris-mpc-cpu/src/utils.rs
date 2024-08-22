use crate::{
    error::Error,
    shares::{int_ring::IntRing2k, ring_impl::RingElement, vecshare::RingBytesIter},
};
use bytes::{Buf, Bytes, BytesMut};

pub struct Utils {}

impl Utils {
    pub fn ring_slice_to_bytes<T: IntRing2k>(vec: &[RingElement<T>]) -> Bytes {
        let slice = RingElement::convert_slice(vec);
        let slice_: &[u8] = bytemuck::cast_slice(slice);
        Bytes::copy_from_slice(slice_)
    }

    pub fn ring_iter_from_bytes<T: IntRing2k>(
        bytes: BytesMut,
        n: usize,
    ) -> Result<RingBytesIter<T>, Error> {
        if bytes.remaining() != n * T::BYTES {
            return Err(Error::InvalidSize);
        }

        Ok(RingBytesIter::new(bytes))
    }
}
