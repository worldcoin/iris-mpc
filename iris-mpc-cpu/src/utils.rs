use crate::{
    error::Error,
    networks::network_trait::NetworkTrait,
    shares::{int_ring::IntRing2k, ring_impl::RingElement, vecshare::RingBytesIter},
};
use bytes::{Buf, Bytes, BytesMut};
use std::io;

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

    pub fn blocking_send_and_receive<N: NetworkTrait>(
        network: &mut N,
        data: Bytes,
    ) -> Result<BytesMut, io::Error> {
        network.blocking_send_next_id(data)?;
        let data = network.blocking_receive_prev_id()?;
        Ok(data)
    }

    pub fn blocking_send_slice_and_receive<N: NetworkTrait, T: IntRing2k>(
        network: &mut N,
        values: &[RingElement<T>],
    ) -> Result<BytesMut, Error> {
        let data = Self::ring_slice_to_bytes(values);
        Ok(Self::blocking_send_and_receive(network, data)?)
    }
}
