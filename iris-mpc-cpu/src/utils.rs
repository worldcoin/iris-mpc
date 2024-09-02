use crate::{
    networks::network_trait::NetworkTrait,
    shares::{int_ring::IntRing2k, ring_impl::RingElement, vecshare::RingBytesIter},
};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use eyre::{eyre, Error, Result};
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
            return Err(eyre!("InvalidSize"));
        }

        Ok(RingBytesIter::new(bytes))
    }

    pub fn ring_to_bytes<T: IntRing2k>(value: &RingElement<T>) -> Bytes {
        let mut out = BytesMut::with_capacity(T::BYTES);
        out.put(bytemuck::bytes_of(&value.0));
        out.freeze()
    }

    pub fn ring_from_bytes<T: IntRing2k>(value: BytesMut) -> Result<RingElement<T>, Error> {
        if value.remaining() != T::BYTES {
            return Err(eyre!("InvalidSize"));
        }
        let slice = value.as_ref();
        let slice_: &T = bytemuck::from_bytes(slice);
        Ok(RingElement(*slice_))
    }

    pub fn ring_iter_to_bytes<'a, T: 'a + IntRing2k>(
        iter: impl ExactSizeIterator<Item = &'a RingElement<T>>,
    ) -> Bytes {
        let mut out = BytesMut::with_capacity(T::BYTES * iter.len());
        for v in iter {
            out.put(bytemuck::bytes_of(&v.0));
        }
        out.freeze()
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

    pub fn blocking_send_and_receive_value<N: NetworkTrait, T: IntRing2k>(
        network: &mut N,
        value: &RingElement<T>,
    ) -> Result<RingElement<T>, Error> {
        let response = Self::blocking_send_and_receive(network, Self::ring_to_bytes(value))?;
        Self::ring_from_bytes(response)
    }

    pub fn blocking_send_iter_and_receive<'a, N: NetworkTrait, T: IntRing2k + 'a>(
        network: &mut N,
        values: impl ExactSizeIterator<Item = &'a RingElement<T>>,
    ) -> Result<BytesMut, Error> {
        Ok(Self::blocking_send_and_receive(
            network,
            Self::ring_iter_to_bytes(values),
        )?)
    }
}
