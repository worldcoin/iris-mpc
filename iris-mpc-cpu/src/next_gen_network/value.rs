use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use eyre::eyre;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// Value sent over the network
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum NetworkValue {
    Ring16(std::num::Wrapping<u16>),
    Ring32(std::num::Wrapping<u32>),
    RingElement16(RingElement<u16>),
    RingElement32(RingElement<u32>),
    RingElement64(RingElement<u64>),
    VecRing32(Vec<RingElement<u32>>),
    VecRing64(Vec<RingElement<u64>>),
    PrfKey([u8; 16]),
}

impl NetworkValue {
    pub fn to_network(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    pub fn from_network(serialized: eyre::Result<Vec<u8>>) -> eyre::Result<Self> {
        bincode::deserialize::<Self>(&serialized?).map_err(|_e| eyre!("failed to parse value"))
    }
}
