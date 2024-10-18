use crate::shares::{bit::Bit, ring_impl::RingElement};
use eyre::eyre;
use serde::{Deserialize, Serialize};

/// Value sent over the network
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum NetworkValue {
    PrfKey([u8; 16]),
    Ring16(std::num::Wrapping<u16>),
    Ring32(std::num::Wrapping<u32>),
    RingElementBit(RingElement<Bit>),
    RingElement16(RingElement<u16>),
    RingElement32(RingElement<u32>),
    RingElement64(RingElement<u64>),
    VecRing16(Vec<RingElement<u16>>),
    VecRing32(Vec<RingElement<u32>>),
    VecRing64(Vec<RingElement<u64>>),
}

impl NetworkValue {
    pub fn to_network(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    pub fn from_network(serialized: eyre::Result<Vec<u8>>) -> eyre::Result<Self> {
        bincode::deserialize::<Self>(&serialized?).map_err(|_e| eyre!("failed to parse value"))
    }
}

impl From<Vec<RingElement<u16>>> for NetworkValue {
    fn from(value: Vec<RingElement<u16>>) -> Self {
        NetworkValue::VecRing16(value)
    }
}

impl TryFrom<NetworkValue> for Vec<RingElement<u16>> {
    type Error = eyre::Error;
    fn try_from(value: NetworkValue) -> eyre::Result<Self> {
        match value {
            NetworkValue::VecRing16(x) => Ok(x),
            _ => Err(eyre!(
                "could not convert Network Value into Vec<RingElement<u16>>"
            )),
        }
    }
}
