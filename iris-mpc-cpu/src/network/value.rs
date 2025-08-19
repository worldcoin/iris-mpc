use crate::shares::{self, bit::Bit, ring_impl::RingElement, IntRing2k};
use bytes::BytesMut;
use eyre::{bail, eyre, Result};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::mem::size_of;

/// Size of a PRF key in bytes
const PRF_KEY_SIZE: usize = 16;

/// Value sent over the network
#[derive(PartialEq, Clone, Debug)]
pub enum NetworkValue {
    PrfKey([u8; PRF_KEY_SIZE]),
    RingElementBit(RingElement<Bit>),
    RingElement16(RingElement<u16>),
    RingElement32(RingElement<u32>),
    RingElement64(RingElement<u64>),
    VecRing16(Vec<RingElement<u16>>),
    VecRing32(Vec<RingElement<u32>>),
    VecRing64(Vec<RingElement<u64>>),
    StateChecksum(StateChecksum),
    // outside of this module, use vec_to_network() and vec_from_network() instead of accessing this variant directly.
    NetworkVec(Vec<Self>),
    // used to verify that the PRFs aren't out of sync
    PrfCheck(RingElement<u128>),
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive)]
pub enum DescriptorByte {
    PrfKey = 0x01,
    RingElementBit1 = 0x12,
    RingElementBit0 = 0x02,
    RingElement16 = 0x03,
    RingElement32 = 0x04,
    RingElement64 = 0x05,
    VecRing16 = 0x06,
    VecRing32 = 0x07,
    VecRing64 = 0x08,
    StateChecksum = 0x09,
    // this is used by the TCP framing protocol.
    NetworkVec = 0x0A,
    PrfCheck = 0x0B,
}

impl DescriptorByte {
    // warning: the Vec* variants (including NetworkVec) have a len field which needs to be parsed to get the total len.
    pub fn base_len(&self) -> usize {
        match self {
            DescriptorByte::PrfKey => 1 + PRF_KEY_SIZE,
            DescriptorByte::RingElementBit1 | DescriptorByte::RingElementBit0 => 1,
            DescriptorByte::RingElement16 => 3,
            DescriptorByte::RingElement32 => 5,
            DescriptorByte::RingElement64 => 9,
            DescriptorByte::VecRing16 | DescriptorByte::VecRing32 | DescriptorByte::VecRing64 => 5,
            DescriptorByte::StateChecksum => 1 + 8 + 8,
            DescriptorByte::NetworkVec => 5,
            DescriptorByte::PrfCheck => 1 + size_of::<u128>(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateChecksum {
    pub irises: u64,
    pub graph: u64,
}

impl StateChecksum {
    pub fn check_left_vs_right(&self, right: &Self) -> Result<()> {
        // The checksum of the iris stores cover the vector IDs which should be the same.
        if self.irises != right.irises {
            bail!("Left/Right iris stores are inconsistent: left={self:?}, right={right:?}");
        }
        Ok(())
    }
}

impl NetworkValue {
    fn get_descriptor_byte(&self) -> u8 {
        let descriptor_byte = match self {
            NetworkValue::PrfKey(_) => DescriptorByte::PrfKey,
            NetworkValue::RingElementBit(bit) => {
                if bit.convert().convert() {
                    DescriptorByte::RingElementBit1
                } else {
                    DescriptorByte::RingElementBit0
                }
            }
            NetworkValue::RingElement16(_) => DescriptorByte::RingElement16,
            NetworkValue::RingElement32(_) => DescriptorByte::RingElement32,
            NetworkValue::RingElement64(_) => DescriptorByte::RingElement64,
            NetworkValue::VecRing16(_) => DescriptorByte::VecRing16,
            NetworkValue::VecRing32(_) => DescriptorByte::VecRing32,
            NetworkValue::VecRing64(_) => DescriptorByte::VecRing64,
            NetworkValue::StateChecksum(_) => DescriptorByte::StateChecksum,
            NetworkValue::NetworkVec(_) => DescriptorByte::NetworkVec,
            NetworkValue::PrfCheck(_) => DescriptorByte::PrfCheck,
        };
        descriptor_byte.into()
    }

    pub fn byte_len(&self) -> usize {
        let db =
            DescriptorByte::try_from(self.get_descriptor_byte()).expect("invalid descriptor byte");
        let base_len = db.base_len();

        match self {
            NetworkValue::VecRing16(v) => base_len + size_of::<u16>() * v.len(),
            NetworkValue::VecRing32(v) => base_len + size_of::<u32>() * v.len(),
            NetworkValue::VecRing64(v) => base_len + size_of::<u64>() * v.len(),
            NetworkValue::NetworkVec(v) => base_len + v.iter().map(|x| x.byte_len()).sum::<usize>(),
            _ => base_len,
        }
    }

    // serialize_internal() doesn't allow for NetworkValue::NetworkVec. a separate code path
    // is used to handle that variant.
    pub fn serialize(&self, res: &mut BytesMut) {
        match &self {
            NetworkValue::NetworkVec(v) => Self::serialize_vec(v, res),
            _ => {
                self.serialize_internal(res);
            }
        }
    }

    fn serialize_vec(values: &Vec<Self>, res: &mut BytesMut) {
        let res_start = res.len();
        res.extend_from_slice(&[DescriptorByte::NetworkVec.into()]);
        // this is a placeholder
        let len_idx = res.len();
        res.extend_from_slice(&[0_u8; 4]);
        for value in values {
            value.serialize_internal(res);
        }
        let msg_len = res.len() - res_start;
        res[len_idx..len_idx + 4].copy_from_slice(&((msg_len - 5) as u32).to_le_bytes());
    }

    fn serialize_internal(&self, res: &mut BytesMut) {
        res.extend_from_slice(&[self.get_descriptor_byte()]);

        match self {
            NetworkValue::PrfKey(key) => res.extend_from_slice(key),
            NetworkValue::PrfCheck(v) => res.extend_from_slice(&v.convert().to_le_bytes()),
            NetworkValue::RingElementBit(_) => {
                // Do nothing, the descriptor byte already contains the bit
                // value
            }
            NetworkValue::RingElement16(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::RingElement32(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::RingElement64(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::VecRing16(v) => {
                res.extend_from_slice(&((v.len() * size_of::<u16>()) as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::VecRing32(v) => {
                res.extend_from_slice(&((v.len() * size_of::<u32>()) as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::VecRing64(v) => {
                res.extend_from_slice(&((v.len() * size_of::<u64>()) as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::StateChecksum(checksums) => {
                res.extend_from_slice(&u64::to_le_bytes(checksums.irises));
                res.extend_from_slice(&u64::to_le_bytes(checksums.graph));
            }
            NetworkValue::NetworkVec(_v) => unreachable!(),
        }
    }

    pub fn deserialize(serialized: &[u8]) -> Result<Self> {
        if serialized
            .first()
            .map(|byte| *byte == DescriptorByte::NetworkVec as u8)
            .unwrap_or_default()
        {
            Self::deserialize_vec(serialized).map(NetworkValue::NetworkVec)
        } else {
            Self::deserialize_internal(serialized)
        }
    }

    pub fn deserialize_vec(serialized: &[u8]) -> Result<Vec<Self>> {
        if serialized.len() < 5 {
            bail!("Can't parse vector length: buffer too short");
        }
        if serialized[0] != DescriptorByte::NetworkVec as u8 {
            return Err(eyre!("called deserialize_vec() on invalid buffer"));
        }
        let payload_len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?) as usize;
        if serialized.len() != 5 + payload_len {
            bail!(
                "NetworkVec length mismatch: {} vs expected {}",
                serialized.len() - 5,
                payload_len,
            );
        }

        let mut res = Vec::new();
        let mut idx = 5;
        let end_idx = 5 + payload_len;

        while idx < end_idx {
            let descriptor_byte: DescriptorByte = serialized[idx].try_into()?;
            let value_len = match descriptor_byte {
                DescriptorByte::RingElementBit0 | DescriptorByte::RingElementBit1 => 1,
                DescriptorByte::RingElement16 => 3,
                DescriptorByte::RingElement32 => 5,
                DescriptorByte::RingElement64 => 9,
                DescriptorByte::VecRing16
                | DescriptorByte::VecRing32
                | DescriptorByte::VecRing64 => {
                    if end_idx < idx + 5 {
                        bail!("Invalid length for VecRing: can't parse vector length");
                    }
                    let len =
                        u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[idx + 1..idx + 5])?)
                            as usize;
                    5 + len
                }
                _ => bail!("Invalid type for NetworkVec"),
            };
            res.push(NetworkValue::deserialize_internal(
                &serialized[idx..idx + value_len],
            )?);
            idx += value_len;
        }
        Ok(res)
    }

    fn deserialize_internal(serialized: &[u8]) -> Result<Self> {
        if serialized.is_empty() {
            bail!("Empty serialized data");
        }
        let descriptor = serialized[0];
        let descriptor_byte: DescriptorByte = descriptor.try_into()?;
        match descriptor_byte {
            DescriptorByte::PrfKey => {
                if serialized.len() != 1 + PRF_KEY_SIZE {
                    bail!("Invalid length for PrfKey");
                }
                Ok(NetworkValue::PrfKey(<[u8; PRF_KEY_SIZE]>::try_from(
                    &serialized[1..1 + PRF_KEY_SIZE],
                )?))
            }
            DescriptorByte::PrfCheck => {
                if serialized.len() != 1 + size_of::<u128>() {
                    bail!("Invalid length for PrfSqueeze");
                }
                Ok(NetworkValue::PrfCheck(RingElement(u128::from_le_bytes(
                    <[u8; 16]>::try_from(&serialized[1..1 + 16])?,
                ))))
            }
            DescriptorByte::RingElementBit1 | DescriptorByte::RingElementBit0 => {
                if serialized.len() != 1 {
                    bail!("Invalid length for RingElementBit");
                }
                let bit = if descriptor == Into::<u8>::into(DescriptorByte::RingElementBit1) {
                    Bit::new(true)
                } else {
                    Bit::new(false)
                };
                Ok(NetworkValue::RingElementBit(RingElement(bit)))
            }
            DescriptorByte::RingElement16 => {
                if serialized.len() != 3 {
                    bail!("Invalid length for RingElement16");
                }
                Ok(NetworkValue::RingElement16(RingElement(
                    u16::from_le_bytes(<[u8; 2]>::try_from(&serialized[1..3])?),
                )))
            }
            DescriptorByte::RingElement32 => {
                if serialized.len() != 5 {
                    bail!("Invalid length for RingElement32");
                }
                Ok(NetworkValue::RingElement32(RingElement(
                    u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?),
                )))
            }
            DescriptorByte::RingElement64 => {
                if serialized.len() != 9 {
                    bail!("Invalid length for RingElement64");
                }
                Ok(NetworkValue::RingElement64(RingElement(
                    u64::from_le_bytes(<[u8; 8]>::try_from(&serialized[1..9])?),
                )))
            }
            DescriptorByte::VecRing16 => {
                let res = deserialize_vec_ring::<u16, 2, _>(serialized, u16::from_le_bytes)?;
                Ok(NetworkValue::VecRing16(res))
            }
            DescriptorByte::VecRing32 => {
                let res = deserialize_vec_ring::<u32, 4, _>(serialized, u32::from_le_bytes)?;
                Ok(NetworkValue::VecRing32(res))
            }
            DescriptorByte::VecRing64 => {
                let res = deserialize_vec_ring::<u64, 8, _>(serialized, u64::from_le_bytes)?;
                Ok(NetworkValue::VecRing64(res))
            }
            DescriptorByte::StateChecksum => {
                let (a, b, c) = (1, 9, 17);
                if serialized.len() != 1 + size_of::<u64>() * 2 {
                    bail!("Invalid length for StateChecksum");
                }
                Ok(NetworkValue::StateChecksum(StateChecksum {
                    irises: u64::from_le_bytes(<[u8; 8]>::try_from(&serialized[a..b])?),
                    graph: u64::from_le_bytes(<[u8; 8]>::try_from(&serialized[b..c])?),
                }))
            }
            _ => Err(eyre!("Invalid network value type")),
        }
    }

    // below are functions whose names have been preserved to not impact existing code.

    pub fn to_network(&self) -> Vec<u8> {
        let mut res = BytesMut::with_capacity(self.byte_len());
        self.serialize(&mut res);
        res.freeze().into()
    }

    pub fn vec_to_network(v: Vec<Self>) -> Self {
        Self::NetworkVec(v)
    }

    pub fn vec_from_network(nv: Self) -> Result<Vec<Self>> {
        match nv {
            NetworkValue::NetworkVec(v) => Ok(v),
            _ => Err(eyre!(
                "expected NetworkVec but got {}",
                nv.get_descriptor_byte()
            )),
        }
    }
}

fn deserialize_vec_ring<T, const N: usize, F>(
    serialized: &[u8],
    from_bytes: F,
) -> Result<Vec<RingElement<T>>>
where
    F: Fn([u8; N]) -> T,
    T: shares::int_ring::IntRing2k,
{
    // this variable is for error messages
    let type_bits = size_of::<T>() * 8;

    if serialized.len() < 5 {
        bail!(
            "Invalid length for VecRing{}: too short: {}",
            type_bits,
            serialized.len()
        );
    }
    let len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?) as usize;
    if serialized.len() != 5 + len {
        bail!(
            "Invalid length for VecRing{}: length mismatch {} but expected {}",
            type_bits,
            serialized.len(),
            5 + len
        );
    }
    if len % size_of::<T>() != 0 {
        bail!(
            "invalid length for VecRing{}: length {} does not divide type length {}",
            type_bits,
            len,
            type_bits
        );
    }
    let num_elements = len / size_of::<T>();
    let mut res = Vec::with_capacity(num_elements);
    for chunk in serialized[5..].chunks_exact(N) {
        let slice = <[u8; N]>::try_from(chunk)?;
        let value: T = from_bytes(slice);
        res.push(RingElement(value));
    }
    Ok(res)
}

pub trait NetworkInt
where
    Self: IntRing2k,
{
    fn new_network_element(element: RingElement<Self>) -> NetworkValue;
    fn new_network_vec(elements: Vec<RingElement<Self>>) -> NetworkValue;
    fn into_vec(value: NetworkValue) -> Result<Vec<RingElement<Self>>>;
}

macro_rules! impl_network_int {
    ($t:ty, $elem:ident, $vec:ident) => {
        impl NetworkInt for $t {
            fn new_network_element(e: RingElement<$t>) -> NetworkValue {
                NetworkValue::$elem(e)
            }
            fn new_network_vec(v: Vec<RingElement<$t>>) -> NetworkValue {
                NetworkValue::$vec(v)
            }
            fn into_vec(val: NetworkValue) -> Result<Vec<RingElement<$t>>> {
                match val {
                    NetworkValue::$elem(e) => Ok(vec![e]),
                    NetworkValue::$vec(v) => Ok(v),
                    _ => Err(eyre!(
                        "Invalid conversion to Vec<RingElement<{}>>",
                        stringify!($t)
                    )),
                }
            }
        }
    };
}

impl_network_int!(u16, RingElement16, VecRing16);
impl_network_int!(u32, RingElement32, VecRing32);
impl_network_int!(u64, RingElement64, VecRing64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() -> Result<()> {
        let values = (0..2).map(RingElement).collect::<Vec<_>>();
        let network_values = values
            .iter()
            .map(|v| NetworkValue::RingElement16(*v))
            .collect::<Vec<_>>();
        let mut bm = BytesMut::new();
        NetworkValue::NetworkVec(network_values.clone()).serialize(&mut bm);
        let serialized = bm.freeze().to_vec();
        let result_vec = NetworkValue::deserialize_vec(&serialized)?;
        assert_eq!(network_values, result_vec);

        Ok(())
    }

    /// Test from_network with empty data
    #[test]
    fn test_from_network_empty() -> Result<()> {
        let result = NetworkValue::deserialize(&[]);
        assert_eq!(result.unwrap_err().to_string(), "Empty serialized data");
        Ok(())
    }
}
