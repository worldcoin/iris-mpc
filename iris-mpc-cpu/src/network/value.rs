use crate::shares::{bit::Bit, ring_impl::RingElement, IntRing2k};
use eyre::{bail, eyre, Result};

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
        match self {
            NetworkValue::PrfKey(_) => 0x01,
            NetworkValue::RingElementBit(bit) => {
                if bit.convert().convert() {
                    0x12
                } else {
                    0x02
                }
            }
            NetworkValue::RingElement16(_) => 0x03,
            NetworkValue::RingElement32(_) => 0x04,
            NetworkValue::RingElement64(_) => 0x05,
            NetworkValue::VecRing16(_) => 0x06,
            NetworkValue::VecRing32(_) => 0x07,
            NetworkValue::VecRing64(_) => 0x08,
            NetworkValue::StateChecksum(_) => 0x09,
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            NetworkValue::PrfKey(_) => 1 + PRF_KEY_SIZE,
            NetworkValue::RingElementBit(_) => 1,
            NetworkValue::RingElement16(_) => 3,
            NetworkValue::RingElement32(_) => 5,
            NetworkValue::RingElement64(_) => 9,
            NetworkValue::VecRing16(v) => 5 + 2 * v.len(),
            NetworkValue::VecRing32(v) => 5 + 4 * v.len(),
            NetworkValue::VecRing64(v) => 5 + 8 * v.len(),
            NetworkValue::StateChecksum(_) => 1 + 8 + 8,
        }
    }

    fn to_network_inner(&self, res: &mut Vec<u8>) {
        res.push(self.get_descriptor_byte());

        match self {
            NetworkValue::PrfKey(key) => res.extend_from_slice(key),
            NetworkValue::RingElementBit(_) => {
                // Do nothing, the descriptor byte already contains the bit
                // value
            }
            NetworkValue::RingElement16(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::RingElement32(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::RingElement64(x) => res.extend_from_slice(&x.convert().to_le_bytes()),
            NetworkValue::VecRing16(v) => {
                res.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::VecRing32(v) => {
                res.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::VecRing64(v) => {
                res.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    res.extend_from_slice(&x.convert().to_le_bytes());
                }
            }
            NetworkValue::StateChecksum(checksums) => {
                res.extend_from_slice(&u64::to_le_bytes(checksums.irises));
                res.extend_from_slice(&u64::to_le_bytes(checksums.graph));
            }
        }
    }

    pub fn to_network(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(self.byte_len());
        self.to_network_inner(&mut res);
        res
    }

    pub fn from_network(serialized: Result<Vec<u8>>) -> Result<Self> {
        let serialized = serialized?;
        if serialized.is_empty() {
            bail!("Empty serialized data");
        }
        let descriptor = serialized[0];
        match descriptor {
            0x01 => {
                if serialized.len() != 1 + PRF_KEY_SIZE {
                    bail!("Invalid length for PrfKey");
                }
                Ok(NetworkValue::PrfKey(<[u8; PRF_KEY_SIZE]>::try_from(
                    &serialized[1..1 + PRF_KEY_SIZE],
                )?))
            }
            0x02 | 0x12 => {
                if serialized.len() != 1 {
                    bail!("Invalid length for RingElementBit");
                }
                let bit = if descriptor == 0x12 {
                    Bit::new(true)
                } else {
                    Bit::new(false)
                };
                Ok(NetworkValue::RingElementBit(RingElement(bit)))
            }
            0x03 => {
                if serialized.len() != 3 {
                    bail!("Invalid length for RingElement16");
                }
                Ok(NetworkValue::RingElement16(RingElement(
                    u16::from_le_bytes(<[u8; 2]>::try_from(&serialized[1..3])?),
                )))
            }
            0x04 => {
                if serialized.len() != 5 {
                    bail!("Invalid length for RingElement32");
                }
                Ok(NetworkValue::RingElement32(RingElement(
                    u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?),
                )))
            }
            0x05 => {
                if serialized.len() != 9 {
                    bail!("Invalid length for RingElement64");
                }
                Ok(NetworkValue::RingElement64(RingElement(
                    u64::from_le_bytes(<[u8; 8]>::try_from(&serialized[1..9])?),
                )))
            }
            0x06 => {
                if serialized.len() < 5 {
                    bail!("Invalid length for VecRing16: can't parse vector length");
                }
                let len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?) as usize;
                if serialized.len() != 5 + 2 * len {
                    bail!("Invalid length for VecRing16");
                }
                let mut res = Vec::with_capacity(len);
                for i in 0..len {
                    res.push(RingElement(u16::from_le_bytes(<[u8; 2]>::try_from(
                        &serialized[5 + 2 * i..5 + 2 * (i + 1)],
                    )?)));
                }
                Ok(NetworkValue::VecRing16(res))
            }
            0x07 => {
                if serialized.len() < 5 {
                    bail!("Invalid length for VecRing32: can't parse vector length");
                }
                let len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?) as usize;
                if serialized.len() != 5 + 4 * len {
                    bail!("Invalid length for VecRing32");
                }
                let mut res = Vec::with_capacity(len);
                for i in 0..len {
                    res.push(RingElement(u32::from_le_bytes(<[u8; 4]>::try_from(
                        &serialized[5 + 4 * i..5 + 4 * (i + 1)],
                    )?)));
                }
                Ok(NetworkValue::VecRing32(res))
            }
            0x08 => {
                if serialized.len() < 5 {
                    bail!("Invalid length for VecRing64: can't parse vector length");
                }
                let len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[1..5])?) as usize;
                if serialized.len() != 5 + 8 * len {
                    bail!("Invalid length for VecRing64");
                }
                let mut res = Vec::with_capacity(len);
                for i in 0..len {
                    res.push(RingElement(u64::from_le_bytes(<[u8; 8]>::try_from(
                        &serialized[5 + 8 * i..5 + 8 * (i + 1)],
                    )?)));
                }
                Ok(NetworkValue::VecRing64(res))
            }
            0x09 => {
                let (a, b, c) = (1, 9, 17);
                if serialized.len() != c {
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

    pub fn vec_to_network(values: &Vec<Self>) -> Vec<u8> {
        // 4 extra bytes for the length of the vector
        let len = values.iter().map(|v| v.byte_len()).sum::<usize>() + 4;
        let mut res = Vec::with_capacity(len);
        res.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            value.to_network_inner(&mut res);
        }
        res
    }

    pub fn vec_from_network(serialized: Result<Vec<u8>>) -> Result<Vec<Self>> {
        let serialized = serialized?;
        if serialized.len() < 4 {
            bail!("Can't parse vector length");
        }
        let len = u32::from_le_bytes(<[u8; 4]>::try_from(&serialized[0..4])?) as usize;
        let mut res = Vec::with_capacity(len);
        let mut offset = 4;
        for _ in 0..len {
            let descriptor = serialized[offset];
            let value_len = match descriptor {
                0x01 => 1 + PRF_KEY_SIZE,
                0x02 | 0x12 => 1, // RingElementBit
                0x03 => 3,        // RingElement16
                0x04 => 5,        // RingElement32
                0x05 => 9,        // RingElement64
                0x06 => {
                    if serialized.len() < offset + 5 {
                        bail!("Invalid length for VecRing16: can't parse vector length");
                    }
                    let len = u32::from_le_bytes(<[u8; 4]>::try_from(
                        &serialized[offset + 1..offset + 5],
                    )?) as usize;
                    5 + 2 * len
                }
                0x07 => {
                    if serialized.len() < offset + 5 {
                        bail!("Invalid length for VecRing32: can't parse vector length");
                    }
                    let len = u32::from_le_bytes(<[u8; 4]>::try_from(
                        &serialized[offset + 1..offset + 5],
                    )?) as usize;
                    5 + 4 * len
                }
                0x08 => {
                    if serialized.len() < offset + 5 {
                        bail!("Invalid length for VecRing64: can't parse vector length");
                    }
                    let len = u32::from_le_bytes(<[u8; 4]>::try_from(
                        &serialized[offset + 1..offset + 5],
                    )?) as usize;
                    5 + 8 * len
                }
                _ => bail!("Invalid network value type"),
            };
            res.push(NetworkValue::from_network(Ok(serialized
                [offset..offset + value_len]
                .to_vec()))?);
            offset += value_len;
        }
        Ok(res)
    }
}

pub trait NetworkInt
where
    Self: IntRing2k,
{
    fn new_network_element(element: RingElement<Self>) -> NetworkValue;
    fn new_network_vec(elements: Vec<RingElement<Self>>) -> NetworkValue;
    fn into_vec(value: NetworkValue) -> Result<Vec<RingElement<Self>>>;
}

impl NetworkInt for u16 {
    fn new_network_element(element: RingElement<Self>) -> NetworkValue {
        NetworkValue::RingElement16(element)
    }

    fn new_network_vec(elements: Vec<RingElement<Self>>) -> NetworkValue {
        NetworkValue::VecRing16(elements)
    }

    fn into_vec(value: NetworkValue) -> Result<Vec<RingElement<Self>>> {
        match value {
            NetworkValue::VecRing16(x) => Ok(x),
            NetworkValue::RingElement16(x) => Ok(vec![x]),
            _ => Err(eyre!("Invalid conversion to Vec<RingElement<u16>>")),
        }
    }
}
impl NetworkInt for u32 {
    fn new_network_element(element: RingElement<Self>) -> NetworkValue {
        NetworkValue::RingElement32(element)
    }

    fn new_network_vec(elements: Vec<RingElement<Self>>) -> NetworkValue {
        NetworkValue::VecRing32(elements)
    }

    fn into_vec(value: NetworkValue) -> Result<Vec<RingElement<Self>>> {
        match value {
            NetworkValue::VecRing32(x) => Ok(x),
            NetworkValue::RingElement32(x) => Ok(vec![x]),
            _ => Err(eyre!("Invalid conversion to Vec<RingElement<u32>>")),
        }
    }
}
impl NetworkInt for u64 {
    fn new_network_element(element: RingElement<Self>) -> NetworkValue {
        NetworkValue::RingElement64(element)
    }

    fn new_network_vec(elements: Vec<RingElement<Self>>) -> NetworkValue {
        NetworkValue::VecRing64(elements)
    }

    fn into_vec(value: NetworkValue) -> Result<Vec<RingElement<Self>>> {
        match value {
            NetworkValue::VecRing64(x) => Ok(x),
            NetworkValue::RingElement64(x) => Ok(vec![x]),
            _ => Err(eyre!("Invalid conversion to Vec<RingElement<u64>>")),
        }
    }
}

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
        let serialized = NetworkValue::vec_to_network(&network_values);
        let result_vec = NetworkValue::vec_from_network(Ok(serialized))?;
        assert_eq!(network_values, result_vec);

        Ok(())
    }

    /// Test from_network with empty data
    #[test]
    fn test_from_network_empty() -> Result<()> {
        let result = NetworkValue::from_network(Ok(vec![]));
        assert_eq!(result.unwrap_err().to_string(), "Empty serialized data");
        Ok(())
    }
}
