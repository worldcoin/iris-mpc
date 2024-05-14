use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::share::RepShare;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoToThreeIrisCodeMessage {
    pub id: u64,
    pub party_id: u8,
    pub from: u8,
    pub data: Vec<RepShare<u16>>,
}
impl Default for TwoToThreeIrisCodeMessage {
    fn default() -> Self {
        Self {
            id: 0,
            party_id: 0,
            from: 0,
            data: vec![RepShare::new(0, 0); 12800],
        }
    }
}

impl TwoToThreeIrisCodeMessage {
    pub fn new(id: u64, party_id: u8, from: u8, data: Vec<RepShare<u16>>) -> Self {
        Self {
            id,
            party_id,
            from,
            data,
        }
    }
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        for share in self.data.iter() {
            writer.write_u16(share.a).await?;
            writer.write_u16(share.b).await?;
        }
        writer.flush().await
    }
    /// assumes that internal data field is of the correct size (e.g. `self` is created with Default::default())
    pub async fn recv(&mut self, reader: &mut (impl AsyncReadExt + Unpin)) -> std::io::Result<()> {
        self.id = reader.read_u64().await?;
        self.party_id = reader.read_u8().await?;
        self.from = reader.read_u8().await?;
        for share in self.data.iter_mut() {
            share.a = reader.read_u16().await?;
            share.b = reader.read_u16().await?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedBinaryAndMessage {
    pub id: u64,
    pub party_id: u8,
    pub from: u8,
    pub data: Vec<u64>,
}

impl Default for PackedBinaryAndMessage {
    fn default() -> Self {
        Self {
            id: 0,
            party_id: 0,
            from: 0,
            data: vec![0; 12800 / 64],
        }
    }
}

impl PackedBinaryAndMessage {
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        let data: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        writer.write_all(data).await?;
        writer.flush().await
    }
    /// assumes that internal data field is of the correct size (e.g. `self` is created with Default::default())
    pub async fn recv(&mut self, reader: &mut (impl AsyncReadExt + Unpin)) -> std::io::Result<()> {
        self.id = reader.read_u64().await?;
        self.party_id = reader.read_u8().await?;
        self.from = reader.read_u8().await?;
        let data: &mut [u8] = bytemuck::cast_slice_mut(self.data.as_mut_slice());
        reader.read_exact(data).await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpMulManyMessage {
    pub id: u64,
    pub party_id: u8,
    pub from: u8,
    pub data: Vec<u16>,
}

impl Default for FpMulManyMessage {
    fn default() -> Self {
        Self {
            id: 0,
            party_id: 0,
            from: 0,
            data: vec![0; 12800],
        }
    }
}

impl FpMulManyMessage {
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        let data: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        writer.write_all(data).await?;
        writer.flush().await
    }
    /// assumes that internal data field is of the correct size (e.g. `self` is created with Default::default())
    pub async fn recv(&mut self, reader: &mut (impl AsyncReadExt + Unpin)) -> std::io::Result<()> {
        self.id = reader.read_u64().await?;
        self.party_id = reader.read_u8().await?;
        self.from = reader.read_u8().await?;
        let data: &mut [u8] = bytemuck::cast_slice_mut(self.data.as_mut_slice());
        reader.read_exact(data).await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShamirSharesMessage {
    pub id: u64,
    pub party_id: u8,
    pub from: u8,
    pub data: Vec<u16>,
}
impl Default for ShamirSharesMessage {
    fn default() -> Self {
        Self {
            id: 0,
            party_id: 0,
            from: 0,
            data: vec![0; 12800],
        }
    }
}

impl ShamirSharesMessage {
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        let data: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        writer.write_all(data).await?;
        writer.flush().await
    }
    /// assumes that internal data field is of the correct size (e.g. `self` is created with Default::default())
    pub async fn recv(&mut self, reader: &mut (impl AsyncReadExt + Unpin)) -> std::io::Result<()> {
        self.id = reader.read_u64().await?;
        self.party_id = reader.read_u8().await?;
        self.from = reader.read_u8().await?;
        let data: &mut [u8] = bytemuck::cast_slice_mut(self.data.as_mut_slice());
        reader.read_exact(data).await?;
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum MpcMessages {
    Stage1(PackedBinaryAndMessage),
    Stage2(FpMulManyMessage),
    Stage3(ShamirSharesMessage),
}

impl From<PackedBinaryAndMessage> for MpcMessages {
    fn from(x: PackedBinaryAndMessage) -> Self {
        MpcMessages::Stage1(x)
    }
}

impl From<FpMulManyMessage> for MpcMessages {
    fn from(x: FpMulManyMessage) -> Self {
        MpcMessages::Stage2(x)
    }
}

impl From<ShamirSharesMessage> for MpcMessages {
    fn from(x: ShamirSharesMessage) -> Self {
        MpcMessages::Stage3(x)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ClientMessages {
    Shares(TwoToThreeIrisCodeMessage),
    Masks(ShamirSharesMessage),
}

impl From<TwoToThreeIrisCodeMessage> for ClientMessages {
    fn from(x: TwoToThreeIrisCodeMessage) -> Self {
        ClientMessages::Shares(x)
    }
}
impl From<ShamirSharesMessage> for ClientMessages {
    fn from(x: ShamirSharesMessage) -> Self {
        ClientMessages::Masks(x)
    }
}
