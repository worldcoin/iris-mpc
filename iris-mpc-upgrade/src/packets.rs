use iris_mpc_common::IRIS_CODE_LENGTH;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoToThreeIrisCodeMessage {
    pub id:       u64,
    pub party_id: u8,
    pub from:     u8,
    #[serde(with = "BigArray")]
    pub data:     [u16; IRIS_CODE_LENGTH],
}
impl Default for TwoToThreeIrisCodeMessage {
    fn default() -> Self {
        Self {
            id:       0,
            party_id: 0,
            from:     0,
            data:     [0; IRIS_CODE_LENGTH],
        }
    }
}

impl TwoToThreeIrisCodeMessage {
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        let data: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        writer.write_all(data).await?;
        writer.flush().await
    }
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
pub struct MaskShareMessage {
    pub id:       u64,
    pub party_id: u8,
    pub from:     u8,
    #[serde(with = "BigArray")]
    pub data:     [u16; IRIS_CODE_LENGTH],
}
impl Default for MaskShareMessage {
    fn default() -> Self {
        Self {
            id:       0,
            party_id: 0,
            from:     0,
            data:     [0; IRIS_CODE_LENGTH],
        }
    }
}

impl MaskShareMessage {
    pub async fn send(&self, writer: &mut (impl AsyncWriteExt + Unpin)) -> std::io::Result<()> {
        writer.write_u64(self.id).await?;
        writer.write_u8(self.party_id).await?;
        writer.write_u8(self.from).await?;
        let data: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        writer.write_all(data).await?;
        writer.flush().await
    }
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
pub enum ClientMessages {
    Shares(TwoToThreeIrisCodeMessage),
    Masks(MaskShareMessage),
}

impl From<TwoToThreeIrisCodeMessage> for ClientMessages {
    fn from(x: TwoToThreeIrisCodeMessage) -> Self {
        ClientMessages::Shares(x)
    }
}
impl From<MaskShareMessage> for ClientMessages {
    fn from(x: MaskShareMessage) -> Self {
        ClientMessages::Masks(x)
    }
}
