use bytes::{Bytes, BytesMut};
use eyre::{Error, Result};
use iris_mpc_common::id::PartyID;
pub type IoError = std::io::Error;

#[allow(async_fn_in_trait)]
pub trait NetworkTrait: Send + Sync {
    fn get_id(&self) -> PartyID;

    async fn shutdown(self) -> Result<(), IoError>;

    async fn send(&mut self, id: PartyID, data: Bytes) -> Result<(), IoError>;
    async fn send_next_id(&mut self, data: Bytes) -> Result<(), IoError>;
    async fn send_prev_id(&mut self, data: Bytes) -> Result<(), IoError>;

    async fn receive(&mut self, id: PartyID) -> Result<BytesMut, IoError>;
    async fn receive_prev_id(&mut self) -> Result<BytesMut, IoError>;
    async fn receive_next_id(&mut self) -> Result<BytesMut, IoError>;

    async fn broadcast(&mut self, data: Bytes) -> Result<Vec<BytesMut>, IoError>;
    //======= sync world =========
    fn blocking_send(&mut self, id: PartyID, data: Bytes) -> Result<(), IoError>;
    fn blocking_send_next_id(&mut self, data: Bytes) -> Result<(), IoError>;
    fn blocking_send_prev_id(&mut self, data: Bytes) -> Result<(), IoError>;

    fn blocking_receive(&mut self, id: PartyID) -> Result<BytesMut, IoError>;
    fn blocking_receive_prev_id(&mut self) -> Result<BytesMut, IoError>;
    fn blocking_receive_next_id(&mut self) -> Result<BytesMut, IoError>;

    fn blocking_broadcast(&mut self, data: Bytes) -> Result<Vec<BytesMut>, IoError>;
}

#[allow(async_fn_in_trait)]
pub trait NetworkEstablisher<N: NetworkTrait> {
    fn get_id(&self) -> PartyID;
    fn get_num_parties(&self) -> usize;
    async fn open_channel(&mut self) -> Result<N, Error>;
    async fn shutdown(self) -> Result<(), Error>;
    //======= sync world =========
    fn print_connection_stats(&self, out: &mut impl std::io::Write) -> std::io::Result<()>;
    fn get_send_receive(&self, i: usize) -> std::io::Result<(u64, u64)>;
}
