use crate::execution::{player::Identity, session::SessionId};
use async_trait::async_trait;

/// Requirements for networking.
#[async_trait]
pub trait Networking {
    async fn send(
        &self,
        value: Vec<u8>,
        receiver: &Identity,
        session_id: &SessionId,
    ) -> eyre::Result<()>;

    async fn receive(&self, sender: &Identity, session_id: &SessionId) -> eyre::Result<Vec<u8>>;
}

pub mod local;
pub mod value;
