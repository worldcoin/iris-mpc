use crate::execution::player::Identity;
use std::collections::HashMap;
use tokio::net::TcpStream;

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
pub struct StreamId(pub u32);

impl StreamId {
    pub fn new(val: u32) -> Self {
        Self(val)
    }
}

impl From<u32> for StreamId {
    fn from(val: u32) -> Self {
        StreamId::new(val)
    }
}

pub type PeerConnections = HashMap<Identity, HashMap<StreamId, TcpConnection>>;

pub struct TcpConnection {
    pub peer: Identity,
    pub stream: TcpStream,
    pub stream_id: StreamId,
}

impl TcpConnection {
    pub fn new(peer: Identity, stream: TcpStream, stream_id: StreamId) -> Self {
        Self {
            peer,
            stream,
            stream_id,
        }
    }

    pub fn peer_id(&self) -> Identity {
        self.peer.clone()
    }
}

impl std::fmt::Debug for TcpConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpConnection")
            .field("peer", &self.peer)
            .field("stream_id", &self.stream_id)
            .finish()
    }
}
