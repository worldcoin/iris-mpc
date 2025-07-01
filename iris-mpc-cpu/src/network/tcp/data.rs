use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{
    execution::{player::Identity, session::SessionId},
    network::value::NetworkValue,
};
use std::collections::HashMap;

// session multiplexing over a socket requires a SessionId
pub type OutboundMsg = (SessionId, NetworkValue);
pub type OutStream = UnboundedSender<NetworkValue>;
pub type InStream = UnboundedReceiver<NetworkValue>;

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

pub type PeerConnections<T> = HashMap<Identity, HashMap<StreamId, Connection<T>>>;

pub struct Connection<T> {
    pub peer: Identity,
    pub stream: T,
    pub stream_id: StreamId,
}

impl<T> Connection<T> {
    pub fn new(peer: Identity, stream: T, stream_id: StreamId) -> Self {
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

impl<T> std::fmt::Debug for Connection<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("peer", &self.peer)
            .field("stream_id", &self.stream_id)
            .finish()
    }
}
