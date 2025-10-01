use crate::{execution::{player::Identity, session::StreamId}, network::tcp2::NetworkConnection};

pub struct <T: NetworkConnection> Connection {
    stream: T,
    stream_id: StreamId,
    own_id: Identity,
    peer_id: Identity,
}
