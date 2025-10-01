mod connection_state;
pub use connection_state::ConnectionState;

use crate::{
    execution::{player::Identity, session::StreamId},
    network::tcp2::NetworkConnection,
};

pub struct Connection<T: NetworkConnection> {
    stream: T,
    stream_id: StreamId,
    own_id: Identity,
    peer_id: Identity,
}
