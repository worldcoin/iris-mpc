use crate::network::tcp::data::StreamId;
use std::{cmp, time::Duration};

#[derive(Default, Clone, Debug)]
pub struct TcpConfig {
    pub timeout_duration: Duration,
    // the number of sessions managed at once
    pub num_sessions: usize,
    // number of TCP connections
    pub num_connections: usize,
}

impl TcpConfig {
    pub fn new(timeout_duration: Duration, num_connections: usize, num_sessions: usize) -> Self {
        // don't allow fewer requests than connections...
        let connection_parallelism = cmp::min(num_connections, num_sessions);

        Self {
            timeout_duration,
            num_sessions,
            num_connections: connection_parallelism,
        }
    }

    // This function determines how many sessions are handled by a TcpStream based on the stream id
    pub fn get_sessions_for_stream(&self, stream_id: &StreamId) -> usize {
        (0..self.num_sessions)
            .filter(|i| i % self.num_connections == stream_id.0 as usize)
            .count()
    }
}
