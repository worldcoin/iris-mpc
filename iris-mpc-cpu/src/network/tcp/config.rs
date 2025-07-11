use crate::network::tcp::data::StreamId;
use std::{cmp, time::Duration};

#[derive(Default, Clone, Debug)]
pub struct TcpConfig {
    pub timeout_duration: Duration,
    // the number of requests processed at once
    pub request_parallelism: usize,
    // number of TCP connections
    pub num_connections: usize,
}

impl TcpConfig {
    pub fn new(
        timeout_duration: Duration,
        num_connections: usize,
        request_parallelism: usize,
    ) -> Self {
        // don't allow fewer requests than connections...
        let connection_parallelism = cmp::min(num_connections, request_parallelism);

        Self {
            timeout_duration,
            request_parallelism,
            num_connections: connection_parallelism,
        }
    }

    // This function determines how many sessions are handled by a TcpStream based on the stream id
    pub fn get_sessions_for_stream(&self, stream_id: &StreamId) -> usize {
        (0..self.request_parallelism)
            .filter(|i| i % self.num_connections == stream_id.0 as usize)
            .count()
    }
}
