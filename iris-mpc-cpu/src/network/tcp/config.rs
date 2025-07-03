use crate::network::tcp::data::StreamId;
use std::{cmp, time::Duration};

#[derive(Default, Clone, Debug)]
pub struct TcpConfig {
    pub timeout_duration: Duration,
    // the number of requests processed at once
    pub request_parallelism: usize,
    // number of TCP connections
    pub num_connections: usize,
    // number of sessions per connection
    pub max_sessions_per_connection: usize,
}

impl TcpConfig {
    pub fn new(
        timeout_duration: Duration,
        num_connections: usize,
        request_parallelism: usize,
    ) -> Self {
        // don't allow fewer requests than connections...
        let connection_parallelism = cmp::min(num_connections, request_parallelism);
        // x2 for both eyes
        let request_parallelism = request_parallelism * 2 * iris_mpc_common::ROTATIONS;

        Self {
            timeout_duration,
            request_parallelism,
            num_connections: connection_parallelism,
            max_sessions_per_connection: request_parallelism / connection_parallelism,
        }
    }

    // max_sessions_per_connection doesn't have to divide request_parallelism. in that case, one TcpStream may have
    // fewer sessions than the others. This function determines how many sessions are handled by a TcpStream based on the stream id
    pub fn get_sessions_for_stream(&self, stream_id: &StreamId) -> usize {
        match self.num_connections {
            0 => unreachable!(),
            1 => self.request_parallelism,
            num_connections => {
                let stream_id = stream_id.0 as usize;
                if stream_id <= num_connections - 2
                    || self.request_parallelism % self.max_sessions_per_connection == 0
                {
                    self.max_sessions_per_connection
                } else {
                    self.request_parallelism % self.max_sessions_per_connection
                }
            }
        }
    }
}
