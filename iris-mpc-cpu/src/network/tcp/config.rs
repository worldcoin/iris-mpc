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

    pub fn get_sessions_for_connection(&self, idx: usize) -> usize {
        let num_sessions = self.num_sessions;
        let num_connections = self.num_connections;
        num_sessions / num_connections
            + if idx < (num_sessions % num_connections) {
                1
            } else {
                0
            }
    }
}
