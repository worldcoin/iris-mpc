use std::fmt;

use serde::{Deserialize, Serialize};

/// Enumeration over request processing states plus time of state change for statisitical purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    // Associated responses have been received.
    Complete,
    // Enqueued upon system ingress queue.
    Enqueued,
    // Ready to be processed (shares uploaded if applicable).
    Ready,
    // Iris shares uploaded in advance of enqueuement.
    SharesUploaded,
}

impl RequestStatus {
    pub(super) const VARIANT_COUNT: usize = 4;
}

impl Default for RequestStatus {
    fn default() -> Self {
        Self::Ready
    }
}

impl fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Complete => write!(f, "Complete"),
            Self::Enqueued => write!(f, "Enqueued"),
            Self::Ready => write!(f, "Ready"),
            Self::SharesUploaded => write!(f, "SharesUploaded"),
        }
    }
}
