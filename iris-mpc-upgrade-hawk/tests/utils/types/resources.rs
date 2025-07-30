use serde::{Deserialize, Serialize};

/// used as inputs to iris-mpc-store > insert_modification()
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct ModificationInput {
    pub serial_id: Option<i64>,
    pub request_type: String,
    pub s3_url: Option<String>,
}
