use serde::{Deserialize, Serialize};

/// used as inputs to iris-mpc-store > insert_modification()
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct ModificationInput {
    pub serial_id: Option<i64>,
    pub request_type: String,
    #[serde(default = "default_modification_status")]
    pub status: String,
    #[serde(default = "default_false")]
    pub persisted: bool,
}

fn default_modification_status() -> String {
    "IN_PROGRESS".into()
}

fn default_false() -> bool {
    false
}
