use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use serde::{Deserialize, Serialize};

// Struct for S3 deserialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrisDeletionsForS3 {
    pub deleted_serial_ids: Vec<IrisSerialId>,
}

/// AWS S3 bucket for iris deletions.
pub fn get_s3_bucket_for_iris_deletions(config: &NodeConfig) -> String {
    format!("wf-smpcv2-{}-sync-protocol", config.environment)
}

/// AWS S3 key for iris deletions.
pub fn get_s3_key_for_iris_deletions(config: &NodeConfig) -> String {
    format!("{}_deleted_serial_ids.json", config.environment)
}
