use super::super::clients::Clients as AwsClients;
use crate::misc::log_error;
use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use eyre::{eyre, Result};
use iris_mpc_common::IrisSerialId;
use serde::Serialize;

/// Component name for logging purposes.
const COMPONENT: &str = "State-AWS-S3";

impl AwsClients {
    /// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
    pub async fn s3_upload_iris_deletions(&self, data: &Vec<IrisSerialId>) -> Result<()> {
        // Payload struct.
        #[derive(Serialize, Debug, Clone)]
        struct Payload {
            deleted_serial_ids: Vec<IrisSerialId>,
        }

        // Set payload.
        let payload = S3_ByteStream::from(
            serde_json::to_string(&Payload {
                deleted_serial_ids: data.to_owned(),
            })
            .unwrap()
            .into_bytes(),
        );

        // Set key/bucket.
        let bucket = format!("wf-smpcv2-{}-sync-protocol", self.config().environment());
        let key = format!("{}_deleted_serial_ids.json", self.config().environment());
        self.log_info(
            format!(
                "Uploading deleted serial ids to S3 bucket: {}, key: {}",
                bucket, key
            )
            .as_str(),
        );

        // Upload payload.
        self.s3()
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(payload)
            .send()
            .await
            .map_err(|err| {
                log_error(COMPONENT, format!("Failed to upload file to S3: {}", err));
                eyre!("Failed to upload Iris deletions to S3")
            })?;

        Ok(())
    }
}
