use super::client::NodeAwsClient;
use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use eyre::{eyre, Result};
use iris_mpc_common::IrisSerialId;
use serde::Serialize;

#[async_trait]
pub trait ServiceOperations {
    /// Uploads a set of Iris serial identifiers to be marked as deleted.
    async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<()>;
}

#[async_trait]
impl ServiceOperations for NodeAwsClient {
    /// Uploads a set of Iris serial identifiers to be marked as deleted.
    async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<()> {
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

        // Upload payload.
        self.s3()
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(payload)
            .send()
            .await
            .map_err(|err| {
                self.log_error(format!("Failed to upload file to S3: {}", err).as_str());
                eyre!("Failed to upload Iris deletions to S3")
            })?;

        Ok(())
    }
}
