use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use base64::{engine::general_purpose, Engine};
use eyre::{eyre, Result};
use serde::Serialize;
use sodiumoxide::crypto::box_::PublicKey;

use iris_mpc_common::{helpers::key_pair::download_public_key, IrisSerialId};

use super::client::NodeAwsClients;
use crate::types::NetEncryptionPublicKeys;

#[async_trait]
pub trait ServiceOperations {
    /// Uploads a set of Iris serial identifiers to be marked as deleted.
    async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<()>;
}

#[async_trait]
impl ServiceOperations for NodeAwsClients {
    async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<()> {
        // Payload struct.
        #[derive(Serialize, Debug, Clone)]
        struct Payload {
            deleted_serial_ids: Vec<IrisSerialId>,
        }

        // Set S3 key/bucket.
        let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", self.config().environment());
        let s3_key = format!("{}_deleted_serial_ids.json", self.config().environment());
        self.log_info(
            format!(
                "Uploading deleted serial ids to S3 bucket: {}, key: {}",
                s3_bucket, s3_key
            )
            .as_str(),
        );

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
            .bucket(s3_bucket)
            .key(s3_key)
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

/// Downloads network wide set of node encryption public keys.
pub async fn download_net_encryption_public_keys(
    public_key_base_url: &String,
) -> Result<NetEncryptionPublicKeys> {
    async fn get_public_key(party_idx: usize, public_key_base_url: &String) -> PublicKey {
        let pbk_raw = download_public_key(public_key_base_url.clone(), party_idx.to_string())
            .await
            .unwrap();
        let pbk_bytes = general_purpose::STANDARD.decode(pbk_raw).unwrap();

        PublicKey::from_slice(&pbk_bytes).unwrap()
    }

    Ok([
        get_public_key(0, public_key_base_url).await,
        get_public_key(1, public_key_base_url).await,
        get_public_key(2, public_key_base_url).await,
    ])
}
