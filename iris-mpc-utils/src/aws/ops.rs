use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use base64::{engine::general_purpose, Engine};
use eyre::{eyre, Result};
use serde::Serialize;
use serde_json;
use sodiumoxide::crypto::box_::PublicKey;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::{helpers::key_pair::download_public_key, IrisSerialId};

use super::{client::AwsClient, factory};
use crate::types::NetworkEncryptionPublicKeys;

impl AwsClient {
    /// Downloads MPC encryption public keys.
    pub async fn download_encryption_public_keys(
        public_key_base_url: &str,
    ) -> Result<NetworkEncryptionPublicKeys> {
        async fn get_public_key(party_idx: usize, public_key_base_url: &str) -> PublicKey {
            let pbk_raw =
                download_public_key(public_key_base_url.to_string(), party_idx.to_string())
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

    /// Encrypts and uploads Iris shares.
    pub async fn encrypt_and_upload_iris_shares(
        &self,
        shares: &IrisCodePartyShares,
    ) -> Result<String> {
        let s3_bucket = self.config().request_bucket_name();
        let s3_key = shares.signup_id.as_str();
        let s3_shares = factory::create_iris_party_shares_for_s3(shares, self.encryption_keys());
        let s3_payload = serde_json::to_vec(&s3_shares)?;

        self.upload_to_s3(s3_bucket, s3_key, &s3_payload)
            .await
            .map_err(|err| eyre!("{}", err))?;

        Ok(s3_bucket.clone())
    }

    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<()> {
        #[derive(Serialize, Debug, Clone)]
        struct Payload {
            deleted_serial_ids: Vec<IrisSerialId>,
        }

        let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", self.config().environment());
        let s3_key = format!("{}_deleted_serial_ids.json", self.config().environment());
        let s3_payload = S3_ByteStream::from(
            serde_json::to_string(&Payload {
                deleted_serial_ids: data.to_owned(),
            })
            .unwrap()
            .into_bytes(),
        );

        self.log_info(
            format!(
                "Uploading deleted serial ids to S3 bucket: {}, key: {}",
                s3_bucket, s3_key
            )
            .as_str(),
        );

        self.s3()
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(s3_payload)
            .send()
            .await
            .map_err(|err| {
                self.log_error(format!("Failed to upload file to S3: {}", err).as_str());
                eyre!("Failed to upload Iris deletions to S3")
            })?;

        Ok(())
    }
}
