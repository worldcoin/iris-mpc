use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use base64::{engine::general_purpose, Engine};
use serde::Serialize;
use serde_json;
use sodiumoxide::crypto::box_::PublicKey;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::{helpers::key_pair::download_public_key, IrisSerialId};

use super::{client::AwsClient, errors::AwsClientError, factory};
use crate::types::EncryptionPublicKeyset;

impl AwsClient {
    /// Downloads a party's encryption public key.
    pub async fn download_encryption_public_key(
        public_key_base_url: String,
        party_idx: usize,
    ) -> Result<PublicKey, AwsClientError> {
        match download_public_key(public_key_base_url.to_string(), party_idx.to_string()).await {
            Ok(pbk_raw) => {
                tracing::info!("Downloaded public key of MPC party {}.", party_idx);
                Ok(
                    PublicKey::from_slice(&general_purpose::STANDARD.decode(pbk_raw).unwrap())
                        .unwrap(),
                )
            }
            Err(e) => {
                tracing::error!(
                    "Encryption keys of party {} download error: {}",
                    party_idx,
                    e
                );
                Err(AwsClientError::EncryptionKeysDownloadError {
                    error: e.to_string(),
                })
            }
        }
    }

    /// Downloads MPC encryption public keys.
    pub async fn download_encryption_public_keys(
        public_key_base_url: String,
    ) -> Result<EncryptionPublicKeyset, AwsClientError> {
        Ok([
            Self::download_encryption_public_key(public_key_base_url.clone(), 0)
                .await
                .unwrap(),
            Self::download_encryption_public_key(public_key_base_url.clone(), 1)
                .await
                .unwrap(),
            Self::download_encryption_public_key(public_key_base_url.clone(), 2)
                .await
                .unwrap(),
        ])
    }

    /// Encrypts and uploads Iris shares.
    pub async fn encrypt_and_upload_iris_shares(
        &self,
        encryption_keys: &EncryptionPublicKeyset,
        shares: &IrisCodePartyShares,
    ) -> Result<String, AwsClientError> {
        let s3_bucket = self.config().s3_request_bucket_name();
        let s3_key = shares.signup_id.as_str();
        let s3_shares = factory::create_iris_party_shares_for_s3(shares, encryption_keys);
        let s3_data = serde_json::to_vec(&s3_shares).unwrap();

        match self.s3_upload(s3_bucket, s3_key, &s3_data).await {
            Ok(_) => Ok(s3_bucket.to_owned()),
            Err(e) => {
                tracing::error!("SNS publish error: {}", e);
                Err(AwsClientError::IrisSharesEncryptAndUploadError {
                    error: e.to_string(),
                })
            }
        }
    }

    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<(), AwsClientError> {
        #[derive(Serialize, Debug, Clone)]
        struct S3Data {
            deleted_serial_ids: Vec<IrisSerialId>,
        }

        let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", self.config().environment());
        let s3_key = format!("{}_deleted_serial_ids.json", self.config().environment());
        let s3_data = S3_ByteStream::from(
            serde_json::to_string(&S3Data {
                deleted_serial_ids: data.to_owned(),
            })
            .unwrap()
            .into_bytes(),
        );

        match self
            .s3()
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(s3_data)
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("SNS publish error: {}", e);
                Err(AwsClientError::IrisDeletionsUploadError {
                    error: e.to_string(),
                })
            }
        }
    }
}
