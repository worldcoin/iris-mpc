use serde::Serialize;
use serde_json;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::IrisSerialId;

use super::{client::AwsClient, errors::AwsClientError, factory, types::S3ObjectInfo};

impl AwsClient {
    /// Encrypts and uploads Iris shares.
    pub async fn encrypt_and_upload_iris_shares(
        &self,
        shares: &IrisCodePartyShares,
    ) -> Result<String, AwsClientError> {
        let s3_bucket = self.config().s3_request_bucket_name().to_owned();
        let s3_key = shares.signup_id.to_owned();
        let s3_body = factory::create_iris_party_shares_for_s3(shares, &self.public_keyset());
        let s3_obj = S3ObjectInfo::new_from_jsonic(&s3_bucket, &s3_key, &s3_body);

        match self.s3_put_object(&s3_obj).await {
            Ok(_) => Ok(s3_bucket),
            Err(e) => {
                tracing::error!("SNS publish error: {}", e);
                Err(AwsClientError::IrisSharesEncryptAndUploadError(
                    e.to_string(),
                ))
            }
        }
    }

    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<(), AwsClientError> {
        #[derive(Serialize, Debug, Clone)]
        struct S3Data {
            deleted_serial_ids: Vec<IrisSerialId>,
        }

        let data = S3Data {
            deleted_serial_ids: data.to_owned(),
        };

        let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", self.config().environment());
        let s3_key = format!("{}_deleted_serial_ids.json", self.config().environment());
        let s3_obj = S3ObjectInfo::new(s3_bucket, s3_key, serde_json::to_vec(&data).unwrap());

        match self.s3_put_object(&s3_obj).await {
            Ok(_) => Ok(()),
            Err(e) => Err(AwsClientError::IrisDeletionsUploadError(e.to_string())),
        }
    }
}
