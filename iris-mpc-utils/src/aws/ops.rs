use serde::Serialize;

use iris_mpc_common::SerialId;
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::{
    client::AwsClient,
    errors::AwsClientError,
    factory::{create_iris_code_shares, create_iris_code_shares_s3},
    types::S3ObjectInfo,
};
use crate::{constants::N_PARTIES, irises::GaloisRingSharedIrisForUpload};

impl AwsClient {
    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn s3_upload_iris_deletions(&self, data: &[SerialId]) -> Result<(), AwsClientError> {
        #[derive(Serialize)]
        struct S3Data<'a> {
            deleted_serial_ids: &'a [SerialId],
        }

        let data = S3Data {
            deleted_serial_ids: data,
        };

        let environment = self.config().environment();
        let object_key = format!("{}_deleted_serial_ids.json", environment);
        let object = S3ObjectInfo::new(
            self.config().iris_deletions_store_location(),
            &object_key,
            &data,
        );
        self.s3_put_object(&object)
            .await
            .map_err(|e| AwsClientError::IrisDeletionsUploadError(e.to_string()))
    }

    // Uploads JSON-encoded Iris shares to the configured object store.
    pub async fn s3_upload_iris_shares(
        &self,
        signup_id: &uuid::Uuid,
        shares: &BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>,
    ) -> Result<S3ObjectInfo, AwsClientError> {
        // Build JSON-compatible shares for the existing wire format.
        let shares = create_iris_code_shares_s3(
            &create_iris_code_shares(signup_id, shares),
            &self.public_keyset(),
        );

        // Upload to object storage.
        let s3_obj_info = S3ObjectInfo::new(
            self.config().s3_request_bucket_name(),
            &signup_id.to_string(),
            &shares,
        );
        self.s3_put_object(&s3_obj_info).await.map(|_| s3_obj_info)
    }
}
