use serde::Serialize;

use iris_mpc_common::IrisSerialId;

use super::{client::AwsClient, errors::AwsClientError, types::S3ObjectInfo};

impl AwsClient {
    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn upload_iris_deletions(&self, data: &[IrisSerialId]) -> Result<(), AwsClientError> {
        #[derive(Serialize)]
        struct S3Data<'a> {
            deleted_serial_ids: &'a [IrisSerialId],
        }

        let data = S3Data {
            deleted_serial_ids: data,
        };

        let environment = self.config().environment();
        let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", environment);
        let s3_key = format!("{}_deleted_serial_ids.json", environment);
        let s3_obj = S3ObjectInfo::new(&s3_bucket, &s3_key, &data);

        self.s3_put_object(s3_obj)
            .await
            .map_err(|e| AwsClientError::IrisDeletionsUploadError(e.to_string()))
    }
}
