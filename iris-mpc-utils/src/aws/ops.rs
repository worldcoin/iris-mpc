use serde::Serialize;

use iris_mpc_common::{galois_engine::degree4::GaloisRingIrisCodeShare, IrisSerialId};
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::{
    client::AwsClient,
    errors::AwsClientError,
    factory::{create_iris_code_party_shares, create_iris_party_shares_for_s3},
    types::S3ObjectInfo,
};
use crate::constants::N_PARTIES;

impl AwsClient {
    /// Uploads Iris serial identifiers marked for deletion.
    pub async fn s3_upload_iris_deletions(
        &self,
        data: &[IrisSerialId],
    ) -> Result<(), AwsClientError> {
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
        self.s3_put_object(&s3_obj)
            .await
            .map_err(|e| AwsClientError::IrisDeletionsUploadError(e.to_string()))
    }

    pub async fn s3_upload_iris_shares(
        &self,
        signup_id: &uuid::Uuid,
        shares: &BothEyes<[[GaloisRingIrisCodeShare; N_PARTIES]; 2]>,
    ) -> Result<S3ObjectInfo, AwsClientError> {
        // Set AWS-S3 JSON compatible shares.
        let [[l_code, l_mask], [r_code, r_mask]] = shares;
        let shares = create_iris_party_shares_for_s3(
            &create_iris_code_party_shares(
                signup_id.to_owned(),
                l_code.to_owned(),
                l_mask.to_owned(),
                r_code.to_owned(),
                r_mask.to_owned(),
            ),
            &self.public_keyset(),
        );

        // Upload to AWS-S3.
        let s3_obj_info = S3ObjectInfo::new(
            self.config().s3_request_bucket_name(),
            &signup_id.to_string(),
            &shares,
        );
        self.s3_put_object(&s3_obj_info).await.map(|_| s3_obj_info)
    }
}
