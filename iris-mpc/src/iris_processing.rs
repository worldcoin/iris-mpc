#![allow(clippy::needless_range_loop)]

use aws_sdk_s3::Client as S3Client;
use eyre::{Context, Report};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        key_pair::SharesEncryptionKeyPairs,
        smpc_request::{
            decrypt_iris_share, get_iris_data_by_party_id, validate_iris_share, ReceiveRequestError,
        },
    },
};
use std::sync::Arc;
use tokio::{
    sync::Semaphore,
    task::{spawn_blocking, JoinHandle},
};

pub type GaloisShares = (
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    Vec<GaloisRingIrisCodeShare>,
    Vec<GaloisRingTrimmedMaskCodeShare>,
    Vec<GaloisRingIrisCodeShare>,
    Vec<GaloisRingTrimmedMaskCodeShare>,
);
pub type ParseSharesTaskResult = Result<(GaloisShares, GaloisShares), Report>;

pub struct IrisShareProcessor {
    party_id:    usize,
    key_pairs:   SharesEncryptionKeyPairs,
    s3_client:   S3Client,
    bucket_name: String,
}

impl IrisShareProcessor {
    pub fn new(
        party_id: usize,
        key_pairs: SharesEncryptionKeyPairs,
        s3_client: S3Client,
        bucket_name: String,
    ) -> Self {
        Self {
            party_id,
            key_pairs,
            s3_client,
            bucket_name,
        }
    }

    pub async fn process_share(
        &self,
        s3_key: String,
    ) -> Result<(GaloisShares, GaloisShares), eyre::Error> {
        // Process the iris share (extract from current code)
        // ...
        let (share_b64, hash) = match get_iris_data_by_party_id(
            &s3_key,
            self.party_id,
            &self.bucket_name,
            &self.s3_client,
        )
        .await
        {
            Ok(iris_message_share) => iris_message_share,
            Err(e) => {
                tracing::error!("Failed to get iris shares: {:?}", e);
                eyre::bail!("Failed to get iris shares: {:?}", e);
            }
        };

        let iris_message_share = match decrypt_iris_share(share_b64, self.key_pairs.clone()) {
            Ok(iris_data) => iris_data,
            Err(e) => {
                tracing::error!("Failed to decrypt iris shares: {:?}", e);
                eyre::bail!("Failed to decrypt iris shares: {:?}", e);
            }
        };

        match validate_iris_share(hash, iris_message_share.clone()) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("Failed to validate iris shares: {:?}", e);
                eyre::bail!("Failed to validate iris shares: {:?}", e);
            }
        }

        let (left_code, left_mask) = decode_iris_message_shares(
            iris_message_share.left_iris_code_shares,
            iris_message_share.left_mask_code_shares,
        )?;

        let (right_code, right_mask) = decode_iris_message_shares(
            iris_message_share.right_iris_code_shares,
            iris_message_share.right_mask_code_shares,
        )?;

        // Preprocess shares for left eye.
        let left_future =
            spawn_blocking(move || preprocess_iris_message_shares(left_code, left_mask));

        // Preprocess shares for right eye.
        let right_future =
            spawn_blocking(move || preprocess_iris_message_shares(right_code, right_mask));

        let (left_result, right_result) = tokio::join!(left_future, right_future);

        Ok((
            left_result.context("while processing left iris shares")??,
            right_result.context("while processing right iris shares")??,
        ))
    }
}

impl Clone for IrisShareProcessor {
    fn clone(&self) -> Self {
        Self {
            party_id:    self.party_id,
            key_pairs:   self.key_pairs.clone(),
            s3_client:   self.s3_client.clone(),
            bucket_name: self.bucket_name.clone(),
        }
    }
}

pub fn decode_iris_message_shares(
    code_share: String,
    mask_share: String,
) -> eyre::Result<(GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare)> {
    let iris_share = GaloisRingIrisCodeShare::from_base64(&code_share)
        .context("Failed to base64 parse iris code")?;
    let mask_share: GaloisRingTrimmedMaskCodeShare =
        GaloisRingIrisCodeShare::from_base64(&mask_share)
            .context("Failed to base64 parse iris mask")?
            .into();

    Ok((iris_share, mask_share))
}

pub fn preprocess_iris_message_shares(
    code_share: GaloisRingIrisCodeShare,
    mask_share: GaloisRingTrimmedMaskCodeShare,
) -> eyre::Result<GaloisShares> {
    let mut code_share = code_share;
    let mut mask_share = mask_share;

    // Original for storage.
    let store_iris_shares = code_share.clone();
    let store_mask_shares = mask_share.clone();

    // With rotations for in-memory database.
    let db_iris_shares = code_share.all_rotations();
    let db_mask_shares = mask_share.all_rotations();

    // With Lagrange interpolation.
    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut code_share);
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(&mut mask_share);

    Ok((
        store_iris_shares,
        store_mask_shares,
        db_iris_shares,
        db_mask_shares,
        code_share.all_rotations(),
        mask_share.all_rotations(),
    ))
}

pub fn get_iris_shares_parse_task(
    semaphore: Arc<Semaphore>,
    s3_key: String,
    iris_share_processor: IrisShareProcessor,
) -> Result<JoinHandle<ParseSharesTaskResult>, ReceiveRequestError> {
    let handle = tokio::spawn(async move {
        let _ = semaphore.acquire().await?;
        let result = iris_share_processor.process_share(s3_key).await?;
        Ok(result)
    });
    Ok(handle)
}
