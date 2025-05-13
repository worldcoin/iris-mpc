pub mod batch;
pub mod job;
pub mod result_message;

use aws_sdk_s3::Client as S3Client;
use eyre::Result;
use eyre::{Context, Report};
use iris_mpc_common::galois_engine::degree4::{
    preprocess_iris_message_shares, GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
    GaloisShares,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::smpc_request::{
    decrypt_iris_share, get_iris_data_by_party_id, validate_iris_share, ReceiveRequestError,
};
use iris_mpc_common::job::BatchQuery;
use iris_mpc_store::Store;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::{spawn_blocking, JoinHandle};

type ParseSharesTaskResult = Result<(GaloisShares, GaloisShares), Report>;

fn decode_iris_message_shares(
    code_share: String,
    mask_share: String,
) -> Result<(
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
)> {
    let iris_share = GaloisRingIrisCodeShare::from_base64(&code_share)
        .context("Failed to base64 parse iris code")?;
    let mask_share: GaloisRingIrisCodeShare = GaloisRingIrisCodeShare::from_base64(&mask_share)
        .context("Failed to base64 parse iris mask")?;

    let iris_share_mirrored = iris_share.mirrored();
    let mask_share_mirrored = mask_share.mirrored();

    Ok((
        iris_share,
        mask_share.into(),
        iris_share_mirrored,
        mask_share_mirrored.into(),
    ))
}

fn get_iris_shares_parse_task(
    party_id: usize,
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    semaphore: Arc<Semaphore>,
    s3_client_arc: S3Client,
    bucket_name: String,
    s3_key: String,
) -> Result<JoinHandle<ParseSharesTaskResult>, ReceiveRequestError> {
    let handle =
        tokio::spawn(async move {
            let _ = semaphore.acquire().await?;

            let (share_b64, hash) =
                match get_iris_data_by_party_id(&s3_key, party_id, &bucket_name, &s3_client_arc)
                    .await
                {
                    Ok(iris_message_share) => iris_message_share,
                    Err(e) => {
                        tracing::error!("Failed to get iris shares: {:?}", e);
                        eyre::bail!("Failed to get iris shares: {:?}", e);
                    }
                };

            let iris_message_share =
                match decrypt_iris_share(share_b64, shares_encryption_key_pairs.clone()) {
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

            let (left_code, left_mask, left_code_mirrored, left_mask_mirrored) =
                decode_iris_message_shares(
                    iris_message_share.left_iris_code_shares,
                    iris_message_share.left_mask_code_shares,
                )?;

            let (right_code, right_mask, right_code_mirrored, right_mask_mirrored) =
                decode_iris_message_shares(
                    iris_message_share.right_iris_code_shares,
                    iris_message_share.right_mask_code_shares,
                )?;

            // Preprocess shares for left eye.
            let left_future = spawn_blocking(move || {
                preprocess_iris_message_shares(
                    left_code,
                    left_mask,
                    left_code_mirrored,
                    left_mask_mirrored,
                )
            });

            // Preprocess shares for right eye.
            let right_future = spawn_blocking(move || {
                preprocess_iris_message_shares(
                    right_code,
                    right_mask,
                    right_code_mirrored,
                    right_mask_mirrored,
                )
            });

            let (left_result, right_result) = tokio::join!(left_future, right_future);
            Ok((
                left_result.context("while processing left iris shares")??,
                right_result.context("while processing right iris shares")??,
            ))
        });
    Ok(handle)
}

pub async fn process_identity_deletions(
    batch: &BatchQuery,
    store: &Store,
    dummy_iris_share: &GaloisRingIrisCodeShare,
    dummy_mask_share: &GaloisRingTrimmedMaskCodeShare,
) -> Result<()> {
    if batch.deletion_requests_indices.is_empty() {
        return Ok(());
    }

    for (&entry_idx, tracing_payload) in batch
        .deletion_requests_indices
        .iter()
        .zip(batch.deletion_requests_metadata.iter())
    {
        let serial_id = entry_idx + 1; // DB serial_id is 1-indexed
        tracing::info!(
            node_id = tracing_payload.node_id,
            dd.trace_id = tracing_payload.trace_id,
            dd.span_id = tracing_payload.span_id,
            "Started processing deletion request",
        );

        // overwrite postgres db with dummy values.
        // note that both serial_id and postgres db are 1-indexed.
        store
            .update_iris(
                None,
                serial_id as i64,
                dummy_iris_share,
                dummy_mask_share,
                dummy_iris_share,
                dummy_mask_share,
            )
            .await?;

        tracing::info!(
            node_id = tracing_payload.node_id,
            dd.trace_id = tracing_payload.trace_id,
            dd.span_id = tracing_payload.span_id,
            "Deleted identity with serial id {}",
            serial_id,
        );
    }

    Ok(())
}
