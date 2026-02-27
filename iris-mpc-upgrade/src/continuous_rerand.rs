use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use bytemuck::cast_slice;
use eyre::Result;
use futures::StreamExt;
use iris_mpc_store::rerand::{
    apply_staging_chunk, get_rerand_progress, insert_staging_irises, set_all_confirmed,
    set_staging_written, staging_schema_name, upsert_rerand_progress, StagingIrisEntry,
};
use iris_mpc_store::Store;
use sqlx::PgPool;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::config::RerandomizeContinuousConfig;
use crate::epoch;
use crate::rerandomization::randomize_iris;
use crate::s3_coordination::{self, Manifest};

/// Run the continuous rerandomization loop.
///
/// If `cancel` is provided, the loop checks for cancellation between chunk
/// stages and exits cleanly with `Ok(())` when cancelled. Pass `None` for
/// production use where the loop runs until the process is killed.
pub async fn run_continuous_rerand(
    config: &RerandomizeContinuousConfig,
    s3: &S3Client,
    sm: &SecretsManagerClient,
    store: &Store,
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    let pool = &store.pool;
    let staging_schema = staging_schema_name(&store.schema_name);
    let poll_interval = Duration::from_millis(config.s3_poll_interval_ms);
    let chunk_delay = Duration::from_secs(config.chunk_delay_secs);

    loop {
        if is_cancelled(cancel) {
            return Ok(());
        }

        let active_epoch = epoch::determine_active_epoch(s3, &config.s3_bucket).await?;
        tracing::info!("Active epoch: {}", active_epoch);

        let shared_secret = epoch::derive_shared_secret(
            sm,
            s3,
            &config.s3_bucket,
            &config.env,
            active_epoch,
            config.party_id,
            poll_interval,
        )
        .await?;

        let manifest =
            get_or_create_manifest(s3, store, config, active_epoch, poll_interval).await?;
        tracing::info!(
            "Epoch {} manifest: chunk_size={}, max_id_inclusive={}",
            active_epoch,
            manifest.chunk_size,
            manifest.max_id_inclusive
        );

        let mut chunk_id: u32 = 0;
        loop {
            if is_cancelled(cancel) {
                return Ok(());
            }

            if manifest.chunk_is_empty(chunk_id) {
                break;
            }

            let progress = get_rerand_progress(pool, active_epoch as i32, chunk_id as i32).await?;

            if progress.as_ref().is_some_and(|p| p.live_applied) {
                chunk_id += 1;
                continue;
            }

            upsert_rerand_progress(pool, active_epoch as i32, chunk_id as i32).await?;

            if !progress.as_ref().is_some_and(|p| p.staging_written) {
                process_chunk_staging(
                    pool,
                    store,
                    &staging_schema,
                    &shared_secret,
                    config.party_id,
                    active_epoch,
                    chunk_id,
                    &manifest,
                )
                .await?;

                set_staging_written(pool, active_epoch as i32, chunk_id as i32).await?;

                s3_coordination::upload_chunk_staged(
                    s3,
                    &config.s3_bucket,
                    active_epoch,
                    config.party_id,
                    chunk_id,
                )
                .await?;
                tracing::info!(
                    "Epoch {} chunk {}: staging written, S3 marker uploaded",
                    active_epoch,
                    chunk_id
                );
            }

            if is_cancelled(cancel) {
                return Ok(());
            }

            if !progress.as_ref().is_some_and(|p| p.all_confirmed) {
                s3_coordination::poll_chunk_staged_all(
                    s3,
                    &config.s3_bucket,
                    active_epoch,
                    chunk_id,
                    poll_interval,
                )
                .await?;

                set_all_confirmed(pool, active_epoch as i32, chunk_id as i32).await?;
                tracing::info!(
                    "Epoch {} chunk {}: all parties confirmed",
                    active_epoch,
                    chunk_id
                );
            }

            if is_cancelled(cancel) {
                return Ok(());
            }

            let rows =
                apply_staging_chunk(pool, &staging_schema, active_epoch as i32, chunk_id as i32)
                    .await?;
            tracing::info!(
                "Epoch {} chunk {}: applied to live DB ({} rows updated)",
                active_epoch,
                chunk_id,
                rows
            );

            chunk_id += 1;

            if chunk_delay > Duration::ZERO {
                sleep(chunk_delay).await;
            }
        }

        if chunk_id == 0 && chunk_delay > Duration::ZERO {
            tracing::info!(
                "Epoch {} is empty, sleeping to avoid spinning",
                active_epoch
            );
            sleep(chunk_delay).await;
        }

        epoch::complete_epoch(
            sm,
            s3,
            &config.s3_bucket,
            &config.env,
            active_epoch,
            config.party_id,
            poll_interval,
        )
        .await?;
        tracing::info!("Epoch {} completed, moving to next epoch", active_epoch);
    }
}

fn is_cancelled(cancel: Option<&CancellationToken>) -> bool {
    cancel.is_some_and(|c| c.is_cancelled())
}

async fn get_or_create_manifest(
    s3: &S3Client,
    store: &Store,
    config: &RerandomizeContinuousConfig,
    epoch: u32,
    poll_interval: Duration,
) -> Result<Manifest> {
    if s3_coordination::manifest_exists(s3, &config.s3_bucket, epoch).await? {
        return s3_coordination::download_manifest(s3, &config.s3_bucket, epoch, poll_interval)
            .await;
    }

    if config.party_id == 0 {
        let local_max = store.get_max_serial_id().await? as u64;
        s3_coordination::upload_max_id(s3, &config.s3_bucket, epoch, 0, local_max).await?;

        let all_max_ids =
            s3_coordination::download_all_max_ids(s3, &config.s3_bucket, epoch, poll_interval)
                .await?;
        let min_max = *all_max_ids.iter().min().unwrap();
        let max_id_inclusive = min_max.saturating_sub(config.safety_buffer_ids);

        let manifest = Manifest {
            epoch,
            chunk_size: config.chunk_size,
            max_id_inclusive,
        };
        s3_coordination::upload_manifest(s3, &config.s3_bucket, epoch, &manifest).await?;
        tracing::info!(
            "Epoch {}: manifest created (max_id_inclusive={}, chunk_size={})",
            epoch,
            max_id_inclusive,
            config.chunk_size
        );
        Ok(manifest)
    } else {
        let local_max = store.get_max_serial_id().await? as u64;
        s3_coordination::upload_max_id(s3, &config.s3_bucket, epoch, config.party_id, local_max)
            .await?;

        s3_coordination::download_manifest(s3, &config.s3_bucket, epoch, poll_interval).await
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_chunk_staging(
    pool: &PgPool,
    store: &Store,
    staging_schema: &str,
    shared_secret: &[u8; 32],
    party_id: u8,
    epoch: u32,
    chunk_id: u32,
    manifest: &Manifest,
) -> Result<()> {
    let (start, end) = manifest.chunk_range(chunk_id);

    const BATCH_SIZE: usize = 500;

    let mut stream = store.stream_irises_in_range(start..end);
    let mut batch: Vec<StagingIrisEntry> = Vec::with_capacity(BATCH_SIZE);

    while let Some(iris) = stream.next().await.transpose()? {
        let version_id = iris.version_id();
        let iris_id = iris.id();
        let (_, lc, lm, rc, rm) = randomize_iris(iris, shared_secret, party_id as usize);

        batch.push(StagingIrisEntry {
            epoch: epoch as i32,
            id: iris_id,
            chunk_id: chunk_id as i32,
            left_code: cast_slice::<u16, u8>(&lc.coefs).to_vec(),
            left_mask: cast_slice::<u16, u8>(&lm.coefs).to_vec(),
            right_code: cast_slice::<u16, u8>(&rc.coefs).to_vec(),
            right_mask: cast_slice::<u16, u8>(&rm.coefs).to_vec(),
            original_version_id: version_id,
            rerand_epoch: (epoch + 1) as i32,
        });

        if batch.len() >= BATCH_SIZE {
            insert_staging_irises(pool, staging_schema, &batch).await?;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        insert_staging_irises(pool, staging_schema, &batch).await?;
    }

    Ok(())
}
