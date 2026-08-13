mod multipart;
mod streaming;
mod streaming_download;
use std::{
    io::Cursor,
    time::{Duration, Instant},
};

use aws_config::SdkConfig;
use aws_sdk_s3::config::{
    retry::RetryConfig, timeout::TimeoutConfig, Builder as S3ConfigBuilder, Region,
};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use bytes::Bytes;
use eyre::{bail, eyre, Result};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hnsw::{
        graph::{
            graph_store::{self, GraphCheckpointRow, GraphPg},
            layered_graph::GraphMem,
        },
        VectorStore,
    },
    utils::serialization::graph::{GraphFormat, LegacyPruneContext, PruneReport},
};

use crate::graph_checkpoint::data::*;
use iris_mpc_common::SerialId;
pub use multipart::*;
pub use streaming::*;
pub use streaming_download::*;

/// Retry budget applied to every S3 client built by [`create_s3_client`].
const S3_MAX_ATTEMPTS: u32 = 5;
/// Per-attempt timeout: a hung request is abandoned so the retry can fire,
/// instead of stalling the whole run behind one dead connection.
const S3_OPERATION_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(120);

/// Build an S3 client with the shared retry + per-attempt-timeout policy.
///
/// The single place genesis and hawk-main construct S3 clients, so the retry
/// budget and attempt timeout stay identical across both (the sidecar builds
/// its own plain client).
/// `region_override` selects a bucket region that differs from `sdk_config`'s
/// (the graph-checkpoint bucket); pass `None` to inherit the SDK region.
pub fn create_s3_client(
    sdk_config: &SdkConfig,
    force_path_style: bool,
    region_override: Option<Region>,
) -> S3Client {
    let mut builder = S3ConfigBuilder::from(sdk_config)
        .force_path_style(force_path_style)
        .retry_config(RetryConfig::standard().with_max_attempts(S3_MAX_ATTEMPTS))
        .timeout_config(
            TimeoutConfig::builder()
                .operation_attempt_timeout(S3_OPERATION_ATTEMPT_TIMEOUT)
                .build(),
        );
    if let Some(region) = region_override {
        builder = builder.region(region);
    }
    S3Client::from_conf(builder.build())
}

/// Exponential backoff with full jitter for the in-client range/part retry
/// loops. `attempt` is 1-based; the delay doubles from 500 ms, caps at 30 s,
/// and jitters uniformly in `[0, delay]` to spread reconnect storms.
pub(super) fn retry_backoff(attempt: u32) -> Duration {
    const BASE_MS: u64 = 500;
    const CAP: Duration = Duration::from_secs(30);
    let exp = BASE_MS.saturating_mul(1u64 << (attempt.saturating_sub(1)).min(6));
    let capped = Duration::from_millis(exp).min(CAP);
    let jitter = rand::random::<f64>() * capped.as_secs_f64();
    Duration::from_secs_f64(jitter)
}

/// Creates an S3 graph checkpoint.
#[allow(clippy::too_many_arguments)]
pub async fn upload_graph_checkpoint(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphRef>,
    s3_client: &S3Client,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
    is_archival: bool,
) -> Result<GraphCheckpointState> {
    let start = Instant::now();
    tracing::info!(
        "Creating S3 graph checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}, is_archival={}",
        last_indexed_iris_id,
        last_indexed_modification_id,
        is_archival,
    );

    let left_graph = graph_mem[LEFT].read().await;
    let right_graph = graph_mem[RIGHT].read().await;
    let data = Bytes::from(bincode::serialize(&[&*left_graph, &*right_graph])?);
    _upload_graph_checkpoint(
        bucket,
        party_id,
        s3_client,
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        is_archival,
        data,
        "genesis",
        "graph",
        start,
    )
    .await
}

/// Creates an S3 graph checkpoint from plaintext graphs (for testing).
#[allow(clippy::too_many_arguments)]
pub async fn upload_graph_checkpoint_plaintext(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphMem>,
    s3_client: &S3Client,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
    is_archival: bool,
) -> Result<GraphCheckpointState> {
    let start = Instant::now();
    tracing::info!(
        "Creating S3 plaintext graph checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}, is_archival={}",
        last_indexed_iris_id,
        last_indexed_modification_id,
        is_archival,
    );

    let data = Bytes::from(bincode::serialize(graph_mem)?);
    _upload_graph_checkpoint(
        bucket,
        party_id,
        s3_client,
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        is_archival,
        data,
        "plaintext",
        "plaintext graph",
        start,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn _upload_graph_checkpoint(
    bucket: &str,
    party_id: usize,
    s3_client: &S3Client,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
    is_archival: bool,
    data: Bytes,
    s3_prefix: &str,
    log_label: &str,
    start: Instant,
) -> Result<GraphCheckpointState> {
    let data_len = data.len();
    tracing::info!(
        "Serialized graphs to {} bytes in {:?}",
        data_len,
        start.elapsed()
    );

    // Compute BLAKE3 hash before upload
    let hash_start = Instant::now();
    let blake3_hash = blake3::hash(&data).to_hex().to_string();
    tracing::info!(
        "Computed BLAKE3 hash: {} in {:?}",
        blake3_hash,
        hash_start.elapsed()
    );

    let s3_key = format!(
        "{}/{}/checkpoint_{}.bin",
        s3_prefix,
        party_id,
        uuid::Uuid::new_v4()
    );

    upload_graph(s3_client, bucket, &s3_key, data).await?;

    let checkpoint = GraphCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        blake3_hash,
        graph_version: GraphFormat::Current.version(),
        is_archival,
    };

    tracing::info!(
        "S3 {} checkpoint created successfully: s3_key={}, is_archival={}, duration={:?}",
        log_label,
        s3_key,
        is_archival,
        start.elapsed()
    );

    metrics::histogram!("graph_checkpoint_upload_duration").record(start.elapsed().as_secs_f64());
    metrics::gauge!("graph_checkpoint_size_bytes").set(data_len as f64);
    metrics::gauge!("graph_checkpoint_last_indexed_id").set(last_indexed_iris_id as f64);
    metrics::gauge!("graph_checkpoint_last_modification_id")
        .set(last_indexed_modification_id as f64);

    Ok(checkpoint)
}

pub async fn download_graph_checkpoint(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
    prune: Option<LegacyPruneContext>,
) -> Result<BothEyes<GraphMem>> {
    let (graphs, _reports) =
        download_graph_checkpoint_pruned(s3_client, bucket, state, prune).await?;
    Ok(graphs)
}

/// Like [`download_graph_checkpoint`], also returning the per-eye
/// [`PruneReport`] when a legacy (V3/V4) base was pruned at read.
pub async fn download_graph_checkpoint_pruned(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
    prune: Option<LegacyPruneContext>,
) -> Result<(BothEyes<GraphMem>, Option<[PruneReport; 2]>)> {
    let format = GraphFormat::try_from(state.graph_version)?;
    if format == GraphFormat::Raw {
        bail!("Unexpected graph checkpoint format: Raw");
    }
    // V2 bakes versions into edges (like V3/V4) but has no v5 prune path, so a
    // serial-only conversion would silently keep version-stale edges. Reject it
    // outright — there is no supported V2 base migration.
    if format == GraphFormat::V2 {
        bail!("refusing V2 checkpoint: version-baked edges with no v5 migration path");
    }
    // Only genesis may consume a legacy V3/V4 base: it supplies a prune context
    // and rewrites the result as V5. Every other consumer (hawk restart,
    // diagnostics) must load a native V5 checkpoint, since v5 serial-only edges
    // can't reproduce the version-skip an unpruned legacy base would need.
    if prune.is_none() && matches!(format, GraphFormat::V3 | GraphFormat::V4) {
        bail!(
            "refusing to load unmigrated legacy {format:?} checkpoint; only genesis \
             may consume V3/V4 (it prunes and rewrites as V5)"
        );
    }
    let start = Instant::now();
    let (graphs, reports, hash_bytes) = stream_download_and_deserialize_graph_pair_with(
        s3_client,
        bucket,
        &state.s3_key,
        format,
        prune,
        DEFAULT_DOWNLOAD_PIPE_CAPACITY,
        DEFAULT_DOWNLOAD_RANGE_SIZE,
        DEFAULT_DOWNLOAD_PARALLELISM,
    )
    .await?;
    metrics::histogram!("genesis_checkpoint_download_duration")
        .record(start.elapsed().as_secs_f64());

    // Verify BLAKE3 hash after download
    let computed_hash = blake3::Hash::from_bytes(hash_bytes).to_hex().to_string();
    if computed_hash != state.blake3_hash {
        return Err(eyre!(
            "BLAKE3 hash mismatch: expected {}, got {}",
            state.blake3_hash,
            computed_hash
        ));
    }
    tracing::info!("BLAKE3 hash verified successfully: {}", computed_hash);
    Ok((graphs, reports))
}

// this is used for the genesis integration tests.
// it does not convert between graph types
pub async fn download_genesis_checkpoint_plaintext(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<BothEyes<GraphMem>> {
    if state.graph_version != GraphFormat::Current.version() {
        bail!("unexpected graph version: {}", state.graph_version);
    }
    let binary_graph = download_and_hash(s3_client, bucket, state).await?;
    let mut cursor = Cursor::new(&binary_graph);
    let graphs: BothEyes<GraphMem> = bincode::deserialize_from(&mut cursor)?;
    Ok(graphs)
}

/// Gets the latest checkpoint from genesis_graph_checkpoint table.
pub async fn get_latest_checkpoint_state<V: VectorStore>(
    graph_store: &GraphPg<V>,
) -> Result<Option<GraphCheckpointState>> {
    tracing::info!("Retrieving latest graph checkpoint metadata from genesis_graph_checkpoint");

    let row = graph_store.get_latest_genesis_graph_checkpoint().await?;
    let metadata: Option<GraphCheckpointState> = row.map(|row| row.try_into()).transpose()?;

    if let Some(m) = &metadata {
        tracing::info!(
            "Found existing checkpoint: s3_key={}, last_indexed_iris_id={}, is_archival={}",
            m.s3_key,
            m.last_indexed_iris_id,
            m.is_archival,
        );
    } else {
        tracing::info!("No existing checkpoint found");
    }

    Ok(metadata)
}

/// stores checkpoint in genesis_graph_checkpoint table.
pub async fn save_checkpoint_state<V: VectorStore>(
    tx: graph_store::GraphTx<'_, V>,
    state: &GraphCheckpointState,
) -> Result<()> {
    tracing::info!(
        "Persisting graph checkpoint state: s3_key={}, last_indexed_iris_id={}, is_archival={}",
        state.s3_key,
        state.last_indexed_iris_id,
        state.is_archival,
    );

    let mut tx = tx.tx;
    GraphPg::<V>::insert_genesis_graph_checkpoint(
        &mut tx,
        &state.s3_key,
        i64::from(state.last_indexed_iris_id),
        state.last_indexed_modification_id,
        state.graph_mutation_id,
        &state.blake3_hash,
        state.is_archival,
        state.graph_version,
    )
    .await
    .map_err(|e| eyre!("Failed to persist checkpoint state: {:?}", e))?;
    tx.commit().await?;
    Ok(())
}

/// DANGER: requires that the 3 parties have
/// reached consensus on a common checkpoint.
/// if the parties have not reached agreement, then
/// calling this function could make rollback impossible
///
/// `retain_from_id`: local row-id watermark; rows with `id >= retain_from_id`
/// are kept regardless of pruning mode. Callers pruning right after recording
/// a checkpoint that peers have NOT yet confirmed durable (the sidecar's
/// `UploadAndRecord`) must pass the cycle's agreed base id here: the base was
/// advertised by every party in Phase 1, so retaining it guarantees a common
/// checkpoint survives even if a peer crashes before recording the new one.
/// `None` is only safe when `current_state` itself is known to be durably
/// recorded by all parties (the startup-agreed common checkpoint).
pub async fn cleanup_checkpoints<V: VectorStore>(
    bucket: &str,
    s3_client: &S3Client,
    current_state: &GraphCheckpointState,
    retain_from_id: Option<i64>,
    graph_store: &GraphPg<V>,
    pruning_mode: PruningMode,
    tiered_pruning: TieredPruningConfig,
) -> Result<()> {
    tracing::info!(
        "cleaning up old genesis graph checkpoints (mode: {})",
        pruning_mode
    );

    if pruning_mode == PruningMode::None {
        tracing::info!("pruning mode is 'none', skipping cleanup");
        return Ok(());
    }
    if pruning_mode == PruningMode::Tiered {
        tiered_pruning.validate()?;
    }

    // Rank ages over the *full* history (live rows + soft-deleted tombstones),
    // newest-first.
    let all_checkpoints = graph_store
        .get_genesis_graph_checkpoints_including_deleted()
        .await?;
    let current_checkpoint_date = all_checkpoints
        .iter()
        .find(|x| x.s3_key == current_state.s3_key)
        .ok_or_else(|| eyre!("current checkpoint not found in the db"))?
        .created_at;
    let checkpoints = checkpoints_to_prune(
        &all_checkpoints,
        &current_state.s3_key,
        retain_from_id,
        pruning_mode,
        &tiered_pruning,
        current_checkpoint_date,
    )?;
    for checkpoint in checkpoints {
        delete_graph(s3_client, bucket, &checkpoint.s3_key).await?;
        // Soft-delete the row (tombstone for audit)
        graph_store.delete_genesis_checkpoint(checkpoint.id).await?;
    }
    Ok(())
}

/// Deletes the WAL rows that precede the oldest live checkpoint, and returns the
/// number of rows removed.
pub async fn prune_wal_before_earliest_checkpoint<V: VectorStore>(
    graph_store: &GraphPg<V>,
) -> Result<u64> {
    let checkpoints = graph_store.get_genesis_graph_checkpoints().await?;

    // Ordered newest-first, so the oldest survivor is last.
    let Some(earliest) = checkpoints.last() else {
        // With no checkpoint at all the WAL is the only record of the graph.
        tracing::info!("no live checkpoint; retaining the whole WAL");
        return Ok(0);
    };

    // Pre-WAL checkpoints (genesis-era rows) carry no height, so there is no
    // way to say which mutations they already contain.
    let Some(cutoff) = earliest.graph_mutation_id else {
        tracing::info!(
            "oldest live checkpoint {} has no graph_mutation_id; retaining the whole WAL",
            earliest.id
        );
        return Ok(0);
    };

    let mut tx = graph_store.begin_tx().await?;
    let deleted = graph_store
        .delete_hawk_graph_mutations(&mut tx, Some(cutoff - 1))
        .await?;
    tx.commit().await?;

    tracing::info!(
        "pruned {} WAL rows below modification_id {} (oldest live checkpoint {})",
        deleted,
        cutoff,
        earliest.id,
    );
    Ok(deleted)
}

/// Selects which checkpoints to prune this run, given the full history
/// (`all_checkpoints`) ordered newest-first and *including* soft-deleted
/// tombstones.
///
/// The enumeration index over the full history is the checkpoint's recency rank
/// (0 = newest). Ranking over the full history — rather than over live rows
/// only — is what makes pruning idempotent: a survivor keeps its rank even after
/// earlier rows are tombstoned, so re-running over the same (or a moving) range
/// never re-classifies and deletes a checkpoint a prior run kept.
///
/// Rows are excluded from the result when they are:
/// - the current checkpoint (never deleted),
/// - at/above the `retain_from_id` watermark, or
/// - kept by the [`PruningMode`] / [`TieredPruningConfig`] policy.
fn checkpoints_to_prune<'a>(
    all_checkpoints: &'a [GraphCheckpointRow],
    current_s3_key: &str,
    retain_from_id: Option<i64>,
    pruning_mode: PruningMode,
    tiered_pruning: &TieredPruningConfig,
    current_checkpoint_date: chrono::DateTime<chrono::Utc>,
) -> Result<Vec<&'a GraphCheckpointRow>> {
    // Validate all_checkpoints are sorted by id descending
    for i in 0..all_checkpoints.len() - 1 {
        if all_checkpoints[i].id <= all_checkpoints[i + 1].id {
            return Err(eyre!(
                "checkpoints are not sorted by id descending: {} <= {}",
                all_checkpoints[i].id,
                all_checkpoints[i + 1].id
            ));
        }
    }
    // Total history length (live + tombstoned). Used to calculate the version age.
    let total_checkpoints = all_checkpoints.len();
    // Sort by id descending to iterate from newest to oldest
    let checkpoints = all_checkpoints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.s3_key != current_s3_key)
        .filter(|(_, c)| retain_from_id.is_none_or(|min_id| c.id < min_id))
        .filter(|(recency_rank, c)| {
            should_delete_checkpoint(
                pruning_mode,
                total_checkpoints,
                *recency_rank,
                c,
                current_checkpoint_date,
                tiered_pruning,
            )
        })
        .map(|(_, c)| c)
        .collect::<Vec<&GraphCheckpointRow>>();
    Ok(checkpoints)
}

/// Decides whether a single checkpoint should be pruned under the given
/// [`PruningMode`].
fn should_delete_checkpoint(
    pruning_mode: PruningMode,
    total_checkpoints: usize,
    recency_rank: usize,
    c: &GraphCheckpointRow,
    current_checkpoint_date: chrono::DateTime<chrono::Utc>,
    tiered: &TieredPruningConfig,
) -> bool {
    if c.is_deleted {
        return false;
    }
    match pruning_mode {
        PruningMode::None => false,
        PruningMode::AllOlder => true,
        PruningMode::OlderNonArchival => !c.is_archival,
        PruningMode::Tiered => {
            if c.is_archival {
                return false;
            }

            // Recent tier: always keep the `keep_recent_count` newest checkpoints.
            if recency_rank < tiered.keep_recent_count {
                return false;
            }

            let checkpoint_age_days = (current_checkpoint_date - c.created_at).num_days();
            if checkpoint_age_days >= tiered.delete_older_than_days as i64 {
                // Ancient tier: delete everything.
                return true;
            }
            // Sequence rank is the index into the (oldest-first) slice (0 = oldest);
            // Sparse tier: keep one out of every `keep_every_nth`.
            let sequence_rank = total_checkpoints - 1 - recency_rank;
            !sequence_rank.is_multiple_of(tiered.keep_every_nth)
        }
    }
}

/// Verifies that the S3 client has read, write, and delete access to the
/// checkpoint bucket. Uploads a small sentinel object, reads it back, and
/// deletes it. This catches misconfigured buckets/regions/IAM before any
/// mutations occur.
pub async fn verify_s3_checkpoint_access(
    s3_client: &S3Client,
    bucket: &str,
    party_id: usize,
) -> Result<()> {
    let key = format!("genesis/{party_id}/_access_check");
    let body = b"access_check";

    // Write
    s3_client
        .put_object()
        .bucket(bucket)
        .key(&key)
        .body(ByteStream::from_static(body))
        .send()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket write check failed: {e}"))?;

    // Read
    let resp = s3_client
        .get_object()
        .bucket(bucket)
        .key(&key)
        .send()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket read check failed: {e}"))?;
    let data = resp
        .body
        .collect()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket read check failed to collect body: {e}"))?;
    if data.into_bytes().as_ref() != body {
        bail!("S3 checkpoint bucket read check returned unexpected content");
    }

    // Delete
    if let Err(e) = s3_client
        .delete_object()
        .bucket(bucket)
        .key(&key)
        .send()
        .await
    {
        tracing::warn!("S3 checkpoint bucket delete check failed: {e}");
    }

    Ok(())
}

/// Checks whether a given key exists in an S3 bucket using a `HeadObject`
/// request. Returns `true` if the key is present, `false` if it does not
/// exist (HTTP 404 / `NoSuchKey`), or an error for any other failure.
///
/// # Arguments
///
/// * `s3_client` - Authenticated S3 client.
/// * `bucket`    - Name of the S3 bucket to query.
/// * `key`       - Object key to check for existence.
pub async fn s3_key_exists(s3_client: &S3Client, bucket: &str, key: &str) -> Result<bool> {
    match s3_client.head_object().bucket(bucket).key(key).send().await {
        Ok(_) => Ok(true),
        Err(e) => {
            // `head_object` returns a 404 when the key does not exist.
            // The SDK surfaces this as a `NotFound` service error.
            if e.as_service_error()
                .map(|se| se.is_not_found())
                .unwrap_or(false)
            {
                Ok(false)
            } else {
                Err(eyre!("S3 head_object failed for s3://{bucket}/{key}: {e}"))
            }
        }
    }
}

async fn download_and_hash(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<Bytes> {
    let start = Instant::now();

    let binary_graph = download_graph(s3_client, bucket, &state.s3_key).await?;

    metrics::histogram!("genesis_checkpoint_download_duration")
        .record(start.elapsed().as_secs_f64());

    // Verify BLAKE3 hash after download
    let computed_hash = blake3::hash(&binary_graph).to_hex().to_string();
    if computed_hash != state.blake3_hash {
        return Err(eyre!(
            "BLAKE3 hash mismatch: expected {}, got {}",
            state.blake3_hash,
            computed_hash
        ));
    }
    tracing::info!("BLAKE3 hash verified successfully: {}", computed_hash);
    Ok(binary_graph)
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod wal_prune_tests {
    use super::*;
    use crate::hawkers::plaintext_store::PlaintextStore;
    use crate::hnsw::graph::graph_store::test_utils::TestGraphPg;

    type Store = GraphPg<PlaintextStore>;

    /// Writes one WAL row per `modification_id` in `1..=n`.
    async fn seed_wal(store: &Store, n: i64) -> Result<()> {
        let mut tx = store.begin_tx().await?;
        for id in 1..=n {
            store
                .upsert_hawk_graph_mutations(&mut tx, id, b"payload")
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Records a checkpoint that covers the WAL through `graph_mutation_id`.
    async fn seed_checkpoint(store: &Store, graph_mutation_id: Option<i64>) -> Result<()> {
        let height = graph_mutation_id.unwrap_or_default();
        let mut tx = store.begin_tx().await?;
        Store::insert_genesis_graph_checkpoint(
            &mut tx,
            &format!("cp/{height}"),
            height,
            height,
            graph_mutation_id,
            &format!("hash-{height}"),
            false,
            GraphFormat::Current.version(),
        )
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Surviving `modification_id`s, ascending.
    async fn surviving_wal(store: &Store) -> Result<Vec<i64>> {
        Ok(store
            .get_hawk_graph_mutations_after(None)
            .await?
            .into_iter()
            .map(|r| r.modification_id)
            .collect())
    }

    /// With no checkpoint the WAL is the only record of the graph, so none of it
    /// may be dropped.
    #[tokio::test]
    async fn keeps_whole_wal_without_a_checkpoint() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 5).await?;

        let deleted = prune_wal_before_earliest_checkpoint(&store.graph).await?;

        assert_eq!(deleted, 0);
        assert_eq!(surviving_wal(&store).await?, vec![1, 2, 3, 4, 5]);
        store.cleanup().await
    }

    /// The cut is taken at the *oldest* live checkpoint, so every retained
    /// checkpoint keeps the rows it needs to roll forward. The row at the
    /// checkpoint's own height stays as the WAL anchor.
    #[tokio::test]
    async fn cuts_at_oldest_live_checkpoint() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 10).await?;
        seed_checkpoint(&store, Some(3)).await?;
        seed_checkpoint(&store, Some(7)).await?;

        let deleted = prune_wal_before_earliest_checkpoint(&store.graph).await?;

        assert_eq!(deleted, 2);
        assert_eq!(surviving_wal(&store).await?, vec![3, 4, 5, 6, 7, 8, 9, 10]);
        store.cleanup().await
    }

    /// A tombstoned checkpoint's S3 object is gone, so it can never be replayed
    /// from and must not hold the WAL back.
    #[tokio::test]
    async fn ignores_tombstoned_checkpoints() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 10).await?;
        seed_checkpoint(&store, Some(3)).await?;
        seed_checkpoint(&store, Some(7)).await?;

        let oldest = store
            .get_genesis_graph_checkpoints()
            .await?
            .last()
            .expect("two checkpoints were just inserted")
            .id;
        store.delete_genesis_checkpoint(oldest).await?;

        let deleted = prune_wal_before_earliest_checkpoint(&store.graph).await?;

        assert_eq!(deleted, 6);
        assert_eq!(surviving_wal(&store).await?, vec![7, 8, 9, 10]);
        store.cleanup().await
    }

    /// A checkpoint with no recorded height gives no way to tell which mutations
    /// it already contains, so nothing is dropped.
    #[tokio::test]
    async fn keeps_whole_wal_when_oldest_checkpoint_has_no_height() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 5).await?;
        seed_checkpoint(&store, None).await?;
        seed_checkpoint(&store, Some(4)).await?;

        let deleted = prune_wal_before_earliest_checkpoint(&store.graph).await?;

        assert_eq!(deleted, 0);
        assert_eq!(surviving_wal(&store).await?, vec![1, 2, 3, 4, 5]);
        store.cleanup().await
    }

    /// Re-running against an unchanged checkpoint set deletes nothing more.
    #[tokio::test]
    async fn is_idempotent() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 10).await?;
        seed_checkpoint(&store, Some(6)).await?;

        assert_eq!(prune_wal_before_earliest_checkpoint(&store.graph).await?, 5);
        assert_eq!(prune_wal_before_earliest_checkpoint(&store.graph).await?, 0);
        assert_eq!(surviving_wal(&store).await?, vec![6, 7, 8, 9, 10]);
        store.cleanup().await
    }

    /// A checkpoint that covers the entire WAL leaves the anchor row behind, so
    /// `MAX(modification_id)` never falls below the checkpoint's height.
    #[tokio::test]
    async fn keeps_anchor_row_when_checkpoint_covers_whole_wal() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        seed_wal(&store, 4).await?;
        seed_checkpoint(&store, Some(4)).await?;

        let deleted = prune_wal_before_earliest_checkpoint(&store.graph).await?;

        assert_eq!(deleted, 3);
        assert_eq!(surviving_wal(&store).await?, vec![4]);
        assert_eq!(store.get_max_hawk_graph_mutation_id().await?, Some(4));
        store.cleanup().await
    }
}

#[cfg(test)]
mod tests {
    use super::{checkpoints_to_prune, GraphCheckpointRow, PruningMode, TieredPruningConfig};
    use chrono::{Duration, Utc};

    /// Builds a checkpoint row. `s3_key` is derived from `id` as `cp/{id}`, so
    /// tests can refer to the current checkpoint by key without a separate map.
    fn row(
        id: i64,
        is_archival: bool,
        is_deleted: bool,
        created_at: chrono::DateTime<chrono::Utc>,
    ) -> GraphCheckpointRow {
        GraphCheckpointRow {
            id,
            s3_key: format!("cp/{id}"),
            last_indexed_iris_id: id,
            last_indexed_modification_id: id,
            graph_mutation_id: Some(id),
            blake3_hash: "deadbeef".to_string(),
            graph_version: 1,
            is_archival,
            created_at,
            is_deleted,
        }
    }

    /// Sorted list of the ids selected for pruning (order-independent compare).
    fn pruned_ids(rows: &[&GraphCheckpointRow]) -> Vec<i64> {
        let mut ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        ids
    }

    /// `Tiered` keeps the `keep_recent_count` newest checkpoints (recent tier),
    /// thins older ones by `keep_every_nth` (sparse tier), and deletes everything
    /// older than `delete_older_than_days` (ancient tier).
    ///
    /// `version_age` is the index into the (oldest-first) slice (0 = oldest);
    /// `recency_rank = len - 1 - version_age` counts newest-first (0 = newest).
    #[test]
    fn test_checkpoints_to_prune_tiered_tiers() {
        let now = Utc::now();
        let days_ago = |d: i64| now - Duration::days(d);
        let cfg = TieredPruningConfig {
            delete_older_than_days: 60,
            keep_recent_count: 2,
            keep_every_nth: 4,
        };
        // newest-first (id DESC), matching the caller. version_age = len - 1 - slice index:
        // id 8 = v_age 8 ... id 1 = v_age 1; recency_rank = slice index.
        let all = vec![
            row(8, false, false, days_ago(0)), // current (newest)        -> excluded
            row(7, false, false, days_ago(10)), // recent (rank 1 < 2)     -> keep
            row(6, false, false, days_ago(20)), // recent (rank 2 < 2)     -> keep
            row(5, false, false, days_ago(30)), // sparse, v_age 2         -> delete
            row(4, false, false, days_ago(40)), // sparse, v_age 3         -> delete
            row(3, false, false, days_ago(50)), // sparse, v_age 4%4       -> keep
            row(2, false, false, days_ago(61)), // sparse, v_age 5         -> delete
            row(1, false, false, days_ago(70)), // ancient (rank 7)        -> delete
            row(0, false, false, days_ago(80)), // ancient (rank 8)        -> excluded
        ];
        let pruned = checkpoints_to_prune(&all, "cp/8", None, PruningMode::Tiered, &cfg, now);
        assert!(pruned.is_ok());
        assert_eq!(pruned_ids(&pruned.unwrap()), vec![0, 1, 2, 3, 5, 6]);
    }

    /// `Tiered` never deletes an archival row, even in the ancient tier.
    #[test]
    fn test_checkpoints_to_prune_tiered_keeps_archival() {
        let now = Utc::now();
        let cfg = TieredPruningConfig {
            delete_older_than_days: 60,
            keep_recent_count: 1,
            keep_every_nth: 4,
        };
        let all = vec![
            row(3, false, false, now),                       // current
            row(2, false, false, now - Duration::days(100)), // ancient           -> delete
            row(1, true, false, now - Duration::days(100)),  // ancient + archival -> keep
        ];
        let pruned = checkpoints_to_prune(&all, "cp/3", None, PruningMode::Tiered, &cfg, now);
        assert!(pruned.is_ok());
        assert_eq!(pruned_ids(&pruned.unwrap()), vec![2]);
    }

    #[test]
    fn test_blake3_hash_to_string() {
        let bytes = b"ABCDEFGHIJKLMNOP";
        let hash1 = blake3::hash(bytes);
        let hash_str1 = hash1.to_hex().to_string();

        // now go back
        let hash2 = blake3::Hash::from_hex(hash_str1.as_bytes()).unwrap();
        assert_eq!(hash1.as_bytes(), hash2.as_bytes());
    }

    #[test]
    fn test_pruning_mode_from_str() {
        use super::PruningMode;
        use std::str::FromStr;

        assert_eq!(PruningMode::from_str("none").unwrap(), PruningMode::None);
        assert_eq!(
            PruningMode::from_str("older-non-archival").unwrap(),
            PruningMode::OlderNonArchival
        );
        assert_eq!(
            PruningMode::from_str("all-older").unwrap(),
            PruningMode::AllOlder
        );
        assert!(PruningMode::from_str("invalid").is_err());
    }

    #[test]
    fn test_pruning_mode_display() {
        use super::PruningMode;

        assert_eq!(PruningMode::None.to_string(), "none");
        assert_eq!(
            PruningMode::OlderNonArchival.to_string(),
            "older-non-archival"
        );
        assert_eq!(PruningMode::AllOlder.to_string(), "all-older");
    }
}
