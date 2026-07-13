use ampc_server_utils::ShutdownHandler;
use async_trait::async_trait;
use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::config::StalledStreamProtectionConfig;
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_s3::{primitives::ByteStream, Client};
use eyre::{bail, ensure, eyre, Result, WrapErr};
use futures::{stream, StreamExt};
use iris_mpc_common::{VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, mem, sync::Arc, time::Duration};
use tokio::{io::AsyncReadExt, sync::mpsc::Sender};

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>()
    + mem::size_of::<u16>(); // 75 KB

const SEMANTIC_ID_SIZE: usize = 16;
const SAFE_ELEMENT_SIZE: usize = SINGLE_ELEMENT_SIZE + mem::size_of::<u32>() + SEMANTIC_ID_SIZE;
const SAFE_SNAPSHOT_FORMAT: u32 = 3;

const MAX_RANGE_SIZE: usize = 200; // Download chunks in sub-chunks of 200 elements = 15 MB
const MAX_MANIFEST_SIZE: usize = 16 * 1024 * 1024;

pub struct S3StoredIris {
    #[allow(dead_code)]
    pub(crate) id: i64,
    pub(crate) left_code_even: Vec<u8>,
    pub(crate) left_code_odd: Vec<u8>,
    pub(crate) left_mask_even: Vec<u8>,
    pub(crate) left_mask_odd: Vec<u8>,
    pub(crate) right_code_even: Vec<u8>,
    pub(crate) right_code_odd: Vec<u8>,
    pub(crate) right_mask_even: Vec<u8>,
    pub(crate) right_mask_odd: Vec<u8>,
    pub(crate) version_id: i16,
    pub(crate) rerand_epoch: i32,
    pub(crate) semantic_id: Option<[u8; SEMANTIC_ID_SIZE]>,
}

impl S3StoredIris {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, eyre::Error> {
        let mut cursor = 0;

        // Helper closure to extract a slice of a given size
        let extract_slice =
            |bytes: &[u8], cursor: &mut usize, size: usize| -> Result<Vec<u8>, eyre::Error> {
                if *cursor + size > bytes.len() {
                    bail!("Exceeded total bytes while extracting slice",);
                }
                let slice = &bytes[*cursor..*cursor + size];
                *cursor += size;
                Ok(slice.to_vec())
            };

        // Parse `id` (i64)
        let id_bytes = extract_slice(bytes, &mut cursor, 4)?;
        let id = u32::from_be_bytes(
            id_bytes
                .try_into()
                .map_err(|_| eyre!("Failed to convert id bytes to i64"))?,
        ) as i64;

        // parse codes and masks for each limb separately
        let left_code_odd = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
        let left_code_even = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
        let left_mask_odd = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
        let left_mask_even = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
        let right_code_odd = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
        let right_code_even = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
        let right_mask_odd = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
        let right_mask_even = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;

        // Parse `version_id` (i16)
        let version_id_bytes = extract_slice(bytes, &mut cursor, 2)?;
        let version_id = u16::from_be_bytes(
            version_id_bytes
                .try_into()
                .map_err(|_| eyre!("Failed to convert version id bytes to i16"))?,
        ) as i16;

        let (rerand_epoch, semantic_id) = if cursor == bytes.len() {
            // Legacy non-rerandomized S3 chunks have no epoch or semantic ID.
            (0, None)
        } else {
            let epoch_bytes = extract_slice(bytes, &mut cursor, 4)?;
            let rerand_epoch = i32::try_from(u32::from_be_bytes(
                epoch_bytes
                    .try_into()
                    .map_err(|_| eyre!("Failed to convert rerand epoch bytes to u32"))?,
            ))
            .wrap_err("rerandomization epoch exceeds PostgreSQL INTEGER")?;
            let semantic_id = if cursor == bytes.len() {
                // Accepted only for direct compatibility with the old parser;
                // safe manifests require v3's exact record size below.
                None
            } else {
                let semantic_id = extract_slice(bytes, &mut cursor, SEMANTIC_ID_SIZE)?;
                Some(
                    semantic_id
                        .try_into()
                        .map_err(|_| eyre!("S3 semantic id must be exactly 16 bytes"))?,
                )
            };
            (rerand_epoch, semantic_id)
        };
        ensure!(cursor == bytes.len(), "Unexpected trailing S3 iris bytes");

        Ok(S3StoredIris {
            id,
            left_code_even,
            left_code_odd,
            left_mask_even,
            left_mask_odd,
            right_code_even,
            right_code_odd,
            right_mask_even,
            right_mask_odd,
            version_id,
            rerand_epoch,
            semantic_id,
        })
    }

    pub fn serial_id(&self) -> usize {
        self.id as usize
    }

    pub fn version_id(&self) -> i16 {
        self.version_id
    }

    pub fn rerand_epoch(&self) -> i32 {
        self.rerand_epoch
    }

    pub fn semantic_id(&self) -> Option<[u8; SEMANTIC_ID_SIZE]> {
        self.semantic_id
    }

    pub fn vector_id(&self) -> VectorId {
        VectorId::new(self.id as u32, self.version_id)
    }

    pub fn left_code_odd(&self) -> &Vec<u8> {
        &self.left_code_odd
    }

    pub fn left_code_even(&self) -> &Vec<u8> {
        &self.left_code_even
    }

    pub fn left_mask_odd(&self) -> &Vec<u8> {
        &self.left_mask_odd
    }

    pub fn left_mask_even(&self) -> &Vec<u8> {
        &self.left_mask_even
    }

    pub fn right_code_odd(&self) -> &Vec<u8> {
        &self.right_code_odd
    }

    pub fn right_code_even(&self) -> &Vec<u8> {
        &self.right_code_even
    }

    pub fn right_mask_odd(&self) -> &Vec<u8> {
        &self.right_mask_odd
    }

    pub fn right_mask_even(&self) -> &Vec<u8> {
        &self.right_mask_even
    }

    pub fn id(&self) -> i64 {
        self.id
    }
}

/// Creates an S3 client specifically for database chunks with additional
/// configuration
pub fn create_db_chunks_s3_client(
    shared_config: &aws_config::SdkConfig,
    force_path_style: bool,
) -> S3Client {
    let retry_config = RetryConfig::standard().with_max_attempts(5);

    // Increase S3 connect timeouts to 10s
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(10))
        .build();

    let db_chunks_s3_config = S3ConfigBuilder::from(shared_config)
        // disable stalled stream protection to avoid panics during s3 import
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
        .retry_config(retry_config)
        .timeout_config(timeout_config)
        .force_path_style(force_path_style)
        .build();

    S3Client::from_conf(db_chunks_s3_config)
}

#[async_trait]
pub trait ObjectStore: Send + Sync + 'static {
    async fn get_object(&self, key: &str, range: (usize, usize)) -> Result<ByteStream>;
    async fn list_objects(&self, prefix: &str) -> Result<Vec<String>>;
}

pub struct S3Store {
    client: Client,
    bucket: String,
}

impl S3Store {
    pub fn new(client: Client, bucket: String) -> Self {
        Self { client, bucket }
    }
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn get_object(&self, key: &str, range: (usize, usize)) -> Result<ByteStream> {
        let res = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .range(format!("bytes={}-{}", range.0, range.1 - 1))
            .send()
            .await?;

        Ok(res.body)
    }

    async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
        let mut objects = Vec::new();
        let mut continuation_token = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;

            objects.extend(
                response
                    .contents()
                    .iter()
                    .filter_map(|obj| obj.key().map(String::from)),
            );

            match response.next_continuation_token() {
                Some(token) => continuation_token = Some(token.to_string()),
                None => break,
            }
        }

        Ok(objects)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafeSnapshotChunk {
    pub first_id: usize,
    pub last_id: usize,
    pub key: String,
    pub sha256: String,
    pub size_bytes: usize,
    pub record_count: usize,
    pub epochs: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafeSnapshotManifest {
    pub format_version: u32,
    pub store_id: String,
    pub row_count: usize,
    pub record_size: usize,
    pub epochs: Vec<u32>,
    pub chunks: Vec<SafeSnapshotChunk>,
}

impl SafeSnapshotManifest {
    fn validate_epochs(epochs: &[u32]) -> Result<()> {
        ensure!(!epochs.is_empty(), "snapshot epoch inventory is empty");
        ensure!(
            epochs.windows(2).all(|pair| pair[0] < pair[1]),
            "snapshot epochs must be sorted and unique"
        );
        ensure!(
            epochs.iter().all(|epoch| *epoch <= i32::MAX as u32),
            "snapshot epoch exceeds PostgreSQL INTEGER"
        );
        Ok(())
    }

    fn validate(&self, prefix: &str, expected_store_id: &str) -> Result<()> {
        ensure!(
            self.format_version == SAFE_SNAPSHOT_FORMAT,
            "unsupported safe snapshot format {}",
            self.format_version
        );
        ensure!(
            !expected_store_id.is_empty() && self.store_id == expected_store_id,
            "snapshot store identity {:?} does not match expected {:?}",
            self.store_id,
            expected_store_id
        );
        ensure!(self.row_count > 0, "snapshot row count must be positive");
        ensure!(
            self.record_size == SAFE_ELEMENT_SIZE,
            "snapshot record size {} does not match compiled size {SAFE_ELEMENT_SIZE}",
            self.record_size
        );
        ensure!(!self.chunks.is_empty(), "snapshot has no chunks");
        Self::validate_epochs(&self.epochs)?;

        let data_prefix = format!("{prefix}/snapshots/data/");
        let mut next_id = 1usize;
        let mut keys = HashSet::new();
        let mut all_epochs = HashSet::new();
        for chunk in &self.chunks {
            Self::validate_epochs(&chunk.epochs)?;
            all_epochs.extend(chunk.epochs.iter().copied());
            ensure!(
                chunk.first_id == next_id,
                "snapshot inventory is not contiguous: expected id {next_id}, got {}",
                chunk.first_id
            );
            ensure!(
                chunk.last_id >= chunk.first_id,
                "snapshot chunk has an invalid id range"
            );
            let record_count = chunk
                .last_id
                .checked_sub(chunk.first_id)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| eyre!("snapshot chunk id range overflow"))?;
            ensure!(
                chunk.record_count == record_count,
                "snapshot chunk record count does not match its id range"
            );
            let expected_size = record_count
                .checked_mul(SAFE_ELEMENT_SIZE)
                .ok_or_else(|| eyre!("snapshot chunk size overflow"))?;
            ensure!(
                chunk.size_bytes == expected_size,
                "snapshot chunk size does not match its id range"
            );
            ensure!(
                chunk.key.starts_with(&data_prefix),
                "snapshot chunk key is outside the snapshot data prefix"
            );
            ensure!(keys.insert(&chunk.key), "snapshot repeats a chunk key");
            ensure!(
                chunk.sha256.len() == 64
                    && chunk
                        .sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "snapshot chunk has an invalid SHA-256 digest"
            );
            next_id = chunk
                .last_id
                .checked_add(1)
                .ok_or_else(|| eyre!("snapshot row id overflow"))?;
        }
        ensure!(
            next_id == self.row_count + 1,
            "snapshot chunks cover {} rows but manifest declares {}",
            next_id - 1,
            self.row_count
        );
        let mut all_epochs: Vec<_> = all_epochs.into_iter().collect();
        all_epochs.sort_unstable();
        ensure!(
            all_epochs == self.epochs,
            "snapshot epoch inventory does not match its chunks"
        );
        Ok(())
    }
}

#[derive(Debug)]
struct CompletionMarker {
    timestamp: u64,
    digest: String,
    manifest_size: usize,
}

impl CompletionMarker {
    fn parse(key: &str) -> Result<Self> {
        let name = key
            .rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| eyre!("empty safe snapshot completion key"))?;
        let parts: Vec<_> = name.split('_').collect();
        ensure!(parts.len() == 3, "invalid safe snapshot completion marker");
        let timestamp = parts[0].parse().wrap_err("invalid completion timestamp")?;
        let digest = parts[1].to_owned();
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "invalid manifest digest"
        );
        let manifest_size = parts[2].parse().wrap_err("invalid manifest size")?;
        ensure!(
            manifest_size > 0 && manifest_size <= MAX_MANIFEST_SIZE,
            "manifest size is outside the allowed range"
        );
        Ok(Self {
            timestamp,
            digest,
            manifest_size,
        })
    }
}

async fn read_object_exact(store: &impl ObjectStore, key: &str, size: usize) -> Result<Vec<u8>> {
    read_object_range_exact(store, key, 0, size).await
}

async fn read_object_range_exact(
    store: &impl ObjectStore,
    key: &str,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    ensure!(end >= start, "invalid S3 byte range");
    let expected = end - start;
    let mut reader = store.get_object(key, (start, end)).await?.into_async_read();
    let mut bytes = Vec::with_capacity(expected);
    reader.read_to_end(&mut bytes).await?;
    ensure!(
        bytes.len() == expected,
        "S3 object {key:?} range {start}..{end} has {} bytes, expected {expected}",
        bytes.len()
    );
    Ok(bytes)
}

async fn read_object_range_with_retry(
    store: &impl ObjectStore,
    key: &str,
    start: usize,
    end: usize,
    max_retries: usize,
    initial_backoff_ms: u64,
) -> Result<Vec<u8>> {
    let mut backoff_ms = initial_backoff_ms;
    for attempt in 1..=max_retries {
        match read_object_range_exact(store, key, start, end).await {
            Ok(bytes) => return Ok(bytes),
            Err(error) if attempt < max_retries => {
                tracing::warn!(
                    ?error,
                    attempt,
                    %key,
                    start,
                    end,
                    "Retrying safe snapshot range"
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = backoff_ms.saturating_mul(2);
            }
            Err(error) => {
                return Err(error).wrap_err_with(|| {
                    format!("failed to read safe snapshot range {key:?} {start}..{end}")
                })
            }
        }
    }
    unreachable!("positive retry count was validated by the caller")
}

/// Finds the newest content-addressed snapshot for this exact physical store.
/// Invalid or foreign candidates are skipped; absence is a normal cache miss.
pub async fn latest_safe_snapshot(
    store: &impl ObjectStore,
    prefix: &str,
    expected_store_id: &str,
) -> Result<SafeSnapshotManifest> {
    let completion_prefix = format!("{prefix}/snapshots/complete/");
    let mut markers: Vec<_> = store
        .list_objects(&completion_prefix)
        .await?
        .into_iter()
        .filter_map(|key| match CompletionMarker::parse(&key) {
            Ok(marker) => Some(marker),
            Err(error) => {
                tracing::warn!(?error, %key, "Ignoring invalid safe snapshot marker");
                None
            }
        })
        .collect();
    markers.sort_unstable_by_key(|marker| std::cmp::Reverse(marker.timestamp));

    for marker in markers {
        let key = format!("{prefix}/snapshots/manifests/{}.json", marker.digest);
        let candidate: Result<SafeSnapshotManifest> = async {
            let bytes = read_object_exact(store, &key, marker.manifest_size).await?;
            ensure!(
                hex::encode(Sha256::digest(&bytes)) == marker.digest,
                "snapshot manifest digest mismatch"
            );
            let manifest: SafeSnapshotManifest =
                serde_json::from_slice(&bytes).wrap_err("invalid snapshot manifest JSON")?;
            manifest.validate(prefix, expected_store_id)?;
            Ok(manifest)
        }
        .await;
        match candidate {
            Ok(manifest) => return Ok(manifest),
            Err(error) => {
                tracing::warn!(?error, %key, "Ignoring unusable safe snapshot candidate");
            }
        }
    }
    bail!("no valid safe snapshot exists for store {expected_store_id:?}")
}

#[allow(clippy::too_many_arguments)]
pub async fn fetch_and_parse_safe_snapshot(
    store: Arc<impl ObjectStore>,
    concurrency: usize,
    manifest: SafeSnapshotManifest,
    max_serial_id_to_load: usize,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
    shutdown_handler: Arc<ShutdownHandler>,
) -> Result<()> {
    ensure!(concurrency > 0, "S3 import concurrency must be positive");
    ensure!(max_retries > 0, "S3 import retry count must be positive");
    let effective_max = manifest.row_count.min(max_serial_id_to_load);
    let chunks = manifest
        .chunks
        .into_iter()
        .take_while(|chunk| chunk.first_id <= effective_max);
    let tasks = stream::iter(chunks).map(|chunk| {
        let store = Arc::clone(&store);
        let tx = tx.clone();
        let shutdown = Arc::clone(&shutdown_handler);
        async move {
            tokio::select! {
                result = fetch_safe_chunk(store, chunk, effective_max, tx, max_retries, initial_backoff_ms) => result,
                _ = shutdown.wait_for_shutdown() => bail!("Shutdown requested"),
            }
        }
    });
    tokio::pin!(tasks);
    let mut tasks = tasks.buffer_unordered(concurrency);
    while let Some(result) = tasks.next().await {
        result?;
    }
    Ok(())
}

async fn fetch_safe_chunk(
    store: Arc<impl ObjectStore>,
    chunk: SafeSnapshotChunk,
    effective_max: usize,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
) -> Result<()> {
    let mut digest = Sha256::new();
    let mut epochs = HashSet::new();

    for first_record in (0..chunk.record_count).step_by(MAX_RANGE_SIZE) {
        let record_count = (chunk.record_count - first_record).min(MAX_RANGE_SIZE);
        let start = first_record
            .checked_mul(SAFE_ELEMENT_SIZE)
            .ok_or_else(|| eyre!("snapshot chunk byte offset overflow"))?;
        let end = first_record
            .checked_add(record_count)
            .and_then(|value| value.checked_mul(SAFE_ELEMENT_SIZE))
            .ok_or_else(|| eyre!("snapshot chunk byte range overflow"))?;

        let bytes = read_object_range_with_retry(
            store.as_ref(),
            &chunk.key,
            start,
            end,
            max_retries,
            initial_backoff_ms,
        )
        .await?;

        digest.update(&bytes);
        for (offset, row) in bytes.chunks_exact(SAFE_ELEMENT_SIZE).enumerate() {
            let expected_id = chunk
                .first_id
                .checked_add(first_record)
                .and_then(|value| value.checked_add(offset))
                .ok_or_else(|| eyre!("snapshot row id overflow"))?;
            let iris = S3StoredIris::from_bytes(row)?;
            epochs.insert(iris.rerand_epoch() as u32);
            ensure!(
                iris.serial_id() == expected_id,
                "snapshot chunk {:?} expected id {expected_id}, got {}",
                chunk.key,
                iris.serial_id()
            );
            if expected_id <= effective_max {
                tx.send(iris).await?;
            }
        }
    }

    ensure!(
        hex::encode(digest.finalize()) == chunk.sha256,
        "snapshot chunk {:?} digest mismatch",
        chunk.key
    );
    let mut epochs: Vec<_> = epochs.into_iter().collect();
    epochs.sort_unstable();
    ensure!(
        epochs == chunk.epochs,
        "snapshot chunk {:?} epoch inventory mismatch",
        chunk.key
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct LastSnapshotDetails {
    pub timestamp: i64,
    pub last_serial_id: i64,
    pub chunk_size: i64,
}

impl LastSnapshotDetails {
    // Parse last snapshot from s3 file name.
    // It is in {unixTime}_{batchSize}_{lastSerialId} format.
    pub fn new_from_str(last_snapshot_str: &str) -> Option<Self> {
        let parts: Vec<&str> = last_snapshot_str.split('_').collect();
        match parts.len() {
            3 => Some(Self {
                timestamp: parts[0].parse().unwrap(),
                chunk_size: parts[1].parse().unwrap(),
                last_serial_id: parts[2].parse().unwrap(),
            }),
            _ => {
                tracing::warn!("Invalid export timestamp file name: {}", last_snapshot_str);
                None
            }
        }
    }
}

pub async fn last_snapshot_timestamp(
    store: &impl ObjectStore,
    prefix_name: String,
) -> Result<LastSnapshotDetails> {
    tracing::info!("Looking for last snapshot time in prefix: {}", prefix_name);
    let timestamps_path = format!("{}/timestamps/", prefix_name);
    store
        .list_objects(timestamps_path.as_str())
        .await?
        .into_iter()
        .filter_map(|f| match f.split('/').next_back() {
            Some(file_name) => LastSnapshotDetails::new_from_str(file_name),
            _ => None,
        })
        .max_by_key(|s| s.timestamp)
        .ok_or_else(|| eyre::eyre!("No snapshot found"))
}

#[allow(clippy::too_many_arguments)]
pub async fn fetch_and_parse_chunks(
    store: Arc<impl ObjectStore>,
    concurrency: usize,
    prefix_name: String,
    last_snapshot_details: LastSnapshotDetails,
    max_serial_id_to_load: Option<usize>,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
    shutdown_handler: Arc<ShutdownHandler>,
) -> Result<()> {
    let effective_last_serial_id = max_serial_id_to_load
        .map(|max_serial_id| {
            last_snapshot_details
                .last_serial_id
                .min(max_serial_id as i64)
        })
        .unwrap_or(last_snapshot_details.last_serial_id);
    if let Some(max_serial_id_to_load) = max_serial_id_to_load {
        tracing::info!(
            "Generating chunk files using {:?}, requested cap {}, effective cap {}",
            last_snapshot_details,
            max_serial_id_to_load,
            effective_last_serial_id
        );
    } else {
        tracing::info!(
            "Generating chunk files using {:?} without a serial id cap",
            last_snapshot_details
        );
    }
    let range_size = if last_snapshot_details.chunk_size as usize > MAX_RANGE_SIZE {
        MAX_RANGE_SIZE
    } else {
        last_snapshot_details.chunk_size as usize
    };

    let chunk_iterator = (1_i64..=effective_last_serial_id).step_by(range_size);
    let stream = stream::iter(chunk_iterator).map(|chunk| {
        let chunk_id =
            (chunk / last_snapshot_details.chunk_size) * last_snapshot_details.chunk_size + 1;
        let prefix_name = prefix_name.clone();
        let offset_within_chunk = (chunk - chunk_id) as usize;
        let remaining_items = (effective_last_serial_id - chunk + 1) as usize;
        let requested_range_size = remaining_items.min(range_size);

        let store = Arc::clone(&store);
        let tx = tx.clone();
        let shutdown = Arc::clone(&shutdown_handler);
        let key = format!("{}/{}.bin", prefix_name, chunk_id);

        async move {
            tokio::spawn(async move {
                tokio::select! {
                    res = fetch_single_chunk(store, key, offset_within_chunk, requested_range_size, tx, max_retries, initial_backoff_ms) => res,
                    _ = shutdown.wait_for_shutdown() => Err(eyre::eyre!("Shutdown requested")),
                }
            })
            .await
            .map_err(|e| eyre::eyre!("Task join error: {e}"))?
        }
    });

    let mut results = stream.buffer_unordered(concurrency);
    while let Some(result) = results.next().await {
        result?;
    }

    tracing::info!("All s3 import tasks are finished.");
    Ok(())
}

async fn fetch_single_chunk(
    store: Arc<impl ObjectStore>,
    key: String,
    offset: usize,
    size: usize,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
) -> Result<()> {
    let mut attempt = 0;
    let mut backoff_ms = initial_backoff_ms;

    loop {
        attempt += 1;
        match read_range_in_chunk(Arc::clone(&store), &key, offset, size, tx.clone()).await {
            Ok(_) => {
                return Ok(());
            }
            Err(e) => {
                // If we've tried all attempts, bail
                if attempt >= max_retries {
                    return Err(eyre::eyre!(
                        "Failed to read {} after {} retries: {:?}",
                        key,
                        attempt,
                        e
                    ));
                }

                tracing::warn!(
                    "Error reading {} (attempt {} of {}): {:?}; retrying in {} ms",
                    key,
                    attempt,
                    max_retries,
                    e,
                    backoff_ms
                );

                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 2; // exponential backoff
            }
        }
    }
}

// Read [offset_within_chunk, offset_within_chunk + range_size) range from the
// chunk (s3 object) and send the parsed iris to the channel.
async fn read_range_in_chunk(
    store: Arc<impl ObjectStore>,
    key: &str,
    offset_within_chunk: usize,
    range_size: usize,
    tx: Sender<S3StoredIris>,
) -> Result<()> {
    let mut stream = store
        .get_object(
            key,
            (
                offset_within_chunk * SINGLE_ELEMENT_SIZE,
                (offset_within_chunk + range_size) * SINGLE_ELEMENT_SIZE,
            ),
        )
        .await?
        .into_async_read();

    let mut slice = vec![0_u8; SINGLE_ELEMENT_SIZE];

    loop {
        match stream.read_exact(&mut slice).await {
            Ok(_) => {
                let iris = S3StoredIris::from_bytes(&slice)?;
                tx.send(iris).await?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DbStoredIris;
    use aws_sdk_s3::primitives::SdkBody;
    use rand::Rng;
    use std::{
        cmp::min,
        collections::{HashMap, HashSet},
        time::Instant,
    };
    use tokio::sync::{mpsc, Mutex};

    #[derive(Default, Clone)]
    pub struct MockStore {
        objects: HashMap<String, Vec<u8>>,
        requested_ranges: Arc<std::sync::Mutex<Vec<(String, (usize, usize))>>>,
    }

    impl MockStore {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn add_timestamp_file(&mut self, key: &str) {
            self.objects.insert(key.to_string(), Vec::new());
        }

        pub fn add_test_data(&mut self, key: &str, records: Vec<DbStoredIris>) {
            let mut result = Vec::new();
            for record in records {
                result.extend_from_slice(&(record.id as u32).to_be_bytes());
                result.extend_from_slice(&record.left_code);
                result.extend_from_slice(&record.left_mask);
                result.extend_from_slice(&record.right_code);
                result.extend_from_slice(&record.right_mask);
                result.extend_from_slice(&(record.version_id as u16).to_be_bytes());
            }
            self.objects.insert(key.to_string(), result);
        }
    }

    #[async_trait]
    impl ObjectStore for MockStore {
        async fn get_object(&self, key: &str, range: (usize, usize)) -> Result<ByteStream> {
            self.requested_ranges
                .lock()
                .expect("range log lock poisoned")
                .push((key.to_owned(), range));
            let bytes = self
                .objects
                .get(key)
                .cloned()
                .ok_or_else(|| eyre::eyre!("Object not found: {}", key))?;

            // Handle the range parameter by slicing the bytes
            let start = range.0;
            let end = range.1.min(bytes.len());
            let sliced_bytes = bytes[start..end].to_vec();

            Ok(ByteStream::from(SdkBody::from(sliced_bytes)))
        }

        async fn list_objects(&self, _: &str) -> Result<Vec<String>> {
            Ok(self.objects.keys().cloned().collect())
        }
    }

    #[derive(Clone)]
    pub struct IntentionalFailureStore {
        inner: MockStore,
        remaining_failures: Arc<Mutex<HashMap<String, i8>>>,
        n_failures: i8,
    }

    impl IntentionalFailureStore {
        pub fn new(inner: MockStore, n_failures: i8) -> Self {
            Self {
                inner,
                remaining_failures: Arc::new(Mutex::new(HashMap::new())),
                n_failures,
            }
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for IntentionalFailureStore {
        async fn get_object(&self, key: &str, range: (usize, usize)) -> Result<ByteStream> {
            let range_hash = format!("{}_{},{}", key, range.0, range.1);
            let mut failures = self.remaining_failures.lock().await;
            let n_remaining = failures
                .entry(range_hash)
                .or_insert_with(|| self.n_failures);
            if *n_remaining > 0 {
                *n_remaining -= 1;
                return Err(eyre::eyre!("Intentional failure for testing retries"));
            }

            // All retries were consumed, delegate to the inner store
            self.inner.get_object(key, range).await
        }

        async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
            self.inner.list_objects(prefix).await
        }
    }

    /// A store whose `get_object` hangs indefinitely, simulating a stalled S3 read.
    #[derive(Clone, Default)]
    pub struct HangingStore;

    #[async_trait]
    impl ObjectStore for HangingStore {
        async fn get_object(&self, _key: &str, _range: (usize, usize)) -> Result<ByteStream> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Err(eyre::eyre!(
                "HangingStore: should have been cancelled before this"
            ))
        }

        async fn list_objects(&self, _prefix: &str) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut v = vec![0u8; len];
        v.fill_with(|| rng.gen());
        v
    }

    fn dummy_entry(id: usize) -> DbStoredIris {
        DbStoredIris {
            id: id as i64,
            version_id: 0,
            left_code: random_bytes(IRIS_CODE_LENGTH * mem::size_of::<u16>()),
            left_mask: random_bytes(MASK_CODE_LENGTH * mem::size_of::<u16>()),
            right_code: random_bytes(IRIS_CODE_LENGTH * mem::size_of::<u16>()),
            right_mask: random_bytes(MASK_CODE_LENGTH * mem::size_of::<u16>()),
            rerand_epoch: 0,
        }
    }

    /// Helper: a LastSnapshotDetails covering `n` entries in chunks of `chunk_size`.
    fn snapshot(n: usize, chunk_size: usize) -> LastSnapshotDetails {
        LastSnapshotDetails {
            timestamp: 0,
            last_serial_id: n as i64,
            chunk_size: chunk_size as i64,
        }
    }

    fn safe_manifest(
        store_id: &str,
        chunks: Vec<SafeSnapshotChunk>,
        rows: usize,
    ) -> SafeSnapshotManifest {
        SafeSnapshotManifest {
            format_version: SAFE_SNAPSHOT_FORMAT,
            store_id: store_id.to_owned(),
            row_count: rows,
            record_size: SAFE_ELEMENT_SIZE,
            epochs: vec![0],
            chunks,
        }
    }

    #[test]
    fn safe_manifest_requires_exact_inventory_and_identity() {
        let chunk = SafeSnapshotChunk {
            first_id: 1,
            last_id: 2,
            key: "out/snapshots/data/run/1.bin".to_owned(),
            sha256: "a".repeat(64),
            size_bytes: SAFE_ELEMENT_SIZE * 2,
            record_count: 2,
            epochs: vec![0],
        };
        assert!(safe_manifest("gpu-0", vec![chunk.clone()], 2)
            .validate("out", "gpu-0")
            .is_ok());
        assert!(safe_manifest("gpu-0", vec![chunk.clone()], 2)
            .validate("out", "gpu-1")
            .is_err());

        let mut gap = chunk;
        gap.first_id = 2;
        assert!(safe_manifest("gpu-0", vec![gap], 2)
            .validate("out", "gpu-0")
            .is_err());
    }

    #[test]
    fn safe_record_rejects_epoch_outside_postgres_integer() {
        let mut bytes = vec![0u8; SAFE_ELEMENT_SIZE];
        bytes[SINGLE_ELEMENT_SIZE..SINGLE_ELEMENT_SIZE + 4]
            .copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(S3StoredIris::from_bytes(&bytes).is_err());
    }

    #[test]
    fn safe_record_requires_exact_semantic_id_length() {
        let bytes = vec![0u8; SAFE_ELEMENT_SIZE - 1];
        assert!(S3StoredIris::from_bytes(&bytes).is_err());
    }

    #[tokio::test]
    async fn safe_snapshot_verifies_manifest_chunk_and_epoch_inventory() {
        let mut store = MockStore::new();
        let mut bytes = Vec::new();
        for (id, epoch) in [(1usize, 0u32), (2, 7)] {
            let record = dummy_entry(id);
            let semantic_id = [id as u8; SEMANTIC_ID_SIZE];
            bytes.extend_from_slice(&(record.id as u32).to_be_bytes());
            bytes.extend_from_slice(&record.left_code);
            bytes.extend_from_slice(&record.left_mask);
            bytes.extend_from_slice(&record.right_code);
            bytes.extend_from_slice(&record.right_mask);
            bytes.extend_from_slice(&(record.version_id as u16).to_be_bytes());
            bytes.extend_from_slice(&epoch.to_be_bytes());
            bytes.extend_from_slice(&semantic_id);
        }
        let chunk_digest = hex::encode(Sha256::digest(&bytes));
        let chunk_key = "out/snapshots/data/run/1.bin";
        store.objects.insert(chunk_key.to_owned(), bytes.clone());
        let manifest = serde_json::json!({
            "format_version": SAFE_SNAPSHOT_FORMAT,
            "store_id": "gpu-0",
            "row_count": 2,
            "record_size": SAFE_ELEMENT_SIZE,
            "epochs": [0, 7],
            "chunks": [{
                "first_id": 1,
                "last_id": 2,
                "key": chunk_key,
                "sha256": chunk_digest,
                "size_bytes": bytes.len(),
                "record_count": 2,
                "epochs": [0, 7]
            }]
        });
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let manifest_digest = hex::encode(Sha256::digest(&manifest_bytes));
        store.objects.insert(
            format!("out/snapshots/manifests/{manifest_digest}.json"),
            manifest_bytes.clone(),
        );
        store.objects.insert(
            format!(
                "out/snapshots/complete/1_{manifest_digest}_{}",
                manifest_bytes.len()
            ),
            Vec::new(),
        );

        let manifest = latest_safe_snapshot(&store, "out", "gpu-0").await.unwrap();
        assert_eq!(manifest.epochs, vec![0, 7]);
        let (tx, mut rx) = mpsc::channel(2);
        fetch_and_parse_safe_snapshot(
            Arc::new(store),
            1,
            manifest,
            2,
            tx,
            1,
            0,
            Arc::new(ShutdownHandler::new(1)),
        )
        .await
        .unwrap();
        let mut rows = vec![rx.recv().await.unwrap(), rx.recv().await.unwrap()];
        rows.sort_unstable_by_key(S3StoredIris::serial_id);
        assert_eq!(rows[0].rerand_epoch(), 0);
        assert_eq!(rows[1].rerand_epoch(), 7);
        assert_eq!(rows[0].semantic_id(), Some([1; SEMANTIC_ID_SIZE]));
        assert_eq!(rows[1].semantic_id(), Some([2; SEMANTIC_ID_SIZE]));
    }

    #[tokio::test]
    async fn safe_snapshot_chunks_are_fetched_in_bounded_ranges() {
        let record_count = MAX_RANGE_SIZE + 1;
        let key = "out/snapshots/data/run/1.bin";
        let mut store = MockStore::new();
        let mut bytes = Vec::with_capacity(record_count * SAFE_ELEMENT_SIZE);
        for id in 1..=record_count {
            let record = dummy_entry(id);
            bytes.extend_from_slice(&(record.id as u32).to_be_bytes());
            bytes.extend_from_slice(&record.left_code);
            bytes.extend_from_slice(&record.left_mask);
            bytes.extend_from_slice(&record.right_code);
            bytes.extend_from_slice(&record.right_mask);
            bytes.extend_from_slice(&(record.version_id as u16).to_be_bytes());
            bytes.extend_from_slice(&0u32.to_be_bytes());
            bytes.extend_from_slice(&[id as u8; SEMANTIC_ID_SIZE]);
        }
        let chunk = SafeSnapshotChunk {
            first_id: 1,
            last_id: record_count,
            key: key.to_owned(),
            sha256: hex::encode(Sha256::digest(&bytes)),
            size_bytes: bytes.len(),
            record_count,
            epochs: vec![0],
        };
        store.objects.insert(key.to_owned(), bytes);
        let range_log = Arc::clone(&store.requested_ranges);
        let (tx, mut rx) = mpsc::channel(record_count);

        fetch_safe_chunk(Arc::new(store), chunk, record_count, tx, 1, 0)
            .await
            .unwrap();
        let mut received = 0;
        while rx.recv().await.is_some() {
            received += 1;
        }
        assert_eq!(received, record_count);
        let ranges = range_log.lock().unwrap();
        let data_ranges = ranges
            .iter()
            .filter(|(requested_key, _)| requested_key == key)
            .map(|(_, range)| *range)
            .collect::<Vec<_>>();
        assert_eq!(data_ranges.len(), 2);
        assert!(data_ranges
            .iter()
            .all(|(start, end)| end - start <= MAX_RANGE_SIZE * SAFE_ELEMENT_SIZE));
    }

    #[tokio::test]
    async fn test_last_snapshot_timestamp() {
        let mut store = MockStore::new();
        store.add_timestamp_file("out/timestamps/123_100_954");
        store.add_timestamp_file("out/timestamps/124_100_958");
        store.add_timestamp_file("out/timestamps/125_100_958");

        let last_snapshot = last_snapshot_timestamp(&store, "out".to_string())
            .await
            .unwrap();
        assert_eq!(last_snapshot.timestamp, 125);
        assert_eq!(last_snapshot.last_serial_id, 958);
        assert_eq!(last_snapshot.chunk_size, 100);
    }

    #[tokio::test]
    async fn test_fetch_and_parse_chunks() {
        const MOCK_ENTRIES: usize = 107;
        const MOCK_CHUNK_SIZE: usize = 10;
        let mut store = MockStore::new();
        let n_chunks = MOCK_ENTRIES.div_ceil(MOCK_CHUNK_SIZE);
        for i in 0..n_chunks {
            let start_serial_id = i * MOCK_CHUNK_SIZE + 1;
            let end_serial_id = min((i + 1) * MOCK_CHUNK_SIZE, MOCK_ENTRIES);
            store.add_test_data(
                &format!("out/{start_serial_id}.bin"),
                (start_serial_id..=end_serial_id).map(dummy_entry).collect(),
            );
        }

        assert_eq!(store.list_objects("").await.unwrap().len(), n_chunks);
        let last_snapshot_details = LastSnapshotDetails {
            timestamp: 0,
            last_serial_id: MOCK_ENTRIES as i64,
            chunk_size: MOCK_CHUNK_SIZE as i64,
        };
        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(1024);
        let store_arc = Arc::new(store);
        let _res = fetch_and_parse_chunks(
            store_arc,
            1,
            "out".to_string(),
            last_snapshot_details,
            None,
            tx,
            1,
            0,
            Arc::new(ShutdownHandler::new(1)),
        )
        .await;
        let mut count = 0;
        let mut ids: HashSet<usize> = HashSet::from_iter(1..MOCK_ENTRIES);
        while let Some(chunk) = rx.recv().await {
            ids.remove(&(chunk.serial_id()));
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_and_parse_chunks_respects_max_serial_id_to_load() {
        const SNAPSHOT_ENTRIES: usize = 36;
        const MAX_SERIAL_ID_TO_LOAD: usize = 25;
        const MOCK_CHUNK_SIZE: usize = 10;
        let mut store = MockStore::new();

        // Intentionally omit 31.bin: if the importer reads past the cap, MockStore
        // will return "Object not found" and this test must fail.
        for start_serial_id in [1, 11, 21] {
            let end_serial_id = min(start_serial_id + MOCK_CHUNK_SIZE - 1, SNAPSHOT_ENTRIES);
            store.add_test_data(
                &format!("out/{start_serial_id}.bin"),
                (start_serial_id..=end_serial_id).map(dummy_entry).collect(),
            );
        }

        let last_snapshot_details = LastSnapshotDetails {
            timestamp: 0,
            last_serial_id: SNAPSHOT_ENTRIES as i64,
            chunk_size: MOCK_CHUNK_SIZE as i64,
        };
        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(1024);
        let store_arc = Arc::new(store);
        let result = fetch_and_parse_chunks(
            store_arc,
            1,
            "out".to_string(),
            last_snapshot_details,
            Some(MAX_SERIAL_ID_TO_LOAD),
            tx,
            1,
            0,
            Arc::new(ShutdownHandler::new(1)),
        )
        .await;

        assert!(
            result.is_ok(),
            "Expected fetch_and_parse_chunks to stay below the cap and therefore never request missing chunk 31.bin"
        );

        let mut count = 0;
        let mut ids: HashSet<usize> = (1..=MAX_SERIAL_ID_TO_LOAD).collect();
        while let Some(chunk) = rx.recv().await {
            ids.remove(&chunk.serial_id());
            count += 1;
        }
        assert_eq!(count, MAX_SERIAL_ID_TO_LOAD);
        assert!(ids.is_empty(), "Expected to receive only capped entries");
    }

    #[tokio::test]
    async fn test_fetch_and_parse_chunks_with_retry() {
        const MOCK_ENTRIES: usize = 36;
        const MOCK_CHUNK_SIZE: usize = 10;
        let mut mock_store = MockStore::new();
        let n_chunks = MOCK_ENTRIES.div_ceil(MOCK_CHUNK_SIZE);
        for i in 0..n_chunks {
            let start_serial_id = i * MOCK_CHUNK_SIZE + 1;
            let end_serial_id = min((i + 1) * MOCK_CHUNK_SIZE, MOCK_ENTRIES);
            mock_store.add_test_data(
                &format!("out/{start_serial_id}.bin"),
                (start_serial_id..=end_serial_id).map(dummy_entry).collect(),
            );
        }

        // Fail the first two attempts to read a chunk
        // With 100ms backoff, we should get a successful read after 100 + 200 = 300ms
        let n_failures = 2;
        let expected_backoff = Duration::from_millis(300);
        let store = IntentionalFailureStore::new(mock_store, n_failures);

        let last_snapshot_details = LastSnapshotDetails {
            timestamp: 0,
            last_serial_id: MOCK_ENTRIES as i64,
            chunk_size: MOCK_CHUNK_SIZE as i64,
        };

        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(1024);
        let store_arc = Arc::new(store);
        let now = Instant::now();
        let result = fetch_and_parse_chunks(
            store_arc,
            5,
            "out".to_string(),
            last_snapshot_details,
            None,
            tx,
            5,
            100,
            Arc::new(ShutdownHandler::new(1)),
        )
        .await;

        assert!(
            result.is_ok(),
            "Expected fetch_and_parse_chunks to succeed after retry"
        );
        assert!(
            now.elapsed() >= expected_backoff,
            "Expected to take more time to fetch"
        );

        // Make sure all the data is received
        let mut count = 0;
        let mut ids: HashSet<usize> = (1..=MOCK_ENTRIES).collect();
        while let Some(chunk) = rx.recv().await {
            ids.remove(&chunk.serial_id());
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty(), "All entries should have been received");
    }

    #[tokio::test(start_paused = true)]
    async fn test_fetch_and_parse_chunks_cancels_on_shutdown_during_read() {
        let store = Arc::new(HangingStore);
        let (tx, _rx) = mpsc::channel::<S3StoredIris>(1024);
        let shutdown = Arc::new(ShutdownHandler::new(1));

        // The HangingStore will never return, ensuring that shutdown triggers while a read is in progress.
        let shutdown_clone = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            shutdown_clone.trigger_manual_shutdown();
        });

        let result = tokio::time::timeout(
            Duration::from_millis(500),
            fetch_and_parse_chunks(
                store,
                1,
                "out".to_string(),
                snapshot(10, 10),
                None,
                tx,
                3,
                10_000, // 10s backoff — should never be reached
                shutdown,
            ),
        )
        .await
        .expect("cancellation should complete within 500ms of virtual time");

        assert!(result.is_err(), "Expected Err on shutdown during read");
    }

    #[tokio::test(start_paused = true)]
    async fn test_fetch_and_parse_chunks_cancels_on_shutdown_during_backoff() {
        // Build a store whose first read always fails so the task enters a long backoff sleep.
        let mut mock_store = MockStore::new();
        mock_store.add_test_data("out/1.bin", (1..=10).map(dummy_entry).collect());
        // i8::MAX failures ensures the task never succeeds within our retry budget.
        let store = Arc::new(IntentionalFailureStore::new(mock_store, i8::MAX));

        let (tx, _rx) = mpsc::channel::<S3StoredIris>(1024);
        let shutdown = Arc::new(ShutdownHandler::new(1));

        // IntentionalFailureStore returns Err immediately (no timer, no sleep), so by the
        // time any Tokio timer fires the code is already inside the backoff sleep.
        // The 50ms shutdown fires well before the 10s backoff expires.
        let shutdown_clone = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            shutdown_clone.trigger_manual_shutdown();
        });

        let result = tokio::time::timeout(
            Duration::from_millis(500),
            fetch_and_parse_chunks(
                store,
                1,
                "out".to_string(),
                snapshot(10, 10),
                None,
                tx,
                20,     // plenty of retries — shutdown should interrupt first
                10_000, // 10s initial backoff
                shutdown,
            ),
        )
        .await
        .expect("cancellation should complete within 500ms of virtual time");

        let err = result.expect_err("result should be err");
        assert!(
            format!("{err:#}").contains("Shutdown"),
            "Expected Err on shutdown during backoff"
        );
    }
}
