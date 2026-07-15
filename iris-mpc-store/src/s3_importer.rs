use ampc_server_utils::ShutdownHandler;
use async_trait::async_trait;
use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::config::StalledStreamProtectionConfig;
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_s3::{primitives::ByteStream, Client};
use eyre::{bail, eyre, Result};
use futures::{stream, StreamExt};
use iris_mpc_common::{VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{mem, sync::Arc, time::Duration};
use tokio::{io::AsyncReadExt, sync::mpsc::Sender};

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>()
    + mem::size_of::<u16>(); // 75 KB, identical for both snapshot formats

const MAX_RANGE_SIZE: usize = 200; // Download chunks in sub-chunks of 200 elements = 15 MB

/// On-disk encoding of the share buffers inside a snapshot chunk. Both
/// formats have identical record sizes, so the format cannot be inferred from
/// the data — it is carried by the snapshot marker file name (see
/// [`LastSnapshotDetails`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotFormat {
    /// Format 1 (legacy, implied by 3-part marker names): each share is
    /// stored as two byte planes with each u16 half encoded as `byte ^ 0x80`
    /// — the zero-point limb encoding of the old GPU kernel.
    V1LegacyLimbs,
    /// Format 2: each share is a plain little-endian u16 array, identical to
    /// the database representation. Consumers derive their compute encoding
    /// at load time.
    V2PlainU16,
}

pub struct S3StoredIris {
    #[allow(dead_code)]
    id: i64,
    version_id: i16,
    shares: S3IrisShares,
}

/// The share buffers of one S3 record, in whichever encoding the snapshot
/// uses.
pub enum S3IrisShares {
    /// [`SnapshotFormat::V1LegacyLimbs`]: `^ 0x80` byte planes
    /// (odd = low bytes, even = high bytes of the u16 shares).
    LegacyLimbs {
        left_code_odd: Vec<u8>,
        left_code_even: Vec<u8>,
        left_mask_odd: Vec<u8>,
        left_mask_even: Vec<u8>,
        right_code_odd: Vec<u8>,
        right_code_even: Vec<u8>,
        right_mask_odd: Vec<u8>,
        right_mask_even: Vec<u8>,
    },
    /// [`SnapshotFormat::V2PlainU16`]: plain u16 shares (database
    /// representation).
    PlainU16 {
        left_code: Vec<u16>,
        left_mask: Vec<u16>,
        right_code: Vec<u16>,
        right_mask: Vec<u16>,
    },
}

impl S3StoredIris {
    pub fn from_bytes(bytes: &[u8], format: SnapshotFormat) -> Result<Self, eyre::Error> {
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
        // Helper closure to extract `len` little-endian u16s
        let extract_u16s =
            |bytes: &[u8], cursor: &mut usize, len: usize| -> Result<Vec<u16>, eyre::Error> {
                if *cursor + len * 2 > bytes.len() {
                    bail!("Exceeded total bytes while extracting u16 slice",);
                }
                let out = bytes[*cursor..*cursor + len * 2]
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                *cursor += len * 2;
                Ok(out)
            };

        // Parse `id` (i64)
        let id_bytes = extract_slice(bytes, &mut cursor, 4)?;
        let id = u32::from_be_bytes(
            id_bytes
                .try_into()
                .map_err(|_| eyre!("Failed to convert id bytes to i64"))?,
        ) as i64;

        let shares = match format {
            SnapshotFormat::V1LegacyLimbs => {
                // parse codes and masks for each limb plane separately
                let left_code_odd = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
                let left_code_even = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
                let left_mask_odd = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
                let left_mask_even = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
                let right_code_odd = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
                let right_code_even = extract_slice(bytes, &mut cursor, IRIS_CODE_LENGTH)?;
                let right_mask_odd = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
                let right_mask_even = extract_slice(bytes, &mut cursor, MASK_CODE_LENGTH)?;
                S3IrisShares::LegacyLimbs {
                    left_code_odd,
                    left_code_even,
                    left_mask_odd,
                    left_mask_even,
                    right_code_odd,
                    right_code_even,
                    right_mask_odd,
                    right_mask_even,
                }
            }
            SnapshotFormat::V2PlainU16 => S3IrisShares::PlainU16 {
                left_code: extract_u16s(bytes, &mut cursor, IRIS_CODE_LENGTH)?,
                left_mask: extract_u16s(bytes, &mut cursor, MASK_CODE_LENGTH)?,
                right_code: extract_u16s(bytes, &mut cursor, IRIS_CODE_LENGTH)?,
                right_mask: extract_u16s(bytes, &mut cursor, MASK_CODE_LENGTH)?,
            },
        };

        // Parse `version_id` (i16)
        let version_id_bytes = extract_slice(bytes, &mut cursor, 2)?;
        let version_id = u16::from_be_bytes(
            version_id_bytes
                .try_into()
                .map_err(|_| eyre!("Failed to convert version id bytes to i16"))?,
        ) as i16;

        Ok(S3StoredIris {
            id,
            version_id,
            shares,
        })
    }

    pub fn serial_id(&self) -> usize {
        self.id as usize
    }

    pub fn version_id(&self) -> i16 {
        self.version_id
    }

    pub fn vector_id(&self) -> VectorId {
        VectorId::new(self.id as u32, self.version_id)
    }

    pub fn shares(&self) -> &S3IrisShares {
        &self.shares
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

#[derive(Debug, Clone)]
pub struct LastSnapshotDetails {
    pub timestamp: i64,
    pub last_serial_id: i64,
    pub chunk_size: i64,
    pub format: SnapshotFormat,
}

impl LastSnapshotDetails {
    // Parse last snapshot from s3 file name.
    // It is in {unixTime}_{batchSize}_{lastSerialId} format, with an optional
    // fourth {formatVersion} part ({unixTime}_{batchSize}_{lastSerialId}_{format}).
    // A 3-part name implies format 1 (legacy limb planes). Marker files with an
    // unknown format version are skipped, so an older importer never
    // misinterprets snapshots it cannot decode (record sizes are identical
    // across formats, so a wrong guess would silently load garbage shares).
    pub fn new_from_str(last_snapshot_str: &str) -> Option<Self> {
        let parts: Vec<&str> = last_snapshot_str.split('_').collect();
        let format = match parts.len() {
            3 => SnapshotFormat::V1LegacyLimbs,
            4 => match parts[3] {
                "1" => SnapshotFormat::V1LegacyLimbs,
                "2" => SnapshotFormat::V2PlainU16,
                other => {
                    tracing::warn!(
                        "Skipping snapshot with unsupported format version {}: {}",
                        other,
                        last_snapshot_str
                    );
                    return None;
                }
            },
            _ => {
                tracing::warn!("Invalid export timestamp file name: {}", last_snapshot_str);
                return None;
            }
        };
        Some(Self {
            timestamp: parts[0].parse().unwrap(),
            chunk_size: parts[1].parse().unwrap(),
            last_serial_id: parts[2].parse().unwrap(),
            format,
        })
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
        let format = last_snapshot_details.format;

        async move {
            tokio::spawn(async move {
                tokio::select! {
                    res = fetch_single_chunk(store, key, offset_within_chunk, requested_range_size, format, tx, max_retries, initial_backoff_ms) => res,
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

#[allow(clippy::too_many_arguments)]
async fn fetch_single_chunk(
    store: Arc<impl ObjectStore>,
    key: String,
    offset: usize,
    size: usize,
    format: SnapshotFormat,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
) -> Result<()> {
    let mut attempt = 0;
    let mut backoff_ms = initial_backoff_ms;

    loop {
        attempt += 1;
        match read_range_in_chunk(Arc::clone(&store), &key, offset, size, format, tx.clone()).await
        {
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
    format: SnapshotFormat,
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
                let iris = S3StoredIris::from_bytes(&slice, format)?;
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
        }
    }

    /// Helper: a LastSnapshotDetails covering `n` entries in chunks of `chunk_size`.
    fn snapshot(n: usize, chunk_size: usize) -> LastSnapshotDetails {
        LastSnapshotDetails {
            timestamp: 0,
            last_serial_id: n as i64,
            chunk_size: chunk_size as i64,
            format: SnapshotFormat::V1LegacyLimbs,
        }
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
        assert_eq!(last_snapshot.format, SnapshotFormat::V1LegacyLimbs);
    }

    #[tokio::test]
    async fn test_last_snapshot_timestamp_versioned_markers() {
        let mut store = MockStore::new();
        store.add_timestamp_file("out/timestamps/123_100_954");
        store.add_timestamp_file("out/timestamps/126_100_960_2");
        // Unknown format versions must be skipped, not misread.
        store.add_timestamp_file("out/timestamps/127_100_961_9");

        let last_snapshot = last_snapshot_timestamp(&store, "out".to_string())
            .await
            .unwrap();
        assert_eq!(last_snapshot.timestamp, 126);
        assert_eq!(last_snapshot.last_serial_id, 960);
        assert_eq!(last_snapshot.format, SnapshotFormat::V2PlainU16);
    }

    #[test]
    fn test_snapshot_marker_format_parsing() {
        let v1 = LastSnapshotDetails::new_from_str("123_100_954").unwrap();
        assert_eq!(v1.format, SnapshotFormat::V1LegacyLimbs);
        let v1_explicit = LastSnapshotDetails::new_from_str("123_100_954_1").unwrap();
        assert_eq!(v1_explicit.format, SnapshotFormat::V1LegacyLimbs);
        let v2 = LastSnapshotDetails::new_from_str("123_100_954_2").unwrap();
        assert_eq!(v2.format, SnapshotFormat::V2PlainU16);
        assert_eq!(v2.timestamp, 123);
        assert_eq!(v2.chunk_size, 100);
        assert_eq!(v2.last_serial_id, 954);
        assert!(LastSnapshotDetails::new_from_str("123_100_954_3").is_none());
        assert!(LastSnapshotDetails::new_from_str("123_100").is_none());
    }

    /// Serialize one record in v2 (plain LE u16) layout.
    fn v2_record_bytes(
        id: u32,
        version_id: u16,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(SINGLE_ELEMENT_SIZE);
        out.extend_from_slice(&id.to_be_bytes());
        for share in [left_code, left_mask, right_code, right_mask] {
            for v in share {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&version_id.to_be_bytes());
        out
    }

    #[test]
    fn test_from_bytes_v2_roundtrip() {
        let mut rng = rand::thread_rng();
        let left_code: Vec<u16> = (0..IRIS_CODE_LENGTH).map(|_| rng.gen()).collect();
        let left_mask: Vec<u16> = (0..MASK_CODE_LENGTH).map(|_| rng.gen()).collect();
        let right_code: Vec<u16> = (0..IRIS_CODE_LENGTH).map(|_| rng.gen()).collect();
        let right_mask: Vec<u16> = (0..MASK_CODE_LENGTH).map(|_| rng.gen()).collect();

        let bytes = v2_record_bytes(42, 7, &left_code, &left_mask, &right_code, &right_mask);
        assert_eq!(bytes.len(), SINGLE_ELEMENT_SIZE);

        let iris = S3StoredIris::from_bytes(&bytes, SnapshotFormat::V2PlainU16).unwrap();
        assert_eq!(iris.serial_id(), 42);
        assert_eq!(iris.version_id(), 7);
        match iris.shares() {
            S3IrisShares::PlainU16 {
                left_code: lc,
                left_mask: lm,
                right_code: rc,
                right_mask: rm,
            } => {
                assert_eq!(lc, &left_code);
                assert_eq!(lm, &left_mask);
                assert_eq!(rc, &right_code);
                assert_eq!(rm, &right_mask);
            }
            _ => panic!("expected PlainU16 shares"),
        }
    }

    /// The same logical shares written in both formats must decode to the same
    /// u16 values (v1 planes hold `byte ^ 0x80` of each u16 half).
    #[test]
    fn test_v1_v2_encode_same_logical_shares() {
        let mut rng = rand::thread_rng();
        let shares: [Vec<u16>; 4] = [
            (0..IRIS_CODE_LENGTH).map(|_| rng.gen()).collect(),
            (0..MASK_CODE_LENGTH).map(|_| rng.gen()).collect(),
            (0..IRIS_CODE_LENGTH).map(|_| rng.gen()).collect(),
            (0..MASK_CODE_LENGTH).map(|_| rng.gen()).collect(),
        ];

        // v1: per share, the odd plane then the even plane, each `^ 0x80`.
        let mut v1 = Vec::with_capacity(SINGLE_ELEMENT_SIZE);
        v1.extend_from_slice(&5u32.to_be_bytes());
        for share in &shares {
            v1.extend(share.iter().map(|v| (*v as u8) ^ 0x80));
            v1.extend(share.iter().map(|v| ((*v >> 8) as u8) ^ 0x80));
        }
        v1.extend_from_slice(&3u16.to_be_bytes());
        assert_eq!(v1.len(), SINGLE_ELEMENT_SIZE);

        let v2 = v2_record_bytes(5, 3, &shares[0], &shares[1], &shares[2], &shares[3]);

        let iris_v1 = S3StoredIris::from_bytes(&v1, SnapshotFormat::V1LegacyLimbs).unwrap();
        let iris_v2 = S3StoredIris::from_bytes(&v2, SnapshotFormat::V2PlainU16).unwrap();

        let decode_plane = |odd: &[u8], even: &[u8]| -> Vec<u16> {
            odd.iter()
                .zip(even)
                .map(|(o, e)| u16::from_le_bytes([o ^ 0x80, e ^ 0x80]))
                .collect()
        };
        let (
            S3IrisShares::LegacyLimbs {
                left_code_odd,
                left_code_even,
                left_mask_odd,
                left_mask_even,
                right_code_odd,
                right_code_even,
                right_mask_odd,
                right_mask_even,
            },
            S3IrisShares::PlainU16 {
                left_code,
                left_mask,
                right_code,
                right_mask,
            },
        ) = (iris_v1.shares(), iris_v2.shares())
        else {
            panic!("unexpected share variants");
        };
        assert_eq!(&decode_plane(left_code_odd, left_code_even), left_code);
        assert_eq!(&decode_plane(left_mask_odd, left_mask_even), left_mask);
        assert_eq!(&decode_plane(right_code_odd, right_code_even), right_code);
        assert_eq!(&decode_plane(right_mask_odd, right_mask_even), right_mask);
        assert_eq!(left_code, &shares[0]);
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
            format: SnapshotFormat::V1LegacyLimbs,
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
    async fn test_fetch_and_parse_chunks_v2() {
        const MOCK_ENTRIES: usize = 27;
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

        let mut last_snapshot_details = snapshot(MOCK_ENTRIES, MOCK_CHUNK_SIZE);
        last_snapshot_details.format = SnapshotFormat::V2PlainU16;
        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(1024);
        let _res = fetch_and_parse_chunks(
            Arc::new(store),
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
        let mut ids: HashSet<usize> = HashSet::from_iter(1..=MOCK_ENTRIES);
        while let Some(iris) = rx.recv().await {
            assert!(
                matches!(iris.shares(), S3IrisShares::PlainU16 { .. }),
                "v2 snapshots must parse into plain u16 shares"
            );
            ids.remove(&iris.serial_id());
        }
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
            format: SnapshotFormat::V1LegacyLimbs,
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
            format: SnapshotFormat::V1LegacyLimbs,
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
