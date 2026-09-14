use ampc_server_utils::ShutdownHandler;
use async_trait::async_trait;
use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::config::StalledStreamProtectionConfig;
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_s3::{primitives::ByteStream, Client};
use eyre::{bail, eyre, Result};
use futures::{stream, StreamExt};
use iris_mpc_common::{VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{cmp::Ordering, mem, sync::Arc, time::Duration};
use tokio::{io::AsyncReadExt, sync::mpsc::Sender};

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>()
    + mem::size_of::<u16>(); // 75 KB

const MAX_RANGE_SIZE: usize = 200; // Download chunks in sub-chunks of 200 elements = 15 MB

pub struct S3StoredIris {
    #[allow(dead_code)]
    id: i64,
    left_code_even: Vec<u8>,
    left_code_odd: Vec<u8>,
    left_mask_even: Vec<u8>,
    left_mask_odd: Vec<u8>,
    right_code_even: Vec<u8>,
    right_code_odd: Vec<u8>,
    right_mask_even: Vec<u8>,
    right_mask_odd: Vec<u8>,
    version_id: i16,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotLayout {
    Legacy,
    Generation { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastSnapshotDetails {
    pub timestamp: i64,
    pub last_serial_id: i64,
    pub chunk_size: i64,
    pub layout: SnapshotLayout,
}

impl LastSnapshotDetails {
    // Parse either a legacy marker ({unixTime}_{batchSize}_{lastSerialId}) or a
    // generation marker ({unixTime}_{batchSize}_{lastSerialId}_v2-bin-{id}).
    pub fn new_from_str(last_snapshot_str: &str) -> Option<Self> {
        let parts: Vec<&str> = last_snapshot_str.split('_').collect();
        let layout = match parts.as_slice() {
            [_, _, _] => SnapshotLayout::Legacy,
            [_, _, _, descriptor] => {
                let descriptor_parts: Vec<&str> = descriptor.split('-').collect();
                match descriptor_parts.as_slice() {
                    ["v2", "bin", id]
                        if id.len() == 32
                            && id.bytes().all(|byte| {
                                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                            }) =>
                    {
                        SnapshotLayout::Generation {
                            id: (*id).to_owned(),
                        }
                    }
                    _ => return Self::invalid_marker(last_snapshot_str),
                }
            }
            _ => return Self::invalid_marker(last_snapshot_str),
        };

        let parse_positive = |value: &str| value.parse::<i64>().ok().filter(|parsed| *parsed > 0);
        let (Some(timestamp), Some(chunk_size), Some(last_serial_id)) = (
            parse_positive(parts[0]),
            parse_positive(parts[1]),
            parse_positive(parts[2]),
        ) else {
            return Self::invalid_marker(last_snapshot_str);
        };

        Some(Self {
            timestamp,
            chunk_size,
            last_serial_id,
            layout,
        })
    }

    fn invalid_marker(last_snapshot_str: &str) -> Option<Self> {
        tracing::warn!("Invalid export timestamp file name: {}", last_snapshot_str);
        None
    }

    fn chunk_prefix(&self, stable_prefix: &str) -> String {
        match &self.layout {
            SnapshotLayout::Legacy => stable_prefix.to_owned(),
            SnapshotLayout::Generation { id } => {
                format!("{stable_prefix}/generations/{id}")
            }
        }
    }

    fn compare_recency(&self, other: &Self) -> Ordering {
        self.timestamp
            .cmp(&other.timestamp)
            .then_with(|| match (&self.layout, &other.layout) {
                (SnapshotLayout::Legacy, SnapshotLayout::Legacy) => Ordering::Equal,
                (SnapshotLayout::Legacy, SnapshotLayout::Generation { .. }) => Ordering::Less,
                (SnapshotLayout::Generation { .. }, SnapshotLayout::Legacy) => Ordering::Greater,
                (
                    SnapshotLayout::Generation { id: self_id },
                    SnapshotLayout::Generation { id: other_id },
                ) => self_id.cmp(other_id),
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
        .max_by(|left, right| left.compare_recency(right))
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
    let chunk_prefix = last_snapshot_details.chunk_prefix(&prefix_name);

    let chunk_iterator = (1_i64..=effective_last_serial_id).step_by(range_size);
    let stream = stream::iter(chunk_iterator).map(|chunk| {
        let chunk_id =
            (chunk / last_snapshot_details.chunk_size) * last_snapshot_details.chunk_size + 1;
        let chunk_prefix = chunk_prefix.clone();
        let offset_within_chunk = (chunk - chunk_id) as usize;
        let remaining_items = (effective_last_serial_id - chunk + 1) as usize;
        let requested_range_size = remaining_items.min(range_size);

        let store = Arc::clone(&store);
        let tx = tx.clone();
        let shutdown = Arc::clone(&shutdown_handler);
        let key = format!("{}/{}.bin", chunk_prefix, chunk_id);

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

    let mut records = Vec::with_capacity(range_size);
    for record_index in 0..range_size {
        let mut slice = vec![0_u8; SINGLE_ELEMENT_SIZE];
        stream.read_exact(&mut slice).await.map_err(|error| {
            eyre!(
                "short read for {key}: requested {range_size} records at offset {offset_within_chunk}, failed at record {record_index}: {error}"
            )
        })?;
        records.push(S3StoredIris::from_bytes(&slice)?);
    }
    for iris in records {
        tx.send(iris).await?;
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
            layout: SnapshotLayout::Legacy,
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
        assert_eq!(last_snapshot.layout, SnapshotLayout::Legacy);
    }

    #[test]
    fn test_snapshot_marker_parses_legacy_and_generation_layouts() {
        assert_eq!(
            LastSnapshotDetails::new_from_str("123_100_958"),
            Some(LastSnapshotDetails {
                timestamp: 123,
                chunk_size: 100,
                last_serial_id: 958,
                layout: SnapshotLayout::Legacy,
            })
        );
        assert_eq!(
            LastSnapshotDetails::new_from_str(
                "123_100_958_v2-bin-0123456789abcdef0123456789abcdef"
            ),
            Some(LastSnapshotDetails {
                timestamp: 123,
                chunk_size: 100,
                last_serial_id: 958,
                layout: SnapshotLayout::Generation {
                    id: "0123456789abcdef0123456789abcdef".to_owned(),
                },
            })
        );
    }

    #[test]
    fn test_snapshot_marker_rejects_malformed_values() {
        for marker in [
            "not-a-number_100_958",
            "123_not-a-number_958",
            "123_100_not-a-number",
            "0_100_958",
            "123_0_958",
            "123_-1_958",
            "123_100_0",
            "123_100_-1",
            "123_100_958_v3-bin-0123456789abcdef0123456789abcdef",
            "123_100_958_v2-json-0123456789abcdef0123456789abcdef",
            "123_100_958_v2-bin-short",
            "123_100_958_v2-bin-0123456789ABCDEF0123456789ABCDEF",
            "123_100_958_v2-bin-0123456789abcdef0123456789abcdeg",
            "123_100_958_v2-bin-0123456789abcdef0123456789abcdef_extra",
        ] {
            assert_eq!(
                LastSnapshotDetails::new_from_str(marker),
                None,
                "marker should be rejected: {marker}"
            );
        }
    }

    #[tokio::test]
    async fn test_snapshot_order_prefers_generation_then_generation_id_on_ties() {
        let mut store = MockStore::new();
        store.add_timestamp_file("out/timestamps/124_100_958");
        store.add_timestamp_file(
            "out/timestamps/124_100_958_v2-bin-00000000000000000000000000000001",
        );
        store.add_timestamp_file(
            "out/timestamps/124_100_958_v2-bin-00000000000000000000000000000002",
        );
        store.add_timestamp_file(
            "out/timestamps/123_100_958_v2-bin-ffffffffffffffffffffffffffffffff",
        );

        let snapshot = last_snapshot_timestamp(&store, "out".to_owned())
            .await
            .unwrap();
        assert_eq!(snapshot.timestamp, 124);
        assert_eq!(
            snapshot.layout,
            SnapshotLayout::Generation {
                id: "00000000000000000000000000000002".to_owned()
            }
        );
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
            layout: SnapshotLayout::Legacy,
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
    async fn test_fetch_and_parse_chunks_uses_generation_prefix() {
        const GENERATION_ID: &str = "0123456789abcdef0123456789abcdef";
        let mut store = MockStore::new();
        store.add_test_data(
            &format!("out/generations/{GENERATION_ID}/1.bin"),
            (1..=2).map(dummy_entry).collect(),
        );
        let details = LastSnapshotDetails {
            timestamp: 123,
            last_serial_id: 2,
            chunk_size: 2,
            layout: SnapshotLayout::Generation {
                id: GENERATION_ID.to_owned(),
            },
        };
        let (tx, mut rx) = mpsc::channel(2);

        fetch_and_parse_chunks(
            Arc::new(store),
            1,
            "out".to_owned(),
            details,
            None,
            tx,
            1,
            0,
            Arc::new(ShutdownHandler::new(1)),
        )
        .await
        .unwrap();

        assert_eq!(rx.recv().await.unwrap().serial_id(), 1);
        assert_eq!(rx.recv().await.unwrap().serial_id(), 2);
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_range_rejects_premature_eof_without_sending_partial_records() {
        let mut store = MockStore::new();
        store.add_test_data("out/1.bin", vec![dummy_entry(1)]);
        let (tx, mut rx) = mpsc::channel(2);

        let error = read_range_in_chunk(Arc::new(store), "out/1.bin", 0, 2, tx)
            .await
            .expect_err("one record cannot satisfy a two-record read");

        assert!(format!("{error:#}").contains("short read"));
        assert!(rx.recv().await.is_none());
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
            layout: SnapshotLayout::Legacy,
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
            layout: SnapshotLayout::Legacy,
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
