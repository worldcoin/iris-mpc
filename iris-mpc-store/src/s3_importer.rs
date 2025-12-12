use async_trait::async_trait;
use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::config::StalledStreamProtectionConfig;
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_s3::{primitives::ByteStream, Client};
use eyre::{bail, eyre, Result};
use iris_mpc_common::{vector_id::VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{collections::VecDeque, mem, sync::Arc, time::Duration};
use tokio::{io::AsyncReadExt, sync::mpsc::Sender, task};

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

pub async fn fetch_and_parse_chunks(
    store: Arc<impl ObjectStore>,
    concurrency: usize,
    prefix_name: String,
    last_snapshot_details: LastSnapshotDetails,
    tx: Sender<S3StoredIris>,
    max_retries: usize,
    initial_backoff_ms: u64,
) -> Result<()> {
    tracing::info!("Generating chunk files using: {:?}", last_snapshot_details);
    let range_size = if last_snapshot_details.chunk_size as usize > MAX_RANGE_SIZE {
        MAX_RANGE_SIZE
    } else {
        last_snapshot_details.chunk_size as usize
    };
    let mut handles: VecDeque<task::JoinHandle<Result<(), eyre::Error>>> =
        VecDeque::with_capacity(concurrency);

    for chunk in (1..=last_snapshot_details.last_serial_id).step_by(range_size) {
        let chunk_id =
            (chunk / last_snapshot_details.chunk_size) * last_snapshot_details.chunk_size + 1;
        let prefix_name = prefix_name.clone();
        let offset_within_chunk = (chunk - chunk_id) as usize;

        // Wait if we've hit the concurrency limit
        if handles.len() >= concurrency {
            let handle = handles.pop_front().expect("No s3 import handles to pop");
            handle.await??;
        }

        handles.push_back(task::spawn({
            let store = Arc::clone(&store);
            let tx = tx.clone();
            async move {
                let mut attempt = 0;
                let mut backoff_ms = initial_backoff_ms;
                let key = format!("{}/{}.bin", prefix_name, chunk_id);

                // Retry reading the range with exponential backoff
                loop {
                    attempt += 1;
                    match read_range_in_chunk(
                        Arc::clone(&store),
                        &key,
                        offset_within_chunk,
                        range_size,
                        tx.clone(),
                    )
                    .await
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
        }));
    }

    tracing::info!("All s3 import tasks are spawned. Waiting for them to finish");
    // Wait for remaining handles
    for handle in handles {
        handle.await??;
    }
    tracing::info!("All s3 import tasks are finished.");
    Ok(())
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
            tx,
            1,
            0,
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
            tx,
            5,
            100,
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
}
