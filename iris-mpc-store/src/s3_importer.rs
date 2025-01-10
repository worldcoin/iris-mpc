use crate::StoredIris;
use async_trait::async_trait;
use aws_sdk_s3::{operation::RequestId, primitives::ByteStream, Client};
use eyre::eyre;
use futures::{stream, Stream, StreamExt};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{
    mem,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio::io::AsyncReadExt;

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>(); // 75 KB

const MAX_RANGE_SIZE: usize = 200; // Download chunks in sub-chunks of 200 elements = 15 MB

pub struct S3StoredIris {
    #[allow(dead_code)]
    id:              i64,
    left_code_even:  Vec<u8>,
    left_code_odd:   Vec<u8>,
    left_mask_even:  Vec<u8>,
    left_mask_odd:   Vec<u8>,
    right_code_even: Vec<u8>,
    right_code_odd:  Vec<u8>,
    right_mask_even: Vec<u8>,
    right_mask_odd:  Vec<u8>,
}

impl S3StoredIris {
    pub fn from_bytes(bytes: &[u8]) -> eyre::Result<Self, eyre::Error> {
        let mut cursor = 0;

        // Helper closure to extract a slice of a given size
        let extract_slice =
            |bytes: &[u8], cursor: &mut usize, size: usize| -> eyre::Result<Vec<u8>, eyre::Error> {
                if *cursor + size > bytes.len() {
                    return Err(eyre!("Exceeded total bytes while extracting slice",));
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
        })
    }

    pub fn index(&self) -> usize {
        self.id as usize
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

#[async_trait]
pub trait ObjectStore: Send + Sync + 'static {
    async fn get_object(&self, key: &str, range: (usize, usize)) -> eyre::Result<ByteStream>;
    async fn list_objects(&self, prefix: &str) -> eyre::Result<Vec<String>>;
}

pub struct S3Store {
    client: Arc<Client>,
    bucket: String,
}

impl S3Store {
    pub fn new(client: Arc<Client>, bucket: String) -> Self {
        Self { client, bucket }
    }
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn get_object(&self, key: &str, range: (usize, usize)) -> eyre::Result<ByteStream> {
        let res = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .range(format!("bytes={}-{}", range.0, range.1 - 1))
            .send()
            .await?;
        if let Some(request_id) = res.request_id() {
            tracing::debug!("get_object request ID for key {}: {}", key, request_id);
        }
        Ok(res.body)
    }

    async fn list_objects(&self, prefix: &str) -> eyre::Result<Vec<String>> {
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

#[derive(Debug)]
pub struct LastSnapshotDetails {
    pub timestamp:      i64,
    pub last_serial_id: i64,
    pub chunk_size:     i64,
}

impl LastSnapshotDetails {
    // Parse last snapshot from s3 file name.
    // It is in {unixTime}_{batchSize}_{lastSerialId} format.
    pub fn new_from_str(last_snapshot_str: &str) -> Option<Self> {
        let parts: Vec<&str> = last_snapshot_str.split('_').collect();
        match parts.len() {
            3 => Some(Self {
                timestamp:      parts[0].parse().unwrap(),
                chunk_size:     parts[1].parse().unwrap(),
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
) -> eyre::Result<LastSnapshotDetails> {
    tracing::info!("Looking for last snapshot time in prefix: {}", prefix_name);
    let timestamps_path = format!("{}/timestamps/", prefix_name);
    store
        .list_objects(timestamps_path.as_str())
        .await?
        .into_iter()
        .filter_map(|f| match f.split('/').last() {
            Some(file_name) => LastSnapshotDetails::new_from_str(file_name),
            _ => None,
        })
        .max_by_key(|s| s.timestamp)
        .ok_or_else(|| eyre::eyre!("No snapshot found"))
}

pub async fn fetch_and_parse_chunks(
    store: &impl ObjectStore,
    concurrency: usize,
    prefix_name: String,
    last_snapshot_details: LastSnapshotDetails,
) -> Pin<Box<dyn Stream<Item = eyre::Result<StoredIris>> + Send + '_>> {
    tracing::info!("Generating chunk files using: {:?}", last_snapshot_details);
    let range_size = if last_snapshot_details.chunk_size as usize > MAX_RANGE_SIZE {
        MAX_RANGE_SIZE
    } else {
        last_snapshot_details.chunk_size as usize
    };
    let total_bytes = Arc::new(AtomicUsize::new(0));
    let now = Instant::now();

    let result_stream =
        stream::iter((1..=last_snapshot_details.last_serial_id).step_by(range_size))
            .map({
                let total_bytes_clone = total_bytes.clone();
                move |chunk| {
                    let counter = total_bytes_clone.clone();
                    let prefix_name = prefix_name.clone();
                    async move {
                        let chunk_id = (chunk / last_snapshot_details.chunk_size)
                            * last_snapshot_details.chunk_size
                            + 1;
                        let offset_within_chunk = (chunk - chunk_id) as usize;
                        let mut object_stream = store
                            .get_object(
                                &format!("{}/{}.bin", prefix_name, chunk_id),
                                (
                                    offset_within_chunk * SINGLE_ELEMENT_SIZE,
                                    (offset_within_chunk + range_size) * SINGLE_ELEMENT_SIZE,
                                ),
                            )
                            .await?
                            .into_async_read();
                        let mut records = Vec::with_capacity(range_size);
                        let mut buf = vec![0u8; SINGLE_ELEMENT_SIZE];
                        loop {
                            match object_stream.read_exact(&mut buf).await {
                                Ok(_) => {
                                    let iris = S3StoredIris::from_bytes(&buf);
                                    records.push(iris);
                                    counter.fetch_add(SINGLE_ELEMENT_SIZE, Ordering::Relaxed);
                                }
                                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                                Err(e) => return Err(e.into()),
                            }
                        }
                        let stream_of_stored_iris =
                            stream::iter(records).map(|res_s3| res_s3.map(StoredIris::S3));

                        Ok::<_, eyre::Error>(stream_of_stored_iris)
                    }
                }
            })
            .buffer_unordered(concurrency)
            .flat_map(|result| match result {
                Ok(stream) => stream.boxed(),
                Err(e) => stream::once(async move { Err(e) }).boxed(),
            })
            .inspect({
                let counter = Arc::new(AtomicUsize::new(0));
                move |_| {
                    if counter.fetch_add(1, Ordering::Relaxed) % 1_000_000 == 0 {
                        let elapsed = now.elapsed().as_secs_f32();
                        if elapsed > 0.0 {
                            let bytes = total_bytes.load(Ordering::Relaxed);
                            tracing::info!(
                                "Current download throughput: {:.2} Gbps",
                                bytes as f32 * 8.0 / 1e9 / elapsed
                            );
                        }
                    }
                }
            })
            .boxed();

    result_stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DbStoredIris;
    use aws_sdk_s3::primitives::SdkBody;
    use rand::Rng;
    use std::{cmp::min, collections::HashSet};

    #[derive(Default, Clone)]
    pub struct MockStore {
        objects: std::collections::HashMap<String, Vec<u8>>,
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
            }
            self.objects.insert(key.to_string(), result);
        }
    }

    #[async_trait]
    impl ObjectStore for MockStore {
        async fn get_object(&self, key: &str, range: (usize, usize)) -> eyre::Result<ByteStream> {
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

        async fn list_objects(&self, _: &str) -> eyre::Result<Vec<String>> {
            Ok(self.objects.keys().cloned().collect())
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
            id:         id as i64,
            left_code:  random_bytes(IRIS_CODE_LENGTH * mem::size_of::<u16>()),
            left_mask:  random_bytes(MASK_CODE_LENGTH * mem::size_of::<u16>()),
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
            timestamp:      0,
            last_serial_id: MOCK_ENTRIES as i64,
            chunk_size:     MOCK_CHUNK_SIZE as i64,
        };
        let mut chunks =
            fetch_and_parse_chunks(&store, 1, "out".to_string(), last_snapshot_details).await;
        let mut count = 0;
        let mut ids: HashSet<usize> = HashSet::from_iter(1..MOCK_ENTRIES);
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            ids.remove(&(chunk.index()));
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty());
    }
}
