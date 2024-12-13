use crate::StoredIris;
use async_trait::async_trait;
use aws_sdk_s3::{config::Region, Client as S3Client};
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{cmp::min, mem, pin::Pin, time::Instant};
use tokio::task;

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>(); // 75 KB

#[async_trait]
pub trait ObjectStore: Send + Sync + 'static {
    async fn get_object(&self, key: &str, client_idx: usize) -> eyre::Result<Bytes>;
    async fn list_objects(&self, prefix: &str) -> eyre::Result<Vec<String>>;
}

pub struct S3Store {
    clients:   Vec<S3Client>,
    bucket:    String,
    n_clients: usize,
}

impl S3Store {
    pub async fn new(n_clients: usize, bucket: String) -> Self {
        let client_futures: Vec<_> = (0..n_clients)
            .map(|_| async move {
                let region_provider = Region::new("eu-north-1");
                let config = aws_config::from_env().region(region_provider).load().await;

                S3Client::new(&config)
            })
            .collect();
        let clients = futures::future::join_all(client_futures).await;
        Self {
            clients,
            bucket,
            n_clients,
        }
    }
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn get_object(&self, key: &str, client_idx: usize) -> eyre::Result<Bytes> {
        let client_idx_modulo = client_idx % self.n_clients;
        let result = self.clients[client_idx_modulo]
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?;

        let data = result.body.collect().await?;
        Ok(data.into_bytes())
    }

    async fn list_objects(&self, prefix: &str) -> eyre::Result<Vec<String>> {
        let mut objects = Vec::new();
        let mut continuation_token = None;

        loop {
            let mut request = self.clients[0]
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

#[derive(Debug, Copy, Clone)]
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
    partition_size: i64,
) -> impl Stream<Item = eyre::Result<StoredIris>> + '_ {
    tracing::info!("Generating chunk files using: {:?}", last_snapshot_details);
    let n_partitions = last_snapshot_details
        .last_serial_id
        .div_ceil(partition_size);
    let partition_concurrency = concurrency.div_ceil(n_partitions as usize);
    let mut results = Vec::with_capacity(n_partitions as usize);
    for partition in 0..n_partitions {
        tracing::info!("Initiating stream for partition: {}", partition);
        let stream = fetch_chunks_for_partition(
            store,
            prefix_name.clone(),
            partition,
            partition_size,
            last_snapshot_details,
            partition_concurrency,
        )
        .await;
        results.push(stream);
        tracing::info!("Initiated stream for partition: {}", partition);
    }

    stream::select_all(results)
}

async fn fetch_chunks_for_partition(
    store: &impl ObjectStore,
    prefix_name: String,
    partition: i64,
    partition_size: i64,
    last_snapshot_details: LastSnapshotDetails,
    concurrency: usize,
) -> Pin<Box<dyn Stream<Item = eyre::Result<StoredIris>> + Send + '_>> {
    let first_serial_id = partition * partition_size + 1;
    let last_serial_id = min(
        (partition + 1) * partition_size,
        last_snapshot_details.last_serial_id,
    );
    tracing::info!(
        "Partition {} will work on range [{},{}] with {} concurrency",
        partition,
        first_serial_id,
        last_serial_id,
        concurrency,
    );
    let chunks: Vec<String> = (first_serial_id..=last_serial_id)
        .step_by(last_snapshot_details.chunk_size as usize)
        .map(|num| format!("{}/{}/{}.bin", prefix_name, partition, num))
        .collect();
    let result_stream = stream::iter(chunks)
        .map(move |chunk| async move {
            let mut now = Instant::now();
            let result = store.get_object(&chunk, partition as usize).await?;
            let get_object_time = now.elapsed();
            tracing::info!("Got chunk object: {} in {:?}", chunk, get_object_time,);

            now = Instant::now();
            let task = task::spawn_blocking(move || {
                let n_records = result.len().div_floor(SINGLE_ELEMENT_SIZE);

                let mut records = Vec::with_capacity(n_records);
                for i in 0..n_records {
                    let start = i * SINGLE_ELEMENT_SIZE;
                    let end = (i + 1) * SINGLE_ELEMENT_SIZE;
                    let chunk = &result[start..end];
                    let iris = StoredIris::from_bytes(chunk);
                    records.push(iris);
                }

                Ok::<_, eyre::Error>(stream::iter(records))
            })
            .await?;
            let parse_time = now.elapsed();
            tracing::info!("Parsed chunk: {} in {:?}", chunk, parse_time,);
            task
        })
        .buffer_unordered(concurrency)
        .flat_map(|result| match result {
            Ok(stream) => stream.boxed(),
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        })
        .boxed();

    result_stream
}

#[cfg(test)]
mod tests {
    use super::*;
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

        pub fn add_test_data(&mut self, key: &str, records: Vec<StoredIris>) {
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
        async fn get_object(&self, key: &str, _: usize) -> eyre::Result<Bytes> {
            self.objects
                .get(key)
                .cloned()
                .map(Bytes::from)
                .ok_or_else(|| eyre::eyre!("Object not found: {}", key))
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

    fn dummy_entry(id: usize) -> StoredIris {
        StoredIris {
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
        const MOCK_PARTITION_SIZE: i64 = 20;
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
        let mut chunks = fetch_and_parse_chunks(
            &store,
            1,
            "out".to_string(),
            last_snapshot_details,
            MOCK_PARTITION_SIZE,
        )
        .await;
        let mut count = 0;
        let mut ids: HashSet<usize> = HashSet::from_iter(1..MOCK_ENTRIES);
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            ids.remove(&(chunk.id as usize));
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty());
    }
}
