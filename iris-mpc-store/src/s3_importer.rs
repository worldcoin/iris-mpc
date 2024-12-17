use crate::StoredIris;
use async_trait::async_trait;
use aws_sdk_s3::{primitives::ByteStream, Client};
use futures::{stream, Stream, StreamExt};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::{mem, pin::Pin, sync::Arc};
use tokio::io::AsyncReadExt;

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>(); // 75 KB

#[async_trait]
pub trait ObjectStore: Send + Sync + 'static {
    async fn get_object(&self, key: &str) -> eyre::Result<ByteStream>;
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
    async fn get_object(&self, key: &str) -> eyre::Result<ByteStream> {
        let res = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?;

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
    let chunks: Vec<String> = (1..=last_snapshot_details.last_serial_id)
        .step_by(last_snapshot_details.chunk_size as usize)
        .map(|num| format!("{}/{}.bin", prefix_name, num))
        .collect();
    tracing::info!("Generated {} chunk names", chunks.len());

    let result_stream = stream::iter(chunks)
        .map(move |chunk| async move {
            let mut object_stream = store.get_object(&chunk).await?.into_async_read();
            let mut records = Vec::with_capacity(last_snapshot_details.chunk_size as usize);
            let mut buf = vec![0u8; SINGLE_ELEMENT_SIZE];
            loop {
                match object_stream.read_exact(&mut buf).await {
                    Ok(_) => {
                        let iris = StoredIris::from_bytes(&buf);
                        records.push(iris);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e.into()),
                }
            }

            Ok::<_, eyre::Error>(stream::iter(records))
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
        async fn get_object(&self, key: &str) -> eyre::Result<ByteStream> {
            let bytes = self
                .objects
                .get(key)
                .cloned()
                .ok_or_else(|| eyre::eyre!("Object not found: {}", key))?;
            Ok(ByteStream::from(SdkBody::from(bytes)))
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
            ids.remove(&(chunk.id as usize));
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty());
    }
}
