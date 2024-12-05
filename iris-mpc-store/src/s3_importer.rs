use crate::StoredIris;
use async_trait::async_trait;
use aws_sdk_s3::Client;
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use rayon::{iter::ParallelIterator, prelude::ParallelBridge};
use serde::Deserialize;
use std::{io::Cursor, mem, pin::Pin};
use tokio::task;
use tracing::log;

const SINGLE_ELEMENT_SIZE: usize = IRIS_CODE_LENGTH * mem::size_of::<u16>() * 2
    + MASK_CODE_LENGTH * mem::size_of::<u16>() * 2
    + mem::size_of::<u32>(); // 75 KB
const CSV_BUFFER_CAPACITY: usize = SINGLE_ELEMENT_SIZE * 10;

#[async_trait]
pub trait ObjectStore: Send + Sync + 'static {
    async fn get_object(&self, key: &str) -> eyre::Result<Bytes>;
    async fn list_objects(&self) -> eyre::Result<Vec<String>>;
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
    async fn get_object(&self, key: &str) -> eyre::Result<Bytes> {
        let result = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?;

        let data = result.body.collect().await?;
        Ok(data.into_bytes())
    }

    async fn list_objects(&self) -> eyre::Result<Vec<String>> {
        let mut objects = Vec::new();
        let mut continuation_token = None;

        loop {
            let mut request = self.client.list_objects_v2().bucket(&self.bucket);

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

#[derive(Debug, Deserialize)]
struct CsvIrisRecord {
    id:         String,
    left_code:  String,
    left_mask:  String,
    right_code: String,
    right_mask: String,
}

fn hex_to_bytes(hex: &str, byte_len: usize) -> eyre::Result<Vec<u8>> {
    if hex.is_empty() {
        return Ok(vec![]);
    }
    let mut bytes = vec![0; byte_len];
    hex::decode_to_slice(hex, &mut bytes)?;
    Ok(bytes)
}

pub async fn last_snapshot_timestamp(store: &impl ObjectStore) -> eyre::Result<i64> {
    let objects = store.list_objects().await?;
    tracing::info!("All objects in db chunks s3: {:?}", objects);
    objects
        .into_iter()
        .filter(|f| f.ends_with(".timestamp"))
        .filter_map(|f| f.replace(".timestamp", "").parse::<i64>().ok())
        .max()
        .ok_or_else(|| eyre::eyre!("No snapshot found"))
}

pub async fn fetch_and_parse_chunks(
    store: &impl ObjectStore,
    concurrency: usize,
) -> Pin<Box<dyn Stream<Item = eyre::Result<StoredIris>> + Send + '_>> {
    let chunks = store.list_objects().await.unwrap();
    stream::iter(chunks)
        .filter_map(|chunk| async move {
            if chunk.ends_with(".csv") {
                Some(chunk)
            } else {
                None
            }
        })
        .map(move |chunk| async move {
            let result = store.get_object(&chunk).await?;
            task::spawn_blocking(move || {
                let cursor = Cursor::new(result);
                let reader = csv::ReaderBuilder::new()
                    .has_headers(true)
                    .buffer_capacity(CSV_BUFFER_CAPACITY)
                    .from_reader(cursor);

                let records: Vec<eyre::Result<StoredIris>> = reader
                    .into_deserialize()
                    .par_bridge()
                    .map(|r: Result<CsvIrisRecord, _>| {
                        let raw = r.map_err(|e| eyre::eyre!("CSV parse error: {}", e))?;

                        Ok(StoredIris {
                            id:         raw.id.parse()?,
                            left_code:  hex_to_bytes(
                                &raw.left_code,
                                IRIS_CODE_LENGTH * mem::size_of::<u16>(),
                            )?,
                            left_mask:  hex_to_bytes(
                                &raw.left_mask,
                                MASK_CODE_LENGTH * mem::size_of::<u16>(),
                            )?,
                            right_code: hex_to_bytes(
                                &raw.right_code,
                                IRIS_CODE_LENGTH * mem::size_of::<u16>(),
                            )?,
                            right_mask: hex_to_bytes(
                                &raw.right_mask,
                                MASK_CODE_LENGTH * mem::size_of::<u16>(),
                            )?,
                        })
                    })
                    .collect();

                Ok::<_, eyre::Error>(stream::iter(records))
            })
            .await?
        })
        .buffer_unordered(concurrency)
        .flat_map(|result| match result {
            Ok(stream) => stream.boxed(),
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        })
        .boxed()
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

        pub fn add_test_data(&mut self, key: &str, records: Vec<StoredIris>) {
            let mut csv = Vec::new();
            {
                let mut writer = csv::Writer::from_writer(&mut csv);
                writer
                    .write_record(["id", "left_code", "left_mask", "right_code", "right_mask"])
                    .unwrap();

                for record in records {
                    writer
                        .write_record(&[
                            record.id.to_string(),
                            hex::encode(record.left_code),
                            hex::encode(record.left_mask),
                            hex::encode(record.right_code),
                            hex::encode(record.right_mask),
                        ])
                        .unwrap();
                }
            }
            self.objects.insert(key.to_string(), csv);
        }
    }

    #[async_trait]
    impl ObjectStore for MockStore {
        async fn get_object(&self, key: &str) -> eyre::Result<Bytes> {
            self.objects
                .get(key)
                .cloned()
                .map(Bytes::from)
                .ok_or_else(|| eyre::eyre!("Object not found: {}", key))
        }

        async fn list_objects(&self) -> eyre::Result<Vec<String>> {
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
    async fn test_fetch_and_parse_chunks() {
        const MOCK_ENTRIES: usize = 107;
        const MOCK_CHUNK_SIZE: usize = 10;
        let mut store = MockStore::new();

        for i in 0..MOCK_ENTRIES.div_ceil(MOCK_CHUNK_SIZE) {
            let start_idx = i * MOCK_CHUNK_SIZE;
            let end_idx = min((i + 1) * MOCK_CHUNK_SIZE, MOCK_ENTRIES) - 1;
            store.add_test_data(
                &format!("{start_idx}_{end_idx}.bin"),
                (start_idx..=end_idx).map(dummy_entry).collect(),
            );
        }

        assert_eq!(
            store.list_objects().await.unwrap().len(),
            MOCK_ENTRIES.div_ceil(MOCK_CHUNK_SIZE)
        );

        let mut chunks = fetch_and_parse_chunks(&store, 1).await;
        let mut count = 0;
        let mut ids: HashSet<usize> = HashSet::from_iter(0..MOCK_ENTRIES);
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            ids.remove(&(chunk.id as usize));
            count += 1;
        }
        assert_eq!(count, MOCK_ENTRIES);
        assert!(ids.is_empty());
    }
}
