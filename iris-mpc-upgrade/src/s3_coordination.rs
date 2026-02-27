use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::{sleep, Instant};

const NUM_PARTIES: u8 = 3;
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub epoch: u32,
    pub chunk_size: u64,
    pub max_id_inclusive: u64,
}

impl Manifest {
    pub fn num_chunks(&self) -> u32 {
        if self.max_id_inclusive == 0 {
            return 0;
        }
        self.max_id_inclusive.div_ceil(self.chunk_size) as u32
    }

    /// Returns (start_id_inclusive, end_id_exclusive) for a given chunk_id.
    /// IDs are 1-based.
    pub fn chunk_range(&self, chunk_id: u32) -> (u64, u64) {
        let start = 1 + (chunk_id as u64) * self.chunk_size;
        let end = std::cmp::min(start + self.chunk_size, self.max_id_inclusive + 1);
        (start, end)
    }

    pub fn chunk_is_empty(&self, chunk_id: u32) -> bool {
        let (start, end) = self.chunk_range(chunk_id);
        start >= end
    }
}

fn epoch_party_prefix(epoch: u32, party: u8) -> String {
    format!("rerand/epoch-{}/party-{}", epoch, party)
}

pub async fn upload_marker(s3: &S3Client, bucket: &str, key: &str, body: Vec<u8>) -> Result<()> {
    s3.put_object()
        .bucket(bucket)
        .key(key)
        .body(body.into())
        .send()
        .await
        .map_err(|e| eyre!("S3 PutObject failed for key {}: {}", key, e))?;
    Ok(())
}

pub async fn marker_exists(s3: &S3Client, bucket: &str, key: &str) -> Result<bool> {
    match s3.head_object().bucket(bucket).key(key).send().await {
        Ok(_) => Ok(true),
        Err(e) => {
            let svc_err = e.into_service_error();
            if svc_err.is_not_found() {
                Ok(false)
            } else {
                Err(eyre!("S3 HeadObject failed for key {}: {}", key, svc_err))
            }
        }
    }
}

pub async fn download_marker(s3: &S3Client, bucket: &str, key: &str) -> Result<Vec<u8>> {
    let resp = s3
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("S3 GetObject failed for key {}: {}", key, e))?;
    let bytes = resp
        .body
        .collect()
        .await
        .map_err(|e| eyre!("Failed to read S3 body for key {}: {}", key, e))?;
    Ok(bytes.to_vec())
}

pub async fn poll_until_marker_exists(
    s3: &S3Client,
    bucket: &str,
    key: &str,
    poll_interval: Duration,
) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_POLL_TIMEOUT;
    loop {
        if marker_exists(s3, bucket, key).await? {
            return Ok(());
        }
        if Instant::now() > deadline {
            eyre::bail!(
                "Timeout after {:?} waiting for S3 marker: {}",
                DEFAULT_POLL_TIMEOUT,
                key
            );
        }
        tracing::debug!("Waiting for S3 marker: {}", key);
        sleep(poll_interval).await;
    }
}

/// Polls until all three parties have uploaded a given marker suffix for an epoch.
pub async fn poll_until_all_parties_marker(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    marker_suffix: &str,
    poll_interval: Duration,
) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_POLL_TIMEOUT;
    loop {
        let mut all_present = true;
        for party in 0..NUM_PARTIES {
            let key = format!("{}/{}", epoch_party_prefix(epoch, party), marker_suffix);
            if !marker_exists(s3, bucket, &key).await? {
                all_present = false;
                break;
            }
        }
        if all_present {
            return Ok(());
        }
        if Instant::now() > deadline {
            eyre::bail!(
                "Timeout after {:?} waiting for all parties' {} markers for epoch {}",
                DEFAULT_POLL_TIMEOUT,
                marker_suffix,
                epoch
            );
        }
        tracing::debug!(
            "Waiting for all parties' {} markers for epoch {}",
            marker_suffix,
            epoch
        );
        sleep(poll_interval).await;
    }
}

// ---- Public key ----

pub async fn upload_public_key(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    key_b64: &str,
) -> Result<()> {
    let key = format!("{}/public-key", epoch_party_prefix(epoch, party));
    upload_marker(s3, bucket, &key, key_b64.as_bytes().to_vec()).await
}

pub async fn download_public_key_for_party(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    poll_interval: Duration,
) -> Result<String> {
    let key = format!("{}/public-key", epoch_party_prefix(epoch, party));
    poll_until_marker_exists(s3, bucket, &key, poll_interval).await?;
    let bytes = download_marker(s3, bucket, &key).await?;
    Ok(String::from_utf8(bytes)?)
}

// ---- Max ID watermark ----

pub async fn upload_max_id(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    max_id: u64,
) -> Result<()> {
    let key = format!("{}/max-id", epoch_party_prefix(epoch, party));
    upload_marker(s3, bucket, &key, max_id.to_string().into_bytes()).await
}

pub async fn download_all_max_ids(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    poll_interval: Duration,
) -> Result<[u64; 3]> {
    let mut ids = [0u64; 3];
    for party in 0..NUM_PARTIES {
        let key = format!("{}/max-id", epoch_party_prefix(epoch, party));
        poll_until_marker_exists(s3, bucket, &key, poll_interval).await?;
        let bytes = download_marker(s3, bucket, &key).await?;
        let s = String::from_utf8(bytes)?;
        ids[party as usize] = s
            .trim()
            .parse()
            .map_err(|e| eyre!("Failed to parse max-id from party {}: {}", party, e))?;
    }
    Ok(ids)
}

// ---- Manifest ----

pub async fn upload_manifest(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    manifest: &Manifest,
) -> Result<()> {
    let key = format!("{}/manifest.json", epoch_party_prefix(epoch, 0));
    let body = serde_json::to_vec(manifest)?;
    upload_marker(s3, bucket, &key, body).await
}

pub async fn download_manifest(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    poll_interval: Duration,
) -> Result<Manifest> {
    let key = format!("{}/manifest.json", epoch_party_prefix(epoch, 0));
    poll_until_marker_exists(s3, bucket, &key, poll_interval).await?;
    let bytes = download_marker(s3, bucket, &key).await?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    Ok(manifest)
}

pub async fn manifest_exists(s3: &S3Client, bucket: &str, epoch: u32) -> Result<bool> {
    let key = format!("{}/manifest.json", epoch_party_prefix(epoch, 0));
    marker_exists(s3, bucket, &key).await
}

// ---- Chunk staged markers ----

pub async fn upload_chunk_staged(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    chunk_id: u32,
) -> Result<()> {
    let key = format!(
        "{}/chunk-{}/staged",
        epoch_party_prefix(epoch, party),
        chunk_id
    );
    upload_marker(s3, bucket, &key, b"ok".to_vec()).await
}

pub async fn poll_chunk_staged_all(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    chunk_id: u32,
    poll_interval: Duration,
) -> Result<()> {
    let suffix = format!("chunk-{}/staged", chunk_id);
    poll_until_all_parties_marker(s3, bucket, epoch, &suffix, poll_interval).await
}

// ---- Chunk version map (modification fence) ----

fn version_map_hash(version_map: &[(i64, i16)]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for (id, ver) in version_map {
        hasher.update(&id.to_le_bytes());
        hasher.update(&ver.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Upload the version map and its blake3 hash for a chunk.
pub async fn upload_chunk_version_map(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    chunk_id: u32,
    version_map: &[(i64, i16)],
) -> Result<()> {
    let prefix = format!("{}/chunk-{}", epoch_party_prefix(epoch, party), chunk_id);

    let hash = version_map_hash(version_map);
    upload_marker(s3, bucket, &format!("{prefix}/version-hash"), hash.to_vec()).await?;

    let body = serde_json::to_vec(version_map)?;
    upload_marker(s3, bucket, &format!("{prefix}/version-map"), body).await
}

async fn download_chunk_version_hash(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    chunk_id: u32,
    poll_interval: Duration,
) -> Result<[u8; 32]> {
    let key = format!(
        "{}/chunk-{}/version-hash",
        epoch_party_prefix(epoch, party),
        chunk_id
    );
    poll_until_marker_exists(s3, bucket, &key, poll_interval).await?;
    let bytes = download_marker(s3, bucket, &key).await?;
    let hash: [u8; 32] = bytes
        .try_into()
        .map_err(|b: Vec<u8>| eyre!("version-hash has wrong length: {}", b.len()))?;
    Ok(hash)
}

async fn download_chunk_version_map(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
    chunk_id: u32,
    poll_interval: Duration,
) -> Result<Vec<(i64, i16)>> {
    let key = format!(
        "{}/chunk-{}/version-map",
        epoch_party_prefix(epoch, party),
        chunk_id
    );
    poll_until_marker_exists(s3, bucket, &key, poll_interval).await?;
    let bytes = download_marker(s3, bucket, &key).await?;
    let map: Vec<(i64, i16)> = serde_json::from_slice(&bytes)?;
    Ok(map)
}

/// Compare version maps across all 3 parties and return IDs where any
/// party disagrees on the `original_version_id`.
///
/// Fast path: download only the 32-byte blake3 hashes. If all match,
/// return empty (no disagreements). Slow path (hash mismatch): download
/// the full maps and compute the exact disagreement set.
pub async fn compute_cross_party_divergent_ids(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    chunk_id: u32,
    poll_interval: Duration,
) -> Result<Vec<i64>> {
    let mut hashes = Vec::new();
    for party in 0..NUM_PARTIES {
        hashes.push(
            download_chunk_version_hash(s3, bucket, epoch, party, chunk_id, poll_interval).await?,
        );
    }
    if hashes[0] == hashes[1] && hashes[1] == hashes[2] {
        return Ok(Vec::new());
    }

    tracing::info!(
        "Epoch {} chunk {}: version-map hashes differ, downloading full maps",
        epoch,
        chunk_id,
    );

    use std::collections::HashMap;
    let mut all_maps: Vec<HashMap<i64, i16>> = Vec::new();
    for party in 0..NUM_PARTIES {
        let map =
            download_chunk_version_map(s3, bucket, epoch, party, chunk_id, poll_interval).await?;
        all_maps.push(map.into_iter().collect());
    }

    let mut divergent = Vec::new();
    let all_ids: std::collections::BTreeSet<i64> =
        all_maps.iter().flat_map(|m| m.keys().copied()).collect();

    for id in all_ids {
        let versions: Vec<Option<&i16>> = all_maps.iter().map(|m| m.get(&id)).collect();
        let first = versions[0];
        if !versions.iter().all(|v| *v == first) {
            divergent.push(id);
        }
    }
    Ok(divergent)
}

// ---- Epoch completion ----

pub async fn upload_epoch_complete(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    party: u8,
) -> Result<()> {
    let key = format!("{}/complete", epoch_party_prefix(epoch, party));
    upload_marker(s3, bucket, &key, b"done".to_vec()).await
}

pub async fn all_parties_complete(s3: &S3Client, bucket: &str, epoch: u32) -> Result<bool> {
    for party in 0..NUM_PARTIES {
        let key = format!("{}/complete", epoch_party_prefix(epoch, party));
        if !marker_exists(s3, bucket, &key).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub async fn poll_epoch_complete_all(
    s3: &S3Client,
    bucket: &str,
    epoch: u32,
    poll_interval: Duration,
) -> Result<()> {
    poll_until_all_parties_marker(s3, bucket, epoch, "complete", poll_interval).await
}
