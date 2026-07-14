//! Small immutable S3 coordination surface.

use std::collections::HashSet;
use std::time::Duration;

use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::Client;
use eyre::{ensure, eyre, Result};
use iris_mpc_store::rerand::RERAND_CHECK_PROTOCOL_VERSION;
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Instant};

use super::RerandStoreKind;

pub const PARTY_COUNT: u8 = 3;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30 * 60);
pub const OFFSET_GENERATION: u32 = 1;

pub fn validate_environment(environment: &str) -> Result<()> {
    ensure!(
        !environment.is_empty()
            && environment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "invalid rerandomization environment label"
    );
    Ok(())
}

pub fn validate_coordination_id(coordination_id: &str) -> Result<()> {
    ensure!(
        !coordination_id.is_empty()
            && coordination_id.len() <= 128
            && coordination_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "invalid rerandomization coordination ID"
    );
    Ok(())
}

fn prefix(environment: &str, coordination_id: &str, epoch: u32) -> String {
    format!("rerand-v2/{environment}/{coordination_id}/epoch-{epoch}")
}

pub fn public_key_key(environment: &str, coordination_id: &str, epoch: u32, party: u8) -> String {
    format!(
        "{}/party-{party}/public-key",
        prefix(environment, coordination_id, epoch)
    )
}

pub fn commitment_key(environment: &str, coordination_id: &str, epoch: u32, party: u8) -> String {
    format!(
        "{}/party-{party}/seed-commitment",
        prefix(environment, coordination_id, epoch)
    )
}

fn completion_key(
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    party: u8,
    kind: RerandStoreKind,
) -> String {
    format!(
        "{}/party-{party}/store-{kind}-complete",
        prefix(environment, coordination_id, epoch)
    )
}

fn epoch_check_key(
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    kind: RerandStoreKind,
) -> String {
    format!(
        "{}/store-{kind}-check-passed",
        prefix(environment, coordination_id, epoch)
    )
}

pub async fn put_immutable(client: &Client, bucket: &str, key: &str, body: &[u8]) -> Result<()> {
    match client
        .put_object()
        .bucket(bucket)
        .key(key)
        .if_none_match("*")
        .body(body.to_vec().into())
        .send()
        .await
    {
        Ok(_) => Ok(()),
        Err(put_error) => match get(client, bucket, key).await {
            Ok(existing) if existing == body => Ok(()),
            Ok(_) => Err(eyre!(
                "immutable S3 marker {key} already has different contents"
            )),
            Err(read_error) => Err(eyre!(
                "failed to create immutable S3 marker {key}: {put_error}; reread: {read_error}"
            )),
        },
    }
}

pub async fn get(client: &Client, bucket: &str, key: &str) -> Result<Vec<u8>> {
    let object = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("S3 GetObject {key}: {e}"))?;
    Ok(object
        .body
        .collect()
        .await
        .map_err(|e| eyre!("reading S3 marker {key}: {e}"))?
        .to_vec())
}

async fn get_optional(client: &Client, bucket: &str, key: &str) -> Result<Option<Vec<u8>>> {
    match client.get_object().bucket(bucket).key(key).send().await {
        Ok(object) => Ok(Some(
            object
                .body
                .collect()
                .await
                .map_err(|e| eyre!("reading S3 marker {key}: {e}"))?
                .to_vec(),
        )),
        Err(error) => {
            let service = error.into_service_error();
            if matches!(service.code(), Some("NoSuchKey" | "NotFound")) {
                Ok(None)
            } else {
                Err(eyre!("S3 GetObject {key}: {service}"))
            }
        }
    }
}

pub async fn wait_for(
    client: &Client,
    bucket: &str,
    key: &str,
    poll_interval: Duration,
) -> Result<Vec<u8>> {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        if let Some(body) = get_optional(client, bucket, key).await? {
            return Ok(body);
        }
        ensure!(
            Instant::now() < deadline,
            "timed out waiting for S3 marker {key}"
        );
        sleep(poll_interval).await;
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    pub environment: String,
    pub coordination_id: String,
    pub offset_generation: u32,
    pub epoch: u32,
    pub party_id: u8,
    pub store_kind: RerandStoreKind,
    pub store_id: String,
    pub writer_role: String,
    pub seed_commitment: String,
}

/// Durable common evidence that one participant reconstructed a successful
/// private persisted-data check. Every participant writes the same body. The
/// marker deliberately contains no syndrome, challenge, or row inventory.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EpochCheckPassed {
    pub environment: String,
    pub coordination_id: String,
    pub offset_generation: u32,
    pub check_protocol: u32,
    pub epoch: u32,
    pub store_kind: RerandStoreKind,
    pub store_registry_commitment: String,
    pub seed_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StoreBinding {
    pub party_id: u8,
    pub store_kind: RerandStoreKind,
    pub store_id: String,
    pub writer_role: String,
}

#[derive(Debug, Clone)]
pub struct StoreRegistry(Vec<StoreBinding>);

impl StoreRegistry {
    pub fn parse(json: &str) -> Result<Self> {
        let entries: Vec<StoreBinding> = serde_json::from_str(json)?;
        ensure!(
            entries.len() == 6,
            "store registry must contain exactly six entries"
        );
        let mut slots = HashSet::new();
        let mut store_ids = HashSet::new();
        for entry in &entries {
            let valid_store_id = entry.store_id.len() <= 128
                && entry
                    .store_id
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric())
                && entry
                    .store_id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'));
            let valid_role = !entry.writer_role.is_empty()
                && entry.writer_role.len() <= 63
                && entry
                    .writer_role
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
            ensure!(
                entry.party_id < PARTY_COUNT && valid_store_id && valid_role,
                "invalid store registry entry"
            );
            ensure!(
                slots.insert((entry.party_id, entry.store_kind)),
                "duplicate party/store-kind registry entry"
            );
            ensure!(
                store_ids.insert(entry.store_id.as_str()),
                "store IDs must be globally unique"
            );
        }
        Ok(Self(entries))
    }

    pub fn expected(&self, party: u8, kind: RerandStoreKind) -> Result<&StoreBinding> {
        self.0
            .iter()
            .find(|entry| entry.party_id == party && entry.store_kind == kind)
            .ok_or_else(|| eyre!("missing party/store-kind registry entry"))
    }

    /// Canonical public binding for the exact three physical stores taking
    /// part in one same-kind epoch check.
    pub fn store_kind_commitment(&self, kind: RerandStoreKind) -> Result<[u8; 32]> {
        let mut hasher =
            blake3::Hasher::new_derive_key("iris-mpc/rerand-v2/epoch-check-store-registry/v1");
        hasher.update(&[match kind {
            RerandStoreKind::Gpu => 0,
            RerandStoreKind::Hnsw => 1,
        }]);
        for party_id in 0..PARTY_COUNT {
            let entry = self.expected(party_id, kind)?;
            hasher.update(&[party_id]);
            for value in [&entry.store_id, &entry.writer_role] {
                hasher.update(&(value.len() as u64).to_le_bytes());
                hasher.update(value.as_bytes());
            }
        }
        Ok(*hasher.finalize().as_bytes())
    }
}

/// Immutable inputs shared by epoch completion and persisted-data checks.
#[derive(Clone, Copy)]
pub struct EpochCoordinationBinding<'a> {
    environment: &'a str,
    coordination_id: &'a str,
    epoch: u32,
    expected_commitment: &'a [u8; 32],
    registry: &'a StoreRegistry,
}

impl<'a> EpochCoordinationBinding<'a> {
    pub fn new(
        environment: &'a str,
        coordination_id: &'a str,
        epoch: u32,
        expected_commitment: &'a [u8; 32],
        registry: &'a StoreRegistry,
    ) -> Self {
        Self {
            environment,
            coordination_id,
            epoch,
            expected_commitment,
            registry,
        }
    }
}

pub async fn publish_completion(
    client: &Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    completion: &Completion,
) -> Result<()> {
    ensure!(
        completion.environment == environment
            && completion.coordination_id == coordination_id
            && completion.offset_generation == OFFSET_GENERATION,
        "completion marker generation mismatch"
    );
    let body = serde_json::to_vec(completion)?;
    put_immutable(
        client,
        bucket,
        &completion_key(
            environment,
            coordination_id,
            completion.epoch,
            completion.party_id,
            completion.store_kind,
        ),
        &body,
    )
    .await
}

fn validate_epoch_check_passed(
    marker: &EpochCheckPassed,
    key: &str,
    binding: EpochCoordinationBinding<'_>,
    store_kind: RerandStoreKind,
    expected_registry_commitment: &str,
    expected_commitment: &str,
) -> Result<()> {
    ensure!(
        marker.environment == binding.environment
            && marker.coordination_id == binding.coordination_id
            && marker.offset_generation == OFFSET_GENERATION
            && marker.check_protocol == RERAND_CHECK_PROTOCOL_VERSION
            && marker.epoch == binding.epoch
            && marker.store_kind == store_kind
            && marker.store_registry_commitment == expected_registry_commitment
            && marker.seed_commitment == expected_commitment,
        "invalid epoch-check marker {key}"
    );
    Ok(())
}

pub async fn publish_epoch_check_passed(
    client: &Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    marker: &EpochCheckPassed,
) -> Result<()> {
    ensure!(
        marker.environment == environment
            && marker.coordination_id == coordination_id
            && marker.offset_generation == OFFSET_GENERATION
            && marker.check_protocol == RERAND_CHECK_PROTOCOL_VERSION,
        "epoch-check marker generation mismatch"
    );
    let body = serde_json::to_vec(marker)?;
    put_immutable(
        client,
        bucket,
        &epoch_check_key(
            environment,
            coordination_id,
            marker.epoch,
            marker.store_kind,
        ),
        &body,
    )
    .await
}

/// Return whether the immutable common success marker exists and exactly
/// matches this epoch. Under the protocol's honest-but-curious model, any one
/// participant that reconstructed zero proves all three joined the check.
pub async fn epoch_check_passed(
    client: &Client,
    bucket: &str,
    binding: EpochCoordinationBinding<'_>,
    store_kind: RerandStoreKind,
) -> Result<bool> {
    let expected_commitment = hex::encode(binding.expected_commitment);
    let expected_registry_commitment =
        hex::encode(binding.registry.store_kind_commitment(store_kind)?);
    let key = epoch_check_key(
        binding.environment,
        binding.coordination_id,
        binding.epoch,
        store_kind,
    );
    let Some(body) = get_optional(client, bucket, &key).await? else {
        return Ok(false);
    };
    let marker: EpochCheckPassed = serde_json::from_slice(&body)?;
    validate_epoch_check_passed(
        &marker,
        &key,
        binding,
        store_kind,
        &expected_registry_commitment,
        &expected_commitment,
    )?;
    Ok(true)
}

fn validate_completion(
    marker: &Completion,
    key: &str,
    binding: EpochCoordinationBinding<'_>,
    party_id: u8,
    store_kind: RerandStoreKind,
    expected: &StoreBinding,
    expected_commitment: &str,
) -> Result<()> {
    ensure!(
        marker.environment == binding.environment
            && marker.coordination_id == binding.coordination_id
            && marker.offset_generation == OFFSET_GENERATION
            && marker.epoch == binding.epoch
            && marker.party_id == party_id
            && marker.store_kind == store_kind
            && marker.store_id == expected.store_id
            && marker.writer_role == expected.writer_role
            && marker.seed_commitment == expected_commitment,
        "invalid completion marker {key}"
    );
    Ok(())
}

/// Require exact completion by GPU and HNSW stores of all three parties.
pub async fn wait_for_epoch_completion(
    client: &Client,
    bucket: &str,
    binding: EpochCoordinationBinding<'_>,
    poll_interval: Duration,
) -> Result<()> {
    let expected_commitment_hex = hex::encode(binding.expected_commitment);
    for party_id in 0..PARTY_COUNT {
        for store_kind in [RerandStoreKind::Gpu, RerandStoreKind::Hnsw] {
            let expected = binding.registry.expected(party_id, store_kind)?;
            let key = completion_key(
                binding.environment,
                binding.coordination_id,
                binding.epoch,
                party_id,
                store_kind,
            );
            let body = wait_for(client, bucket, &key, poll_interval).await?;
            let marker: Completion = serde_json::from_slice(&body)?;
            validate_completion(
                &marker,
                &key,
                binding,
                party_id,
                store_kind,
                expected,
                &expected_commitment_hex,
            )?;
        }
    }
    for store_kind in [RerandStoreKind::Gpu, RerandStoreKind::Hnsw] {
        ensure!(
            epoch_check_passed(client, bucket, binding, store_kind).await?,
            "epoch {} {store_kind} completed without a valid persisted-data check marker",
            binding.epoch
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_entries() -> Vec<StoreBinding> {
        let mut entries = Vec::new();
        for party_id in 0..PARTY_COUNT {
            for store_kind in [RerandStoreKind::Gpu, RerandStoreKind::Hnsw] {
                entries.push(StoreBinding {
                    party_id,
                    store_kind,
                    store_id: format!("party-{party_id}-{store_kind}"),
                    writer_role: format!("rerand_party_{party_id}_{store_kind}"),
                });
            }
        }
        entries
    }

    #[test]
    fn registry_requires_exact_unique_six_store_binding() {
        let entries = registry_entries();
        let registry = StoreRegistry::parse(&serde_json::to_string(&entries).unwrap()).unwrap();
        assert_eq!(
            registry
                .expected(2, RerandStoreKind::Hnsw)
                .unwrap()
                .store_id,
            "party-2-hnsw"
        );

        let mut duplicate = entries.clone();
        duplicate[5].store_id = duplicate[0].store_id.clone();
        assert!(StoreRegistry::parse(&serde_json::to_string(&duplicate).unwrap()).is_err());
        assert!(StoreRegistry::parse(&serde_json::to_string(&entries[..5]).unwrap()).is_err());
    }

    #[test]
    fn same_kind_registry_commitment_is_canonical_and_exact() {
        let entries = registry_entries();
        let expected = StoreRegistry::parse(&serde_json::to_string(&entries).unwrap()).unwrap();
        let mut reordered = entries.clone();
        reordered.reverse();
        let reordered = StoreRegistry::parse(&serde_json::to_string(&reordered).unwrap()).unwrap();
        assert_eq!(
            expected
                .store_kind_commitment(RerandStoreKind::Gpu)
                .unwrap(),
            reordered
                .store_kind_commitment(RerandStoreKind::Gpu)
                .unwrap()
        );
        assert_ne!(
            expected
                .store_kind_commitment(RerandStoreKind::Gpu)
                .unwrap(),
            expected
                .store_kind_commitment(RerandStoreKind::Hnsw)
                .unwrap()
        );

        let mut changed = entries;
        changed[0].writer_role.push_str("_replacement");
        let changed = StoreRegistry::parse(&serde_json::to_string(&changed).unwrap()).unwrap();
        assert_ne!(
            expected
                .store_kind_commitment(RerandStoreKind::Gpu)
                .unwrap(),
            changed.store_kind_commitment(RerandStoreKind::Gpu).unwrap()
        );
    }

    #[test]
    fn common_epoch_check_marker_binds_protocol_registry_and_seed() {
        let registry =
            StoreRegistry::parse(&serde_json::to_string(&registry_entries()).unwrap()).unwrap();
        let expected_commitment = [9; 32];
        let binding = EpochCoordinationBinding::new(
            "stage",
            "generation-7",
            4,
            &expected_commitment,
            &registry,
        );
        let registry_commitment = hex::encode(
            registry
                .store_kind_commitment(RerandStoreKind::Gpu)
                .unwrap(),
        );
        let marker = EpochCheckPassed {
            environment: "stage".into(),
            coordination_id: "generation-7".into(),
            offset_generation: OFFSET_GENERATION,
            check_protocol: RERAND_CHECK_PROTOCOL_VERSION,
            epoch: 4,
            store_kind: RerandStoreKind::Gpu,
            store_registry_commitment: registry_commitment.clone(),
            seed_commitment: hex::encode(expected_commitment),
        };
        validate_epoch_check_passed(
            &marker,
            "key",
            binding,
            RerandStoreKind::Gpu,
            &registry_commitment,
            &hex::encode(expected_commitment),
        )
        .unwrap();

        let mut wrong_protocol = marker;
        wrong_protocol.check_protocol += 1;
        assert!(validate_epoch_check_passed(
            &wrong_protocol,
            "key",
            binding,
            RerandStoreKind::Gpu,
            &registry_commitment,
            &hex::encode(expected_commitment),
        )
        .is_err());
    }
}
