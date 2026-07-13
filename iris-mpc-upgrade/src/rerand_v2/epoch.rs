//! Crash-safe epoch seed creation and cross-party commitment verification.
//!
//! Seeds are intentionally retained. Deletion is a separate protocol and is
//! not implemented here; an operational error therefore cannot make rows
//! undecodable.

use std::collections::BTreeSet;
use std::time::Duration;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsClient;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use eyre::{ensure, eyre, Result};
use iris_mpc_common::rerand_offsets::{epoch_seed_from_dh, seed_commitment, EpochSeed};
use iris_mpc_store::rerand::{
    build_epoch_zero_rerand_context, get_rerand_epoch_inventory, verify_store_identity,
    RerandContext,
};
use iris_mpc_store::Store;

use super::coordination::{self, PARTY_COUNT};
use crate::tripartite_dh;

fn seed_secret_id(environment: &str, coordination_id: &str, epoch: u32, party: u8) -> String {
    format!("{environment}/iris-mpc/rerand-v2/{coordination_id}/epoch-{epoch}/seed-party-{party}")
}

fn scalar_secret_id(environment: &str, coordination_id: &str, epoch: u32, party: u8) -> String {
    format!("{environment}/iris-mpc/rerand-v2/{coordination_id}/epoch-{epoch}/scalar-party-{party}")
}

async fn load_secret(client: &SecretsClient, id: &str) -> Result<Option<Vec<u8>>> {
    match client
        .get_secret_value()
        .secret_id(id)
        .version_stage("AWSCURRENT")
        .send()
        .await
    {
        Ok(value) => {
            Ok(Some(STANDARD.decode(value.secret_string().ok_or_else(
                || eyre!("secret {id} has no string value"),
            )?)?))
        }
        Err(error) => {
            let service = error.into_service_error();
            if service.is_resource_not_found_exception() {
                Ok(None)
            } else {
                Err(eyre!("Secrets Manager GetSecretValue {id}: {service}"))
            }
        }
    }
}

/// Create-only. Returning false means another local worker won the race.
async fn create_secret(client: &SecretsClient, id: &str, value: &[u8]) -> Result<bool> {
    match client
        .create_secret()
        .name(id)
        .secret_string(STANDARD.encode(value))
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(error) => {
            let service = error.into_service_error();
            if service.is_resource_exists_exception() {
                Ok(false)
            } else {
                Err(eyre!("Secrets Manager CreateSecret {id}: {service}"))
            }
        }
    }
}

fn parse_seed(id: &str, bytes: Vec<u8>) -> Result<EpochSeed> {
    bytes
        .try_into()
        .map_err(|_| eyre!("secret {id} is not a 32-byte epoch seed"))
}

pub async fn load_epoch_seed(
    secrets: &SecretsClient,
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    party: u8,
) -> Result<EpochSeed> {
    ensure!(epoch > 0, "epoch zero has no seed");
    let id = seed_secret_id(environment, coordination_id, epoch, party);
    parse_seed(
        &id,
        load_secret(secrets, &id)
            .await?
            .ok_or_else(|| eyre!("required epoch seed {id} is missing"))?,
    )
}

async fn publish_and_verify_commitment(
    s3: &S3Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    party: u8,
    seed: &EpochSeed,
    poll_interval: Duration,
) -> Result<[u8; 32]> {
    let own = seed_commitment(seed);
    coordination::put_immutable(
        s3,
        bucket,
        &coordination::commitment_key(environment, coordination_id, epoch, party),
        hex::encode(own).as_bytes(),
    )
    .await?;
    for peer in 0..PARTY_COUNT {
        let key = coordination::commitment_key(environment, coordination_id, epoch, peer);
        let published = coordination::wait_for(s3, bucket, &key, poll_interval).await?;
        ensure!(
            published == hex::encode(own).as_bytes(),
            "seed commitment mismatch for epoch {epoch}, party {peer}"
        );
    }
    Ok(own)
}

/// Load a source seed and prove it matches every party's immutable marker.
pub async fn load_verified_epoch_seed(
    secrets: &SecretsClient,
    s3: &S3Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    party: u8,
) -> Result<EpochSeed> {
    let seed = load_epoch_seed(secrets, environment, coordination_id, epoch, party).await?;
    verify_commitments(s3, bucket, environment, coordination_id, epoch, &seed).await?;
    Ok(seed)
}

/// Serving is read-only with respect to the coordination namespace. A missing
/// marker is an incomplete epoch, never something serving credentials repair.
async fn verify_commitments(
    s3: &S3Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    seed: &EpochSeed,
) -> Result<[u8; 32]> {
    let expected = hex::encode(seed_commitment(seed));
    for peer in 0..PARTY_COUNT {
        let published = coordination::get(
            s3,
            bucket,
            &coordination::commitment_key(environment, coordination_id, epoch, peer),
        )
        .await?;
        ensure!(
            published == expected.as_bytes(),
            "seed commitment mismatch for epoch {epoch}, party {peer}"
        );
    }
    Ok(seed_commitment(seed))
}

/// Serving-side fail-closed preparation: bind the physical store identity,
/// enumerate every authoritative row epoch, commitment-verify every required
/// seed, then construct the fixed epoch-zero normalization context.
#[allow(clippy::too_many_arguments)]
pub async fn build_verified_context(
    store: &Store,
    expected_store_id: &str,
    expected_store_kind: &str,
    secrets: &SecretsClient,
    s3: &S3Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    party: u8,
    cache_epochs: &[u32],
) -> Result<RerandContext> {
    ensure!(party < PARTY_COUNT, "invalid party id");
    coordination::validate_environment(environment)?;
    coordination::validate_coordination_id(coordination_id)?;
    ensure!(!bucket.is_empty(), "rerandomization S3 bucket is empty");
    let state = verify_store_identity(
        &store.pool,
        expected_store_id,
        environment,
        coordination_id,
        party,
        expected_store_kind,
    )
    .await?;
    let inventory = get_rerand_epoch_inventory(&store.pool).await?;
    validate_authoritative_epochs(&inventory, &state)?;
    let mut seeds = std::collections::HashMap::new();
    let (mandatory, cache_only) =
        split_required_epochs(&inventory, state.active_epoch, cache_epochs)?;
    for epoch in mandatory {
        let seed = load_verified_epoch_seed(
            secrets,
            s3,
            bucket,
            environment,
            coordination_id,
            epoch,
            party,
        )
        .await?;
        let expected = if epoch == state.last_completed_epoch {
            state.last_seed_commitment
        } else if Some(epoch) == state.active_epoch {
            state.active_seed_commitment
        } else {
            None
        };
        ensure!(
            expected == Some(seed_commitment(&seed)),
            "database seed commitment mismatch for epoch {epoch}"
        );
        seeds.insert(epoch, seed);
    }
    for epoch in cache_only {
        match load_verified_epoch_seed(
            secrets, s3, bucket, environment, coordination_id, epoch, party,
        )
        .await
        {
            Ok(seed) => {
                seeds.insert(epoch, seed);
            }
            Err(error) => tracing::warn!(
                epoch,
                ?error,
                "cached rerandomization epoch is unavailable; affected rows will fall back to Aurora"
            ),
        }
    }
    build_epoch_zero_rerand_context(party as usize, seeds, &inventory)
}

fn split_required_epochs(
    inventory: &[(i32, i64)],
    active_epoch: Option<u32>,
    cache_epochs: &[u32],
) -> Result<(BTreeSet<u32>, BTreeSet<u32>)> {
    ensure!(
        inventory.iter().all(|(epoch, _)| *epoch >= 0),
        "negative authoritative row epoch"
    );
    ensure!(
        cache_epochs.iter().all(|epoch| *epoch <= i32::MAX as u32),
        "cached row epoch exceeds supported range"
    );
    let mandatory: BTreeSet<u32> = inventory
        .iter()
        .filter_map(|(epoch, _)| (*epoch > 0).then_some(*epoch as u32))
        .chain(active_epoch)
        .collect();
    let cache_only = cache_epochs
        .iter()
        .copied()
        .filter(|epoch| *epoch > 0 && !mandatory.contains(epoch))
        .collect();
    Ok((mandatory, cache_only))
}

fn validate_authoritative_epochs(
    inventory: &[(i32, i64)],
    state: &iris_mpc_store::rerand::RerandStoreState,
) -> Result<()> {
    for (epoch, _) in inventory {
        ensure!(*epoch >= 0, "negative authoritative row epoch");
        if *epoch > 0 {
            let epoch = *epoch as u32;
            ensure!(
                epoch == state.last_completed_epoch || Some(epoch) == state.active_epoch,
                "authoritative row references impossible epoch {epoch}"
            );
        }
    }
    ensure!(
        (state.last_completed_epoch == 0) == state.last_seed_commitment.is_none(),
        "completed epoch commitment state is inconsistent"
    );
    ensure!(
        state.active_epoch.is_some() == state.active_seed_commitment.is_some(),
        "active epoch commitment state is inconsistent"
    );
    Ok(())
}

/// Create or load this party's seed and verify all three commitments before
/// returning it to a writer.
pub async fn ensure_epoch_seed(
    secrets: &SecretsClient,
    s3: &S3Client,
    bucket: &str,
    environment: &str,
    coordination_id: &str,
    epoch: u32,
    party: u8,
    poll_interval: Duration,
) -> Result<(EpochSeed, [u8; 32])> {
    ensure!(party < PARTY_COUNT && epoch > 0, "invalid party or epoch");
    let seed_id = seed_secret_id(environment, coordination_id, epoch, party);
    let seed = if let Some(seed) = load_secret(secrets, &seed_id).await? {
        parse_seed(&seed_id, seed)?
    } else {
        let scalar_id = scalar_secret_id(environment, coordination_id, epoch, party);
        let private = match load_secret(secrets, &scalar_id).await? {
            Some(bytes) => tripartite_dh::PrivateKey::deserialize(&bytes)
                .map_err(|error| eyre!("invalid DH scalar {scalar_id}: {error:?}"))?,
            None => {
                let generated = tripartite_dh::PrivateKey::random(&mut rand::rngs::OsRng);
                if create_secret(secrets, &scalar_id, &generated.serialize()).await? {
                    generated
                } else {
                    let stored = load_secret(secrets, &scalar_id)
                        .await?
                        .ok_or_else(|| eyre!("concurrent scalar {scalar_id} disappeared"))?;
                    tripartite_dh::PrivateKey::deserialize(&stored)
                        .map_err(|error| eyre!("invalid DH scalar {scalar_id}: {error:?}"))?
                }
            }
        };

        coordination::put_immutable(
            s3,
            bucket,
            &coordination::public_key_key(environment, coordination_id, epoch, party),
            &private.public_key().serialize(),
        )
        .await?;
        let next = (party + 1) % PARTY_COUNT;
        let previous = (party + 2) % PARTY_COUNT;
        let next_key = coordination::wait_for(
            s3,
            bucket,
            &coordination::public_key_key(environment, coordination_id, epoch, next),
            poll_interval,
        )
        .await?;
        let previous_key = coordination::wait_for(
            s3,
            bucket,
            &coordination::public_key_key(environment, coordination_id, epoch, previous),
            poll_interval,
        )
        .await?;
        let next_key = tripartite_dh::PublicKeys::deserialize(&next_key)
            .map_err(|error| eyre!("invalid party {next} public key: {error:?}"))?;
        let previous_key = tripartite_dh::PublicKeys::deserialize(&previous_key)
            .map_err(|error| eyre!("invalid party {previous} public key: {error:?}"))?;
        let derived = epoch_seed_from_dh(&private.derive_shared_secret(&next_key, &previous_key));
        if create_secret(secrets, &seed_id, &derived).await? {
            derived
        } else {
            parse_seed(
                &seed_id,
                load_secret(secrets, &seed_id)
                    .await?
                    .ok_or_else(|| eyre!("concurrent seed {seed_id} disappeared"))?,
            )?
        }
    };
    let commitment = publish_and_verify_commitment(
        s3,
        bucket,
        environment,
        coordination_id,
        epoch,
        party,
        &seed,
        poll_interval,
    )
    .await?;
    Ok((seed, commitment))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_and_active_epochs_are_mandatory_but_cache_only_is_not() {
        let (mandatory, cache_only) =
            split_required_epochs(&[(0, 3), (2, 5)], Some(3), &[1, 2, 4]).unwrap();
        assert_eq!(mandatory, BTreeSet::from([2, 3]));
        assert_eq!(cache_only, BTreeSet::from([1, 4]));
    }

    #[test]
    fn invalid_epoch_metadata_is_rejected_before_loading_secrets() {
        assert!(split_required_epochs(&[(-1, 1)], None, &[]).is_err());
        assert!(split_required_epochs(&[], None, &[i32::MAX as u32 + 1]).is_err());
    }

    #[test]
    fn authoritative_rows_may_only_use_current_database_epochs() {
        let state = iris_mpc_store::rerand::RerandStoreState {
            store_id: "store".into(),
            environment: "test".into(),
            coordination_id: "generation".into(),
            party_id: 0,
            store_kind: "gpu".into(),
            writer_role: "writer".into(),
            last_completed_epoch: 2,
            last_seed_commitment: Some([2; 32]),
            active_epoch: Some(3),
            active_seed_commitment: Some([3; 32]),
            next_id: Some(1),
            max_id: Some(1),
        };
        validate_authoritative_epochs(&[(0, 1), (2, 1), (3, 1)], &state).unwrap();
        assert!(validate_authoritative_epochs(&[(1, 1)], &state).is_err());
    }
}
