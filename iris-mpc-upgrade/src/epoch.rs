use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use eyre::{eyre, Result};
use std::time::Duration;

use crate::s3_coordination;
use crate::tripartite_dh;

fn secret_id(env: &str, epoch: u32, party_id: u8) -> String {
    format!(
        "{}/iris-mpc-db-rerandomization/epoch-{}/private-key-party-{}",
        env, epoch, party_id
    )
}

/// Check if a private key for this epoch already exists in Secrets Manager.
async fn load_private_key_from_sm(
    sm: &SecretsManagerClient,
    env: &str,
    epoch: u32,
    party_id: u8,
) -> Result<Option<tripartite_dh::PrivateKey>> {
    let sid = secret_id(env, epoch, party_id);
    match sm
        .get_secret_value()
        .secret_id(&sid)
        .version_stage("AWSCURRENT")
        .send()
        .await
    {
        Ok(output) => {
            let b64 = output
                .secret_string()
                .ok_or_else(|| eyre!("Secret {} has no string value", sid))?;
            let bytes = STANDARD.decode(b64)?;
            let key = tripartite_dh::PrivateKey::deserialize(&bytes)
                .map_err(|e| eyre!("Failed to deserialize private key from SM: {:?}", e))?;
            Ok(Some(key))
        }
        Err(e) => {
            let svc = e.into_service_error();
            if svc.is_resource_not_found_exception() {
                Ok(None)
            } else {
                Err(eyre!("SM GetSecretValue failed for {}: {}", sid, svc))
            }
        }
    }
}

async fn save_private_key_to_sm(
    sm: &SecretsManagerClient,
    env: &str,
    epoch: u32,
    party_id: u8,
    key: &tripartite_dh::PrivateKey,
) -> Result<bool> {
    let sid = secret_id(env, epoch, party_id);
    let b64 = STANDARD.encode(key.serialize());

    match sm
        .create_secret()
        .name(&sid)
        .secret_string(&b64)
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(e) => {
            let svc = e.into_service_error();
            if svc.is_resource_exists_exception() {
                Ok(false)
            } else {
                Err(eyre!("SM CreateSecret failed for {}: {}", sid, svc))
            }
        }
    }
}

async fn delete_private_key_from_sm(
    sm: &SecretsManagerClient,
    env: &str,
    epoch: u32,
    party_id: u8,
) -> Result<()> {
    let sid = secret_id(env, epoch, party_id);
    sm.delete_secret()
        .secret_id(&sid)
        .force_delete_without_recovery(true)
        .send()
        .await
        .map_err(|e| eyre!("SM DeleteSecret failed for {}: {}", sid, e))?;
    tracing::info!("Deleted epoch {} private key from SM", epoch);
    Ok(())
}

/// Idempotent key generation for an epoch.
///
/// 1. Best-effort cleanup of previous epoch's key (covers crash between
///    `poll_epoch_complete_all` and `delete_private_key_from_sm`)
/// 2. Check SM for existing private key
/// 3. If found: load it, derive public key, re-upload to S3 (covers crash between SM write and S3 upload)
/// 4. If not found: generate new keypair, write to SM first, then upload public key to S3
pub async fn idempotent_keygen(
    sm: &SecretsManagerClient,
    s3: &S3Client,
    bucket: &str,
    env: &str,
    epoch: u32,
    party_id: u8,
) -> Result<tripartite_dh::PrivateKey> {
    if epoch > 0 {
        if let Err(e) = delete_private_key_from_sm(sm, env, epoch - 1, party_id).await {
            tracing::debug!("Cleanup of epoch {} key (best-effort): {}", epoch - 1, e);
        }
    }

    if let Some(existing) = load_private_key_from_sm(sm, env, epoch, party_id).await? {
        tracing::info!(
            "Epoch {}: private key found in SM, re-uploading public key to S3",
            epoch
        );
        let public_key = existing.public_key();
        let pk_b64 = STANDARD.encode(public_key.serialize());
        s3_coordination::upload_public_key(s3, bucket, epoch, party_id, &pk_b64).await?;
        return Ok(existing);
    }

    tracing::info!(
        "Epoch {}: generating fresh BLS12-381 keypair for party {}",
        epoch,
        party_id
    );
    let mut rng = rand::rngs::OsRng;
    let private_key = tripartite_dh::PrivateKey::random(&mut rng);

    let saved = save_private_key_to_sm(sm, env, epoch, party_id, &private_key).await?;
    let private_key = if saved {
        private_key
    } else {
        tracing::warn!(
            "Epoch {}: private key already exists in SM (likely concurrent start); reloading it",
            epoch
        );
        load_private_key_from_sm(sm, env, epoch, party_id)
            .await?
            .ok_or_else(|| {
                eyre!(
                    "Secret existed but could not be loaded: {}",
                    secret_id(env, epoch, party_id)
                )
            })?
    };

    let public_key = private_key.public_key();
    let pk_b64 = STANDARD.encode(public_key.serialize());
    s3_coordination::upload_public_key(s3, bucket, epoch, party_id, &pk_b64).await?;

    Ok(private_key)
}

/// Derive the shared secret for an epoch: keygen + download peer keys + BLS pairing.
pub async fn derive_shared_secret(
    sm: &SecretsManagerClient,
    s3: &S3Client,
    bucket: &str,
    env: &str,
    epoch: u32,
    party_id: u8,
    poll_interval: Duration,
) -> Result<[u8; 32]> {
    let private_key = idempotent_keygen(sm, s3, bucket, env, epoch, party_id).await?;

    let next_id = (party_id + 1) % 3;
    let prev_id = (party_id + 2) % 3;

    let pk_next_b64 =
        s3_coordination::download_public_key_for_party(s3, bucket, epoch, next_id, poll_interval)
            .await?;
    let pk_next =
        tripartite_dh::PublicKeys::deserialize(&STANDARD.decode(&pk_next_b64)?).map_err(|e| {
            eyre!(
                "Failed to deserialize public key for party {}: {:?}",
                next_id,
                e
            )
        })?;

    let pk_prev_b64 =
        s3_coordination::download_public_key_for_party(s3, bucket, epoch, prev_id, poll_interval)
            .await?;
    let pk_prev =
        tripartite_dh::PublicKeys::deserialize(&STANDARD.decode(&pk_prev_b64)?).map_err(|e| {
            eyre!(
                "Failed to deserialize public key for party {}: {:?}",
                prev_id,
                e
            )
        })?;

    let shared_secret = private_key.derive_shared_secret(&pk_next, &pk_prev);
    let hash = blake3::hash(&shared_secret);
    tracing::info!(
        "Epoch {}: derived shared secret (blake3 fingerprint: {})",
        epoch,
        hash.to_hex()
    );
    Ok(shared_secret)
}

/// Determine the active epoch by scanning S3 for the highest epoch with a
/// manifest but without all three `complete` markers.
///
/// `start_hint` allows callers to skip already-completed epochs (e.g. from
/// `get_current_epoch`). Falls back to 0 if no hint is available.
pub async fn determine_active_epoch(
    s3: &S3Client,
    bucket: &str,
    start_hint: Option<u32>,
) -> Result<u32> {
    let mut epoch: u32 = start_hint.unwrap_or(0);
    loop {
        if !s3_coordination::manifest_exists(s3, bucket, epoch).await? {
            break;
        }
        if s3_coordination::all_parties_complete(s3, bucket, epoch).await? {
            epoch += 1;
            continue;
        }
        return Ok(epoch);
    }
    Ok(epoch)
}

/// Upload completion marker, poll for all three, then delete the epoch key from SM.
pub async fn complete_epoch(
    sm: &SecretsManagerClient,
    s3: &S3Client,
    bucket: &str,
    env: &str,
    epoch: u32,
    party_id: u8,
    poll_interval: Duration,
) -> Result<()> {
    s3_coordination::upload_epoch_complete(s3, bucket, epoch, party_id).await?;
    tracing::info!(
        "Epoch {}: uploaded completion marker for party {}",
        epoch,
        party_id
    );

    s3_coordination::poll_epoch_complete_all(s3, bucket, epoch, poll_interval).await?;
    tracing::info!("Epoch {}: all parties completed", epoch);

    delete_private_key_from_sm(sm, env, epoch, party_id).await?;
    Ok(())
}
