//! Key rotation helpers for WAL workflow tests.
//!
//! Mirrors the `rotate` subcommand of the `key-manager` binary
//! (`iris-mpc-bins/bin/iris-mpc-common/key_manager.rs`) so tests can rotate
//! ECDH keys into LocalStack without shelling out to a separate process.
//!
//! # One-time setup
//!
//! Call [`global_setup`] once before any test in `e2e_wal.rs` runs.  It:
//! 1. Polls Secrets Manager until LocalStack *and* its init-script are ready.
//! 2. Rotates ECDH keys twice for each of the three MPC parties.

use aws_config::SdkConfig;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use base64::{engine::general_purpose::STANDARD, Engine};
use eyre::Result;
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};
use rand::{thread_rng, Rng};
use sodiumoxide::crypto::box_::{curve25519xsalsa20poly1305, Seed};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// LocalStack S3 bucket that holds the nodes' public ECDH keys.
/// Created by `scripts/tools/init-localstack.sh`.
const PUBLIC_KEY_BUCKET: &str = "wf-dev-public-keys";

/// S3 key prefix for public keys.  The full key is `"public-key-{node_id}"`.
const PUBLIC_KEY_S3_KEY_PREFIX: &str = "public-key";

/// Secrets Manager secret-name prefix for private keys.
/// The full name is `"dev/iris-mpc/ecdh-private-key-{node_id}"`.
const PRIVATE_KEY_SECRET_PREFIX: &str = "dev/iris-mpc/ecdh-private-key";

/// How long to wait for LocalStack + its init-script before giving up.
const LOCALSTACK_WAIT_SECS: u64 = 120;

/// Number of MPC parties.
const PARTY_COUNT: usize = 3;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// One-time global setup: wait for LocalStack then rotate keys twice per party.
///
/// `endpoint_url` is the LocalStack base URL, e.g. `"http://localhost:4566"`.
pub async fn global_setup(endpoint_url: &str) -> Result<()> {
    wait_for_localstack(endpoint_url).await?;

    let sdk_config = base_sdk_config().await;

    for round in 1..=2 {
        for node_id in 0..PARTY_COUNT {
            let bucket_key_name = format!("{PUBLIC_KEY_S3_KEY_PREFIX}-{node_id}");
            let private_key_secret_id = format!("{PRIVATE_KEY_SECRET_PREFIX}-{node_id}");

            tracing::info!(round, node_id, "rotating ECDH keys");
            rotate_keys(
                &sdk_config,
                &bucket_key_name,
                &private_key_secret_id,
                None,
                Some(PUBLIC_KEY_BUCKET.to_string()),
                Some(endpoint_url.to_string()),
            )
            .await?;
        }
    }

    tracing::info!("global_setup: keys rotated 2× for all {PARTY_COUNT} parties");
    Ok(())
}

// ---------------------------------------------------------------------------
// Localstack readiness poll
// ---------------------------------------------------------------------------

/// Poll Secrets Manager until `dev/iris-mpc/ecdh-private-key-0` is readable.
///
/// This confirms two things:
/// - LocalStack is accepting API calls.
/// - The init-script (`init-localstack.sh`) has finished creating secrets.
async fn wait_for_localstack(endpoint_url: &str) -> Result<()> {
    let sdk_config = base_sdk_config().await;

    let sm_config = aws_sdk_secretsmanager::config::Builder::from(&sdk_config)
        .endpoint_url(endpoint_url)
        .build();
    let sm = SecretsManagerClient::from_conf(sm_config);

    let secret_id = format!("{PRIVATE_KEY_SECRET_PREFIX}-0");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(LOCALSTACK_WAIT_SECS);

    loop {
        match sm.get_secret_value().secret_id(&secret_id).send().await {
            Ok(_) => {
                tracing::info!(%endpoint_url, "LocalStack ready (secrets initialised)");
                return Ok(());
            }
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Err(eyre::eyre!(
                        "LocalStack/init-script not ready after {LOCALSTACK_WAIT_SECS}s: {e}"
                    ));
                }
                tracing::debug!(%endpoint_url, %e, "waiting for LocalStack init-script…");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Key rotation
// ---------------------------------------------------------------------------

/// Generate a fresh ECDH key pair, upload the public key to S3, and store
/// the private key in Secrets Manager.
///
/// Signature matches what the `key-manager` binary's `rotate` subcommand does
/// so callers can use the same arguments as the CLI.
pub async fn rotate_keys(
    sdk_config: &SdkConfig,
    bucket_key_name: &str,
    private_key_secret_id: &str,
    dry_run: Option<bool>,
    public_key_bucket_name: Option<String>,
    endpoint_url: Option<String>,
) -> Result<()> {
    let bucket_name = public_key_bucket_name.unwrap_or_else(|| PUBLIC_KEY_BUCKET.to_string());

    // Generate a fresh key pair from a random seed.
    let mut seedbuf = [0u8; 32];
    thread_rng().fill(&mut seedbuf);
    let (public_key, private_key) = curve25519xsalsa20poly1305::keypair_from_seed(&Seed(seedbuf));

    let pub_key_str = STANDARD.encode(public_key);
    let priv_key_str = STANDARD.encode(private_key);

    if dry_run.unwrap_or(false) {
        tracing::info!(%pub_key_str, "dry-run: skipping upload");
        return Ok(());
    }

    // Build SDK clients, optionally overriding the endpoint for LocalStack.
    let mut sm_builder = aws_sdk_secretsmanager::config::Builder::from(sdk_config);
    let mut object_store = ObjectStoreClient::new(
        sdk_config.region().map(ToString::to_string),
        endpoint_url.is_some(),
    );

    if let Some(ref url) = endpoint_url {
        object_store = object_store
            .with_option("aws_endpoint", url)
            .with_option("aws_allow_http", url.starts_with("http://"));
        sm_builder = sm_builder.endpoint_url(url);
    }

    let sm = SecretsManagerClient::from_conf(sm_builder.build());

    // Upload public key to S3.
    object_store
        .store(&bucket_name)?
        .put(&path(bucket_key_name)?, pub_key_str.into_bytes().into())
        .await
        .map_err(|e| {
            eyre::eyre!(
                "failed to upload public key to s3://{bucket_name}/{bucket_key_name}: {e:?}"
            )
        })?;

    tracing::debug!(
        key = bucket_key_name,
        bucket = %bucket_name,
        "public key uploaded to S3",
    );

    // Store private key in Secrets Manager.
    sm.put_secret_value()
        .secret_id(private_key_secret_id)
        .secret_string(priv_key_str)
        .send()
        .await
        .map_err(|e| {
            eyre::eyre!(
                "failed to store private key in Secrets Manager ({private_key_secret_id}): {e:?}"
            )
        })?;

    tracing::debug!(secret_id = %private_key_secret_id, "private key stored in Secrets Manager");

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a base `SdkConfig` with the us-east-1 region (matching LocalStack's
/// `DEFAULT_REGION` in `docker-compose.hawk-db.yaml`).
async fn base_sdk_config() -> SdkConfig {
    aws_config::from_env()
        .region(aws_sdk_secretsmanager::config::Region::new("us-east-1"))
        .load()
        .await
}
