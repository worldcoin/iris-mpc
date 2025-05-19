use super::{errors::IndexationError, logger};
use aws_sdk_s3::Client as S3_Client;
use eyre::Result;
use iris_mpc_common::{config::Config, IrisSerialId};
use iris_mpc_store::{DbStoredIris, Store as IrisPgresStore};
use serde::{Deserialize, Serialize};

// Component name for logging purposes.
const COMPONENT: &str = "Fetcher";

/// Fetch height of indexed from store.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Index of lastest iris.
///
pub async fn fetch_height_of_indexed(iris_store: &IrisPgresStore) -> IrisSerialId {
    let domain = "genesis";
    let key = "indexed_height";
    iris_store
        .get_persistent_state(domain, key)
        .await
        .unwrap_or(Some(1))
        .unwrap_or(1)
}

/// Set height of indexed from store.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `new_height` - the height to be stored in the database.
///
/// # Returns
///
/// Result<()> on success
///
pub async fn set_height_of_indexed(
    iris_store: &IrisPgresStore,
    new_height: IrisSerialId,
) -> Result<()> {
    let domain = "genesis";
    let key = "indexed_height";
    let mut tx = iris_store.tx().await?;
    iris_store
        .set_persistent_state(&mut tx, domain, key, &new_height)
        .await
}

/// Fetch a batch of iris data for indexation.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `identifiers` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// Iris data for indexation.
///
pub(crate) async fn fetch_iris_batch(
    iris_store: &IrisPgresStore,
    identifiers: Vec<IrisSerialId>,
) -> Result<Vec<DbStoredIris>, IndexationError> {
    logger::log_info(
        COMPONENT,
        format!(
            "Fetching Iris batch for indexation: batch-size={}",
            identifiers.len()
        ),
    );

    let data = iris_store
        .fetch_iris_batch(identifiers)
        .await
        .map_err(|err| IndexationError::PostgresFetchIrisBatch(err.to_string()))?;

    Ok(data)
}

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `s3_client` - A configured AWS S3 client instance.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
#[allow(dead_code)]
pub async fn fetch_iris_deletions(
    config: &Config,
    s3_client: &S3_Client,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    // Struct for deserialization.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct S3Object {
        deleted_serial_ids: Vec<IrisSerialId>,
    }

    // Compose bucket and key based on environment
    let s3_bucket = config.get_s3_bucket_for_iris_deletions();
    let s3_key = config.get_s3_key_for_iris_deletions();
    logger::log_info(
        COMPONENT,
        format!(
            "Fetching deleted serial ids from S3 bucket: {}, key: {}",
            s3_bucket, s3_key
        ),
    );

    // Fetch from S3.
    let s3_response = s3_client
        .get_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .send()
        .await
        .map_err(|err| {
            logger::log_error(
                COMPONENT,
                format!("Failed to download file from S3: {}", err),
            );
            IndexationError::AwsS3ObjectDownload
        })?;

    // Consume S3 object stream.
    let s3_object_body = s3_response.body.collect().await.map_err(|err| {
        logger::log_error(COMPONENT, format!("Failed to get object body: {}", err));
        IndexationError::AwsS3ObjectDeserialize
    })?;

    // Decode S3 object bytes.
    let s3_object_bytes = s3_object_body.into_bytes();
    let s3_object: S3Object = serde_json::from_slice(&s3_object_bytes).map_err(|err| {
        logger::log_error(
            COMPONENT,
            format!("Failed to deserialize S3 object: {}", err),
        );
        IndexationError::AwsS3ObjectDeserialize
    })?;

    Ok(s3_object.deleted_serial_ids)
}

// ------------------------------------------------------------------------
// Tests.
// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{fetch_height_of_indexed, fetch_iris_batch};
    use eyre::Result;
    use iris_mpc_common::{
        postgres::{AccessMode, PostgresClient},
        IrisSerialId,
    };
    use iris_mpc_store::{
        test_utils::{cleanup, temporary_name, test_db_url},
        Store as IrisStore,
    };
    use itertools::Itertools;

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;

    // Returns a set of test resources.
    async fn get_resources() -> Result<(IrisStore, PostgresClient, String)> {
        // Set PostgreSQL client + store.
        let pg_schema = temporary_name();
        let pg_client =
            PostgresClient::new(&test_db_url()?, &pg_schema, AccessMode::ReadWrite).await?;

        // Set store.
        let iris_store = IrisStore::new(&pg_client).await?;

        // Set dB with 100 Iris's.
        iris_store
            .init_db_with_random_shares(
                DEFAULT_RNG_SEED,
                DEFAULT_PARTY_ID,
                DEFAULT_SIZE_OF_IRIS_DB,
                true,
            )
            .await?;

        Ok((iris_store, pg_client, pg_schema))
    }

    #[tokio::test]
    async fn test_fetch_height_of_indexed() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let height = fetch_height_of_indexed(&iris_store).await;
        assert_eq!(height, 1);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_iris_batch() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let identifiers: Vec<IrisSerialId> = (1..11).collect_vec();
        let data = fetch_iris_batch(&iris_store, identifiers).await.unwrap();
        assert_eq!(data.len(), 10);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }
}
