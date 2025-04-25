use super::{errors::IndexationError, types::IrisSerialId};
use aws_sdk_s3::Client as S3_Client;
use iris_mpc_store::{DbStoredIris as IrisData, Store as IrisPgresStore};
use rand::prelude::IteratorRandom;

/// Fetches height of indexed from store.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Height of indexed Iris's.
///
pub(crate) async fn fetch_height_of_indexed(
    _: &IrisPgresStore,
) -> Result<IrisSerialId, IndexationError> {
    // TODO: fetch from store.
    Ok(1)
}

/// Fetches height of protocol from store.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Height of stored Iris's.
///
pub(crate) async fn fetch_height_of_protocol(
    iris_store: &IrisPgresStore,
) -> Result<IrisSerialId, IndexationError> {
    iris_store
        .count_irises()
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisById)
        .map(|val| val as IrisSerialId)
}

/// Fetch a batch of iris data for indexation.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
/// * `identifiers` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// Iris data for indexation.
///
#[allow(dead_code)]
pub(crate) async fn fetch_iris_batch(
    iris_store: &IrisPgresStore,
    identifiers: Vec<IrisSerialId>,
) -> Result<Vec<IrisData>, IndexationError> {
    let data = iris_store
        .fetch_iris_batch(identifiers)
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisBatch)
        .unwrap();

    Ok(data)
}

/// Fetch iris data for indexation.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
/// * `serial_id` - Serial identifier of a processedIris.
///
/// # Returns
///
/// Iris data for indexation.
///
#[allow(dead_code)]
pub(crate) async fn fetch_iris_data(
    iris_store: &IrisPgresStore,
    serial_id: IrisSerialId,
) -> Result<IrisData, IndexationError> {
    let data = iris_store
        .fetch_iris_by_serial_id(serial_id)
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisById)
        .unwrap();

    Ok(data)
}

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `s3_client` - A configured AWS S3 client instance.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
pub(crate) async fn fetch_iris_deletions(
    _s3_client: &S3_Client,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    // TODO: Set AWS S3 response.
    // Response will be a simple json file with a single field:
    // {
    //     "deleted_serial_ids": ["1234567890", "0987654321" ... etc]
    // }
    // Response parser will:
    //  - attempt to simply deserialise the response body into Json.Value
    //  - map `deleted_serial_ids` field from Vec<String> -> Vec<IrisSerialId>.
    //  - return mapped Vec<IrisSerialId>
    // Errors:
    //  - AWS S3 bucket fetch error
    //  - JSON parsing error
    //  - Mapping error

    // TODO: remove temporary code that returns a random set of identifiers.
    let mut rng = rand::thread_rng();
    let mut identifiers: Vec<IrisSerialId> = (1..1000).choose_multiple(&mut rng, 50);
    identifiers.sort();

    Ok(identifiers)
}
