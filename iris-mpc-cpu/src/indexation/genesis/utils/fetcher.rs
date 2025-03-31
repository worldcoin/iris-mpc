use super::super::{errors::IndexationError, types::IrisSerialId};
use aws_sdk_s3::{config::Region as S3_Region, Client as S3_CLient};
use iris_mpc_common::config::Config;
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
    store: &IrisPgresStore,
) -> Result<IrisSerialId, IndexationError> {
    store
        .count_irises()
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
        .map(|val| val as IrisSerialId)
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
pub(crate) async fn fetch_iris_data(
    store: &IrisPgresStore,
    serial_id: IrisSerialId,
) -> Result<IrisData, IndexationError> {
    let data = store
        .fetch_iris_by_serial_id(serial_id)
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
        .unwrap();

    Ok(data)
}

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - System configuration information.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
pub(crate) async fn fetch_iris_deletions(
    config: &Config,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    // Destructure AWS configuration settings.
    let aws_endpoint = config
        .aws
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?
        .endpoint
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;
    let aws_region = config
        .aws
        .as_ref()
        .unwrap()
        .region
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;

    // Set AWS S3 client.
    let aws_config = aws_config::from_env()
        .region(S3_Region::new(aws_region.clone()))
        .load()
        .await;
    let s3_cfg = aws_sdk_s3::config::Builder::from(&aws_config)
        .endpoint_url(aws_endpoint.clone())
        .force_path_style(true)
        .build();
    let _ = S3_CLient::from_conf(s3_cfg);

    // Set AWS S3 response.
    // Response will be a simple json file with a single field:
    //  name: deleted_serial_ids
    //  type: Vec<IrisSerialId>
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
