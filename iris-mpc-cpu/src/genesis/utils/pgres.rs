use super::errors::IndexationError;
use eyre::eyre;
use iris_mpc_common::{
    config::Config as ApplicationConfig,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_store::Store as IrisStore;

/// Factory: return instance of application postgres store API.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Height of indexed Iris's.
///
#[allow(dead_code)]
pub(crate) async fn get_store_instance(config: &ApplicationConfig) -> IrisStore {
    let db_schema_name = config.get_database_schema_name();

    let db_config = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))
        .unwrap();

    let db_client = PostgresClient::new(
        db_config.url.as_str(),
        db_schema_name.as_str(),
        AccessMode::ReadWrite,
    )
    .await
    .map_err(|_| IndexationError::PostgresConnection)
    .unwrap();

    IrisStore::new(&db_client).await.unwrap()
}
