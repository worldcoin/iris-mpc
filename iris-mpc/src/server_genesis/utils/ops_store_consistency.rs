use eyre::{Report, Result};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisPgresStore;

/// Validates that the PostgreSQL iris store is in a consistent state.
///
/// # Arguments
///
/// * `config` - System configuration.
/// * `iris_pg_store` - Postgres client.
///
/// # Returns
///
/// Last iris serial identifier.
///
#[allow(dead_code)]
pub(crate) async fn validate_iris_store_consistency(
    config: &Config,
    iris_pg_store: &IrisPgresStore,
) -> Result<usize, Report> {
    let store_len = iris_pg_store.count_irises().await?;
    tracing::info!("Size of the database: {}", store_len);

    // Error if serial id is inconsistent with number of irises.
    let max_serial_id = iris_pg_store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        let error_message = format!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id, store_len
        );
        tracing::error!(error_message);
        eyre::bail!(error_message);
    }

    // Error if number of irises exceeds maximum constraint.
    if store_len > config.max_db_size {
        let error_message = format!(
            "Database size ({}) exceeds maximum allowed size ({})",
            store_len, config.max_db_size
        );
        tracing::error!(error_message);
        eyre::bail!(error_message);
    }

    Ok(store_len)
}
