use eyre::{Report, Result};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisPgresStore;

const RNG_SEED_INIT_DB: u64 = 42;

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
pub(crate) async fn validate_iris_store_consistency(
    config: &Config,
    iris_pg_store: &IrisPgresStore,
) -> Result<usize, Report> {
    let store_len = iris_pg_store.count_irises().await?;

    tracing::info!("Size of the database before init: {}", store_len);

    // Seed the persistent storage with random shares if configured and db is still
    // empty.
    if store_len == 0 && config.init_db_size > 0 {
        tracing::info!(
            "Initialize persistent iris DB with {} randomly generated shares",
            config.init_db_size
        );
        tracing::info!("Resetting the db: {}", config.clear_db_before_init);
        iris_pg_store
            .init_db_with_random_shares(
                RNG_SEED_INIT_DB,
                config.party_id,
                config.init_db_size,
                config.clear_db_before_init,
            )
            .await?;
    }

    // Fetch again in case we've just initialized the DB
    let store_len = iris_pg_store.count_irises().await?;
    tracing::info!("Size of the database after init: {}", store_len);

    // Check if the sequence id is consistent with the number of irises
    let max_serial_id = iris_pg_store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        tracing::error!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );

        eyre::bail!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );
    }

    if store_len > config.max_db_size {
        tracing::error!(
            "Database size ({}) exceeds maximum allowed size ({})",
            store_len,
            config.max_db_size
        );
        eyre::bail!(
            "Database size ({}) exceeds maximum allowed size ({})",
            store_len,
            config.max_db_size
        );
    }

    Ok(store_len)
}
