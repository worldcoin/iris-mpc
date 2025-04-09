use eyre::eyre;
use iris_mpc_common::{
    config::{Config, Config as ApplicationConfig, ModeOfCompute},
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_cpu::execution::hawk_main::GraphStore;
use iris_mpc_store::Store as IrisStore;

/// Gets set of MPC node IP addresses for a particular endpoint.
///
/// # Arguments
///
/// * `config` - System configuration.
/// * `endpoint` - Endpoint being invoked.
///
/// # Returns
///
/// Set of MPC node IP addresses.
///
pub(crate) fn get_check_addresses(config: &Config, endpoint: &str) -> Vec<String> {
    config
        .node_hostnames
        .iter()
        .zip(config.healthcheck_ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host, port, endpoint))
        .collect::<Vec<String>>()
}

/// Factory: returns instance of application's Graph postgres store API pointer.
///
/// # Arguments
///
/// * `config` - Application configuration.
///
/// # Returns
///
/// Graph postgres store API pointer.
///
pub(crate) async fn get_graph_pg_store_instance(config: &ApplicationConfig) -> GraphStore {
    let db_schema_name = config.get_database_schema_name();

    let db_config = config
        .cpu_database
        .as_ref()
        .ok_or(eyre!("Missing database config"))
        .unwrap();

    let db_client = PostgresClient::new(
        db_config.url.as_str(),
        db_schema_name.as_str(),
        AccessMode::ReadWrite,
    )
    .await
    .map_err(|_| "Postgres connection error")
    .unwrap();

    GraphStore::new(&db_client).await.unwrap()
}

/// Factory: returns instance of application's Iris postgres store API pointer.
///
/// # Arguments
///
/// * `config` - Application configuration.
///
/// # Returns
///
/// Iris postgres store API pointer.
///
pub(crate) async fn get_iris_pg_store_instance(config: &ApplicationConfig) -> IrisStore {
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
    .map_err(|_| "Postgres connection error")
    .unwrap();

    IrisStore::new(&db_client).await.unwrap()
}

/// Validates system configuration.
///
/// # Arguments
///
/// * `config` - System configuration being validated.
///
pub(crate) fn validate_config(config: &Config) {
    // Validate modes of compute/deployment.
    if config.mode_of_compute != ModeOfCompute::Cpu {
        panic!(
            "Invalid config setting: compute_mode: actual: {:?} :: expected: ModeOfCompute::CPU",
            config.mode_of_compute
        );
    } else {
        tracing::info!("Mode of compute: {:?}", config.mode_of_compute);
        tracing::info!("Mode of deployment: {:?}", config.mode_of_deployment);
    }
}
