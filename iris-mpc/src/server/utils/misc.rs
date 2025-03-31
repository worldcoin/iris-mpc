use iris_mpc_common::config::{Config, ModeOfCompute};

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
