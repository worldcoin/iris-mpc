use iris_mpc_common::config::{Config, ModeOfCompute};

pub(crate) fn get_check_addresses(config: &Config, endpoint: &str) -> Vec<String> {
    let hosts = config.node_hostnames.clone();
    let ports = config.healthcheck_ports.clone();

    hosts
        .iter()
        .zip(ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host, port, endpoint))
        .collect::<Vec<String>>()
}

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
