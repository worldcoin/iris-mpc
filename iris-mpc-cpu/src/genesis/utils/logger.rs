/// Logs component information messages.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An information message.
///
pub fn log_info(component: &str, msg: String) {
    // In testing print to stdout.
    #[cfg(test)]
    println!("HNSW GENESIS :: {} :: {}", component, msg);

    // Trace as normal.
    tracing::info!("HNSW GENESIS :: {} :: {}", component, msg);
}

/// Logs component error messages.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An error message.
///
pub fn log_error(component: &str, msg: String) {
    // In testing print to stdout.
    #[cfg(test)]
    println!("ERROR :: HNSW GENESIS :: {} :: {}", component, msg);

    // Trace as normal.
    tracing::error!("HNSW GENESIS :: {} :: {}", component, msg);
}
