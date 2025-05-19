/// Logs component error messages.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An error message.
///
pub fn log_error(component: &str, msg: String) -> String {
    let msg = format!("HNSW GENESIS :: {} :: {}", component, msg);

    // In testing print to stdout.
    #[cfg(test)]
    println!("ERROR :: {}", msg);

    // Trace as normal.
    tracing::error!(msg);

    msg
}

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

/// Logs component warning messages.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An information message.
///
pub fn log_warn(component: &str, msg: String) {
    // In testing print to stdout.
    #[cfg(test)]
    println!("WARN :: HNSW GENESIS :: {} :: {}", component, msg);

    // Trace as normal.
    tracing::warn!("HNSW GENESIS :: {} :: {}", component, msg);
}
