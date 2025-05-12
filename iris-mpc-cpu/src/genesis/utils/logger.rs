/// Logs component information messages.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An information message.
///
pub fn log_info(component: &str, msg: &str) {
    // In testing print to stdout.
    #[cfg(test)]
    println!("HNSW GENESIS :: {} :: {}", component, msg);

    // Trace as normal.
    tracing::info!("HNSW GENESIS :: {} :: {}", component, msg);
}
