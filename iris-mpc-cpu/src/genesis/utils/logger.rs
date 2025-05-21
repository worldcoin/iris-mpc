/// Returns a message for logging.
fn get_formatted_message(component: &str, msg: String) -> String {
    format!("HNSW-GENESIS :: {} :: {}", component, msg)
}

/// Logs & returns a component error message.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An error message.
///
pub fn log_error(component: &str, msg: String) -> String {
    let msg = get_formatted_message(component, msg);

    // In testing print to stdout.
    #[cfg(test)]
    println!("ERROR :: {}", msg);

    // Trace as normal.
    tracing::error!(msg);

    msg
}

/// Logs & returns a component information message.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An information message.
///
pub fn log_info(component: &str, msg: String) -> String {
    let msg = get_formatted_message(component, msg);

    // In testing print to stdout.
    #[cfg(test)]
    println!("{}", msg);

    // Trace as normal.
    tracing::info!(msg);

    msg
}

/// Logs & returns a component warning message.
///
/// # Arguments
///
/// * `component` - A component encapsulating a unit of system functionality.
/// * `msg` - An information message.
///
pub fn log_warn(component: &str, msg: String) -> String {
    let msg = get_formatted_message(component, msg);

    // In testing print to stdout.
    #[cfg(test)]
    println!("WARN :: {}", msg);

    tracing::warn!(msg);

    msg
}
