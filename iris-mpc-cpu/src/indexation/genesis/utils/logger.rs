/// Logs an actor informatino message.
///
/// # Arguments
///
/// * `actor` - Name of actor.
/// * `msg_type` - Type of information message.
/// * `msg` - An actor life-cycle information message.
///
pub(crate) fn log_info(actor: &str, msg_type: &str, msg: Option<&str>) {
    match msg {
        Some(info) => tracing::info!("GENESIS::{} :: {} :: {}", actor, msg_type, info),
        None => tracing::info!("GENESIS::{} :: {}", actor, msg_type),
    }
}

/// Logs an actor life-cycle message.
///
/// # Arguments
///
/// * `actor` - Name of actor.
/// * `msg` - An actor life-cycle message.
///
pub(crate) fn log_lifecycle(actor: &str, episode: &str, msg: Option<&str>) {
    log_info(actor, format!("Lifecycle::{}", episode).as_str(), msg)
}

/// Logs an actor message receipt.
///
/// # Arguments
///
/// * `actor` - Name of actor receiving a message.
/// * `msg` - An actor message.
///
pub(crate) fn log_message(actor: &str, event: &str, msg: Option<&str>) {
    log_info(actor, format!("Message::{}", event).as_str(), msg)
}

/// Logs an actor todo message.
///
/// # Arguments
///
/// * `actor` - Name of actor emitting a todo message.
/// * `msg` - An actor todo message.
///
pub(crate) fn log_todo(actor: &str, msg: &str) {
    log_info(actor, format!("TODO::{}", msg).as_str(), None);
}
