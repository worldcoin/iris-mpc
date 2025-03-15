use core::fmt;
use kameo::Actor;
use std::any::type_name;

/// Logs an actor information message.
///
/// # Arguments
///
/// * `msg_type` - Type of information message.
/// * `msg` - An actor life-cycle information message.
///
pub(crate) fn log_info<A>(msg_type: &str, msg: Option<&str>)
where
    A: Actor,
{
    match msg {
        Some(info) => {
            if info.is_empty() {
                tracing::info!("GENESIS :: {} :: {}", A::name(), msg_type);
            } else {
                tracing::info!("GENESIS :: {} :: {} :: {}", A::name(), msg_type, info);
            }
        }
        None => tracing::info!("GENESIS :: {} :: {}", A::name(), msg_type),
    }
}

/// Logs an actor life-cycle message.
///
/// # Arguments
///
/// * `episode` - Actor lifecycle episode.
/// * `msg` - Actor life-cycle message.
/// * `info` - Other pertinent information.
///
pub(crate) fn log_lifecycle<A>(episode: &str, info: Option<&str>)
where
    A: Actor,
{
    log_info::<A>(format!("Lifecycle[{}]", episode).as_str(), info)
}

/// Logs an actor message receipt.
///
/// # Arguments
///
/// * `msg` - A message received by an actor.
///
pub(crate) fn log_message<A, M>(msg: &M)
where
    A: Actor,
    M: fmt::Display,
{
    // TODO: optimise via a cache.
    let f: Vec<&str> = type_name::<M>().split("::").collect();

    log_info::<A>(f.last().unwrap(), Some(msg.to_string().as_str()));
}

/// Logs an actor todo message.
///
/// # Arguments
///
/// * `info` - An actor todo message.
///
pub(crate) fn log_todo<A>(info: &str)
where
    A: Actor,
{
    log_info::<A>(format!("TODO::{}", info).as_str(), None);
}
