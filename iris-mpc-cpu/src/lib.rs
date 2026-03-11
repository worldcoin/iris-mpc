#![deny(
    clippy::iter_over_hash_type,
    reason = "In MPC protocols, this can be dangerous as the iteration order is not guaranteed to be in sync between the parties due to HashMap randomization."
)]
pub mod analysis;
pub mod execution;
pub mod genesis;
pub mod hawkers;
pub mod hnsw;
pub mod protocol;
pub mod py_bindings;
pub mod utils;

pub use ampc_actor_utils::network;
pub use ampc_secret_sharing::shares;

/// Instrument a phase with automatic begin/end events for the phase tracer.
/// No-op when the `phase_trace` feature is disabled.
///
/// Usage:
///   `phase_trace!("dot_product", "cpu");`
///   `phase_trace!("dot_product", "cpu", "n_vectors" => vectors.len());`
#[macro_export]
macro_rules! phase_trace {
    ($name:expr, $cat:expr) => {
        #[cfg(feature = "phase_trace")]
        let _phase_guard =
            $crate::execution::hawk_main::phase_tracer::phase_begin($name, $cat, None);
    };
    ($name:expr, $cat:expr, $($key:expr => $val:expr),+ $(,)?) => {
        #[cfg(feature = "phase_trace")]
        let _phase_guard = $crate::execution::hawk_main::phase_tracer::phase_begin(
            $name,
            $cat,
            Some(serde_json::json!({ $($key: $val),+ })),
        );
    };
}
