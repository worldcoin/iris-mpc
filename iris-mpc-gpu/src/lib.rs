#![allow(clippy::needless_range_loop)]
#![deny(
    clippy::iter_over_hash_type,
    reason = "In MPC protocols, this can be dangerous as the iteration order is not guaranteed to be in sync between the parties due to HashMap randomization."
)]
pub mod dot;
pub mod helpers;
pub mod rng;
pub mod server;
pub mod threshold_ring;
