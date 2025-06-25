mod constants;
mod factory;
mod logger;
pub mod runner;
pub mod types;

pub use factory::{get_net_inputs, get_node_inputs, get_test_info};

/// Sets up execution environment in readiness for a test run.
pub fn setup() -> () {
    println!("TODO");
}
