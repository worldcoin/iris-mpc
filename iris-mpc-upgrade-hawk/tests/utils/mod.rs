pub mod constants;
mod errors;
mod factory;
mod logger;
pub mod resources;
pub mod runner;
pub mod runner1;
pub mod types;

pub use errors::TestError;
pub use factory::{get_net_inputs, get_node_inputs, get_test_info};

/// Sets up execution environment in readiness for a test run.
pub fn setup() -> () {
    println!("TODO");
}
