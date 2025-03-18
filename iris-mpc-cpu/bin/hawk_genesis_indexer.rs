use iris_mpc_common::config::Config;
use iris_mpc_cpu::indexation::genesis::Supervisor;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing.
    tracing_subscriber::fmt()
        .with_env_filter("trace".parse::<EnvFilter>().unwrap())
        .without_time()
        .with_target(false)
        .init();
    tracing::info!("Spinup: tracing initialised.");

    // Set config.
    let config = Config::load_config("SMPC").unwrap();

    // Spawn supervisor.
    let a_ref = kameo::spawn(Supervisor::new(config));

    // Run supervisor until shutdown or killed.
    a_ref.wait_for_stop().await;

    Ok(())
}
