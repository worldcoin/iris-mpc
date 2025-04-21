use iris_mpc_common::config::Config;
use iris_mpc_cpu::indexation::genesis::Supervisor1;
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

    // Initialise config.
    let config = Config::load_config("SMPC").unwrap();
    tracing::info!("Spinup: config initialised.");

    // Spawn job.
    kameo::spawn(Supervisor1::new(config))
        .wait_for_shutdown()
        .await;

    Ok(())
}
