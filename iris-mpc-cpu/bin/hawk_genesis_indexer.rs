use iris_mpc_common::config::Config;
use iris_mpc_cpu::indexation::genesis::{OnBegin, Supervisor};
use std::future::pending;
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
    let a = Supervisor::new(config);
    kameo::spawn(a).tell(OnBegin).await?;

    // TODO: block until a supervisor OnIndexationEnd | OnIndexationError event is emitted.
    pending().await
}
