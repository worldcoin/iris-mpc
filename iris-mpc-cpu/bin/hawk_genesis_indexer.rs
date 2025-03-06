use iris_mpc_common::config::Config;
use iris_mpc_cpu::indexer;
use kameo::actor::pubsub::{PubSub, Publish, Subscribe};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("trace".parse::<EnvFilter>().unwrap())
        .without_time()
        .with_target(false)
        .init();
    tracing::info!("Spinup: tracing initialised.");

    // Set config.
    let config: Config = Config::load_config("SMPC").unwrap();
    tracing::info!("Spinup: configuration loaded.");

    // Set brokers.
    let on_indexation_start = kameo::spawn(PubSub::<indexer::messages::OnIndexationStart>::new());
    let on_iris_id_pulled_from_store =
        kameo::spawn(PubSub::<indexer::messages::OnIrisIdPulledFromStore>::new());

    // Set actors.
    let ref_iris_id_stream_reader = kameo::spawn(
        indexer::actors::IrisIdStreamReader::new(
            config.clone(),
            on_iris_id_pulled_from_store.clone(),
        )
        .await,
    );
    let ref_iris_data_loader = kameo::spawn(indexer::actors::IrisDataLoader::default());

    // Set event subscribers.
    on_indexation_start
        .ask(Subscribe(ref_iris_id_stream_reader))
        .await?;
    on_iris_id_pulled_from_store
        .ask(Subscribe(ref_iris_data_loader))
        .await?;

    // Publish.
    on_indexation_start
        .ask(Publish(indexer::messages::OnIndexationStart))
        .await?;

    Ok(())
}
