use iris_mpc_common::config::Config;
use iris_mpc_cpu::execution::hawk_main::GraphStore as GraphPgresStore;
use iris_mpc_store::{DbStoredIris, Store as IrisPgresStore};
use metrics_exporter_statsd::StatsdBuilder;
use std::{
    backtrace::Backtrace,
    panic::{self, PanicHookInfo},
};
use telemetry_batteries::tracing::{datadog::DatadogBattery, TracingShutdownHandle};
use tokio_stream::StreamExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Name of process for tracing purposes.
const PROCESS_NAME: &str = "hawk-batch-indexer";

// Main entry point.
// Runs a job to index an HNSW graph from Iris data persisted within a postgres
// store.
#[tokio::main]
async fn main() -> eyre::Result<()> {
    fn init_tracing(config: &Config) -> eyre::Result<TracingShutdownHandle> {
        fn on_execution_panic(panic_info: &PanicHookInfo) {
            let message = match panic_info.payload().downcast_ref::<&str>() {
                Some(s) => *s,
                None => match panic_info.payload().downcast_ref::<String>() {
                    Some(s) => s.as_str(),
                    None => "Unknown panic message",
                },
            };

            let location = if let Some(location) = panic_info.location() {
                format!(
                    "{}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column()
                )
            } else {
                "Unknown location".to_string()
            };

            let backtrace = Backtrace::capture();
            let backtrace_string = format!("{:?}", backtrace);
            let backtrace_single_line = backtrace_string.replace('\n', " | ");

            tracing::error!(
                {
                    backtrace = %backtrace_single_line,
                    location = %location
                },
                "{} :: panic occurred with message: {}",
                PROCESS_NAME,
                message
            );
        }

        match &config.service {
            None => {
                tracing_subscriber::registry()
                    .with(tracing_subscriber::fmt::layer().pretty().compact())
                    .with(
                        tracing_subscriber::EnvFilter::try_from_default_env()
                            .unwrap_or_else(|_| "info".into()),
                    )
                    .init();

                Ok(TracingShutdownHandle {})
            }
            Some(service) => {
                let tracing_shutdown_handle = DatadogBattery::init(
                    service.traces_endpoint.as_deref(),
                    &service.service_name,
                    None,
                    true,
                );

                if let Some(metrics_config) = &service.metrics {
                    let recorder = StatsdBuilder::from(&metrics_config.host, metrics_config.port)
                        .with_queue_size(metrics_config.queue_size)
                        .with_buffer_size(metrics_config.buffer_size)
                        .histogram_is_distribution()
                        .build(Some(&metrics_config.prefix))?;
                    metrics::set_global_recorder(recorder)?;
                }

                panic::set_hook(Box::new(on_execution_panic));

                Ok(tracing_shutdown_handle)
            }
        }
    }

    // Set config.
    let config: Config = Config::load_config("SMPC").unwrap();

    // Set tracing.
    let _ = match init_tracing(&config) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    // Await execution.
    tracing::info!("{} :: process execution BEGINS", PROCESS_NAME);
    match execute(config).await {
        Ok(_) => {
            tracing::info!("{} :: process execution ENDS", PROCESS_NAME);
            tracing::info!("{} :: exited normally", PROCESS_NAME);
        }
        Err(report) => {
            tracing::error!("{} :: exited with error: {:?}", PROCESS_NAME, report);
            return Err(report);
        }
    }

    Ok(())
}

// Information flowing through execution path.
#[derive(Debug)]
#[allow(dead_code)]
struct ExecutionContextInfo {
    // Iris data pulled from a remote store.
    iris: DbStoredIris,
}

// Ctor.
impl ExecutionContextInfo {
    fn new(iris: DbStoredIris) -> Self {
        Self { iris }
    }
}

async fn execute(config: Config) -> eyre::Result<()> {
    // Set services.
    let store_of_irises = IrisPgresStore::new_from_config(&config).await?;
    let _ = GraphPgresStore::from_iris_store(&store_of_irises);

    // Process stream of stored iris's.
    // TODO: one task
    let mut stream_of_irises = store_of_irises.stream_irises().await;
    while let Some(item) = stream_of_irises.try_next().await? {
        on_iris_recieved_from_stream(ExecutionContextInfo::new(item)).await;
    }

    Ok(())
}

async fn on_iris_recieved_from_stream(_: ExecutionContextInfo) {
    println!(
        " TODO - run HNSW MPC search (i.e. `HawkMain::search_to_insert`, `HawkMain::insert`),"
    );
    println!(" TODO- insert into graph pgres store (i.e.  `GraphOps::insert_apply`).");
}
