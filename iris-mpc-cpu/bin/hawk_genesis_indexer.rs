use eyre;
use iris_mpc_common::config::Config;
use iris_mpc_store::{DbStoredIris, Store as IrisPgresStore};
use metrics_exporter_statsd::StatsdBuilder;
use std::{backtrace::Backtrace, panic};
use telemetry_batteries::tracing::{datadog::DatadogBattery, TracingShutdownHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Name of process for tracing purposes.
const PROCESS_NAME: &str = "hawk-batch-indexer";

// Main entry point.
// Runs a job to index an HNSW graph from Iris data persisted within a postgres store.
#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Set config.
    let mut config: Config = Config::load_config("SMPC").unwrap();

    // Set tracing.
    let _ = match initialize_tracing(&config) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    // Await execution.
    match execute(config).await {
        Ok(_) => {
            tracing::info!("{} :: exited normally", PROCESS_NAME);
        }
        Err(e) => {
            tracing::error!("{} :: exited with error: {:?}", PROCESS_NAME, e);
            return Err(e);
        }
    }

    Ok(())
}

async fn execute(_: Config) -> eyre::Result<()> {
    println!("For each Iris within pgres dB (i.e. `Store::stream_irises`):");
    println!("   - run HNSW MPC search (i.e. `HawkMain::search_to_insert`, `HawkMain::insert`),");
    println!("   - insert into graph pgres store (i.e.  `GraphOps::insert_apply`).");

    Ok(())
}

fn initialize_tracing(config: &Config) -> eyre::Result<TracingShutdownHandle> {
    if let Some(service) = &config.service {
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

        // Set a custom panic hook to print backtraces on one line
        panic::set_hook(Box::new(|panic_info| {
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
                { backtrace = %backtrace_single_line, location = %location},
                "Panic occurred with message: {}",
                message
            );
        }));
        Ok(tracing_shutdown_handle)
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().pretty().compact())
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();

        Ok(TracingShutdownHandle {})
    }
}
