use ampc_server_utils::ServiceConfig;
use eyre::Result;
use metrics_exporter_statsd::StatsdBuilder;
use std::{backtrace::Backtrace, panic};
use telemetry_batteries::tracing::{datadog::DatadogBattery, TracingShutdownHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn initialize_tracing(service_config: Option<ServiceConfig>) -> Result<TracingShutdownHandle> {
    if let Some(service) = &service_config {
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
