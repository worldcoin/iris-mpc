use aws_sdk_sqs::Client;
use iris_mpc_common::{
    config::Config,
    helpers::{
        kms_dh::derive_shared_secret,
        smpc_request::{CircuitBreakerRequest, ReceiveRequestError, SQSMessage},
    },
};
use metrics_exporter_statsd::StatsdBuilder;
use std::{backtrace::Backtrace, panic};
use telemetry_batteries::tracing::{datadog::DatadogBattery, TracingShutdownHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub async fn handle_circuit_breaker_message(
    message: &SQSMessage,
    sqs_message: &aws_sdk_sqs::types::Message,
    client: &Client,
    queue_url: &str,
    batch_size: &mut usize,
    max_batch_size: usize,
) -> eyre::Result<(), ReceiveRequestError> {
    let circuit_breaker_request: CircuitBreakerRequest = serde_json::from_str(&message.message)
        .map_err(|e| ReceiveRequestError::json_parse_error("circuit_breaker_request", e))?;

    metrics::counter!("request.received", "type" => "circuit_breaker").increment(1);

    client
        .delete_message()
        .queue_url(queue_url)
        .receipt_handle(sqs_message.receipt_handle.clone().unwrap())
        .send()
        .await
        .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

    if let Some(new_batch_size) = circuit_breaker_request.batch_size {
        *batch_size = new_batch_size.clamp(1, max_batch_size);
        tracing::info!(
            "Updating batch size to {} due to circuit breaker message",
            new_batch_size
        );
    }

    Ok(())
}

pub fn initialize_tracing(config: &Config) -> eyre::Result<TracingShutdownHandle> {
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

pub async fn initialize_chacha_seeds(config: Config) -> eyre::Result<([u32; 8], [u32; 8])> {
    // Init RNGs
    let own_key_arn = config
        .kms_key_arns
        .0
        .get(config.party_id)
        .expect("Expected value not found in kms_key_arns");
    let dh_pairs = match config.party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let dh_pair_0: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.0)
        .expect("Expected value not found in kms_key_arns");
    let dh_pair_1: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.1)
        .expect("Expected value not found in kms_key_arns");

    // To be used only for e2e testing where we use localstack. There's a bug in
    // localstack's implementation of `derive_shared_secret`. See: https://github.com/localstack/localstack/pull/12071
    let chacha_seeds: ([u32; 8], [u32; 8]) = if config.fixed_shared_secrets {
        ([0u32; 8], [0u32; 8])
    } else {
        (
            bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_0).await?),
            bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_1).await?),
        )
    };

    Ok(chacha_seeds)
}
