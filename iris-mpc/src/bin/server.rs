#![allow(clippy::needless_range_loop)]

use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use aws_sdk_sqs::{config::Region, Client};
use axum::{routing::get, Router};
use clap::Parser;
use eyre::{eyre, Context};
use futures::TryStreamExt;
use iris_mpc_common::{
    config::{json_wrapper::JsonStrWrapper, Config, Opt},
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        aws::{
            construct_message_attributes, SPAN_ID_MESSAGE_ATTRIBUTE_NAME,
            TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
        },
        key_pair::SharesEncryptionKeyPairs,
        kms_dh::derive_shared_secret,
        smpc_request::{
            create_message_type_attribute_map, CircuitBreakerRequest, IdentityDeletionRequest,
            IdentityDeletionResult, ReceiveRequestError, SQSMessage, UniquenessRequest,
            UniquenessResult, CIRCUIT_BREAKER_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE,
            SMPC_MESSAGE_TYPE_ATTRIBUTE, UNIQUENESS_MESSAGE_TYPE,
        },
        sync::SyncState,
        task_monitor::TaskMonitor,
    },
};
use iris_mpc_gpu::{
    helpers::device_manager::DeviceManager,
    server::{
        get_dummy_shares_for_deletion, sync_nccl, BatchMetadata, BatchQuery,
        BatchQueryEntriesPreprocessed, ServerActor, ServerJobResult,
    },
};
use iris_mpc_store::{Store, StoredIrisRef};
use metrics_exporter_statsd::StatsdBuilder;
use std::{
    backtrace::Backtrace,
    collections::HashMap,
    mem, panic,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};
use telemetry_batteries::tracing::{datadog::DatadogBattery, TracingShutdownHandle};
use tokio::{
    sync::{mpsc, oneshot, Semaphore},
    task::spawn_blocking,
    time::timeout,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REGION: &str = "eu-north-1";
const RNG_SEED_INIT_DB: u64 = 42;
const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
const MAX_CONCURRENT_REQUESTS: usize = 32;

static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

fn decode_iris_message_shares(
    code_share: String,
    mask_share: String,
) -> eyre::Result<(GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare)> {
    let iris_share = GaloisRingIrisCodeShare::from_base64(&code_share)
        .context("Failed to base64 parse iris code")?;
    let mask_share: GaloisRingTrimmedMaskCodeShare =
        GaloisRingIrisCodeShare::from_base64(&mask_share)
            .context("Failed to base64 parse iris mask")?
            .into();

    Ok((iris_share, mask_share))
}

#[allow(clippy::type_complexity)]
fn preprocess_iris_message_shares(
    code_share: GaloisRingIrisCodeShare,
    mask_share: GaloisRingTrimmedMaskCodeShare,
) -> eyre::Result<(
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    Vec<GaloisRingIrisCodeShare>,
    Vec<GaloisRingTrimmedMaskCodeShare>,
    Vec<GaloisRingIrisCodeShare>,
    Vec<GaloisRingTrimmedMaskCodeShare>,
)> {
    let mut code_share = code_share;
    let mut mask_share = mask_share;

    // Original for storage.
    let store_iris_shares = code_share.clone();
    let store_mask_shares = mask_share.clone();

    // With rotations for in-memory database.
    let db_iris_shares = code_share.all_rotations();
    let db_mask_shares = mask_share.all_rotations();

    // With Lagrange interpolation.
    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut code_share);
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(&mut mask_share);

    Ok((
        store_iris_shares,
        store_mask_shares,
        db_iris_shares,
        db_mask_shares,
        code_share.all_rotations(),
        mask_share.all_rotations(),
    ))
}

async fn receive_batch(
    party_id: usize,
    client: &Client,
    queue_url: &String,
    store: &Store,
    skip_request_ids: &[String],
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    max_batch_size: usize,
) -> eyre::Result<BatchQuery, ReceiveRequestError> {
    let mut batch_query = BatchQuery::default();

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));
    let mut handles = vec![];
    let mut msg_counter = 0;

    while msg_counter < *CURRENT_BATCH_SIZE.lock().unwrap() {
        let rcv_message_output = client
            .receive_message()
            .max_number_of_messages(1)
            .queue_url(queue_url)
            .send()
            .await
            .map_err(ReceiveRequestError::FailedToReadFromSQS)?;

        if let Some(messages) = rcv_message_output.messages {
            for sqs_message in messages {
                let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())
                    .map_err(|e| ReceiveRequestError::json_parse_error("SQS body", e))?;

                // messages arrive to SQS through SNS. So, all the attributes set in SNS are
                // moved into the SQS body.
                let message_attributes = message.message_attributes;

                let mut batch_metadata = BatchMetadata::default();

                if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let trace_id = trace_id.string_value().unwrap();
                    batch_metadata.trace_id = trace_id.to_string();
                }
                if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let span_id = span_id.string_value().unwrap();
                    batch_metadata.span_id = span_id.to_string();
                }

                let request_type = message_attributes
                    .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
                    .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
                    .string_value()
                    .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?;

                match request_type {
                    CIRCUIT_BREAKER_MESSAGE_TYPE => {
                        let circuit_breaker_request: CircuitBreakerRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error("circuit_breaker_request", e)
                            })?;
                        metrics::counter!("request.received", "type" => "circuit_breaker")
                            .increment(1);
                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;
                        if let Some(batch_size) = circuit_breaker_request.batch_size {
                            // Updating the batch size to ensure we process the messages in the next
                            // loop
                            *CURRENT_BATCH_SIZE.lock().unwrap() =
                                batch_size.clamp(1, max_batch_size);
                            tracing::info!(
                                "Updating batch size to {} due to circuit breaker message",
                                batch_size
                            );
                        }
                    }

                    IDENTITY_DELETION_MESSAGE_TYPE => {
                        // If it's a deletion request, we just store the serial_id and continue.
                        // Deletion will take place when batch process starts.
                        let identity_deletion_request: IdentityDeletionRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error(
                                    "Identity deletion request",
                                    e,
                                )
                            })?;
                        metrics::counter!("request.received", "type" => "identity_deletion")
                            .increment(1);
                        batch_query
                            .deletion_requests_indices
                            .push(identity_deletion_request.serial_id - 1); // serial_id is 1-indexed
                        batch_query.deletion_requests_metadata.push(batch_metadata);
                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;
                    }
                    UNIQUENESS_MESSAGE_TYPE => {
                        msg_counter += 1;

                        let shares_encryption_key_pairs = shares_encryption_key_pairs.clone();

                        let smpc_request: UniquenessRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error("Uniqueness request", e)
                            })?;
                        metrics::counter!("request.received", "type" => "uniqueness_verification")
                            .increment(1);
                        store
                            .mark_requests_deleted(&[smpc_request.signup_id.clone()])
                            .await
                            .map_err(ReceiveRequestError::FailedToMarkRequestAsDeleted)?;

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if skip_request_ids.contains(&smpc_request.signup_id) {
                            // Some party (maybe us) already meant to delete this request, so we
                            // skip it.
                            continue;
                        }

                        if let Some(batch_size) = smpc_request.batch_size {
                            // Updating the batch size instantly makes it a bit unpredictable, since
                            // if we're already above the new limit, we'll still process the current
                            // batch at the higher limit. On the other
                            // hand, updating it after the batch is
                            // processed would not let us "unblock" the protocol if we're stuck with
                            // low throughput.
                            *CURRENT_BATCH_SIZE.lock().unwrap() =
                                batch_size.clamp(1, max_batch_size);
                            tracing::info!("Updating batch size to {}", batch_size);
                        }

                        batch_query.request_ids.push(smpc_request.signup_id.clone());
                        batch_query.metadata.push(batch_metadata);

                        let semaphore = Arc::clone(&semaphore);
                        let handle = tokio::spawn(async move {
                            let _ = semaphore.acquire().await?;

                            let base_64_encoded_message_payload =
                                match smpc_request.get_iris_data_by_party_id(party_id).await {
                                    Ok(iris_message_share) => iris_message_share,
                                    Err(e) => {
                                        tracing::error!("Failed to get iris shares: {:?}", e);
                                        eyre::bail!("Failed to get iris shares: {:?}", e);
                                    }
                                };

                            let iris_message_share = match smpc_request.decrypt_iris_share(
                                base_64_encoded_message_payload,
                                shares_encryption_key_pairs.clone(),
                            ) {
                                Ok(iris_data) => iris_data,
                                Err(e) => {
                                    tracing::error!("Failed to decrypt iris shares: {:?}", e);
                                    eyre::bail!("Failed to decrypt iris shares: {:?}", e);
                                }
                            };

                            match smpc_request
                                .validate_iris_share(party_id, iris_message_share.clone())
                            {
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::error!("Failed to validate iris shares: {:?}", e);
                                    eyre::bail!("Failed to validate iris shares: {:?}", e);
                                }
                            }

                            let (left_code, left_mask) = decode_iris_message_shares(
                                iris_message_share.left_iris_code_shares,
                                iris_message_share.left_mask_code_shares,
                            )?;

                            let (right_code, right_mask) = decode_iris_message_shares(
                                iris_message_share.right_iris_code_shares,
                                iris_message_share.right_mask_code_shares,
                            )?;

                            // Preprocess shares for left eye.
                            let left_future = spawn_blocking(move || {
                                preprocess_iris_message_shares(left_code, left_mask)
                            });

                            // Preprocess shares for right eye.
                            let right_future = spawn_blocking(move || {
                                preprocess_iris_message_shares(right_code, right_mask)
                            });

                            let (left_result, right_result) =
                                tokio::join!(left_future, right_future);

                            Ok((
                                left_result.context("while processing left iris shares")??,
                                right_result.context("while processing right iris shares")??,
                            ))
                        });

                        handles.push(handle);
                    }
                    _ => {
                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;
                        tracing::error!("Error: {}", ReceiveRequestError::InvalidMessageType);
                    }
                }
            }
        } else {
            tokio::time::sleep(SQS_POLLING_INTERVAL).await;
        }
    }

    for handle in handles {
        let (
            (
                (
                    store_iris_shares_left,
                    store_mask_shares_left,
                    db_iris_shares_left,
                    db_mask_shares_left,
                    iris_shares_left,
                    mask_shares_left,
                ),
                (
                    store_iris_shares_right,
                    store_mask_shares_right,
                    db_iris_shares_right,
                    db_mask_shares_right,
                    iris_shares_right,
                    mask_shares_right,
                ),
            ),
            valid_entry,
        ) = match handle
            .await
            .map_err(ReceiveRequestError::FailedToJoinHandle)?
        {
            Ok(res) => (res, true),
            Err(e) => {
                tracing::error!("Failed to process iris shares: {:?}", e);
                // If we failed to process the iris shares, we include a dummy entry in the
                // batch in order to keep the same order across nodes
                let dummy_code_share = GaloisRingIrisCodeShare::default_for_party(party_id);
                let dummy_mask_share = GaloisRingTrimmedMaskCodeShare::default_for_party(party_id);
                (
                    (
                        (
                            dummy_code_share.clone(),
                            dummy_mask_share.clone(),
                            dummy_code_share.clone().all_rotations(),
                            dummy_mask_share.clone().all_rotations(),
                            dummy_code_share.clone().all_rotations(),
                            dummy_mask_share.clone().all_rotations(),
                        ),
                        (
                            dummy_code_share.clone(),
                            dummy_mask_share.clone(),
                            dummy_code_share.clone().all_rotations(),
                            dummy_mask_share.clone().all_rotations(),
                            dummy_code_share.clone().all_rotations(),
                            dummy_mask_share.clone().all_rotations(),
                        ),
                    ),
                    false,
                )
            }
        };

        batch_query.valid_entries.push(valid_entry);

        batch_query.store_left.code.push(store_iris_shares_left);
        batch_query.store_left.mask.push(store_mask_shares_left);
        batch_query.db_left.code.extend(db_iris_shares_left);
        batch_query.db_left.mask.extend(db_mask_shares_left);
        batch_query.query_left.code.extend(iris_shares_left);
        batch_query.query_left.mask.extend(mask_shares_left);

        batch_query.store_right.code.push(store_iris_shares_right);
        batch_query.store_right.mask.push(store_mask_shares_right);
        batch_query.db_right.code.extend(db_iris_shares_right);
        batch_query.db_right.mask.extend(db_mask_shares_right);
        batch_query.query_right.code.extend(iris_shares_right);
        batch_query.query_right.mask.extend(mask_shares_right);
    }

    // Preprocess query shares here already to avoid blocking the actor
    batch_query.query_left_preprocessed =
        BatchQueryEntriesPreprocessed::from(batch_query.query_left.clone());
    batch_query.query_right_preprocessed =
        BatchQueryEntriesPreprocessed::from(batch_query.query_right.clone());
    batch_query.db_left_preprocessed =
        BatchQueryEntriesPreprocessed::from(batch_query.db_left.clone());
    batch_query.db_right_preprocessed =
        BatchQueryEntriesPreprocessed::from(batch_query.db_right.clone());

    Ok(batch_query)
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

async fn initialize_chacha_seeds(
    kms_key_arns: &JsonStrWrapper<Vec<String>>,
    party_id: usize,
) -> eyre::Result<([u32; 8], [u32; 8])> {
    // Init RNGs
    let own_key_arn = kms_key_arns
        .0
        .get(party_id)
        .expect("Expected value not found in kms_key_arns");
    let dh_pairs = match party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let dh_pair_0: &str = kms_key_arns
        .0
        .get(dh_pairs.0)
        .expect("Expected value not found in kms_key_arns");
    let dh_pair_1: &str = kms_key_arns
        .0
        .get(dh_pairs.1)
        .expect("Expected value not found in kms_key_arns");

    let chacha_seeds = (
        bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_0).await?),
        bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_1).await?),
    );

    Ok(chacha_seeds)
}

async fn send_results_to_sns(
    result_events: Vec<String>,
    metadata: &[BatchMetadata],
    sns_client: &SNSClient,
    config: &Config,
    base_message_attributes: &HashMap<String, MessageAttributeValue>,
    message_type: &str,
) -> eyre::Result<()> {
    for (i, result_event) in result_events.iter().enumerate() {
        let mut message_attributes = base_message_attributes.clone();
        if metadata.len() > i {
            let trace_attributes =
                construct_message_attributes(&metadata[i].trace_id, &metadata[i].span_id)?;
            message_attributes.extend(trace_attributes);
        }
        sns_client
            .publish()
            .topic_arn(&config.results_topic_arn)
            .message(result_event)
            .message_group_id(format!("party-id-{}", config.party_id))
            .set_message_attributes(Some(message_attributes))
            .send()
            .await?;
        metrics::counter!("result.sent", "type" => message_type.to_owned()).increment(1);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    println!("Init tracing");
    let _tracing_shutdown_handle = match initialize_tracing(&config) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    match server_main(config).await {
        Ok(_) => {
            tracing::info!("Server exited normally");
        }
        Err(e) => {
            tracing::error!("Server exited with error: {:?}", e);
            return Err(e);
        }
    }
    Ok(())
}

async fn server_main(config: Config) -> eyre::Result<()> {
    // Load batch_size config
    *CURRENT_BATCH_SIZE.lock().unwrap() = config.max_batch_size;
    let max_sync_lookback: usize = config.max_batch_size * 2;
    let max_rollback: usize = config.max_batch_size * 2;
    assert!(max_sync_lookback <= sync_nccl::MAX_REQUESTS);
    tracing::info!("Set batch size to {}", config.max_batch_size);

    tracing::info!("Creating new storage from: {:?}", config);
    let store = Store::new_from_config(&config).await?;

    tracing::info!("Initialising AWS services");

    // TODO: probably move into separate function
    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let sqs_client = Client::new(&shared_config);
    let sns_client = SNSClient::new(&shared_config);
    let shares_encryption_key_pair =
        match SharesEncryptionKeyPairs::from_storage(config.clone()).await {
            Ok(key_pair) => key_pair,
            Err(e) => {
                tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
                return Ok(());
            }
        };

    let party_id = config.party_id;
    tracing::info!("Deriving shared secrets");
    let chacha_seeds = initialize_chacha_seeds(&config.kms_key_arns, party_id).await?;

    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);
    tracing::info!("Replaying results");
    send_results_to_sns(
        store.last_results(max_sync_lookback).await?,
        &Vec::new(),
        &sns_client,
        &config,
        &uniqueness_result_attributes,
        UNIQUENESS_MESSAGE_TYPE,
    )
    .await?;

    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database before init: {}", store_len);

    // Seed the persistent storage with random shares if configured and db is still
    // empty.
    if store_len == 0 && config.init_db_size > 0 {
        tracing::info!(
            "Initialize persistent iris DB with {} randomly generated shares",
            config.init_db_size
        );
        tracing::info!("Resetting the db: {}", config.clear_db_before_init);
        store
            .init_db_with_random_shares(
                RNG_SEED_INIT_DB,
                party_id,
                config.init_db_size,
                config.clear_db_before_init,
            )
            .await?;
    }

    // Fetch again in case we've just initialized the DB
    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database after init: {}", store_len);

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        eyre::bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

    let my_state = SyncState {
        db_len:              store_len as u64,
        deleted_request_ids: store.last_deleted_requests(max_sync_lookback).await?,
    };

    tracing::info!("Preparing task monitor");
    let mut background_tasks = TaskMonitor::new();

    // Start the actor in separate task.
    // A bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    let (tx, rx) = oneshot::channel();
    background_tasks.spawn_blocking(move || {
        let device_manager = Arc::new(DeviceManager::init());
        let ids = device_manager.get_ids_from_magic(0);

        tracing::info!("Starting NCCL");
        let comms = device_manager.instantiate_network_from_ids(config.party_id, &ids)?;

        tracing::info!("NCCL: getting sync results");
        let sync_result = match sync_nccl::sync(&comms[0], &my_state) {
            Ok(res) => res,
            Err(e) => {
                tx.send(Err(e)).unwrap();
                return Ok(());
            }
        };

        if let Some(db_len) = sync_result.must_rollback_storage() {
            tracing::error!("Databases are out-of-sync: {:?}", sync_result);
            if db_len + max_rollback < store_len {
                return Err(eyre!(
                    "Refusing to rollback so much (from {} to {})",
                    store_len,
                    db_len,
                ));
            }
            tokio::runtime::Handle::current().block_on(async { store.rollback(db_len).await })?;
            tracing::error!("Rolled back to db_len={}", db_len);
        }

        tracing::info!("Starting server actor");
        match ServerActor::new_with_device_manager_and_comms(
            config.party_id,
            chacha_seeds,
            device_manager,
            comms,
            8,
            config.max_db_size,
            config.max_batch_size,
            config.return_partial_results,
            config.disable_persistence,
        ) {
            Ok((mut actor, handle)) => {
                let res = if config.fake_db_size > 0 {
                    tracing::warn!(
                        "Faking db with {} entries, returned results will be random.",
                        config.fake_db_size
                    );
                    actor.set_current_db_sizes(vec![
                        config.fake_db_size
                            / actor.current_db_sizes().len();
                        actor.current_db_sizes().len()
                    ]);
                    Ok(())
                } else {
                    tracing::info!(
                        "Initialize iris db: Loading from DB (parallelism: {})",
                        parallelism
                    );
                    tokio::runtime::Handle::current().block_on(async {
                        let mut stream = store.stream_irises_par(parallelism).await;
                        let mut record_counter = 0;
                        while let Some(iris) = stream.try_next().await? {
                            if record_counter % 100_000 == 0 {
                                tracing::info!(
                                    "Loaded {} records from db into memory",
                                    record_counter
                                );
                            }
                            if iris.index() > store_len {
                                tracing::error!("Inconsistent iris index {}", iris.index());
                                return Err(eyre!("Inconsistent iris index {}", iris.index()));
                            }
                            actor.load_single_record(
                                iris.index() - 1,
                                iris.left_code(),
                                iris.left_mask(),
                                iris.right_code(),
                                iris.right_mask(),
                            );
                            record_counter += 1;
                        }

                        tracing::info!("Preprocessing db");
                        actor.preprocess_db();

                        tracing::info!(
                            "Loaded {} records from db into memory [DB sizes: {:?}]",
                            record_counter,
                            actor.current_db_sizes()
                        );

                        eyre::Ok(())
                    })
                };

                match res {
                    Ok(_) => {
                        tx.send(Ok((handle, sync_result, store))).unwrap();
                    }
                    Err(e) => {
                        tx.send(Err(e)).unwrap();
                        return Ok(());
                    }
                }

                actor.run(); // forever
            }
            Err(e) => {
                tx.send(Err(e)).unwrap();
                return Ok(());
            }
        };
        Ok(())
    });

    let (mut handle, sync_result, store) = rx.await??;

    let mut skip_request_ids = sync_result.deleted_request_ids();

    background_tasks.check_tasks();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let sns_client_bg = sns_client.clone();
    let config_bg = config.clone();
    let store_bg = store.clone();
    let _result_sender_abort = background_tasks.spawn(async move {
        while let Some(ServerJobResult {
            merged_results,
            request_ids,
            metadata,
            matches,
            match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            store_left,
            store_right,
            deleted_ids,
        }) = rx.recv().await
        {
            // returned serial_ids are 0 indexed, but we want them to be 1 indexed
            let uniqueness_results = merged_results
                .iter()
                .enumerate()
                .map(|(i, &idx_result)| {
                    let result_event = UniquenessResult::new(
                        party_id,
                        match matches[i] {
                            true => None,
                            false => Some(idx_result + 1),
                        },
                        matches[i],
                        request_ids[i].clone(),
                        match matches[i] {
                            true => Some(match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>()),
                            false => None,
                        },
                        match partial_match_ids_left[i].is_empty() {
                            false => Some(
                                partial_match_ids_left[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match partial_match_ids_right[i].is_empty() {
                            false => Some(
                                partial_match_ids_right[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                    );

                    serde_json::to_string(&result_event).wrap_err("failed to serialize result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            // Insert non-matching queries into the persistent store.
            let (memory_serial_ids, codes_and_masks): (Vec<u32>, Vec<StoredIrisRef>) = matches
                .iter()
                .enumerate()
                .filter_map(
                    // Find the indices of non-matching queries in the batch.
                    |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
                )
                .map(|query_idx| {
                    // Get the original vectors from `receive_batch`.
                    (merged_results[query_idx] + 1, StoredIrisRef {
                        left_code:  &store_left.code[query_idx].coefs[..],
                        left_mask:  &store_left.mask[query_idx].coefs[..],
                        right_code: &store_right.code[query_idx].coefs[..],
                        right_mask: &store_right.mask[query_idx].coefs[..],
                    })
                })
                .unzip();

            let mut tx = store_bg.tx().await?;

            store_bg
                .insert_results(&mut tx, &uniqueness_results)
                .await?;

            if !codes_and_masks.is_empty() && !config_bg.disable_persistence {
                let db_serial_ids = store_bg
                    .insert_irises(&mut tx, &codes_and_masks)
                    .await
                    .wrap_err("failed to persist queries")?
                    .iter()
                    .map(|&x| x as u32)
                    .collect::<Vec<_>>();

                // Check if the serial_ids match between memory and db.
                if memory_serial_ids != db_serial_ids {
                    tracing::error!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    );
                    return Err(eyre!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    ));
                }
            }

            tx.commit().await?;

            for memory_serial_id in memory_serial_ids {
                tracing::info!("Inserted serial_id: {}", memory_serial_id);
                metrics::gauge!("results_inserted.latest_serial_id").set(memory_serial_id as f64);
            }

            tracing::info!("Sending {} uniqueness results", uniqueness_results.len());
            send_results_to_sns(
                uniqueness_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &uniqueness_result_attributes,
                UNIQUENESS_MESSAGE_TYPE,
            )
            .await?;

            // handling identity deletion results
            let identity_deletion_results = deleted_ids
                .iter()
                .map(|&serial_id| {
                    let result_event = IdentityDeletionResult::new(party_id, serial_id + 1, true);
                    serde_json::to_string(&result_event)
                        .wrap_err("failed to serialize identity deletion result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            tracing::info!(
                "Sending {} identity deletion results",
                identity_deletion_results.len()
            );
            send_results_to_sns(
                identity_deletion_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &identity_deletion_result_attributes,
                IDENTITY_DELETION_MESSAGE_TYPE,
            )
            .await?;
        }

        Ok(())
    });
    background_tasks.check_tasks();

    tracing::info!("All systems ready.");
    tracing::info!("Starting healthcheck server.");

    let _health_check_abort = background_tasks.spawn(async move {
        // Generate a random UUID for each run.
        let uuid = uuid::Uuid::new_v4().to_string();
        let app = Router::new().route("/health", get(|| async { uuid })); // implicit 200 return
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .wrap_err("healthcheck listener bind error")?;
        axum::serve(listener, app)
            .await
            .wrap_err("healthcheck listener server launch error")?;

        Ok(())
    });

    background_tasks.check_tasks();
    tracing::info!("Healthcheck server running on port 3000.");

    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);
    let all_nodes = config.node_hostnames.clone();
    let _heartbeat = background_tasks.spawn(async move {
        let next_node = &all_nodes[(config.party_id + 1) % 3];
        let prev_node = &all_nodes[(config.party_id + 2) % 3];
        let mut last_response = [String::default(), String::default()];
        let mut connected = [false, false];
        let mut retries = [0, 0];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(format!("http://{}:3000/health", host)).await;
                if res.is_err() || !res.as_ref().unwrap().status().is_success() {
                    // If it's the first time after startup, we allow a few retries to let the other
                    // nodes start up as well.
                    if last_response[i] == String::default()
                        && retries[i] < config.heartbeat_initial_retries
                    {
                        retries[i] += 1;
                        tracing::warn!("Node {} did not respond with success, retrying...", host);
                        continue;
                    }
                    // The other node seems to be down or returned an error.
                    panic!(
                        "Node {} did not respond with success, killing server...",
                        host
                    );
                }

                let uuid = res.unwrap().text().await?;
                if last_response[i] == String::default() {
                    last_response[i] = uuid;
                    connected[i] = true;

                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        if let Some(tx) = heartbeat_tx.take() {
                            tx.send(()).unwrap();
                        }
                    }
                } else if uuid != last_response[i] {
                    // If the UUID response is different, the node has restarted without us
                    // noticing. Our main NCCL connections cannot recover from
                    // this, so we panic.
                    panic!("Node {} seems to have restarted, killing server...", host);
                } else {
                    tracing::info!("Heartbeat: Node {} is healthy", host);
                }
            }

            tokio::time::sleep(Duration::from_secs(config.heartbeat_interval_secs)).await;
        }
    });

    tracing::info!("Heartbeat starting...");
    heartbeat_rx.await?;
    tracing::info!("Heartbeat on all nodes started.");
    background_tasks.check_tasks();

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);

    // Main loop
    let res: eyre::Result<()> = async {
        tracing::info!("Entering main loop");
        // **Tensor format of queries**
        //
        // The functions `receive_batch` and `prepare_query_shares` will prepare the
        // _query_ variables as `Vec<Vec<u8>>` formatted as follows:
        //
        // - The inner Vec is a flattening of these dimensions (inner to outer):
        //   - One u8 limb of one iris bit.
        //   - One code: 12800 coefficients.
        //   - One query: all rotated variants of a code.
        //   - One batch: many queries.
        // - The outer Vec is the dimension of the Galois Ring (2):
        //   - A decomposition of each iris bit into two u8 limbs.

        // Skip requests based on the startup sync, only in the first iteration.
        let skip_request_ids = mem::take(&mut skip_request_ids);
        let shares_encryption_key_pair = shares_encryption_key_pair.clone();
        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above
        let mut next_batch = receive_batch(
            party_id,
            &sqs_client,
            &config.requests_queue_url,
            &store,
            &skip_request_ids,
            shares_encryption_key_pair.clone(),
            config.max_batch_size,
        );

        let dummy_shares_for_deletions = get_dummy_shares_for_deletion(party_id);

        loop {
            let now = Instant::now();

            let batch = next_batch.await?;

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());

            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());

            process_identity_deletions(
                &batch,
                &store,
                &dummy_shares_for_deletions.0,
                &dummy_shares_for_deletions.1,
            )
            .await?;

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id and
            // parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Started processing share",
                );
            }

            background_tasks.check_tasks();

            let result_future = handle.submit_batch_query(batch);

            next_batch = receive_batch(
                party_id,
                &sqs_client,
                &config.requests_queue_url,
                &store,
                &skip_request_ids,
                shares_encryption_key_pair.clone(),
                config.max_batch_size,
            );

            // await the result
            let result = timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("ServerActor processing timeout: {:?}", e))?;

            tx.send(result).await?;

            // wrap up span context
        }
    }
    .await;

    match res {
        Ok(_) => {}
        Err(e) => {
            tracing::error!("ServerActor processing error: {:?}", e);
            // drop actor handle to initiate shutdown
            drop(handle);

            // Clean up server tasks, then wait for them to finish
            background_tasks.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Check for background task hangs and shutdown panics
            background_tasks.check_tasks_finished();
        }
    }

    Ok(())
}

async fn process_identity_deletions(
    batch: &BatchQuery,
    store: &Store,
    dummy_iris_share: &GaloisRingIrisCodeShare,
    dummy_mask_share: &GaloisRingTrimmedMaskCodeShare,
) -> eyre::Result<()> {
    if batch.deletion_requests_indices.is_empty() {
        return Ok(());
    }

    for (&entry_idx, tracing_payload) in batch
        .deletion_requests_indices
        .iter()
        .zip(batch.deletion_requests_metadata.iter())
    {
        let serial_id = entry_idx + 1; // DB serial_id is 1-indexed
        tracing::info!(
            node_id = tracing_payload.node_id,
            dd.trace_id = tracing_payload.trace_id,
            dd.span_id = tracing_payload.span_id,
            "Started processing deletion request",
        );

        // overwrite postgres db with dummy values.
        // note that both serial_id and postgres db are 1-indexed.
        store
            .update_iris(
                serial_id as i64,
                dummy_iris_share,
                dummy_mask_share,
                dummy_iris_share,
                dummy_mask_share,
            )
            .await?;

        tracing::info!(
            node_id = tracing_payload.node_id,
            dd.trace_id = tracing_payload.trace_id,
            dd.span_id = tracing_payload.span_id,
            "Deleted identity with serial id {}",
            serial_id,
        );
    }

    Ok(())
}
