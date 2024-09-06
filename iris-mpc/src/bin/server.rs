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
        aws::{SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME},
        key_pair::SharesEncryptionKeyPairs,
        kms_dh::derive_shared_secret,
        smpc_request::{
            create_message_type_attribute_map, IdentityDeletionRequest, IdentityDeletionResult,
            ReceiveRequestError, SQSMessage, UniquenessRequest, UniquenessResult,
            IDENTITY_DELETION_MESSAGE_TYPE, SMPC_MESSAGE_TYPE_ATTRIBUTE, UNIQUENESS_MESSAGE_TYPE,
        },
        sync::SyncState,
        task_monitor::TaskMonitor,
    },
    iris_db::iris::IrisCode,
    IrisCodeDb, IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
};
use iris_mpc_gpu::{
    helpers::device_manager::DeviceManager,
    server::{
        heartbeat_nccl::start_heartbeat, sync_nccl, BatchMetadata, BatchQuery, ServerActor,
        ServerJobResult,
    },
};
use iris_mpc_store::{Store, StoredIrisRef};
use rand::{rngs::StdRng, SeedableRng};
use std::{
    collections::HashMap,
    mem,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};
use telemetry_batteries::{
    metrics::statsd::StatsdBattery,
    tracing::{datadog::DatadogBattery, TracingShutdownHandle},
};
use tokio::{
    sync::{mpsc, oneshot, Semaphore},
    task::spawn_blocking,
    time::timeout,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REGION: &str = "eu-north-1";
const DB_BUFFER: usize = 8 * 1_000;
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

                let message_attributes = sqs_message.message_attributes.unwrap_or_default();

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
                        batch_query
                            .deletion_requests
                            .push(identity_deletion_request.serial_id);
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
                                iris_message_share.left_iris_mask_shares,
                            )?;

                            let (right_code, right_mask) = decode_iris_message_shares(
                                iris_message_share.right_iris_code_shares,
                                iris_message_share.right_iris_mask_shares,
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
            StatsdBattery::init(
                &metrics_config.host,
                metrics_config.port,
                metrics_config.queue_size,
                metrics_config.buffer_size,
                Some(&metrics_config.prefix),
            )?;
        }

        Ok(tracing_shutdown_handle)
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().pretty().compact())
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();

        Ok(TracingShutdownHandle)
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

async fn initialize_iris_dbs(
    party_id: usize,
    store: &Store,
    config: &Config,
) -> eyre::Result<(IrisCodeDb, IrisCodeDb, usize)> {
    // Generate or load DB

    tracing::info!("Initialize persistent iris db with randomly generated shares");
    store
        .init_db_with_random_shares(
            RNG_SEED_INIT_DB,
            party_id,
            config.init_db_size,
            config.clear_db_before_init,
        )
        .await
        .expect("failed to initialise db");

    let count_irises = store.count_irises().await?;
    tracing::info!("Initialize iris db: Counted {} entries in DB", count_irises);

    let mut left_codes_db: Vec<u16> = vec![0u16; count_irises * IRIS_CODE_LENGTH];
    let mut left_masks_db: Vec<u16> = vec![0u16; count_irises * MASK_CODE_LENGTH];
    let mut right_codes_db: Vec<u16> = vec![0u16; count_irises * IRIS_CODE_LENGTH];
    let mut right_masks_db: Vec<u16> = vec![0u16; count_irises * MASK_CODE_LENGTH];

    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    tracing::info!(
        "Initialize iris db: Loading from DB (parallelism: {})",
        parallelism
    );
    // Load DB from persistent storage.
    let mut store_len = 0;
    let mut stream = store.stream_irises_par(parallelism).await;
    while let Some(iris) = stream.try_next().await? {
        if iris.index() >= count_irises {
            return Err(eyre!("Inconsistent iris index {}", iris.index()));
        }

        let start_code = iris.index() * IRIS_CODE_LENGTH;
        let start_mask = iris.index() * MASK_CODE_LENGTH;
        left_codes_db[start_code..start_code + IRIS_CODE_LENGTH].copy_from_slice(iris.left_code());
        left_masks_db[start_mask..start_mask + MASK_CODE_LENGTH].copy_from_slice(iris.left_mask());
        right_codes_db[start_code..start_code + IRIS_CODE_LENGTH]
            .copy_from_slice(iris.right_code());
        right_masks_db[start_mask..start_mask + MASK_CODE_LENGTH]
            .copy_from_slice(iris.right_mask());

        store_len += 1;
        if (store_len % 10000) == 0 {
            tracing::info!("Initialize iris db: Loaded {} entries from DB", store_len);
        }
    }
    tracing::info!(
        "Initialize iris db: Loaded {} entries from DB, done!",
        store_len
    );

    Ok((
        (left_codes_db, left_masks_db),
        (right_codes_db, right_masks_db),
        count_irises,
    ))
}

async fn send_results_to_sns(
    result_events: Vec<String>,
    sns_client: &SNSClient,
    config: &Config,
    message_attributes: &HashMap<String, MessageAttributeValue>,
) -> eyre::Result<()> {
    for result_event in result_events {
        sns_client
            .publish()
            .topic_arn(&config.results_topic_arn)
            .message(result_event)
            .message_group_id(format!("party-id-{}", config.party_id))
            .set_message_attributes(Some(message_attributes.clone()))
            .send()
            .await?;
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
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    tracing::info!("Replaying results");
    send_results_to_sns(
        store.last_results(max_sync_lookback).await?,
        &sns_client,
        &config,
        &uniqueness_result_attributes,
    )
    .await?;

    tracing::info!("Initialize iris db");
    let (mut left_iris_db, mut right_iris_db, store_len) =
        initialize_iris_dbs(party_id, &store, &config).await?;

    let my_state = SyncState {
        db_len:              store_len as u64,
        deleted_request_ids: store.last_deleted_requests(max_sync_lookback).await?,
    };

    tracing::info!("Preparing task monitor");
    let mut background_tasks = TaskMonitor::new();

    let (tx, rx) = oneshot::channel();
    let _heartbeat = background_tasks.spawn(start_heartbeat(config.party_id, tx));

    background_tasks.check_tasks();
    tracing::info!("Heartbeat starting...");
    rx.await??;
    tracing::info!("Heartbeat started.");

    // a bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
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

        tracing::info!("DB: check if rollback needed");
        if let Some(db_len) = sync_result.must_rollback_storage() {
            tracing::warn!(
                "Databases are out-of-sync, rolling back (current len: {}, new len: {})",
                store_len,
                db_len
            );
            // Rollback the data that we have already loaded.
            let bit_len_code = db_len * IRIS_CODE_LENGTH;
            let bit_len_mask = db_len * MASK_CODE_LENGTH;

            // TODO: remove the line below if you removed fake data.
            let bit_len_code = bit_len_code + (left_iris_db.0.len() - store_len * IRIS_CODE_LENGTH);
            let bit_len_mask = bit_len_mask + (left_iris_db.1.len() - store_len * MASK_CODE_LENGTH);
            left_iris_db.0.truncate(bit_len_code);
            left_iris_db.1.truncate(bit_len_mask);
            right_iris_db.0.truncate(bit_len_code);
            right_iris_db.1.truncate(bit_len_mask);
        }

        tracing::info!("Starting server actor");
        match ServerActor::new_with_device_manager_and_comms(
            config.party_id,
            chacha_seeds,
            (&left_iris_db.0, &left_iris_db.1),
            (&right_iris_db.0, &right_iris_db.1),
            device_manager,
            comms,
            8,
            store_len,
            DB_BUFFER,
            config.max_batch_size,
        ) {
            Ok((actor, handle)) => {
                tx.send(Ok((handle, sync_result))).unwrap();
                actor.run(); // forever
            }
            Err(e) => {
                tx.send(Err(e)).unwrap();
                return Ok(());
            }
        };
        Ok(())
    });

    let (mut handle, sync_result) = rx.await??;

    if let Some(db_len) = sync_result.must_rollback_storage() {
        tracing::error!("Databases are out-of-sync: {:?}", sync_result);
        if db_len + max_rollback < store_len {
            return Err(eyre!(
                "Refusing to rollback so much (from {} to {})",
                store_len,
                db_len,
            ));
        }
        store.rollback(db_len).await?;
        tracing::error!("Rolled back to db_len={}", db_len);
    }

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
            matches,
            match_ids,
            store_left,
            store_right,
        }) = rx.recv().await
        {
            // returned serial_ids are 0 indexed, but we want them to be 1 indexed
            let result_events = merged_results
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
                    );

                    serde_json::to_string(&result_event).wrap_err("failed to serialize result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            // Insert non-matching queries into the persistent store.
            let codes_and_masks: Vec<StoredIrisRef> = matches
                .iter()
                .enumerate()
                .filter_map(
                    // Find the indices of non-matching queries in the batch.
                    |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
                )
                .map(|query_idx| {
                    // Get the original vectors from `receive_batch`.
                    StoredIrisRef {
                        left_code:  &store_left.code[query_idx].coefs[..],
                        left_mask:  &store_left.mask[query_idx].coefs[..],
                        right_code: &store_right.code[query_idx].coefs[..],
                        right_mask: &store_right.mask[query_idx].coefs[..],
                    }
                })
                .collect();

            let mut tx = store_bg.tx().await?;

            store_bg.insert_results(&mut tx, &result_events).await?;

            if !codes_and_masks.is_empty() {
                store_bg
                    .insert_irises(&mut tx, &codes_and_masks)
                    .await
                    .wrap_err("failed to persist queries")?;
            }

            tx.commit().await?;

            tracing::info!("Sending {} uniqueness results", result_events.len());
            send_results_to_sns(
                result_events,
                &sns_client_bg,
                &config_bg,
                &uniqueness_result_attributes,
            )
            .await?;
        }

        Ok(())
    });
    background_tasks.check_tasks();

    tracing::info!("All systems ready.");
    tracing::info!("Starting healthcheck server.");

    let _health_check_abort = background_tasks.spawn(async move {
        let app = Router::new().route("/health", get(|| async {})); // implicit 200 return
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

            process_identity_deletions(
                &batch,
                &store,
                &dummy_shares_for_deletions.0,
                &dummy_shares_for_deletions.1,
            )
            .await?;

            let identity_deletion_results = batch
                .deletion_requests
                .iter()
                .map(|serial_id| {
                    let result_event = IdentityDeletionResult::new(party_id, *serial_id, true);
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
                &sns_client,
                &config,
                &identity_deletion_result_attributes,
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

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());
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
    if batch.deletion_requests.is_empty() {
        return Ok(());
    }

    for (serial_id, tracing_payload) in batch
        .deletion_requests
        .iter()
        .zip(batch.deletion_requests_metadata.iter())
    {
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
                *serial_id as i64,
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

fn get_dummy_shares_for_deletion(
    party_id: usize,
) -> (GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare) {
    let mut rng: StdRng = StdRng::seed_from_u64(0);
    let dummy: IrisCode = IrisCode::default();
    let iris_share: GaloisRingIrisCodeShare =
        GaloisRingIrisCodeShare::encode_iris_code(&dummy.code, &dummy.mask, &mut rng)[party_id]
            .clone();
    let mask_share: GaloisRingTrimmedMaskCodeShare =
        GaloisRingIrisCodeShare::encode_mask_code(&dummy.mask, &mut rng)[party_id]
            .clone()
            .into();
    (iris_share, mask_share)
}
