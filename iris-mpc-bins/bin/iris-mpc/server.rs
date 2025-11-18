#![allow(clippy::needless_range_loop, unused)]

use ampc_server_utils::{
    delete_messages_until_sequence_num, get_next_sns_seq_num, shutdown_handler::ShutdownHandler,
    ReadyProbeResponse, TaskMonitor,
};
use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use aws_sdk_sqs::Client;
use axum::{response::IntoResponse, routing::get, Router};
use chrono::Utc;
use clap::Parser;
use eyre::{bail, eyre, Context, Report, Result};
use futures::{stream::BoxStream, StreamExt};
use iris_mpc::services::aws::clients::AwsClients;
use iris_mpc::services::init::initialize_chacha_seeds;
use iris_mpc::services::processors::get_iris_shares_parse_task;
use iris_mpc::services::processors::modifications_sync::{
    send_last_modifications_to_sns, sync_modifications,
};
use iris_mpc::services::processors::result_message::{
    send_error_results_to_sns, send_results_to_sns,
};
use iris_mpc_common::config::CommonConfig;
use iris_mpc_common::galois_engine::degree4::GaloisShares;
use iris_mpc_common::helpers::sync::ModificationKey::{RequestId, RequestSerialId};
use iris_mpc_common::job::{GaloisSharesBothSides, RequestIndex};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_common::{
    config::{Config, Opt},
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        aws::{SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME},
        inmemory_store::InMemoryStore,
        key_pair::SharesEncryptionKeyPairs,
        smpc_request::{
            decrypt_iris_share, get_iris_data_by_party_id, validate_iris_share,
            CircuitBreakerRequest, IdentityDeletionRequest, ReAuthRequest, ReceiveRequestError,
            ResetCheckRequest, ResetUpdateRequest, SQSMessage, UniquenessRequest,
            ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE, ANONYMIZED_STATISTICS_MESSAGE_TYPE,
            CIRCUIT_BREAKER_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
            RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
        },
        smpc_response::{
            create_message_type_attribute_map, IdentityDeletionResult, ReAuthResult,
            ResetCheckResult, ResetUpdateAckResult, UniquenessResult,
            ERROR_FAILED_TO_PROCESS_IRIS_SHARES, ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH,
            SMPC_MESSAGE_TYPE_ATTRIBUTE,
        },
        sqs_s3_helper::upload_file_to_s3,
        sync::{Modification, ModificationKey, SyncResult, SyncState},
    },
    iris_db::get_dummy_shares_for_deletion,
    job::{BatchMetadata, BatchQuery, JobSubmissionHandle, ServerJobResult},
};
use iris_mpc_gpu::server::ServerActor;
use iris_mpc_store::loader::load_iris_db;
use iris_mpc_store::{
    fetch_and_parse_chunks, last_snapshot_timestamp, DbStoredIris, ObjectStore, S3Store,
    S3StoredIris, Store, StoredIrisRef,
};
use itertools::izip;
use metrics_exporter_statsd::StatsdBuilder;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sodiumoxide::hex;
use std::process::exit;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem, panic,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc::Receiver;
use tokio::{
    sync::{mpsc, oneshot, Semaphore},
    task::{spawn_blocking, JoinHandle},
    time::timeout,
};

const RNG_SEED_INIT_DB: u64 = 42;
const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
const MAX_CONCURRENT_REQUESTS: usize = 32;

static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

fn decode_iris_message_shares(
    code_share: String,
    mask_share: String,
) -> Result<(GaloisRingIrisCodeShare, GaloisRingIrisCodeShare)> {
    let iris_share = GaloisRingIrisCodeShare::from_base64(&code_share)
        .context("Failed to base64 parse iris code")?;
    let mask_share = GaloisRingIrisCodeShare::from_base64(&mask_share)
        .context("Failed to base64 parse iris mask")?;

    Ok((iris_share, mask_share))
}

fn trim_mask(mask: GaloisRingIrisCodeShare) -> GaloisRingTrimmedMaskCodeShare {
    mask.into()
}

#[allow(clippy::too_many_arguments)]
pub fn receive_batch_stream(
    party_id: usize,
    client: Client,
    sns_client: SNSClient,
    s3_client: S3Client,
    config: Config,
    store: Store,
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: Arc<ShutdownHandler>,
    uniqueness_error_result_attributes: HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: HashMap<String, MessageAttributeValue>,
    reset_error_result_attributes: HashMap<String, MessageAttributeValue>,
) -> Receiver<Result<Option<BatchQuery>, ReceiveRequestError>> {
    let (tx, rx) = mpsc::channel(1);

    tokio::spawn(async move {
        loop {
            let permit = match tx.reserve().await {
                Ok(p) => p,
                Err(_) => break,
            };

            let batch = receive_batch(
                party_id,
                &client,
                &sns_client,
                &s3_client,
                &config,
                &store,
                shares_encryption_key_pairs.clone(),
                &shutdown_handler,
                &uniqueness_error_result_attributes,
                &reauth_error_result_attributes,
                &reset_error_result_attributes,
            )
            .await;

            let stop = matches!(batch, Err(_) | Ok(None));
            permit.send(batch);

            if stop {
                break;
            }
        }
        tracing::info!("Stopping batch receiver.");
    });

    rx
}

#[allow(clippy::too_many_arguments)]
async fn receive_batch(
    party_id: usize,
    client: &Client,
    sns_client: &SNSClient,
    s3_client: &S3Client,
    config: &Config,
    store: &Store,
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: &ShutdownHandler,
    uniqueness_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reset_error_result_attributes: &HashMap<String, MessageAttributeValue>,
) -> Result<Option<BatchQuery>, ReceiveRequestError> {
    let max_batch_size = config.clone().max_batch_size;
    let queue_url = &config.clone().requests_queue_url;
    if shutdown_handler.is_shutting_down() {
        tracing::info!("Stopping batch receive due to shutdown signal...");
        return Ok(None);
    }

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
                let sns_message_id = message.message_id;

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
                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;
                        metrics::counter!("request.received", "type" => "identity_deletion")
                            .increment(1);
                        if batch_query
                            .modifications
                            .contains_key(&RequestSerialId(identity_deletion_request.serial_id))
                        {
                            tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                identity_deletion_request.serial_id,
                                identity_deletion_request,
                            );
                            continue;
                        }
                        let modification = store
                            .insert_modification(
                                Some(identity_deletion_request.serial_id as i64),
                                IDENTITY_DELETION_MESSAGE_TYPE,
                                None,
                            )
                            .await?;
                        batch_query.modifications.insert(
                            RequestSerialId(identity_deletion_request.serial_id),
                            modification,
                        );

                        batch_query.push_deletion_request(
                            sns_message_id,
                            identity_deletion_request.serial_id - 1,
                            batch_metadata,
                        );
                    }

                    UNIQUENESS_MESSAGE_TYPE => {
                        msg_counter += 1;

                        let shares_encryption_key_pairs = shares_encryption_key_pairs.clone();

                        let uniqueness_request: UniquenessRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error("Uniqueness request", e)
                            })?;
                        metrics::counter!("request.received", "type" => "uniqueness_verification")
                            .increment(1);

                        // If any request in the batch asks to disable anonymized statistics,
                        // apply it for the whole batch.
                        if let Some(disable) = uniqueness_request.disable_anonymized_stats {
                            if disable && !batch_query.disable_anonymized_stats {
                                batch_query.disable_anonymized_stats = true;
                                tracing::debug!(
                                    "Disabling anonymized statistics for current batch due to request {}",
                                    uniqueness_request.signup_id
                                );
                            }
                        }

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if let Some(batch_size) = uniqueness_request.batch_size {
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

                        let modification = store
                            .insert_modification(
                                None,
                                UNIQUENESS_MESSAGE_TYPE,
                                Some(uniqueness_request.s3_key.as_str()),
                            )
                            .await?;
                        batch_query.modifications.insert(
                            RequestId(uniqueness_request.signup_id.clone()),
                            modification,
                        );

                        if let Some(enable_mirror_attacks) =
                            uniqueness_request.full_face_mirror_attacks_detection_enabled
                        {
                            if enable_mirror_attacks
                                != batch_query.full_face_mirror_attacks_detection_enabled
                            {
                                batch_query.full_face_mirror_attacks_detection_enabled =
                                    enable_mirror_attacks;
                                tracing::info!(
                                    "Setting mirror attack to {} for batch due to request from {}",
                                    enable_mirror_attacks,
                                    uniqueness_request.signup_id
                                );
                            }
                        }

                        if let Some(skip_persistence) = uniqueness_request.skip_persistence {
                            tracing::info!(
                                "Setting skip_persistence to {} for request id {}",
                                skip_persistence,
                                uniqueness_request.signup_id
                            );
                        }

                        if config.luc_enabled && config.luc_lookback_records > 0 {
                            batch_query.luc_lookback_records = config.luc_lookback_records;
                        }

                        let or_rule_indices = if config.luc_enabled
                            && config.luc_serial_ids_from_smpc_request
                        {
                            if let Some(serial_ids) = uniqueness_request.or_rule_serial_ids.as_ref()
                            {
                                // convert from 1-based serial id to 0-based index in actor
                                serial_ids.iter().map(|x| x - 1).collect()
                            } else {
                                tracing::warn!(
                                        "LUC serial ids from request enabled, but no serial_ids were passed"
                                    );
                                vec![]
                            }
                        } else {
                            vec![]
                        };

                        batch_query.push_matching_request(
                            sns_message_id,
                            uniqueness_request.signup_id.clone(),
                            UNIQUENESS_MESSAGE_TYPE,
                            batch_metadata,
                            or_rule_indices,
                            uniqueness_request.skip_persistence.unwrap_or(false),
                        );

                        let semaphore = Arc::clone(&semaphore);
                        let s3_client_arc = s3_client.clone();
                        let bucket_name = config.shares_bucket_name.clone();
                        let s3_key = uniqueness_request.s3_key.clone();
                        let handle = get_iris_shares_parse_task(
                            party_id,
                            shares_encryption_key_pairs,
                            semaphore,
                            s3_client_arc,
                            bucket_name,
                            s3_key,
                        )?;

                        handles.push(handle);
                    }

                    REAUTH_MESSAGE_TYPE => {
                        let shares_encryption_key_pairs = shares_encryption_key_pairs.clone();

                        let reauth_request: ReAuthRequest = serde_json::from_str(&message.message)
                            .map_err(|e| {
                                ReceiveRequestError::json_parse_error("Reauth request", e)
                            })?;
                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        metrics::counter!("request.received", "type" => "reauth").increment(1);

                        tracing::debug!("Received reauth request: {:?}", reauth_request);

                        if config.enable_reauth {
                            if reauth_request.use_or_rule
                                && !(config.luc_enabled && config.luc_serial_ids_from_smpc_request)
                            {
                                tracing::error!(
                                "Received a reauth request with use_or_rule set to true, but LUC \
                                 is not enabled. Skipping request."
                            );
                                continue;
                            }

                            if batch_query
                                .modifications
                                .contains_key(&RequestSerialId(reauth_request.serial_id))
                            {
                                tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                reauth_request.serial_id,
                                reauth_request,
                            );
                                continue;
                            }

                            msg_counter += 1;

                            let modification = store
                                .insert_modification(
                                    Some(reauth_request.serial_id as i64),
                                    REAUTH_MESSAGE_TYPE,
                                    Some(reauth_request.s3_key.as_str()),
                                )
                                .await?;
                            batch_query
                                .modifications
                                .insert(RequestSerialId(reauth_request.serial_id), modification);

                            if let Some(batch_size) = reauth_request.batch_size {
                                // Updating the batch size instantly makes it a bit unpredictable,
                                // since if we're already above the
                                // new limit, we'll still process the current
                                // batch at the higher limit. On the other
                                // hand, updating it after the batch is
                                // processed would not let us "unblock" the protocol if we're stuck
                                // with low throughput.
                                *CURRENT_BATCH_SIZE.lock().unwrap() =
                                    batch_size.clamp(1, max_batch_size);
                                tracing::info!("Updating batch size to {}", batch_size);
                            }

                            batch_query.reauth_target_indices.insert(
                                reauth_request.reauth_id.clone(),
                                reauth_request.serial_id - 1,
                            );
                            batch_query.reauth_use_or_rule.insert(
                                reauth_request.reauth_id.clone(),
                                reauth_request.use_or_rule,
                            );

                            let or_rule_indices = if reauth_request.use_or_rule {
                                vec![reauth_request.serial_id - 1]
                            } else {
                                vec![]
                            };

                            batch_query.push_matching_request(
                                sns_message_id,
                                reauth_request.reauth_id.clone(),
                                REAUTH_MESSAGE_TYPE,
                                batch_metadata,
                                or_rule_indices,
                                false, // skip_persistence is only used for uniqueness requests
                            );

                            let semaphore = Arc::clone(&semaphore);
                            let s3_client_clone = s3_client.clone();
                            let bucket_name = config.shares_bucket_name.clone();
                            let s3_key = reauth_request.s3_key.clone();
                            let handle = get_iris_shares_parse_task(
                                party_id,
                                shares_encryption_key_pairs,
                                semaphore,
                                s3_client_clone,
                                bucket_name,
                                s3_key,
                            )?;

                            handles.push(handle);
                        } else {
                            tracing::warn!("Reauth is disabled, skipping reauth request");
                        }
                    }

                    RESET_CHECK_MESSAGE_TYPE => {
                        let shares_encryption_key_pairs = shares_encryption_key_pairs.clone();

                        let reset_check_request: ResetCheckRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error("Reset check request", e)
                            })?;
                        metrics::counter!("request.received", "type" => "reset_check").increment(1);

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if config.enable_reset {
                            msg_counter += 1;

                            // Persist in progress reset_check message.
                            // Note that reset_check is only a query and does not persist anything into the database.
                            // We store modification so that the SNS result can be replayed.
                            let modification = store
                                .insert_modification(
                                    None,
                                    RESET_CHECK_MESSAGE_TYPE,
                                    Some(reset_check_request.s3_key.as_str()),
                                )
                                .await?;
                            batch_query.modifications.insert(
                                RequestId(reset_check_request.reset_id.clone()),
                                modification,
                            );

                            if let Some(batch_size) = reset_check_request.batch_size {
                                *CURRENT_BATCH_SIZE.lock().unwrap() =
                                    batch_size.clamp(1, max_batch_size);
                                tracing::info!("Updating batch size to {}", batch_size);
                            }

                            batch_query.push_matching_request(
                                sns_message_id,
                                reset_check_request.reset_id.clone(),
                                RESET_CHECK_MESSAGE_TYPE,
                                batch_metadata,
                                vec![], // use AND rule for reset check requests
                                false,  // skip_persistence is only used for uniqueness requests
                            );

                            let semaphore = Arc::clone(&semaphore);
                            let s3_client_arc = s3_client.clone();
                            let bucket_name = config.shares_bucket_name.clone();
                            let s3_key = reset_check_request.s3_key.clone();
                            let handle = get_iris_shares_parse_task(
                                party_id,
                                shares_encryption_key_pairs,
                                semaphore,
                                s3_client_arc,
                                bucket_name,
                                s3_key,
                            )?;

                            handles.push(handle);
                        }
                    }

                    RESET_UPDATE_MESSAGE_TYPE => {
                        let shares_encryption_key_pairs = shares_encryption_key_pairs.clone();

                        let reset_update_request: ResetUpdateRequest =
                            serde_json::from_str(&message.message).map_err(|e| {
                                ReceiveRequestError::json_parse_error("Reset update request", e)
                            })?;
                        metrics::counter!("request.received", "type" => "reset_update")
                            .increment(1);

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if config.enable_reset {
                            // Fetch new iris shares from S3
                            let semaphore = Arc::clone(&semaphore);
                            let s3_client_arc = s3_client.clone();
                            let bucket_name = config.shares_bucket_name.clone();
                            let s3_key = reset_update_request.s3_key.clone();

                            let task_handle = get_iris_shares_parse_task(
                                party_id,
                                shares_encryption_key_pairs,
                                semaphore,
                                s3_client_arc,
                                bucket_name,
                                s3_key,
                            )?;

                            let (left_shares, right_shares) = match task_handle.await {
                                Ok(result) => {
                                    match result {
                                        Ok(shares) => shares,
                                        Err(e) => {
                                            tracing::error!("Failed to process iris shares for reset update: {:?}", e);
                                            continue;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to join task handle for reset update: {:?}",
                                        e
                                    );
                                    continue;
                                }
                            };

                            if batch_query
                                .modifications
                                .contains_key(&RequestSerialId(reset_update_request.serial_id))
                            {
                                tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                reset_update_request.serial_id,
                                reset_update_request,
                            );
                                continue;
                            }

                            let modification = store
                                .insert_modification(
                                    Some(reset_update_request.serial_id as i64),
                                    RESET_UPDATE_MESSAGE_TYPE,
                                    Some(reset_update_request.s3_key.as_str()),
                                )
                                .await?;
                            batch_query.modifications.insert(
                                RequestSerialId(reset_update_request.serial_id),
                                modification,
                            );

                            batch_query.push_reset_update_request(
                                sns_message_id,
                                reset_update_request.reset_id,
                                reset_update_request.serial_id - 1,
                                GaloisSharesBothSides {
                                    code_left: left_shares.code,
                                    mask_left: left_shares.mask,
                                    code_right: right_shares.code,
                                    mask_right: right_shares.mask,
                                },
                            );
                        }
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
    for (index, handle) in handles.into_iter().enumerate() {
        let result = handle
            .await
            .map_err(ReceiveRequestError::FailedToJoinHandle)?;
        let (shares, valid_entry) = match result {
            Ok(shares) => (shares, true),
            Err(e) => {
                tracing::error!("Failed to process iris shares: {:?}", e);
                // Return error message back to the signup-service if failed to process iris
                // shares
                let request_id = batch_query.request_ids[index].clone();
                let (result_attributes, message) = match batch_query.request_types[index].as_str() {
                    UNIQUENESS_MESSAGE_TYPE => {
                        let message = UniquenessResult::new_error_result(
                            config.party_id,
                            request_id,
                            ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                        );
                        let serialized = serde_json::to_string(&message).unwrap();
                        (uniqueness_error_result_attributes, serialized)
                    }
                    REAUTH_MESSAGE_TYPE => {
                        let message = ReAuthResult::new_error_result(
                            request_id.clone(),
                            config.party_id,
                            *batch_query.reauth_target_indices.get(&request_id).unwrap(),
                            ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                        );
                        let serialized = serde_json::to_string(&message).unwrap();
                        (reauth_error_result_attributes, serialized)
                    }
                    RESET_CHECK_MESSAGE_TYPE => {
                        let message = ResetCheckResult::new_error_result(
                            request_id.clone(),
                            config.party_id,
                            ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                        );
                        let serialized = serde_json::to_string(&message).unwrap();
                        (reset_error_result_attributes, serialized)
                    }
                    _ => unreachable!(), // we don't push a handle for unknown message types
                };

                send_error_results_to_sns(
                    message,
                    &batch_query.metadata[index],
                    sns_client,
                    config,
                    result_attributes,
                    batch_query.request_types[index].as_str(),
                )
                .await?;
                // If we failed to process the iris shares, we include a dummy entry in the
                // batch in order to keep the same order across nodes
                let dummy_code_share = GaloisRingIrisCodeShare::default_for_party(party_id);
                let dummy_mask_share = GaloisRingTrimmedMaskCodeShare::default_for_party(party_id);
                let dummy_one_side = GaloisShares {
                    code: dummy_code_share.clone(),
                    mask: dummy_mask_share.clone(),
                    code_rotated: dummy_code_share.clone().all_rotations(),
                    mask_rotated: dummy_mask_share.clone().all_rotations(),
                    code_interpolated: dummy_code_share.clone().all_rotations(),
                    mask_interpolated: dummy_mask_share.clone().all_rotations(),
                    code_mirrored: dummy_code_share.clone().all_rotations(),
                    mask_mirrored: dummy_mask_share.clone().all_rotations(),
                };
                ((dummy_one_side.clone(), dummy_one_side), false)
            }
        };

        batch_query.valid_entries.push(valid_entry);

        // push left iris related entries
        batch_query.left_iris_requests.code.push(shares.0.code);
        batch_query.left_iris_requests.mask.push(shares.0.mask);
        batch_query
            .left_iris_rotated_requests
            .code
            .extend(shares.0.code_rotated);
        batch_query
            .left_iris_rotated_requests
            .mask
            .extend(shares.0.mask_rotated);
        batch_query
            .left_iris_interpolated_requests
            .code
            .extend(shares.0.code_interpolated);
        batch_query
            .left_iris_interpolated_requests
            .mask
            .extend(shares.0.mask_interpolated);
        batch_query
            .left_mirrored_iris_interpolated_requests
            .code
            .extend(shares.0.code_mirrored);
        batch_query
            .left_mirrored_iris_interpolated_requests
            .mask
            .extend(shares.0.mask_mirrored);

        // push right iris related entries
        batch_query.right_iris_requests.code.push(shares.1.code);
        batch_query.right_iris_requests.mask.push(shares.1.mask);
        batch_query
            .right_iris_rotated_requests
            .code
            .extend(shares.1.code_rotated);
        batch_query
            .right_iris_rotated_requests
            .mask
            .extend(shares.1.mask_rotated);
        batch_query
            .right_iris_interpolated_requests
            .code
            .extend(shares.1.code_interpolated);
        batch_query
            .right_iris_interpolated_requests
            .mask
            .extend(shares.1.mask_interpolated);
        batch_query
            .right_mirrored_iris_interpolated_requests
            .code
            .extend(shares.1.code_mirrored);
        batch_query
            .right_mirrored_iris_interpolated_requests
            .mask
            .extend(shares.1.mask_mirrored);
    }

    tracing::info!(
        "Batch requests: {:?}",
        batch_query
            .request_ids
            .iter()
            .zip(batch_query.request_types.iter())
            .collect::<Vec<_>>()
    );

    Ok(Some(batch_query))
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    println!("Init tracing");
    let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
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
            exit(1);
        }
    }
    Ok(())
}

async fn server_main(config: Config) -> Result<()> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.register_signal_handler().await;
    // Load batch_size config
    *CURRENT_BATCH_SIZE.lock().unwrap() = config.max_batch_size;
    let max_modification_lookback = config.max_modifications_lookback;
    tracing::info!("Set batch size to {}", config.max_batch_size);

    let schema_name = format!(
        "{}{}_{}_{}",
        config.schema_name, config.gpu_schema_name_suffix, config.environment, config.party_id
    );
    let db_config = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?;

    tracing::info!(
        "Creating new iris storage from: {:?} with schema {}",
        db_config,
        schema_name
    );
    let postgres_client =
        PostgresClient::new(&db_config.url, schema_name.as_str(), AccessMode::ReadWrite).await?;
    let store = Store::new(&postgres_client).await?;

    tracing::info!("Initialising AWS services");
    let aws_clients = AwsClients::new(&config.clone()).await?;
    let next_sns_seq_number_future = get_next_sns_seq_num(
        &aws_clients.sqs_client,
        &config.requests_queue_url,
        config.sqs_sync_long_poll_seconds,
    );

    let shares_encryption_key_pair = match SharesEncryptionKeyPairs::from_storage(
        aws_clients.secrets_manager_client.clone(),
        &config.environment,
        &config.party_id,
    )
    .await
    {
        Ok(key_pair) => key_pair,
        Err(e) => {
            tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
            return Ok(());
        }
    };

    let party_id = config.party_id;
    tracing::info!("Deriving shared secrets");
    let chacha_seeds = initialize_chacha_seeds(config.clone()).await?;

    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_result_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_check_result_attributes = create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let reset_update_result_attributes =
        create_message_type_attribute_map(RESET_UPDATE_MESSAGE_TYPE);
    let anonymized_statistics_attributes =
        create_message_type_attribute_map(ANONYMIZED_STATISTICS_MESSAGE_TYPE);
    let anonymized_statistics_2d_attributes =
        create_message_type_attribute_map(ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);

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

    // Check if the sequence id is consistent with the number of irises
    let max_serial_id = store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        tracing::error!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );

        eyre::bail!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );
    }

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        eyre::bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

    tracing::info!("Preparing task monitor");
    let mut background_tasks = TaskMonitor::new();

    // --------------------------------------------------------------------------
    // ANCHOR: Starting Healthcheck, Readiness and Sync server
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let is_ready_flag = Arc::new(AtomicBool::new(false));
    let is_ready_flag_cloned = Arc::clone(&is_ready_flag);

    let my_state = SyncState {
        db_len: store_len as u64,
        modifications: store.last_modifications(max_modification_lookback).await?,
        next_sns_sequence_num: next_sns_seq_number_future.await?,
        common_config: CommonConfig::from(config.clone()),
    };

    tracing::info!("Sync state: {:?}", my_state);

    let health_shutdown_handler = Arc::clone(&shutdown_handler);

    let _health_check_abort = background_tasks.spawn({
        let uuid = uuid::Uuid::new_v4().to_string();
        let ready_probe_response = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: false,
            uuid: uuid.clone(),
        };
        let ready_probe_response_shutdown = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: true,
            uuid: uuid.clone(),
        };
        let serialized_response = serde_json::to_string(&ready_probe_response)
            .expect("Serialization to JSON to probe response failed");
        let serialized_response_shutdown = serde_json::to_string(&ready_probe_response_shutdown)
            .expect("Serialization to JSON to probe response failed");
        tracing::info!("Healthcheck probe response: {}", serialized_response);
        let my_state = my_state.clone();
        async move {
            // Generate a random UUID for each run.
            let app = Router::new()
                .route(
                    "/health",
                    get(move || {
                        let shutdown_handler_clone = Arc::clone(&health_shutdown_handler);
                        async move {
                            if shutdown_handler_clone.is_shutting_down() {
                                serialized_response_shutdown.clone()
                            } else {
                                serialized_response.clone()
                            }
                        }
                    }),
                )
                .route(
                    "/ready",
                    get({
                        // We are only ready once this flag is set to true.
                        let is_ready_flag = Arc::clone(&is_ready_flag);
                        move || async move {
                            if is_ready_flag.load(Ordering::SeqCst) {
                                "ready".into_response()
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE.into_response()
                            }
                        }
                    }),
                )
                .route(
                    "/startup-sync",
                    get(move || async move { serde_json::to_string(&my_state).unwrap() }),
                );
            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
                .await
                .wrap_err("healthcheck listener bind error")?;
            axum::serve(listener, app)
                .await
                .wrap_err("healthcheck listener server launch error")?;

            Ok::<(), eyre::Error>(())
        }
    });

    background_tasks.check_tasks();
    tracing::info!("Healthcheck and Readiness server running on port 3000.");

    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");
    // Check other nodes and wait until all nodes are ready.
    let all_nodes = config.node_hostnames.clone();
    let unready_check = tokio::spawn(async move {
        let next_node = &all_nodes[(config.party_id + 1) % 3];
        let prev_node = &all_nodes[(config.party_id + 2) % 3];
        let mut connected_but_unready = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(format!("http://{}:3000/ready", host)).await;

                if res.is_ok() && res.unwrap().status() == StatusCode::SERVICE_UNAVAILABLE {
                    connected_but_unready[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected_but_unready.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be unready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        unready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be unready.");
            bail!("Timeout waiting for all nodes to be unready.");
        }
    };
    tracing::info!("All nodes are starting up.");

    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);
    let all_nodes = config.node_hostnames.clone();
    let image_name = config.image_name.clone();
    let heartbeat_shutdown_handler = Arc::clone(&shutdown_handler);
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
                    tracing::info!(
                        "Node {} did not respond with success, starting graceful shutdown",
                        host
                    );
                    // if the nodes are still starting up and they get a failure - we can panic and
                    // not start graceful shutdown
                    if last_response[i] == String::default() {
                        panic!(
                            "Node {} did not respond with success during heartbeat init phase, \
                             killing server...",
                            host
                        );
                    }

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has not completed health check, therefore graceful shutdown \
                             has been triggered",
                            host
                        );
                    } else {
                        tracing::info!("Node {} has already started graceful shutdown.", host);
                    }
                    continue;
                }

                let probe_response = res
                    .unwrap()
                    .json::<ReadyProbeResponse>()
                    .await
                    .expect("Deserialization of probe response failed");
                if probe_response.image_name != image_name {
                    // Do not create a panic as we still can continue to process before its
                    // updated
                    tracing::error!(
                        "Host {} is using image {} which differs from current node image: {}",
                        host,
                        probe_response.image_name.clone(),
                        image_name
                    );
                }
                if last_response[i] == String::default() {
                    last_response[i] = probe_response.uuid;
                    connected[i] = true;

                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        if let Some(tx) = heartbeat_tx.take() {
                            tx.send(()).unwrap();
                        }
                    }
                } else if probe_response.uuid != last_response[i] {
                    // If the UUID response is different, the node has restarted without us
                    // noticing. Our main NCCL connections cannot recover from
                    // this, so we panic.
                    panic!("Node {} seems to have restarted, killing server...", host);
                } else if probe_response.shutting_down {
                    tracing::info!("Node {} has starting graceful shutdown", host);

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has starting graceful shutdown, therefore triggering \
                             graceful shutdown",
                            host
                        );
                    }
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
    let download_shutdown_handler = Arc::clone(&shutdown_handler);

    background_tasks.check_tasks();

    // Start the actor in separate task.
    // A bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    let s3_load_parallelism = config.load_chunks_parallelism;
    let s3_chunks_bucket_name = config.db_chunks_bucket_name.clone();
    let s3_chunks_folder_name = config.db_chunks_folder_name.clone();
    let s3_load_max_retries = config.load_chunks_max_retries;
    let s3_load_initial_backoff_ms = config.load_chunks_initial_backoff_ms;

    // --------------------------------------------------------------------------
    // ANCHOR: Syncing latest node state
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");
    let all_nodes = config.node_hostnames.clone();
    let next_node = &all_nodes[(config.party_id + 1) % 3];
    let prev_node = &all_nodes[(config.party_id + 2) % 3];

    tracing::info!("Database store length is: {}", store_len);
    let mut states = vec![my_state.clone()];
    for host in [next_node, prev_node].iter() {
        let res = reqwest::get(format!("http://{}:3000/startup-sync", host)).await;
        match res {
            Ok(res) => {
                let state: SyncState = match res.json().await {
                    Ok(state) => state,
                    Err(e) => {
                        tracing::error!("Failed to parse sync state from party {}: {:?}", host, e);
                        panic!(
                            "could not get sync state from party {}, trying to restart",
                            host
                        );
                    }
                };
                states.push(state);
            }
            Err(e) => {
                tracing::error!("Failed to fetch sync state from party {}: {:?}", host, e);
                panic!(
                    "could not get sync state from party {}, trying to restart",
                    host
                );
            }
        }
    }
    let sync_result = SyncResult::new(my_state.clone(), states);

    // check if common part of the config is the same across all nodes
    sync_result.check_common_config()?;

    // sync the queues
    let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
    delete_messages_until_sequence_num(
        &aws_clients.sqs_client,
        &config.requests_queue_url,
        my_state.next_sns_sequence_num,
        max_sqs_sequence_num,
        config.sqs_sync_long_poll_seconds,
    )
    .await?;

    let dummy_shares_for_deletions = get_dummy_shares_for_deletion(party_id);

    // Handle modifications sync
    if config.enable_modifications_sync {
        sync_modifications(
            &config,
            &store,
            None,
            &aws_clients,
            &shares_encryption_key_pair,
            sync_result,
        )
        .await?;
    }

    if config.enable_modifications_replay {
        // replay last `max_modification_lookback` modifications to SNS
        if let Err(e) = send_last_modifications_to_sns(
            &store,
            &aws_clients.sns_client,
            &config,
            max_modification_lookback,
        )
        .await
        {
            tracing::error!("Failed to replay last modifications: {:?}", e);
        }
    }

    if download_shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    // refetch store_len in case we rolled back
    let store_len = store.count_irises().await?;
    tracing::info!("Database store length after sync: {}", store_len);

    let (tx, rx) = oneshot::channel();
    let config_clone = config.clone();
    background_tasks.spawn_blocking(move || {
        let config = config_clone;
        // --------------------------------------------------------------------------
        // ANCHOR: Load the database
        // --------------------------------------------------------------------------
        tracing::info!("⚓️ ANCHOR: Starting server actor");
        match ServerActor::new(
            config.party_id,
            chacha_seeds,
            8,
            config.max_db_size,
            config.max_batch_size,
            config.match_distances_buffer_size,
            config.match_distances_buffer_size_extra_percent,
            config.match_distances_2d_buffer_size,
            config.n_buckets,
            config.return_partial_results,
            config.disable_persistence,
            config.enable_debug_timing,
            config.full_scan_side,
            config.full_scan_side_switching_enabled,
        ) {
            Ok((mut actor, handle)) => {
                tracing::info!("⚓️ ANCHOR: Load the database");
                let res = if config.fake_db_size > 0 {
                    // TODO: does this even still work, since we do not page-lock the memory here?
                    actor.fake_db(config.fake_db_size);
                    Ok(())
                } else {
                    tracing::info!(
                        "Initialize iris db: Loading from DB (parallelism: {})",
                        parallelism
                    );
                    let download_shutdown_handler = Arc::clone(&download_shutdown_handler);

                    tokio::runtime::Handle::current().block_on(async {
                        load_iris_db(
                            &mut actor,
                            &store,
                            store_len,
                            parallelism,
                            &config,
                            download_shutdown_handler,
                        )
                        .await
                    })
                };

                match res {
                    Ok(_) => {
                        tx.send(Ok((handle, store))).unwrap();
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

    let (mut handle, store) = rx.await??;

    background_tasks.check_tasks();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let sns_client_bg = aws_clients.sns_client.clone();
    let s3_client_bg = aws_clients.s3_client.clone();
    let config_bg = config.clone();
    let store_bg = store.clone();
    let shutdown_handler_bg = Arc::clone(&shutdown_handler);
    let _result_sender_abort = background_tasks.spawn(async move {
        while let Some(ServerJobResult {
            merged_results,
            request_ids,
            request_types,
            metadata,
            matches,
            matches_with_skip_persistence,
            match_ids,
            full_face_mirror_match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_rotation_indices_left,
            partial_match_rotation_indices_right,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_counters_right,
            left_iris_requests,
            right_iris_requests,
            deleted_ids,
            matched_batch_request_ids,
            anonymized_bucket_statistics_left,
            anonymized_bucket_statistics_right,
            anonymized_bucket_statistics_left_mirror,
            anonymized_bucket_statistics_right_mirror,
            successful_reauths,
            reauth_target_indices,
            reauth_or_rule_used,
            reset_update_indices,
            reset_update_request_ids,
            reset_update_shares,
            mut modifications,
            actor_data: _,
            full_face_mirror_attack_detected,
            anonymized_bucket_statistics_2d,
        }) = rx.recv().await
        {
            let dummy_deletion_shares = get_dummy_shares_for_deletion(party_id);

            // returned serial_ids are 0 indexed, but we want them to be 1 indexed
            let uniqueness_results = merged_results
                .iter()
                .enumerate()
                .filter(|(i, _)| request_types[*i] == UNIQUENESS_MESSAGE_TYPE)
                .map(|(i, &idx_result)| {
                    let result_event = UniquenessResult::new(
                        party_id,
                        match matches[i] {
                            true => None,
                            false => Some(idx_result + 1),
                        },
                        matches_with_skip_persistence[i],
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
                        Some(matched_batch_request_ids[i].clone()),
                        match partial_match_counters_right.is_empty() {
                            false => Some(partial_match_counters_right[i]),
                            true => None,
                        },
                        match partial_match_counters_left.is_empty() {
                            false => Some(partial_match_counters_left[i]),
                            true => None,
                        },
                        match partial_match_rotation_indices_left[i].is_empty() {
                            false => Some(partial_match_rotation_indices_left[i].clone()),
                            true => None,
                        },
                        match partial_match_rotation_indices_right[i].is_empty() {
                            false => Some(partial_match_rotation_indices_right[i].clone()),
                            true => None,
                        },
                        match full_face_mirror_match_ids[i].is_empty() {
                            false => Some(
                                full_face_mirror_match_ids[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_ids_left[i].is_empty() {
                            false => Some(
                                full_face_mirror_partial_match_ids_left[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_ids_right[i].is_empty() {
                            false => Some(
                                full_face_mirror_partial_match_ids_right[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_counters_left.is_empty() {
                            false => Some(full_face_mirror_partial_match_counters_left[i]),
                            true => None,
                        },
                        match full_face_mirror_partial_match_counters_right.is_empty() {
                            false => Some(full_face_mirror_partial_match_counters_right[i]),
                            true => None,
                        },
                        full_face_mirror_attack_detected[i],
                    );
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize reauth result");
                    modifications
                        .get_mut(&RequestId(request_ids[i].clone()))
                        .unwrap()
                        .mark_completed(!result_event.is_match, &result_string, result_event.serial_id, None);
                    result_string
                })
                .collect::<Vec<String>>();

            // Insert non-matching uniqueness queries into the persistent store.
            let (memory_serial_ids, codes_and_masks): (Vec<i64>, Vec<StoredIrisRef>) = matches
                .iter()
                .enumerate()
                .filter_map(
                    // Find the indices of non-matching queries in the batch.
                    |(query_idx, is_match)| {
                        if !is_match {
                                Some(query_idx)
                        } else {
                            // Check for full face mirror attack (only for UNIQUENESS requests) and log it.
                            if request_types[query_idx] == UNIQUENESS_MESSAGE_TYPE && full_face_mirror_attack_detected[query_idx]
                            {
                                tracing::warn!(
                                    "Mirror attack detected for request_id {} - Not persisting to database",
                                    request_ids[query_idx]
                                );
                                metrics::counter!("mirror.attack.rejected").increment(1);
                            }
                            // It matched, don't include.
                            None
                        }
                    },
                )
                .map(|query_idx| {
                    let serial_id = (merged_results[query_idx] + 1) as i64;
                    // Get the original vectors from `receive_batch`.
                    (
                        serial_id,
                        StoredIrisRef {
                            id: serial_id,
                            left_code: &left_iris_requests.code[query_idx].coefs[..],
                            left_mask: &left_iris_requests.mask[query_idx].coefs[..],
                            right_code: &right_iris_requests.code[query_idx].coefs[..],
                            right_mask: &right_iris_requests.mask[query_idx].coefs[..],
                        },
                    )
                })
                .unzip();

            let reauth_results = request_types
                .iter()
                .enumerate()
                .filter(|(_, request_type)| *request_type == REAUTH_MESSAGE_TYPE)
                .map(|(i, _)| {
                    let reauth_id = request_ids[i].clone();
                    let or_rule_used = reauth_or_rule_used.get(&reauth_id).unwrap();
                    let serial_id = reauth_target_indices.get(&reauth_id).unwrap() + 1;
                    let success = successful_reauths[i];
                    let result_event = ReAuthResult::new(
                        reauth_id.clone(),
                        party_id,
                        serial_id,
                        success,
                        match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>(),
                        *reauth_or_rule_used.get(&reauth_id).unwrap(),
                    );
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize reauth result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(success, &result_string, None, None);
                    result_string
                })
                .collect::<Vec<String>>();

            // handling identity deletion results
            let identity_deletion_results = deleted_ids
                .iter()
                .map(|&idx| {
                    let serial_id = idx + 1;
                    let result_event = IdentityDeletionResult::new(party_id, serial_id, true);
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize identity deletion result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(true, &result_string, None, None);
                    result_string
                })
                .collect::<Vec<String>>();

            let reset_check_results = request_types
                .iter()
                .enumerate()
                .filter(|(_, request_type)| *request_type == RESET_CHECK_MESSAGE_TYPE)
                .map(|(i, _)| {
                    let reset_id = request_ids[i].clone();
                    let result_event = ResetCheckResult::new(
                        reset_id.clone(),
                        party_id,
                        Some(match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>()),
                        Some(
                            partial_match_ids_left[i]
                                .iter()
                                .map(|x| x + 1)
                                .collect::<Vec<_>>(),
                        ),
                        Some(
                            partial_match_ids_right[i]
                                .iter()
                                .map(|x| x + 1)
                                .collect::<Vec<_>>(),
                        ),
                        Some(matched_batch_request_ids[i].clone()),
                        Some(partial_match_counters_right[i]),
                        Some(partial_match_counters_left[i]),
                    );
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize reset check result");

                    // Mark the reset check modification as completed.
                    // Note that reset_check is only a query and does not persist anything into the database.
                    // We store modification so that the SNS result can be replayed.
                    modifications
                        .get_mut(&RequestId(reset_id))
                        .unwrap()
                        .mark_completed(false, &result_string, None, None);
                    result_string
                })
                .collect::<Vec<String>>();

            // reset update results
            let reset_update_results = reset_update_request_ids
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let reset_id = reset_update_request_ids[i].clone();
                    let serial_id = reset_update_indices[i] + 1;
                    let result_event =
                        ResetUpdateAckResult::new(reset_id.clone(), party_id, serial_id);
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize reset update result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(true, &result_string, None, None);
                    result_string
                })
                .collect::<Vec<String>>();

            let mut tx = store_bg.tx().await?;

            store_bg
                .update_modifications(&mut tx, &modifications.values().collect::<Vec<_>>())
                .await?;

            // persist uniqueness results into db
            if !codes_and_masks.is_empty() && !config_bg.disable_persistence {
                let db_serial_ids = store_bg.insert_irises(&mut tx, &codes_and_masks).await?;

                // Check if the serial_ids match between memory and db.
                if memory_serial_ids != db_serial_ids {
                    tracing::error!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    );
                    bail!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    );
                }
            }

            if !config_bg.disable_persistence {
                // persist reauth results into db
                for (i, success) in successful_reauths.iter().enumerate() {
                    if !success {
                        continue;
                    }
                    let reauth_id = request_ids[i].clone();
                    // convert from memory index (0-based) to db index (1-based)
                    let serial_id = *reauth_target_indices.get(&reauth_id).unwrap() + 1;
                    tracing::info!(
                        "Persisting successful reauth update {} into postgres on serial id {} ",
                        reauth_id,
                        serial_id
                    );
                    store_bg
                        .update_iris(
                            Some(&mut tx),
                            serial_id as i64,
                            &left_iris_requests.code[i],
                            &left_iris_requests.mask[i],
                            &right_iris_requests.code[i],
                            &right_iris_requests.mask[i],
                        )
                        .await?;
                }

                // persist deletion results into db
                for idx in deleted_ids.iter() {
                    // overwrite postgres db with dummy shares.
                    // note that both serial_id and postgres db are 1-indexed.
                    let serial_id = *idx + 1;
                    tracing::info!(
                        "Persisting identity deletion into postgres on serial id {}",
                        serial_id
                    );
                    store_bg.update_iris(
                        Some(&mut tx),
                        serial_id as i64,
                        &dummy_deletion_shares.0,
                        &dummy_deletion_shares.1,
                        &dummy_deletion_shares.0,
                        &dummy_deletion_shares.1,
                    )
                    .await?;
                }

                // persist reset_update results into db
                for (idx, shares) in izip!(reset_update_indices, reset_update_shares) {
                    // overwrite postgres db with reset update shares.
                    // note that both serial_id and postgres db are 1-indexed.
                    let serial_id = idx + 1;
                    tracing::info!(
                        "Persisting reset update into postgres on serial id {}",
                        serial_id
                    );
                    store_bg.update_iris(
                        Some(&mut tx),
                        serial_id as i64,
                        &shares.code_left,
                        &shares.mask_left,
                        &shares.code_right,
                        &shares.mask_right,
                    )
                    .await?;
                }
            }

            tx.commit().await?;

            for memory_serial_id in memory_serial_ids {
                tracing::info!("Inserted serial_id: {}", memory_serial_id+1);
                metrics::gauge!("results_inserted.latest_serial_id").set((memory_serial_id +1) as f64);
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

            tracing::info!("Sending {} reauth results", reauth_results.len());
            send_results_to_sns(
                reauth_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &reauth_result_attributes,
                REAUTH_MESSAGE_TYPE,
            )
            .await?;

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

            if !reset_check_results.is_empty() {
                tracing::info!("Sending {} reset check results", reset_check_results.len());
                send_results_to_sns(
                    reset_check_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &reset_check_result_attributes,
                    RESET_CHECK_MESSAGE_TYPE,
                )
                .await?;
            }

            if !reset_update_results.is_empty() {
                tracing::info!(
                    "Sending {} reset update results",
                    reset_update_results.len()
                );
                send_results_to_sns(
                    reset_update_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &reset_update_result_attributes,
                    RESET_UPDATE_MESSAGE_TYPE,
                )
                .await?;
            }

            if (config_bg.enable_sending_anonymized_stats_message)
                && (!anonymized_bucket_statistics_left.buckets.is_empty()
                    || !anonymized_bucket_statistics_right.buckets.is_empty())
            {
                tracing::info!("Sending anonymized stats results");
                let anonymized_statistics_results = [
                    anonymized_bucket_statistics_left,
                    anonymized_bucket_statistics_right,
                ];
                // transform to vector of string ands remove None values
                let anonymized_statistics_results = anonymized_statistics_results
                    .iter()
                    .map(|anonymized_bucket_statistics| {
                        serde_json::to_string(anonymized_bucket_statistics)
                            .wrap_err("failed to serialize anonymized statistics result")
                    })
                    .collect::<Result<Vec<_>>>()?;

                send_results_to_sns(
                    anonymized_statistics_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &anonymized_statistics_attributes,
                    ANONYMIZED_STATISTICS_MESSAGE_TYPE,
                )
                .await?;
            }

            // Send mirror orientation statistics separately with their own flag
            if (config_bg.enable_sending_mirror_anonymized_stats_message)
                && (!anonymized_bucket_statistics_left_mirror.buckets.is_empty()
                    || !anonymized_bucket_statistics_right_mirror.buckets.is_empty())
            {
                tracing::info!("Sending mirror orientation anonymized stats results");
                let mirror_anonymized_statistics_results = [
                    anonymized_bucket_statistics_left_mirror,
                    anonymized_bucket_statistics_right_mirror,
                ];
                // transform to vector of string
                let mirror_anonymized_statistics_results = mirror_anonymized_statistics_results
                    .iter()
                    .map(|anonymized_bucket_statistics| {
                        serde_json::to_string(anonymized_bucket_statistics)
                            .wrap_err("failed to serialize mirror anonymized statistics result")
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;

                send_results_to_sns(
                    mirror_anonymized_statistics_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &anonymized_statistics_attributes,
                    ANONYMIZED_STATISTICS_MESSAGE_TYPE,
                )
                .await?;
            }

            // Send 2D anonymized statistics if present with their own flag
            if config_bg.enable_sending_anonymized_stats_2d_message
                && !anonymized_bucket_statistics_2d.buckets.is_empty()
            {
                tracing::info!("Sending 2D anonymized stats results");
                let serialized = serde_json::to_string(&anonymized_bucket_statistics_2d)
                    .wrap_err("failed to serialize 2D anonymized statistics result")?;

                // offloading 2D anon stats file to s3 to avoid sending large messages to SNS
                // with 2D stats we were exceeding the SNS message size limit
                let now_ms = Utc::now().timestamp_millis();
                let sha = iris_mpc_common::helpers::sha256::sha256_bytes(&serialized);
                let content_hash =  hex::encode(sha);
                let s3_key = format!("stats2d/{}_{}.json", now_ms, content_hash);

                upload_file_to_s3(
                    &config_bg.sns_buffer_bucket_name,
                    &s3_key,
                    s3_client_bg.clone(),
                    serialized.as_bytes(),
                )
                .await
                .wrap_err("failed to upload 2D anonymized statistics to s3")?;

                // Publish only the S3 key to SNS
                let payload = serde_json::to_string(&serde_json::json!({
                    "s3_key": s3_key,
                }))?;
                send_results_to_sns(
                    vec![payload],
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &anonymized_statistics_2d_attributes,
                    ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE,
                )
                .await?;
            }

            shutdown_handler_bg.decrement_batches_pending_completion();
        }

        Ok(())
    });
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Enable readiness and check all nodes
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Enable readiness and check all nodes");

    // Set the readiness flag to true, which will make the readiness server return a
    // 200 status code.
    is_ready_flag_cloned.store(true, std::sync::atomic::Ordering::SeqCst);

    // Check other nodes and wait until all nodes are ready.
    let all_nodes = config.node_hostnames.clone();
    let ready_check = tokio::spawn(async move {
        let next_node = &all_nodes[(config.party_id + 1) % 3];
        let prev_node = &all_nodes[(config.party_id + 2) % 3];
        let mut connected = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(format!("http://{}:3000/ready", host)).await;

                if res.is_ok() && res.as_ref().unwrap().status().is_success() {
                    connected[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be ready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        ready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be ready.");
            bail!("Timeout waiting for all nodes to be ready.");
        }
    }
    tracing::info!("All nodes are ready.");
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Start the main loop
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Start the main loop");

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
    let uniqueness_error_result_attribute =
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_error_result_attribute = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_check_error_result_attribute =
        create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let reset_update_error_result_attribute =
        create_message_type_attribute_map(RESET_UPDATE_MESSAGE_TYPE);
    let res: Result<()> = async {
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

        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above
        let mut batch_stream = receive_batch_stream(
            party_id,
            aws_clients.sqs_client.clone(),
            aws_clients.sns_client.clone(),
            aws_clients.s3_client.clone(),
            config.clone(),
            store.clone(),
            shares_encryption_key_pair.clone(),
            shutdown_handler.clone(),
            uniqueness_error_result_attribute,
            reauth_error_result_attribute,
            reset_check_error_result_attribute,
        );

        loop {
            let now = Instant::now();

            let batch = match batch_stream.recv().await {
                Some(Ok(None)) | None => {
                    tracing::info!("No more batches to process, exiting main loop");
                    return Ok(());
                }
                Some(Err(e)) => {
                    return Err(e.into());
                }
                Some(Ok(Some(batch))) => batch,
            };

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());

            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());

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

            // await the result
            let result = timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("ServerActor processing timeout: {:?}", e))??;

            tx.send(result).await?;

            shutdown_handler.increment_batches_pending_completion()
            // wrap up span context
        }
    }
    .await;

    match res {
        Ok(_) => {
            tracing::info!(
                "Main loop exited normally. Waiting for last batch results to be processed before \
                 shutting down..."
            );

            shutdown_handler.wait_for_pending_batches_completion().await;
        }
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
