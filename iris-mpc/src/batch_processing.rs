use crate::{
    aws::sns,
    iris_processing::{get_iris_shares_parse_task, IrisShareProcessor},
    message_receiver::MessageReceiver,
};
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use aws_sdk_sqs::Client;
use iris_mpc_common::{
    config::Config,
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        key_pair::SharesEncryptionKeyPairs,
        shutdown_handler::ShutdownHandler,
        smpc_request::{
            CircuitBreakerRequest, IdentityDeletionRequest, ReAuthRequest, ReceiveRequestError,
            SQSMessage, UniquenessRequest, CIRCUIT_BREAKER_MESSAGE_TYPE,
            IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
        },
        smpc_response::{
            ReAuthResult, UniquenessResult, ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
            ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH, SMPC_MESSAGE_TYPE_ATTRIBUTE,
        },
    },
    job::{BatchMetadata, BatchQuery},
};
use iris_mpc_store::Store;
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};
use tokio::sync::Semaphore;

const MAX_CONCURRENT_REQUESTS: usize = 32;
pub static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

pub struct BatchQueryBuilder {
    pub batch_query: BatchQuery,
}

impl Default for BatchQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchQueryBuilder {
    pub fn new() -> Self {
        Self {
            batch_query: BatchQuery::default(),
        }
    }

    pub fn add_deletion_request(
        &mut self,
        request: IdentityDeletionRequest,
        metadata: BatchMetadata,
    ) {
        self.batch_query
            .deletion_requests_indices
            .push(request.serial_id - 1);
        self.batch_query.deletion_requests_metadata.push(metadata);
    }

    pub fn add_uniqueness_request(
        &mut self,
        uniqueness_request: UniquenessRequest,
        metadata: BatchMetadata,
    ) {
        self.batch_query
            .request_ids
            .push(uniqueness_request.signup_id.clone());
        self.batch_query
            .request_types
            .push(UNIQUENESS_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(metadata);
    }

    pub fn add_reauth_request(&mut self, reauth_request: ReAuthRequest, metadata: BatchMetadata) {
        self.batch_query
            .request_ids
            .push(reauth_request.reauth_id.clone());
        self.batch_query
            .request_types
            .push(REAUTH_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(metadata);
        self.batch_query.reauth_target_indices.insert(
            reauth_request.reauth_id.clone(),
            reauth_request.serial_id - 1,
        );
        self.batch_query
            .reauth_use_or_rule
            .insert(reauth_request.reauth_id.clone(), reauth_request.use_or_rule);

        let or_rule_indices = if reauth_request.use_or_rule {
            vec![reauth_request.serial_id - 1]
        } else {
            vec![]
        };
        self.batch_query.or_rule_indices.push(or_rule_indices);
    }

    pub fn build(self) -> BatchQuery {
        self.batch_query
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn receive_batch(
    party_id: usize,
    client: &Client,
    sns_client: &SNSClient,
    s3_client: &S3Client,
    config: &Config,
    store: &Store,
    skip_request_ids: &[String],
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: &ShutdownHandler,
    uniqueness_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: &HashMap<String, MessageAttributeValue>,
) -> eyre::Result<Option<BatchQuery>, ReceiveRequestError> {
    if shutdown_handler.is_shutting_down() {
        tracing::info!("Stopping batch receive due to shutdown signal...");
        return Ok(None);
    }

    let message_receiver = MessageReceiver::new(client, config.clone().requests_queue_url);

    let mut batch_builder = BatchQueryBuilder::default();
    let iris_processor = IrisShareProcessor::new(
        party_id,
        shares_encryption_key_pairs,
        s3_client.clone(),
        config.shares_bucket_name.clone(),
    );

    let mut handles = vec![];
    let mut msg_counter = 0;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    // Main processing loop
    while msg_counter < *CURRENT_BATCH_SIZE.lock().unwrap() {
        let messages = message_receiver.receive_messages().await?;
        if messages.is_empty() {
            tokio::time::sleep(message_receiver.wait_time).await;
        }
        for sqs_message in messages {
            let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())
                .map_err(|e| ReceiveRequestError::json_parse_error("SQS body", e))?;
            let metadata = BatchMetadata::default();

            match message
                .message_attributes
                .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
                .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
                .string_value()
                .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            {
                CIRCUIT_BREAKER_MESSAGE_TYPE => {
                    handle_circuit_breaker_message(
                        &message,
                        &sqs_message,
                        client,
                        &config.requests_queue_url,
                        config.max_batch_size,
                    )
                    .await?;
                }
                IDENTITY_DELETION_MESSAGE_TYPE => {
                    let request: IdentityDeletionRequest = serde_json::from_str(&message.message)
                        .map_err(|e| {
                        ReceiveRequestError::json_parse_error("Identity deletion request", e)
                    })?;
                    batch_builder.add_deletion_request(request, metadata);
                    message_receiver
                        .delete_message(sqs_message.receipt_handle.unwrap())
                        .await?;
                }
                UNIQUENESS_MESSAGE_TYPE => {
                    msg_counter += 1;

                    let uniqueness_request: UniquenessRequest =
                        serde_json::from_str(&message.message).map_err(|e| {
                            ReceiveRequestError::json_parse_error("Uniqueness request", e)
                        })?;
                    metrics::counter!("request.received", "type" => "uniqueness_verification")
                        .increment(1);
                    store
                        .mark_requests_deleted(&[uniqueness_request.signup_id.clone()])
                        .await
                        .map_err(ReceiveRequestError::FailedToMarkRequestAsDeleted)?;

                    message_receiver
                        .delete_message(sqs_message.receipt_handle.unwrap())
                        .await?;

                    if skip_request_ids.contains(&uniqueness_request.signup_id) {
                        msg_counter -= 1;
                        metrics::counter!("skip.request.deleted.sqs.request").increment(1);
                        tracing::warn!(
                            "Skipping request due to it being from synced deleted ids: {}",
                            uniqueness_request.signup_id
                        );
                        let message = UniquenessResult::new_error_result(
                            config.party_id,
                            uniqueness_request.signup_id,
                            ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH,
                        );
                        sns::send_error_results_to_sns(
                            serde_json::to_string(&message).unwrap(),
                            &metadata,
                            sns_client,
                            config,
                            uniqueness_error_result_attributes,
                            UNIQUENESS_MESSAGE_TYPE,
                        )
                        .await?;
                        continue;
                    }

                    if let Some(batch_size) = uniqueness_request.batch_size {
                        *CURRENT_BATCH_SIZE.lock().unwrap() =
                            batch_size.clamp(1, config.max_batch_size);
                        tracing::info!("Updating batch size to {}", batch_size);
                    }
                    if config.luc_enabled {
                        if config.luc_lookback_records > 0 {
                            batch_builder.batch_query.luc_lookback_records =
                                config.luc_lookback_records;
                        }
                        if config.luc_serial_ids_from_smpc_request {
                            if let Some(serial_ids) = uniqueness_request.or_rule_serial_ids.clone()
                            {
                                batch_builder
                                    .batch_query
                                    .or_rule_indices
                                    .push(serial_ids.iter().map(|x| x - 1).collect());
                            } else {
                                tracing::error!("Received a uniqueness request without serial_ids");
                            }
                        }
                    }
                    batch_builder.add_uniqueness_request(uniqueness_request.clone(), metadata);
                    let semaphore = Arc::clone(&semaphore);
                    let handle = get_iris_shares_parse_task(
                        semaphore,
                        uniqueness_request.s3_key.clone(),
                        iris_processor.clone(),
                    )?;
                    handles.push(handle);
                }
                REAUTH_MESSAGE_TYPE => {
                    let reauth_request: ReAuthRequest = serde_json::from_str(&message.message)
                        .map_err(|e| ReceiveRequestError::json_parse_error("Reauth request", e))?;
                    metrics::counter!("request.received", "type" => "reauth").increment(1);

                    tracing::debug!("Received reauth request: {:?}", reauth_request);

                    message_receiver
                        .delete_message(sqs_message.receipt_handle.unwrap())
                        .await?;

                    if reauth_request.use_or_rule
                        && !(config.luc_enabled && config.luc_serial_ids_from_smpc_request)
                    {
                        tracing::error!(
                            "Received a reauth request with use_or_rule set to true, but LUC is \
                             not enabled. Skipping request."
                        );
                        continue;
                    }

                    if config.enable_reauth {
                        msg_counter += 1;

                        if let Some(batch_size) = reauth_request.batch_size {
                            *CURRENT_BATCH_SIZE.lock().unwrap() =
                                batch_size.clamp(1, config.max_batch_size);
                            tracing::info!("Updating batch size to {}", batch_size);
                        }

                        batch_builder.add_reauth_request(reauth_request.clone(), metadata);
                        let semaphore = Arc::clone(&semaphore);
                        let handle = get_iris_shares_parse_task(
                            semaphore,
                            reauth_request.s3_key.clone(),
                            iris_processor.clone(),
                        )?;
                        handles.push(handle);
                    } else {
                        tracing::warn!("Reauth is disabled, skipping reauth request");
                    }
                }
                _ => {
                    message_receiver
                        .delete_message(sqs_message.receipt_handle.unwrap())
                        .await?;
                    tracing::error!("Error: {}", ReceiveRequestError::InvalidMessageType);
                }
            }
        }
    }

    for (index, handle) in handles.into_iter().enumerate() {
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
                let request_id = batch_builder.batch_query.request_ids[index].clone();
                let (result_attributes, message) =
                    match batch_builder.batch_query.request_types[index].as_str() {
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
                                *batch_builder
                                    .batch_query
                                    .reauth_target_indices
                                    .get(&request_id)
                                    .unwrap(),
                                ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                            );
                            let serialized = serde_json::to_string(&message).unwrap();
                            (reauth_error_result_attributes, serialized)
                        }
                        _ => unreachable!(),
                    };

                sns::send_error_results_to_sns(
                    message,
                    &batch_builder.batch_query.metadata[index],
                    sns_client,
                    config,
                    result_attributes,
                    batch_builder.batch_query.request_types[index].as_str(),
                )
                .await?;
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

        batch_builder.batch_query.valid_entries.push(valid_entry);

        batch_builder
            .batch_query
            .left_iris_requests
            .code
            .push(store_iris_shares_left);
        batch_builder
            .batch_query
            .left_iris_requests
            .mask
            .push(store_mask_shares_left);
        batch_builder
            .batch_query
            .left_iris_rotated_requests
            .code
            .extend(db_iris_shares_left);
        batch_builder
            .batch_query
            .right_iris_rotated_requests
            .mask
            .extend(db_mask_shares_left);
        batch_builder
            .batch_query
            .left_iris_interpolated_requests
            .code
            .extend(iris_shares_left);
        batch_builder
            .batch_query
            .left_iris_interpolated_requests
            .mask
            .extend(mask_shares_left);

        batch_builder
            .batch_query
            .right_iris_requests
            .code
            .push(store_iris_shares_right);
        batch_builder
            .batch_query
            .right_iris_requests
            .mask
            .push(store_mask_shares_right);
        batch_builder
            .batch_query
            .right_iris_rotated_requests
            .code
            .extend(db_iris_shares_right);
        batch_builder
            .batch_query
            .right_iris_rotated_requests
            .mask
            .extend(db_mask_shares_right);
        batch_builder
            .batch_query
            .right_iris_interpolated_requests
            .code
            .extend(iris_shares_right);
        batch_builder
            .batch_query
            .right_iris_interpolated_requests
            .mask
            .extend(mask_shares_right);
    }

    tracing::info!(
        "Batch requests: {:?}",
        batch_builder
            .batch_query
            .request_ids
            .iter()
            .zip(batch_builder.batch_query.request_types.iter())
            .collect::<Vec<_>>()
    );

    Ok(Some(batch_builder.build()))
}

pub async fn handle_circuit_breaker_message(
    message: &SQSMessage,
    sqs_message: &aws_sdk_sqs::types::Message,
    client: &Client,
    queue_url: &str,
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
        *CURRENT_BATCH_SIZE.lock().unwrap() = new_batch_size.clamp(1, max_batch_size);
        tracing::info!(
            "Updating batch size to {} due to circuit breaker message",
            new_batch_size
        );
    }

    Ok(())
}
