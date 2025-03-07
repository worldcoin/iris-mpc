use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::server::{CURRENT_BATCH_SIZE, MAX_CONCURRENT_REQUESTS, SQS_POLLING_INTERVAL};
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::message::send_error_results_to_sns;
use iris_mpc_common::config::Config;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::helpers::aws::{
    SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    CircuitBreakerRequest, IdentityDeletionRequest, ReAuthRequest, ReceiveRequestError, SQSMessage,
    UniquenessRequest, CIRCUIT_BREAKER_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    ReAuthResult, UniquenessResult, ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
    ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH, SMPC_MESSAGE_TYPE_ATTRIBUTE,
};
use iris_mpc_common::job::{BatchMetadata, BatchQuery};
use iris_mpc_store::Store;

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

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if skip_request_ids.contains(&uniqueness_request.signup_id) {
                            // Some party (maybe us) already meant to delete this request, so we
                            // skip it. Ignore this message when calculating the batch size.
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
                            // shares
                            send_error_results_to_sns(
                                serde_json::to_string(&message).unwrap(),
                                &batch_metadata,
                                sns_client,
                                config,
                                uniqueness_error_result_attributes,
                                UNIQUENESS_MESSAGE_TYPE,
                            )
                            .await?;
                            continue;
                        }

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
                        if config.luc_enabled {
                            if config.luc_lookback_records > 0 {
                                batch_query.luc_lookback_records = config.luc_lookback_records;
                            }
                            if config.luc_serial_ids_from_smpc_request {
                                if let Some(serial_ids) =
                                    uniqueness_request.or_rule_serial_ids.clone()
                                {
                                    // convert from 1-based serial id to 0-based index in actor
                                    batch_query
                                        .or_rule_indices
                                        .push(serial_ids.iter().map(|x| x - 1).collect());
                                } else {
                                    tracing::error!(
                                        "Received a uniqueness request without serial_ids"
                                    );
                                }
                            }
                        }

                        batch_query
                            .request_ids
                            .push(uniqueness_request.signup_id.clone());
                        batch_query.request_types.push(
                            iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE
                                .to_string(),
                        );
                        batch_query.metadata.push(batch_metadata);

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
                        metrics::counter!("request.received", "type" => "reauth").increment(1);

                        tracing::debug!("Received reauth request: {:?}", reauth_request);

                        // TODO: populate sync mechanism table (TBD: rollback or rollforward)

                        client
                            .delete_message()
                            .queue_url(queue_url)
                            .receipt_handle(sqs_message.receipt_handle.unwrap())
                            .send()
                            .await
                            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;

                        if reauth_request.use_or_rule
                            && !(config.luc_enabled && config.luc_serial_ids_from_smpc_request)
                        {
                            tracing::error!(
                                "Received a reauth request with use_or_rule set to true, but LUC \
                                 is not enabled. Skipping request."
                            );
                            continue;
                        }

                        if config.enable_reauth {
                            msg_counter += 1;

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

                            batch_query
                                .request_ids
                                .push(reauth_request.reauth_id.clone());
                            batch_query.request_types.push(
                                iris_mpc_common::helpers::smpc_request::REAUTH_MESSAGE_TYPE
                                    .to_string(),
                            );
                            batch_query.metadata.push(batch_metadata);
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
                            batch_query.or_rule_indices.push(or_rule_indices);

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

        batch_query
            .left_iris_requests
            .code
            .push(store_iris_shares_left);
        batch_query
            .left_iris_requests
            .mask
            .push(store_mask_shares_left);
        batch_query
            .left_iris_rotated_requests
            .code
            .extend(db_iris_shares_left);
        batch_query
            .left_iris_rotated_requests
            .mask
            .extend(db_mask_shares_left);
        batch_query
            .left_iris_interpolated_requests
            .code
            .extend(iris_shares_left);
        batch_query
            .left_iris_interpolated_requests
            .mask
            .extend(mask_shares_left);

        batch_query
            .right_iris_requests
            .code
            .push(store_iris_shares_right);
        batch_query
            .right_iris_requests
            .mask
            .push(store_mask_shares_right);
        batch_query
            .right_iris_rotated_requests
            .code
            .extend(db_iris_shares_right);
        batch_query
            .right_iris_rotated_requests
            .mask
            .extend(db_mask_shares_right);
        batch_query
            .right_iris_interpolated_requests
            .code
            .extend(iris_shares_right);
        batch_query
            .right_iris_interpolated_requests
            .mask
            .extend(mask_shares_right);
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
