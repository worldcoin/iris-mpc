use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_error_results_to_sns;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client;
use eyre::Result;
use iris_mpc_common::config::Config;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare, GaloisShares,
};
use iris_mpc_common::helpers::aws::{
    SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};
use iris_mpc_common::helpers::batch_sync::{
    get_batch_sync_states, get_own_batch_sync_state, BatchSyncResult,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest, SQSMessage,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_request::{ReceiveRequestError, RESET_CHECK_MESSAGE_TYPE};
use iris_mpc_common::helpers::smpc_response::{
    ReAuthResult, ResetCheckResult, UniquenessResult, ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
    SMPC_MESSAGE_TYPE_ATTRIBUTE,
};
use iris_mpc_common::helpers::sync::ModificationKey::{RequestId, RequestSerialId};
use iris_mpc_common::job::{BatchMetadata, BatchQuery, GaloisSharesBothSides};
use iris_mpc_store::Store;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

#[allow(clippy::too_many_arguments)]
pub fn receive_batch_stream(
    party_id: usize,
    client: Client,
    sns_client: SNSClient,
    s3_client: S3Client,
    config: Config,
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: Arc<ShutdownHandler>,
    uniqueness_error_result_attributes: HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: HashMap<String, MessageAttributeValue>,
    reset_error_result_attributes: HashMap<String, MessageAttributeValue>,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: Store,
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
                shares_encryption_key_pairs.clone(),
                &shutdown_handler,
                &uniqueness_error_result_attributes,
                &reauth_error_result_attributes,
                &reset_error_result_attributes,
                current_batch_id_atomic.clone(),
                &iris_store,
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
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: &ShutdownHandler,
    uniqueness_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reset_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: &Store,
) -> Result<Option<BatchQuery>, ReceiveRequestError> {
    let mut processor = BatchProcessor::new(
        party_id,
        client,
        sns_client,
        s3_client,
        config,
        shares_encryption_key_pairs,
        shutdown_handler,
        uniqueness_error_result_attributes,
        reauth_error_result_attributes,
        reset_error_result_attributes,
        current_batch_id_atomic,
        iris_store,
    );

    processor.receive_batch().await
}

pub struct BatchProcessor<'a> {
    party_id: usize,
    client: &'a Client,
    sns_client: &'a SNSClient,
    s3_client: &'a S3Client,
    config: &'a Config,
    shares_encryption_key_pairs: SharesEncryptionKeyPairs,
    shutdown_handler: &'a ShutdownHandler,
    uniqueness_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    reauth_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    reset_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    batch_query: BatchQuery,
    semaphore: Arc<Semaphore>,
    handles: Vec<JoinHandle<Result<(GaloisShares, GaloisShares), eyre::Error>>>,
    msg_counter: usize,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: &'a Store,
}

impl<'a> BatchProcessor<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        party_id: usize,
        client: &'a Client,
        sns_client: &'a SNSClient,
        s3_client: &'a S3Client,
        config: &'a Config,
        shares_encryption_key_pairs: SharesEncryptionKeyPairs,
        shutdown_handler: &'a ShutdownHandler,
        uniqueness_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        reauth_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        reset_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        current_batch_id_atomic: Arc<AtomicU64>,
        iris_store: &'a Store,
    ) -> Self {
        Self {
            party_id,
            client,
            sns_client,
            s3_client,
            config,
            shares_encryption_key_pairs,
            shutdown_handler,
            uniqueness_error_result_attributes,
            reauth_error_result_attributes,
            reset_error_result_attributes,
            batch_query: BatchQuery::default(),
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            handles: vec![],
            msg_counter: 0,
            current_batch_id_atomic,
            iris_store,
        }
    }

    pub async fn receive_batch(&mut self) -> Result<Option<BatchQuery>, ReceiveRequestError> {
        if self.shutdown_handler.is_shutting_down() {
            tracing::info!("Stopping batch receive due to shutdown signal...");
            return Ok(None);
        }

        loop {
            let current_batch_id = self.current_batch_id_atomic.load(Ordering::SeqCst);

            // Determine the number of messages to poll based on synchronized state
            let own_state = get_own_batch_sync_state(self.config, self.client, current_batch_id)
                .await
                .map_err(ReceiveRequestError::BatchSyncError)?;

            let all_states =
                get_batch_sync_states(self.config, self.client, Some(&own_state), current_batch_id)
                    .await
                    .map_err(ReceiveRequestError::BatchSyncError)?;

            let batch_sync_result = BatchSyncResult::new(own_state, all_states);
            let max_visible_messages = batch_sync_result.max_approximate_visible_messages();

            let num_to_poll =
                std::cmp::min(max_visible_messages, self.config.max_batch_size as u32);

            tracing::info!(
                "Batch ID: {}. Agreed to poll {} messages (max_visible: {}, max_batch_size: {}).",
                current_batch_id,
                num_to_poll,
                max_visible_messages,
                self.config.max_batch_size
            );

            // Poll the determined number of messages
            if num_to_poll > 0 {
                self.poll_exact_messages(num_to_poll).await?;
                break;
            } else {
                tracing::info!(
                    "Batch ID: {}. No messages to poll based on sync state. Will re-check after a short delay.",
                    current_batch_id
                );
                if self.shutdown_handler.is_shutting_down() {
                    tracing::info!(
                        "Stopping batch receive during polling wait due to shutdown signal..."
                    );
                    return Ok(None);
                }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }

        // Process all parse tasks
        self.process_parse_tasks().await?;

        tracing::info!(
            "Batch ID: {}. Formed batch with requests: {:?}",
            self.current_batch_id_atomic.load(Ordering::SeqCst),
            self.batch_query
                .request_ids
                .iter()
                .zip(self.batch_query.request_types.iter())
                .collect::<Vec<_>>()
        );

        // Increment batch_id for the next batch
        self.current_batch_id_atomic.fetch_add(1, Ordering::SeqCst);

        Ok(Some(self.batch_query.clone()))
    }

    async fn poll_exact_messages(&mut self, num_to_poll: u32) -> Result<(), ReceiveRequestError> {
        let current_batch_id = self.current_batch_id_atomic.load(Ordering::SeqCst);
        tracing::info!(
            "Batch ID: {}. Polling SQS for up to {} messages.",
            current_batch_id,
            num_to_poll
        );
        let queue_url = &self.config.requests_queue_url;

        while self.msg_counter < num_to_poll as usize {
            if self.shutdown_handler.is_shutting_down() {
                tracing::info!(
                    "Stopping batch receive during polling exact messages due to shutdown signal..."
                );
                return Ok(()); // Exit if shutdown is signaled
            }

            let rcv_message_output = self
                .client
                .receive_message()
                // Set a short wait time to avoid busy-looping when the queue is temporarily empty
                // but we still expect more messages for the current batch.
                .wait_time_seconds(1) // Short poll to quickly check for messages
                .max_number_of_messages(1) // Process one message at a time to respect num_to_poll accurately
                .queue_url(queue_url)
                .send()
                .await
                .map_err(ReceiveRequestError::FailedToReadFromSQS)?;

            if let Some(messages) = rcv_message_output.messages {
                if messages.is_empty() {
                    // If the queue is empty, short poll again until num_to_poll is met or shutdown.
                    // This can happen if messages are arriving slower than we are polling.
                    tracing::debug!(
                        "Batch ID: {}. No message received in a polling attempt, will retry as {} out of {} messages have been processed.",
                        current_batch_id,
                        self.msg_counter,
                        num_to_poll
                    );
                    // Add a small delay to prevent tight looping when queue is empty but we are still expecting messages
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }

                for sqs_message in messages {
                    // Should be only one message due to max_number_of_messages(1)
                    self.process_message(sqs_message).await?;
                }
            } else {
                // This case should ideally not be hit often if wait_time_seconds > 0,
                // as SQS long polling usually returns an empty messages array instead of None.
                // However, handling it defensively.
                tracing::debug!(
                    "Batch ID: {}. SQS receive_message returned no messages array, will retry polling.",
                    current_batch_id
                );
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        }

        tracing::info!(
            "Batch ID: {}. Finished polling SQS. Processed {} messages for this batch attempt (target: {}).",
            current_batch_id,
            self.msg_counter,
            num_to_poll
        );
        Ok(())
    }

    async fn process_message(
        &mut self,
        sqs_message: aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())
            .map_err(|e| ReceiveRequestError::json_parse_error("SQS body", e))?;

        let message_attributes = message.message_attributes.clone();
        let batch_metadata = self.extract_batch_metadata(&message_attributes);

        let request_type = message_attributes
            .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            .string_value()
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?;

        match request_type {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                self.process_identity_deletion(&message, batch_metadata, &sqs_message)
                    .await
            }
            UNIQUENESS_MESSAGE_TYPE => {
                self.process_uniqueness_request(&message, batch_metadata, &sqs_message)
                    .await
            }
            REAUTH_MESSAGE_TYPE => {
                self.process_reauth_request(&message, batch_metadata, &sqs_message)
                    .await
            }
            RESET_CHECK_MESSAGE_TYPE => {
                self.process_reset_check_request(&message, batch_metadata, &sqs_message)
                    .await
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                self.process_reset_update_request(&message, batch_metadata, &sqs_message)
                    .await
            }
            _ => {
                self.delete_message(&sqs_message).await?;
                tracing::error!("Error: {}", ReceiveRequestError::InvalidMessageType);
                Ok(())
            }
        }
    }

    async fn process_identity_deletion(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        if self.config.hawk_server_deletions_enabled {
            let identity_deletion_request: IdentityDeletionRequest =
                serde_json::from_str(&message.message).map_err(|e| {
                    ReceiveRequestError::json_parse_error("Identity deletion request", e)
                })?;

            metrics::counter!("request.received", "type" => "identity_deletion").increment(1);

            // Skip the request if serial ID already exists in current batch modifications
            if self
                .batch_query
                .modifications
                .contains_key(&RequestSerialId(identity_deletion_request.serial_id))
            {
                tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                identity_deletion_request.serial_id,
                                identity_deletion_request,
                            );
                return Ok(());
            }

            // Persist in progress modification
            let modification = self
                .iris_store
                .insert_modification(
                    Some(identity_deletion_request.serial_id as i64),
                    IDENTITY_DELETION_MESSAGE_TYPE,
                    None,
                )
                .await?;
            self.batch_query.modifications.insert(
                RequestSerialId(identity_deletion_request.serial_id),
                modification,
            );

            self.batch_query
                .deletion_requests_indices
                .push(identity_deletion_request.serial_id - 1);
            self.batch_query
                .deletion_requests_metadata
                .push(batch_metadata);
        } else {
            tracing::warn!("Identity deletions are disabled");
        }

        self.delete_message(sqs_message).await?;
        Ok(())
    }

    async fn process_uniqueness_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let uniqueness_request: UniquenessRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Uniqueness request", e))?;

        metrics::counter!("request.received", "type" => "uniqueness_verification").increment(1);

        self.delete_message(sqs_message).await?;

        // Persist in progress modification
        let modification = self
            .iris_store
            .insert_modification(
                None,
                UNIQUENESS_MESSAGE_TYPE,
                Some(uniqueness_request.s3_key.as_str()),
            )
            .await?;
        self.batch_query.modifications.insert(
            RequestId(uniqueness_request.signup_id.clone()),
            modification,
        );

        self.update_luc_config_if_needed(&uniqueness_request);

        self.msg_counter += 1;

        self.batch_query
            .request_ids
            .push(uniqueness_request.signup_id.clone());
        self.batch_query
            .request_types
            .push(UNIQUENESS_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(batch_metadata);

        self.add_iris_shares_task(uniqueness_request.s3_key)?;
        // skip_persistence is only used for uniqueness requests
        if let Some(skip_persistence) = uniqueness_request.skip_persistence {
            tracing::info!(
                "Setting skip_persistence to {} for request id {}",
                skip_persistence,
                uniqueness_request.signup_id
            );
            self.batch_query.skip_persistence.push(skip_persistence);
        } else {
            self.batch_query.skip_persistence.push(false);
        }

        Ok(())
    }

    async fn process_reauth_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let reauth_request: ReAuthRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Reauth request", e))?;

        metrics::counter!("request.received", "type" => "reauth").increment(1);
        tracing::debug!("Received reauth request: {:?}", reauth_request);

        self.delete_message(sqs_message).await?;

        if !self.config.hawk_server_reauths_enabled {
            tracing::warn!("Reauth is disabled, skipping reauth request");
            return Ok(());
        }

        if reauth_request.use_or_rule
            && !(self.config.luc_enabled && self.config.luc_serial_ids_from_smpc_request)
        {
            tracing::error!(
                "Received a reauth request with use_or_rule set to true, but LUC is not enabled. Skipping request."
            );
            return Ok(());
        }

        // Skip the request if serial ID already exists in current batch modifications
        if self
            .batch_query
            .modifications
            .contains_key(&RequestSerialId(reauth_request.serial_id))
        {
            tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                reauth_request.serial_id,
                                reauth_request,
                            );
            return Ok(());
        }

        // Persist in progress modification
        let modification = self
            .iris_store
            .insert_modification(
                Some(reauth_request.serial_id as i64),
                REAUTH_MESSAGE_TYPE,
                Some(reauth_request.s3_key.as_str()),
            )
            .await?;
        self.batch_query
            .modifications
            .insert(RequestSerialId(reauth_request.serial_id), modification);

        self.msg_counter += 1;

        self.batch_query
            .request_ids
            .push(reauth_request.reauth_id.clone());
        self.batch_query
            .request_types
            .push(REAUTH_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(batch_metadata);

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
        self.add_iris_shares_task(reauth_request.s3_key)?;
        // skip_persistence is only used for uniqueness requests
        self.batch_query.skip_persistence.push(false);

        Ok(())
    }

    async fn process_reset_check_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let reset_check_request: ResetCheckRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Reset check request", e))?;

        metrics::counter!("request.received", "type" => "reset_check").increment(1);
        tracing::debug!("Received reset check request: {:?}", reset_check_request);

        self.delete_message(sqs_message).await?;

        if !self.config.hawk_server_resets_enabled {
            tracing::warn!("Reset is disabled, skipping reset request");
            return Ok(());
        }

        // Persist in progress modification
        let modification = self
            .iris_store
            .insert_modification(
                None,
                RESET_CHECK_MESSAGE_TYPE,
                Some(reset_check_request.s3_key.as_str()),
            )
            .await?;
        self.batch_query.modifications.insert(
            RequestId(reset_check_request.reset_id.clone()),
            modification,
        );

        self.msg_counter += 1;

        self.batch_query
            .request_ids
            .push(reset_check_request.reset_id.clone());
        self.batch_query
            .request_types
            .push(RESET_CHECK_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(batch_metadata);

        // skip_persistence is only used for uniqueness requests
        self.batch_query.skip_persistence.push(false);

        // We need to use AND rule for reset check requests
        self.batch_query.or_rule_indices.push(vec![]);
        self.add_iris_shares_task(reset_check_request.s3_key)?;

        Ok(())
    }

    async fn process_reset_update_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let sns_message_id = &message.message_id;
        let reset_update_request: ResetUpdateRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Reset update request", e))?;

        metrics::counter!("request.received", "type" => "reset_update").increment(1);
        tracing::debug!("Received reset update request: {:?}", reset_update_request);

        self.delete_message(sqs_message).await?;

        if !self.config.hawk_server_resets_enabled {
            tracing::warn!("Reset is disabled, skipping reset request");
            return Ok(());
        }
        let semaphore = Arc::clone(&self.semaphore);
        let s3_client = self.s3_client.clone();
        let bucket_name = self.config.shares_bucket_name.clone();
        let shares_encryption_key_pairs = self.shares_encryption_key_pairs.clone();

        let task_handle = get_iris_shares_parse_task(
            self.party_id,
            shares_encryption_key_pairs,
            semaphore,
            s3_client,
            bucket_name,
            reset_update_request.s3_key.clone(),
        )?;
        let (left_shares, right_shares) = match task_handle.await {
            Ok(result) => match result {
                Ok(shares) => shares,
                Err(e) => {
                    tracing::error!("Failed to process iris shares for reset update: {:?}", e);
                    return Err(ReceiveRequestError::FailedToProcessIrisShares(e));
                }
            },
            Err(e) => {
                tracing::error!("Failed to join task handle for reset update: {:?}", e);
                return Err(ReceiveRequestError::FailedToJoinHandle(e));
            }
        };

        if self
            .batch_query
            .modifications
            .contains_key(&RequestSerialId(reset_update_request.serial_id))
        {
            tracing::warn!(
                                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                                reset_update_request.serial_id,
                                reset_update_request,
                            );
            return Ok(());
        }

        let modification = self
            .iris_store
            .insert_modification(
                Some(reset_update_request.serial_id as i64),
                RESET_UPDATE_MESSAGE_TYPE,
                Some(reset_update_request.s3_key.as_str()),
            )
            .await?;
        self.batch_query.modifications.insert(
            RequestSerialId(reset_update_request.serial_id),
            modification,
        );

        self.msg_counter += 1;

        self.batch_query
            .reset_update_indices
            .push(reset_update_request.serial_id - 1);
        self.batch_query
            .sns_message_ids
            .push(sns_message_id.clone());
        self.batch_query.metadata.push(batch_metadata);
        self.batch_query
            .reset_update_request_ids
            .push(reset_update_request.reset_id);
        self.batch_query
            .reset_update_shares
            .push(GaloisSharesBothSides {
                code_left: left_shares.code,
                mask_left: left_shares.mask,
                code_right: right_shares.code,
                mask_right: right_shares.mask,
            });
        Ok(())
    }

    async fn process_parse_tasks(&mut self) -> Result<(), ReceiveRequestError> {
        // Use std::mem::take to take ownership of handles while leaving an empty vector in self
        let handles = std::mem::take(&mut self.handles);
        for (index, handle) in handles.into_iter().enumerate() {
            let result = handle
                .await
                .map_err(ReceiveRequestError::FailedToJoinHandle)?;

            match result {
                Ok((share_left, share_right)) => {
                    self.batch_query.valid_entries.push(true);
                    self.add_shares_to_batch_query(share_left, share_right);
                }
                Err(e) => {
                    tracing::error!("Failed to process iris shares: {:?}", e);
                    self.handle_share_processing_error(index).await?;

                    // Create dummy shares for invalid entry
                    let ((dummy_left, dummy_right), valid) = self.create_dummy_shares();
                    self.batch_query.valid_entries.push(valid);
                    self.add_shares_to_batch_query(dummy_left, dummy_right);
                }
            }
        }

        Ok(())
    }
    async fn handle_share_processing_error(&self, index: usize) -> Result<()> {
        let request_id = self.batch_query.request_ids[index].clone();
        let request_type = &self.batch_query.request_types[index];

        let (result_attributes, message) = match request_type.as_str() {
            UNIQUENESS_MESSAGE_TYPE => {
                let message = UniquenessResult::new_error_result(
                    self.config.party_id,
                    request_id,
                    ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                );
                (
                    self.uniqueness_error_result_attributes,
                    serde_json::to_string(&message).unwrap(),
                )
            }
            REAUTH_MESSAGE_TYPE => {
                let message = ReAuthResult::new_error_result(
                    request_id.clone(),
                    self.config.party_id,
                    *self
                        .batch_query
                        .reauth_target_indices
                        .get(&request_id)
                        .unwrap(),
                    ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                );
                (
                    self.reauth_error_result_attributes,
                    serde_json::to_string(&message).unwrap(),
                )
            }
            RESET_CHECK_MESSAGE_TYPE => {
                let message = ResetCheckResult::new_error_result(
                    request_id.clone(),
                    self.config.party_id,
                    ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                );
                (
                    self.reset_error_result_attributes,
                    serde_json::to_string(&message).unwrap(),
                )
            }
            _ => unreachable!(),
        };

        send_error_results_to_sns(
            message,
            &self.batch_query.metadata[index],
            self.sns_client,
            self.config,
            result_attributes,
            request_type.as_str(),
        )
        .await
    }

    // Helper methods
    fn extract_batch_metadata(
        &self,
        message_attributes: &HashMap<String, MessageAttributeValue>,
    ) -> BatchMetadata {
        let mut batch_metadata = BatchMetadata::default();

        if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
            if let Some(trace_id_str) = trace_id.string_value() {
                batch_metadata.trace_id = trace_id_str.to_string();
            }
        }

        if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
            if let Some(span_id_str) = span_id.string_value() {
                batch_metadata.span_id = span_id_str.to_string();
            }
        }

        batch_metadata
    }

    async fn delete_message(
        &self,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        self.client
            .delete_message()
            .queue_url(&self.config.requests_queue_url)
            .receipt_handle(sqs_message.receipt_handle.as_ref().unwrap())
            .send()
            .await
            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)?;
        tracing::debug!("Deleted message: {:?}", sqs_message.message_id);
        Ok(())
    }

    fn update_luc_config_if_needed(&mut self, uniqueness_request: &UniquenessRequest) {
        let config = &self.config;

        if config.luc_enabled && config.luc_lookback_records > 0 {
            self.batch_query.luc_lookback_records = config.luc_lookback_records;
        }

        let or_rule_indices = if config.luc_enabled && config.luc_serial_ids_from_smpc_request {
            if let Some(serial_ids) = uniqueness_request.or_rule_serial_ids.as_ref() {
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

        self.batch_query.or_rule_indices.push(or_rule_indices);
    }

    fn add_iris_shares_task(&mut self, s3_key: String) -> Result<(), ReceiveRequestError> {
        let semaphore = Arc::clone(&self.semaphore);
        let s3_client = self.s3_client.clone();
        let bucket_name = self.config.shares_bucket_name.clone();
        let shares_encryption_key_pairs = self.shares_encryption_key_pairs.clone();

        let handle = get_iris_shares_parse_task(
            self.party_id,
            shares_encryption_key_pairs,
            semaphore,
            s3_client,
            bucket_name,
            s3_key,
        )?;

        self.handles.push(handle);
        Ok(())
    }

    fn create_dummy_shares(&self) -> ((GaloisShares, GaloisShares), bool) {
        let dummy_code_share = GaloisRingIrisCodeShare::default_for_party(self.party_id);
        let dummy_mask_share = GaloisRingTrimmedMaskCodeShare::default_for_party(self.party_id);

        let dummy = GaloisShares {
            code: dummy_code_share.clone(),
            mask: dummy_mask_share.clone(),
            code_rotated: dummy_code_share.all_rotations(),
            mask_rotated: dummy_mask_share.all_rotations(),
            code_interpolated: dummy_code_share.all_rotations(),
            mask_interpolated: dummy_mask_share.all_rotations(),
            code_mirrored: dummy_code_share.all_rotations(),
            mask_mirrored: dummy_mask_share.all_rotations(),
        };

        ((dummy.clone(), dummy), false)
    }

    fn add_shares_to_batch_query(&mut self, share_left: GaloisShares, share_right: GaloisShares) {
        self.batch_query
            .left_iris_requests
            .code
            .push(share_left.code);
        self.batch_query
            .left_iris_requests
            .mask
            .push(share_left.mask);
        self.batch_query
            .left_iris_rotated_requests
            .code
            .extend(share_left.code_rotated);
        self.batch_query
            .left_iris_rotated_requests
            .mask
            .extend(share_left.mask_rotated);
        self.batch_query
            .left_iris_interpolated_requests
            .code
            .extend(share_left.code_interpolated);
        self.batch_query
            .left_iris_interpolated_requests
            .mask
            .extend(share_left.mask_interpolated);

        self.batch_query
            .right_iris_requests
            .code
            .push(share_right.code);
        self.batch_query
            .right_iris_requests
            .mask
            .push(share_right.mask);
        self.batch_query
            .right_iris_rotated_requests
            .code
            .extend(share_right.code_rotated);
        self.batch_query
            .right_iris_rotated_requests
            .mask
            .extend(share_right.mask_rotated);
        self.batch_query
            .right_iris_interpolated_requests
            .code
            .extend(share_right.code_interpolated);
        self.batch_query
            .right_iris_interpolated_requests
            .mask
            .extend(share_right.mask_interpolated);

        self.batch_query
            .left_mirrored_iris_interpolated_requests
            .code
            .extend(share_left.code_mirrored);
        self.batch_query
            .left_mirrored_iris_interpolated_requests
            .mask
            .extend(share_left.mask_mirrored);
        self.batch_query
            .right_mirrored_iris_interpolated_requests
            .code
            .extend(share_right.code_mirrored);
        self.batch_query
            .right_mirrored_iris_interpolated_requests
            .mask
            .extend(share_right.mask_mirrored);
    }
}
