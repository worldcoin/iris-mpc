use crate::server::{MAX_CONCURRENT_REQUESTS, SQS_POLLING_INTERVAL};
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_error_results_to_sns;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::types::MessageSystemAttributeName;
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
    get_batch_sync_states, BatchSyncResult, BatchSyncState,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::ReceiveRequestError;
use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, SQSMessage, UniquenessRequest,
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    ReAuthResult, UniquenessResult, ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
    SMPC_MESSAGE_TYPE_ATTRIBUTE,
};
use iris_mpc_common::job::{BatchMetadata, BatchQuery};
use std::collections::HashMap;
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
    batch_query: BatchQuery,
    semaphore: Arc<Semaphore>,
    handles: Vec<JoinHandle<Result<(GaloisShares, GaloisShares), eyre::Error>>>,
    msg_counter: usize,
    own_batch_sync_state: BatchSyncState,
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
            batch_query: BatchQuery::default(),
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            handles: vec![],
            msg_counter: 0,
            own_batch_sync_state: BatchSyncState {
                next_sns_sequence_num: 0,
            },
        }
    }

    pub async fn receive_batch(&mut self) -> Result<Option<BatchQuery>, ReceiveRequestError> {
        if self.shutdown_handler.is_shutting_down() {
            tracing::info!("Stopping batch receive due to shutdown signal...");
            return Ok(None);
        }

        // Poll messages until we have enough or timeout
        self.poll_messages().await?;

        // Process all parse tasks
        self.process_parse_tasks().await?;

        tracing::info!(
            "Batch requests: {:?}",
            self.batch_query
                .request_ids
                .iter()
                .zip(self.batch_query.request_types.iter())
                .collect::<Vec<_>>()
        );

        Ok(Some(self.batch_query.clone()))
    }

    async fn poll_messages(&mut self) -> Result<(), ReceiveRequestError> {
        let max_batch_size = self.config.max_batch_size;
        let queue_url = &self.config.requests_queue_url;
        match self.poll_until_batch_size(max_batch_size, queue_url).await {
            Ok(results) => results,
            Err(ReceiveRequestError::BatchPollingTimeout(timeout_secs)) => {
                tracing::info!(
                    "Batch polling timeout reached after {} secs, checking other nodes' sync states",
                    timeout_secs
                );

                let states = get_batch_sync_states(
                    self.config,
                    self.client,
                    Some(&self.own_batch_sync_state),
                )
                .await
                .map_err(ReceiveRequestError::BatchSyncError)?;
                let batch_sync_result =
                    BatchSyncResult::new(self.own_batch_sync_state.clone(), states);

                let max_sequence_num = batch_sync_result.max_sns_sequence_num();
                let own_sequence_num = batch_sync_result.my_state.next_sns_sequence_num;

                tracing::info!(
                    "Max sequence number across nodes: {}, own sequence number: {}",
                    max_sequence_num,
                    own_sequence_num
                );

                if (own_sequence_num >= max_sequence_num) && self.msg_counter > 0 {
                    tracing::info!(
                        "Own sequence number is greater than or equal to max sequence number. No need to poll further."
                    );
                    return Ok(());
                }
                self.poll_until_sequence_num(max_sequence_num, queue_url)
                    .await?;
            }
            Err(err) => return Err(err),
        }
        Ok(())
    }
    async fn poll_until_batch_size(
        &mut self,
        max_batch_size: usize,
        queue_url: &str,
    ) -> Result<(), ReceiveRequestError> {
        tracing::info!(
            "Polling SQS for messages until batch size {} is reached, msg counter: {}",
            max_batch_size,
            self.msg_counter
        );
        while self.msg_counter < max_batch_size {
            let rcv_message_output = self
                .client
                .receive_message()
                .wait_time_seconds(self.config.batch_polling_timeout_secs)
                .max_number_of_messages(10)
                .queue_url(queue_url)
                .send()
                .await
                .map_err(ReceiveRequestError::FailedToReadFromSQS)?;

            if let Some(messages) = rcv_message_output.messages {
                if messages.is_empty() {
                    // No more messages in the queue
                    tracing::info!("No more messages in the queue");
                    break;
                }

                for sqs_message in messages {
                    self.process_message(sqs_message).await?;

                    // Check if we've reached the target batch size
                    if self.msg_counter >= max_batch_size {
                        break;
                    }
                }
            } else {
                // No messages received
                tokio::time::sleep(SQS_POLLING_INTERVAL).await;
            }
        }
        Ok(())
    }

    async fn poll_until_sequence_num(
        &mut self,
        target_sequence_num: u128,
        queue_url: &str,
    ) -> Result<(), ReceiveRequestError> {
        let mut current_sequence_num = 0;

        while current_sequence_num < target_sequence_num {
            let rcv_message_output = self
                .client
                .receive_message()
                .wait_time_seconds(self.config.sqs_long_poll_wait_time as i32)
                .max_number_of_messages(1)
                .queue_url(queue_url)
                .send()
                .await
                .map_err(ReceiveRequestError::FailedToReadFromSQS)?;

            if let Some(messages) = rcv_message_output.messages {
                if messages.is_empty() {
                    // No more messages in the queue even though we haven't reached target sequence
                    tracing::warn!("No more messages in queue but target sequence not reached: current={}, target={}",
                                  current_sequence_num, target_sequence_num);
                    break;
                }

                for sqs_message in messages {
                    // Extract sequence number from message attributes if available
                    if let Some(attributes) = &sqs_message.attributes {
                        if let Some(sequence_num_str) =
                            attributes.get(&MessageSystemAttributeName::SequenceNumber)
                        {
                            if let Ok(seq_num) = sequence_num_str.parse::<u128>() {
                                current_sequence_num = seq_num;
                            } else {
                                tracing::error!(
                                    "Failed to parse sequence number from message attributes: {}",
                                    sequence_num_str
                                );
                            }
                        }
                    }

                    // Process the message regardless of sequence number
                    self.process_message(sqs_message.clone()).await?;

                    // If we've reached or exceeded the target sequence number, we're done
                    if current_sequence_num >= target_sequence_num {
                        break;
                    }
                }
            } else {
                // No messages received
                tokio::time::sleep(SQS_POLLING_INTERVAL).await;
            }
        }

        Ok(())
    }

    fn update_own_batch_sync_state(&mut self, sequence_number: u128) {
        self.own_batch_sync_state = BatchSyncState {
            next_sns_sequence_num: sequence_number,
        };
    }

    async fn process_message(
        &mut self,
        sqs_message: aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())
            .map_err(|e| ReceiveRequestError::json_parse_error("SQS body", e))?;

        let sequence_number: u128 = message.sequence_number.parse::<u128>().map_err(|e| {
            ReceiveRequestError::seq_number_parse_error(
                format!("SQS sequence number {}", message.sequence_number),
                e,
            )
        })?;
        self.update_own_batch_sync_state(sequence_number);

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

            self.batch_query
                .deletion_requests_indices
                .push(identity_deletion_request.serial_id - 1);
            self.batch_query
                .deletion_requests_metadata
                .push(batch_metadata);
        } else {
            tracing::warn!("Identity deletions are disabled");
        }

        self.delete_message(sqs_message).await
    }

    async fn process_uniqueness_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        sqs_message: &aws_sdk_sqs::types::Message,
    ) -> Result<(), ReceiveRequestError> {
        self.msg_counter += 1;

        let uniqueness_request: UniquenessRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Uniqueness request", e))?;

        metrics::counter!("request.received", "type" => "uniqueness_verification").increment(1);

        self.delete_message(sqs_message).await?;

        self.update_luc_config_if_needed(&uniqueness_request);

        self.batch_query
            .request_ids
            .push(uniqueness_request.signup_id.clone());
        self.batch_query
            .request_types
            .push(UNIQUENESS_MESSAGE_TYPE.to_string());
        self.batch_query.metadata.push(batch_metadata);

        self.add_iris_shares_task(uniqueness_request.s3_key)?;

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

        if !self.config.enable_reauth {
            tracing::warn!("Reauth processing is disabled, skipping reauth request");
            return Ok(());
        }

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
