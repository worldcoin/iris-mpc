use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_error_results_to_sns;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use ampc_server_utils::{
    get_approximate_number_of_messages, get_batch_sync_states, BatchSyncResult,
    BatchSyncSharedState, BatchSyncState,
};
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client;
use eyre::{eyre, Result};
use iris_mpc_common::config::Config;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare, GaloisShares,
};
use iris_mpc_common::helpers::aws::{
    SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
#[cfg(feature = "explicit-sns-batching")]
use iris_mpc_common::helpers::smpc_request::{
    CompactBatchRequest, CompressedBatchPayload, BATCH_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, IdentityMatchCheckRequest, IdentityUpdateRequest, ReAuthRequest,
    SQSMessage, UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RECOVERY_UPDATE_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_request::{
    ReceiveRequestError, RECOVERY_CHECK_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    IdentityDeletionResult, IdentityMatchCheckResult, IdentityUpdateAckResult, ReAuthResult,
    UniquenessResult, ERROR_FAILED_TO_PROCESS_IRIS_SHARES, SMPC_MESSAGE_TYPE_ATTRIBUTE,
};
use iris_mpc_common::helpers::sync::Modification;
use iris_mpc_common::helpers::sync::ModificationKey::{RequestId, RequestSerialId};
use iris_mpc_common::job::{BatchMetadata, BatchQuery, GaloisSharesBothSides};
use iris_mpc_store::{IngestedRequest, Store};
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

fn messages_to_poll_for_available(
    available_messages: u32,
    max_batch_size: usize,
    predefined_batch_sizes: &[usize],
    current_batch_id: u64,
) -> u32 {
    let index = current_batch_id.saturating_sub(1) as usize;
    let configured_limit = if predefined_batch_sizes.len() > index {
        predefined_batch_sizes[index]
    } else {
        max_batch_size
    };
    std::cmp::min(
        available_messages,
        std::cmp::min(configured_limit, max_batch_size) as u32,
    )
}

#[derive(Debug)]
struct DbIngestBackoff {
    initial_ms: u64,
    max_ms: u64,
    current_ms: u64,
}

impl DbIngestBackoff {
    fn new(initial_ms: u64, max_ms: u64) -> Self {
        let initial_ms = initial_ms.max(1);
        let max_ms = max_ms.max(initial_ms);
        Self {
            initial_ms,
            max_ms,
            current_ms: initial_ms,
        }
    }

    fn reset(&mut self) {
        self.current_ms = self.initial_ms;
    }

    fn next_sleep(&mut self) -> Duration {
        let jitter_ms = rand::thread_rng().gen_range(0..=self.current_ms);
        let sleep_ms = self.current_ms.saturating_add(jitter_ms).min(self.max_ms);
        self.current_ms = self.current_ms.saturating_mul(2).min(self.max_ms);
        Duration::from_millis(sleep_ms)
    }
}

pub fn spawn_db_backed_ingest_task(
    client: Client,
    config: Config,
    iris_store: Store,
    shutdown_handler: Arc<ShutdownHandler>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut backoff = DbIngestBackoff::new(
            config.db_ingest_backoff_initial_ms,
            config.db_ingest_backoff_max_ms,
        );

        tracing::info!("Starting DB-backed SQS ingest task");
        while !shutdown_handler.is_shutting_down() {
            match ingest_one_sqs_message(&client, &config, &iris_store).await {
                Ok(received_message) => {
                    if received_message {
                        tracing::debug!("DB-backed ingest processed one SQS message");
                    }
                    backoff.reset();
                }
                Err(err) => {
                    let sleep = backoff.next_sleep();
                    tracing::warn!(
                        "DB-backed ingest error: {:?}; retrying after {:?}",
                        err,
                        sleep
                    );
                    tokio::time::sleep(sleep).await;
                }
            }
        }
        tracing::info!("Stopping DB-backed SQS ingest task");
    })
}

async fn ingest_one_sqs_message(
    client: &Client,
    config: &Config,
    iris_store: &Store,
) -> Result<bool> {
    let receive_output = client
        .receive_message()
        .wait_time_seconds(config.db_ingest_sqs_wait_secs)
        .max_number_of_messages(1)
        .queue_url(&config.requests_queue_url)
        .send()
        .await?;

    let Some(messages) = receive_output.messages else {
        return Ok(false);
    };

    let mut received_message = false;
    for sqs_message in messages {
        ingest_sqs_message(client, config, iris_store, sqs_message).await?;
        received_message = true;
    }
    Ok(received_message)
}

async fn ingest_sqs_message(
    client: &Client,
    config: &Config,
    iris_store: &Store,
    sqs_message: aws_sdk_sqs::types::Message,
) -> Result<()> {
    let body = sqs_message
        .body()
        .ok_or_else(|| eyre!("SQS message missing body: {:?}", sqs_message.message_id))?
        .to_string();
    let sns_message: SQSMessage = serde_json::from_str(&body)?;

    let inserted = iris_store
        .insert_ingested_request(&sns_message.sequence_number, &body)
        .await?;

    let receipt_handle = sqs_message.receipt_handle.as_deref().ok_or_else(|| {
        eyre!(
            "SQS message missing receipt handle: {:?}",
            sqs_message.message_id
        )
    })?;
    client
        .delete_message()
        .queue_url(&config.requests_queue_url)
        .receipt_handle(receipt_handle)
        .send()
        .await?;

    if inserted {
        tracing::debug!(
            "Inserted ingested request sequence_number={}",
            sns_message.sequence_number
        );
    } else {
        tracing::debug!(
            "Skipped duplicate ingested request sequence_number={}",
            sns_message.sequence_number
        );
    }
    Ok(())
}

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
    reset_check_error_result_attributes: HashMap<String, MessageAttributeValue>,
    recovery_check_error_result_attributes: HashMap<String, MessageAttributeValue>,
    identity_deletion_error_result_attributes: HashMap<String, MessageAttributeValue>,
    reset_update_error_result_attributes: HashMap<String, MessageAttributeValue>,
    recovery_update_error_result_attributes: HashMap<String, MessageAttributeValue>,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: Store,
    batch_sync_shared_state: Arc<tokio::sync::Mutex<BatchSyncSharedState>>,
) -> (
    Receiver<Result<Option<BatchQuery>, ReceiveRequestError>>,
    Arc<Semaphore>,
) {
    let (tx, rx) = mpsc::channel(1);
    let sem = Arc::new(Semaphore::new(1));

    tokio::spawn({
        let sem = sem.clone();
        async move {
            loop {
                match sem.acquire().await {
                    // We successfully acquired the semaphore, proceed with receiving a batch
                    // However, we forget the permit here to avoid giving it back
                    // The main server loop will add new permits when allowed
                    Ok(p) => p.forget(),
                    Err(_) => {
                        break;
                    }
                };
                let permit = match tx.reserve().await {
                    Ok(permit) => permit,
                    Err(_) => {
                        break;
                    }
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
                    &reset_check_error_result_attributes,
                    &recovery_check_error_result_attributes,
                    &identity_deletion_error_result_attributes,
                    &reset_update_error_result_attributes,
                    &recovery_update_error_result_attributes,
                    current_batch_id_atomic.clone(),
                    &iris_store,
                    batch_sync_shared_state.clone(),
                )
                .await;

                let stop = matches!(batch, Err(_) | Ok(None));
                permit.send(batch);

                if stop {
                    break;
                }
            }
            tracing::info!("Stopping batch receiver.");
        }
    });

    (rx, sem)
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
    reset_check_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    recovery_check_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    identity_deletion_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    reset_update_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    recovery_update_error_result_attributes: &HashMap<String, MessageAttributeValue>,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: &Store,
    batch_sync_shared_state: Arc<tokio::sync::Mutex<BatchSyncSharedState>>,
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
        reset_check_error_result_attributes,
        recovery_check_error_result_attributes,
        identity_deletion_error_result_attributes,
        reset_update_error_result_attributes,
        recovery_update_error_result_attributes,
        current_batch_id_atomic,
        iris_store,
        batch_sync_shared_state,
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
    reset_check_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    recovery_check_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    identity_deletion_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    reset_update_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    recovery_update_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
    batch_query: BatchQuery,
    semaphore: Arc<Semaphore>,
    handles: Vec<JoinHandle<Result<(GaloisShares, GaloisShares), eyre::Error>>>,
    msg_counter: usize,
    current_batch_id_atomic: Arc<AtomicU64>,
    iris_store: &'a Store,
    batch_sync_shared_state: Arc<tokio::sync::Mutex<BatchSyncSharedState>>,
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
        reset_check_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        recovery_check_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        identity_deletion_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        reset_update_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        recovery_update_error_result_attributes: &'a HashMap<String, MessageAttributeValue>,
        current_batch_id_atomic: Arc<AtomicU64>,
        iris_store: &'a Store,
        batch_sync_shared_state: Arc<tokio::sync::Mutex<BatchSyncSharedState>>,
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
            reset_check_error_result_attributes,
            recovery_check_error_result_attributes,
            identity_deletion_error_result_attributes,
            reset_update_error_result_attributes,
            recovery_update_error_result_attributes,
            batch_query: BatchQuery::default(),
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            handles: vec![],
            msg_counter: 0,
            current_batch_id_atomic,
            iris_store,
            batch_sync_shared_state,
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
            let mut own_state = get_own_batch_sync_state(
                self.config,
                self.client,
                self.iris_store,
                current_batch_id,
            )
            .await
            .map_err(ReceiveRequestError::BatchSyncError)?;

            // Update the shared state with our current state
            {
                let mut shared_state = self.batch_sync_shared_state.lock().await;
                // we are here for the first time, set everything
                if shared_state.batch_id != own_state.batch_id {
                    shared_state.batch_id = own_state.batch_id;
                    shared_state.messages_to_poll = own_state.messages_to_poll;
                } else if shared_state.messages_to_poll == 0 {
                    // we have been here before, only update messages_to_poll if it was 0, otherwise other parties could have state mismatches
                    shared_state.messages_to_poll = own_state.messages_to_poll;
                } else {
                    // we have already set this for this batch, so it might have gone out to other parties, so we need to update our own state to match what we already sent out
                    own_state.messages_to_poll = shared_state.messages_to_poll;
                }

                let log_msg = format!(
                    "Updated shared batch sync state: batch_id={}, messages_to_poll={}",
                    shared_state.batch_id, shared_state.messages_to_poll,
                );
                match shared_state.messages_to_poll {
                    0 => tracing::debug!(log_msg),
                    _ => tracing::info!(log_msg),
                };
            }

            let server_coord_config = self.config.server_coordination.as_ref().ok_or(
                ReceiveRequestError::BatchSyncError(eyre::eyre!(
                    "Server coordination config is missing"
                )),
            )?;

            let all_states = get_batch_sync_states(
                server_coord_config,
                Some(&own_state),
                self.config.batch_sync_polling_timeout_secs,
            )
            .await
            .map_err(ReceiveRequestError::BatchSyncError)?;

            let batch_sync_result = BatchSyncResult::new(own_state, all_states);
            let messages_to_poll = batch_sync_result.messages_to_poll();

            let log_msg = format!(
                "Batch ID: {}. Agreed to poll {} messages (max_batch_size: {}).",
                current_batch_id, messages_to_poll, self.config.max_batch_size
            );
            match messages_to_poll {
                0 => tracing::debug!(log_msg),
                _ => tracing::info!(log_msg),
            };

            // Poll the determined number of messages
            if messages_to_poll > 0 {
                if self.config.db_backed_ingest {
                    self.process_pending_ingested_requests(messages_to_poll)
                        .await?;
                } else {
                    self.poll_exact_messages(messages_to_poll).await?;
                }
                break;
            } else {
                tracing::debug!(
                    "Batch ID: {}. No messages to poll based on sync state. Will re-check after a short delay.",
                    current_batch_id
                );
                if self.shutdown_handler.is_shutting_down() {
                    tracing::info!(
                        "Stopping batch receive during polling wait due to shutdown signal..."
                    );
                    return Ok(None);
                }
                // Reduce sleep time when no messages are available
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }

        // Process all parse tasks
        self.process_parse_tasks().await?;

        let batch_id = self.current_batch_id_atomic.load(Ordering::SeqCst);
        tracing::info!(
            "Batch ID: {}. First Signup ID: {}. Formed batch with requests: {:?}",
            batch_id,
            self.batch_query
                .request_ids
                .first()
                .unwrap_or(&"<empty>".to_string()),
            self.batch_query
                .request_ids
                .iter()
                .zip(self.batch_query.request_types.iter())
                .collect::<Vec<_>>()
        );

        if !self.batch_query.skip_persistence.is_empty() {
            let skip_ids: Vec<&String> = self
                .batch_query
                .request_ids
                .iter()
                .zip(self.batch_query.skip_persistence.iter())
                .filter_map(|(id, skip)| skip.then_some(id))
                .collect();
            tracing::info!(
                "Batch ID: {}. skip_persistence enabled for {}/{} requests: {:?}",
                batch_id,
                skip_ids.len(),
                self.batch_query.skip_persistence.len(),
                skip_ids,
            );
        }

        if !self.config.disable_persistence && !self.batch_query.modifications.is_empty() {
            let mut mods: Vec<String> = self
                .batch_query
                .modifications
                .values()
                .map(|m| format!("{}#{} serial={:?}", m.request_type, m.id, m.serial_id))
                .collect();
            mods.sort();
            tracing::info!(
                "Batch ID: {}. Inserted {} IN_PROGRESS modifications: {:?}",
                batch_id,
                mods.len(),
                mods,
            );
        }

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
                .map_err(ReceiveRequestError::from)?;

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

    async fn process_pending_ingested_requests(
        &mut self,
        num_to_poll: u32,
    ) -> Result<(), ReceiveRequestError> {
        // Batch ids restart per boot and advance under prefetch, so the claim
        // below uses the id only as an opaque non-NULL marker. Persist-marking
        // is keyed by the batch's sequence numbers; boot recovery releases
        // unpersisted claims.
        let current_batch_id = self.current_batch_id_atomic.load(Ordering::SeqCst);
        tracing::info!(
            "Batch ID: {}. Reading {} pending ingested requests from DB.",
            current_batch_id,
            num_to_poll
        );

        let rows = self
            .iris_store
            .pending_ingested_requests(num_to_poll)
            .await
            .map_err(ReceiveRequestError::BatchSyncError)?;

        if rows.len() != num_to_poll as usize {
            return Err(ReceiveRequestError::BatchSyncError(eyre!(
                "expected {} pending ingested requests for batch {}, found {}",
                num_to_poll,
                current_batch_id,
                rows.len()
            )));
        }

        for row in &rows {
            self.process_ingested_request(row).await?;
        }

        let sequence_numbers: Vec<String> =
            rows.iter().map(|row| row.sequence_number.clone()).collect();
        self.iris_store
            .mark_ingested_requests_consumed(&sequence_numbers, current_batch_id)
            .await
            .map_err(ReceiveRequestError::BatchSyncError)?;
        // Carry the claimed set with the batch so the results transaction can
        // mark exactly these rows persisted (batch ids are not stable enough
        // to correlate claim and persist — they restart per boot and the
        // shared atomic advances under prefetch).
        self.batch_query
            .sqs_sequence_numbers
            .extend(sequence_numbers.iter().cloned());
        tracing::info!(
            "Batch ID: {}. Marked {} ingested requests consumed.",
            current_batch_id,
            sequence_numbers.len()
        );
        Ok(())
    }

    async fn process_ingested_request(
        &mut self,
        ingested_request: &IngestedRequest,
    ) -> Result<(), ReceiveRequestError> {
        let message: SQSMessage = serde_json::from_str(&ingested_request.message_body)
            .map_err(|e| ReceiveRequestError::json_parse_error("ingested SQS body", e))?;

        let message_attributes = message.message_attributes.clone();
        let batch_metadata = self.extract_batch_metadata(&message_attributes);

        let request_type = message_attributes
            .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            .string_value()
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            .to_string();

        #[cfg(feature = "explicit-sns-batching")]
        if request_type == BATCH_MESSAGE_TYPE {
            self.process_batch_message(&message, batch_metadata).await?;
            self.msg_counter += 1;
            return Ok(());
        }

        self.process_message_(&message, &request_type, batch_metadata)
            .await?;
        self.msg_counter += 1;
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

        // SQS delete-after-persist (POP-3781). The message is only
        // deleted once processing has returned Ok — i.e. the modification
        // row is durably persisted. If the process crashes (or returns
        // Err) between receiving and persisting, `?` propagates without
        // acking, SQS visibility timeout expires, and the message is
        // redelivered. The previous delete-before-process ordering lost
        // messages on crashes in that window.
        //
        // `DeleteMessage` itself uses strict `?` propagation. If the ack
        // fails after persistence has succeeded, the Err tears down
        // receive_batch_stream (batch.rs:116 — any Err from receive_batch
        // breaks the spawn loop), which causes the node to restart. On
        // restart, modifications_sync GCs any orphan IN_PROGRESS rows
        // (no peer has them COMPLETED), and the redelivered SQS message
        // is re-ingested cleanly. Without strict propagation, the orphan
        // IN_PROGRESS row would persist (modifications_sync only runs at
        // startup) and visibility-timeout redelivery would produce a
        // second COMPLETED row plus a duplicate SNS result. Strict `?`
        // is the trigger for the cleanup path, not just error escalation.
        #[cfg(feature = "explicit-sns-batching")]
        if request_type == BATCH_MESSAGE_TYPE {
            self.process_batch_message(&message, batch_metadata).await?;
            self.delete_message(&sqs_message).await?;
            self.msg_counter += 1;
            return Ok(());
        }

        self.process_message_(&message, request_type, batch_metadata)
            .await?;
        self.delete_message(&sqs_message).await?;
        self.msg_counter += 1;
        Ok(())
    }

    async fn process_message_(
        &mut self,
        message: &SQSMessage,
        request_type: &str,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        match request_type {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                self.process_identity_deletion(message, batch_metadata)
                    .await
            }
            UNIQUENESS_MESSAGE_TYPE => {
                self.process_uniqueness_request(message, batch_metadata)
                    .await
            }
            REAUTH_MESSAGE_TYPE => self.process_reauth_request(message, batch_metadata).await,
            RECOVERY_CHECK_MESSAGE_TYPE => {
                self.process_identity_match_check_request(
                    message,
                    batch_metadata,
                    RECOVERY_CHECK_MESSAGE_TYPE,
                    self.config.enable_recovery,
                )
                .await
            }
            RESET_CHECK_MESSAGE_TYPE => {
                self.process_identity_match_check_request(
                    message,
                    batch_metadata,
                    RESET_CHECK_MESSAGE_TYPE,
                    self.config.enable_reset,
                )
                .await
            }
            RECOVERY_UPDATE_MESSAGE_TYPE => {
                self.process_identity_update_request(
                    message,
                    batch_metadata,
                    RECOVERY_UPDATE_MESSAGE_TYPE,
                    self.config.enable_recovery,
                )
                .await
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                self.process_identity_update_request(
                    message,
                    batch_metadata,
                    RESET_UPDATE_MESSAGE_TYPE,
                    self.config.enable_reset,
                )
                .await
            }
            _ => {
                tracing::error!("Error: {}", ReceiveRequestError::InvalidMessageType);
                Ok(())
            }
        }
    }

    #[cfg(feature = "explicit-sns-batching")]
    async fn process_batch_message(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        let sns_message_id = message.message_id.clone();

        // Parse the compressed payload wrapper
        let payload: CompressedBatchPayload = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Compressed batch payload", e))?;

        // Decompress and parse the batch
        let batch = CompactBatchRequest::decompress(&payload.data)
            .map_err(|e| ReceiveRequestError::BatchDecompressionError(e.to_string()))?;

        let _ = message; // don't accidentally use it in the below loop

        let total_messages = batch.items.len();
        let mut errors: Vec<(String, ReceiveRequestError)> = Vec::new();

        for (idx, item) in batch.items.into_iter().enumerate() {
            // combine with the SQS message id to ensure unique ids across service client restarts
            let msg_id = format!("{}:{}", &message.message_id, idx);
            let request_type = item.message_type();

            // Convert to SQSMessage for compatibility with existing process_message_
            let msg = match item.into_sqs_message(msg_id.clone()) {
                Ok(m) => m,
                Err(e) => {
                    errors.push((
                        msg_id,
                        ReceiveRequestError::json_parse_error("RequestPayload", e),
                    ));
                    continue;
                }
            };

            if let Err(e) = self
                .process_message_(&msg, request_type, batch_metadata.clone())
                .await
            {
                errors.push((msg_id, e));
            }
        }

        tracing::info!(
            "Processed batch {}: {}/{} messages successful",
            sns_message_id,
            total_messages - errors.len(),
            total_messages
        );

        Ok(())
    }

    async fn process_identity_deletion(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => "identity_deletion").increment(1);
        let sns_message_id = message.message_id.clone();
        let identity_deletion_request: IdentityDeletionRequest =
            serde_json::from_str(&message.message).map_err(|e| {
                ReceiveRequestError::json_parse_error("Identity deletion request", e)
            })?;

        if self.config.enable_deletion {
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
                metrics::counter!("request.skipped_duplicate", "type" => IDENTITY_DELETION_MESSAGE_TYPE).increment(1);
                let error_result = IdentityDeletionResult::new_error_result(
                    self.config.party_id,
                    identity_deletion_request.serial_id,
                    "duplicate serial_id in batch",
                );
                let message = serde_json::to_string(&error_result).unwrap();
                send_error_results_to_sns(
                    message,
                    &batch_metadata,
                    self.sns_client,
                    self.config,
                    self.identity_deletion_error_result_attributes,
                    IDENTITY_DELETION_MESSAGE_TYPE,
                )
                .await?;
                return Ok(());
            }

            // Persist in progress modification
            let modification = persist_modification(
                self.config.disable_persistence,
                self.iris_store,
                Some(identity_deletion_request.serial_id as i64),
                IDENTITY_DELETION_MESSAGE_TYPE,
                None,
            )
            .await?;
            self.batch_query.modifications.insert(
                RequestSerialId(identity_deletion_request.serial_id),
                modification,
            );

            self.batch_query.push_deletion_request(
                sns_message_id,
                identity_deletion_request.serial_id - 1,
                batch_metadata,
            );
        } else {
            tracing::warn!("Identity deletions are disabled");
            let error_result = IdentityDeletionResult::new_error_result(
                self.config.party_id,
                identity_deletion_request.serial_id,
                "Identity deletions are disabled",
            );
            let message = serde_json::to_string(&error_result).unwrap();
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                self.identity_deletion_error_result_attributes,
                IDENTITY_DELETION_MESSAGE_TYPE,
            )
            .await?;
        }

        Ok(())
    }

    async fn process_uniqueness_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => "uniqueness_verification").increment(1);
        let sns_message_id = message.message_id.clone();
        let uniqueness_request: UniquenessRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Uniqueness request", e))?;

        // Persist in progress modification
        let modification = persist_modification(
            self.config.disable_persistence,
            self.iris_store,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some(uniqueness_request.s3_key.as_str()),
        )
        .await?;
        self.batch_query.modifications.insert(
            RequestId(uniqueness_request.signup_id.clone()),
            modification,
        );

        if let Some(enable_mirror_attacks) =
            uniqueness_request.full_face_mirror_attacks_detection_enabled
        {
            if enable_mirror_attacks != self.batch_query.full_face_mirror_attacks_detection_enabled
            {
                self.batch_query.full_face_mirror_attacks_detection_enabled = enable_mirror_attacks;
                tracing::info!(
                    "Setting full-face mirror-attack detection to {} for batch due to request from {}",
                    enable_mirror_attacks,
                    uniqueness_request.signup_id
                );
            }
        }

        let or_rule_indices = self.update_luc_config_if_needed(&uniqueness_request);

        self.batch_query.push_matching_request(
            sns_message_id,
            uniqueness_request.signup_id.clone(),
            UNIQUENESS_MESSAGE_TYPE,
            batch_metadata,
            or_rule_indices,
            uniqueness_request.skip_persistence.unwrap_or(false),
        );

        self.add_iris_shares_task(uniqueness_request.s3_key)?;

        Ok(())
    }

    async fn process_reauth_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => "reauth").increment(1);
        let sns_message_id = message.message_id.clone();
        let reauth_request: ReAuthRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Reauth request", e))?;

        tracing::debug!("Received reauth request: {:?}", reauth_request);

        if !self.config.enable_reauth {
            tracing::warn!("Reauth is disabled");
            let error_result = ReAuthResult::new_error_result(
                reauth_request.reauth_id,
                self.config.party_id,
                reauth_request.serial_id,
                "Reauth is disabled",
            );
            let message = serde_json::to_string(&error_result).unwrap();
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                self.reauth_error_result_attributes,
                REAUTH_MESSAGE_TYPE,
            )
            .await?;
            return Ok(());
        }

        if reauth_request.use_or_rule
            && !(self.config.luc_enabled && self.config.luc_serial_ids_from_smpc_request)
        {
            tracing::error!(
                "Received a reauth request with use_or_rule set to true, but LUC is not enabled"
            );
            let error_result = ReAuthResult::new_error_result(
                reauth_request.reauth_id,
                self.config.party_id,
                reauth_request.serial_id,
                "LUC is not enabled for use_or_rule",
            );
            let message = serde_json::to_string(&error_result).unwrap();
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                self.reauth_error_result_attributes,
                REAUTH_MESSAGE_TYPE,
            )
            .await?;
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
            metrics::counter!("request.skipped_duplicate", "type" => REAUTH_MESSAGE_TYPE)
                .increment(1);
            let error_result = ReAuthResult::new_error_result(
                reauth_request.reauth_id,
                self.config.party_id,
                reauth_request.serial_id,
                "duplicate serial_id in batch",
            );
            let message = serde_json::to_string(&error_result).unwrap();
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                self.reauth_error_result_attributes,
                REAUTH_MESSAGE_TYPE,
            )
            .await?;
            return Ok(());
        }

        // Persist in progress modification
        let modification = persist_modification(
            self.config.disable_persistence,
            self.iris_store,
            Some(reauth_request.serial_id as i64),
            REAUTH_MESSAGE_TYPE,
            Some(reauth_request.s3_key.as_str()),
        )
        .await?;
        self.batch_query
            .modifications
            .insert(RequestSerialId(reauth_request.serial_id), modification);

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

        self.batch_query.push_matching_request(
            sns_message_id,
            reauth_request.reauth_id.clone(),
            REAUTH_MESSAGE_TYPE,
            batch_metadata,
            or_rule_indices,
            reauth_request.skip_persistence.unwrap_or(false),
        );

        self.add_iris_shares_task(reauth_request.s3_key)?;

        Ok(())
    }

    async fn process_identity_match_check_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        request_type: &str,
        is_request_type_enabled: bool,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => request_type.to_string()).increment(1);
        let sns_message_id = message.message_id.clone();
        let identity_match_check_request: IdentityMatchCheckRequest =
            serde_json::from_str(&message.message)
                .map_err(|e| ReceiveRequestError::json_parse_error("Identity check request", e))?;

        tracing::debug!(
            "Received {} request: {:?}",
            request_type,
            identity_match_check_request.clone()
        );

        if !is_request_type_enabled {
            tracing::warn!("{} is disabled", request_type);
            let error_result = IdentityMatchCheckResult::new_error_result(
                identity_match_check_request.request_id,
                self.config.party_id,
                &format!("{} is disabled", request_type),
            );
            let message = serde_json::to_string(&error_result).unwrap();
            let error_attrs = if request_type == RESET_CHECK_MESSAGE_TYPE {
                self.reset_check_error_result_attributes
            } else {
                self.recovery_check_error_result_attributes
            };
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                error_attrs,
                request_type,
            )
            .await?;
            return Ok(());
        }

        // Persist in progress reset_check message.
        // Note that reset_check is only a query and does not persist anything into the database.
        // We store modification so that the SNS result can be replayed.
        let modification = persist_modification(
            self.config.disable_persistence,
            self.iris_store,
            None,
            request_type,
            Some(identity_match_check_request.s3_key.as_str()),
        )
        .await?;
        self.batch_query.modifications.insert(
            RequestId(identity_match_check_request.request_id.clone()),
            modification,
        );

        self.batch_query.push_matching_request(
            sns_message_id,
            identity_match_check_request.request_id.clone(),
            request_type,
            batch_metadata,
            vec![], // reset checks use the AND rule
            false,  // skip_persistence is only used for uniqueness requests
        );

        self.add_iris_shares_task(identity_match_check_request.s3_key)?;

        Ok(())
    }

    async fn process_identity_update_request(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
        request_type: &str,
        is_request_type_enabled: bool,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => request_type.to_string()).increment(1);
        let sns_message_id = message.message_id.clone();
        let identity_update_request: IdentityUpdateRequest = serde_json::from_str(&message.message)
            .map_err(|e| ReceiveRequestError::json_parse_error("Identity update request", e))?;

        tracing::debug!(
            "Received {} request: {:?}",
            request_type,
            identity_update_request
        );

        if !is_request_type_enabled {
            tracing::warn!("{} is disabled", request_type);
            let error_result = IdentityUpdateAckResult::new_error_result(
                identity_update_request.request_id,
                self.config.party_id,
                identity_update_request.serial_id,
                &format!("{} is disabled", request_type),
            );
            let message = serde_json::to_string(&error_result).unwrap();
            let error_attrs = if request_type == RESET_UPDATE_MESSAGE_TYPE {
                self.reset_update_error_result_attributes
            } else {
                self.recovery_update_error_result_attributes
            };
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                error_attrs,
                request_type,
            )
            .await?;
            return Ok(());
        }

        // Check for duplicate serial_id before downloading S3 shares to avoid wasted work
        if self
            .batch_query
            .modifications
            .contains_key(&RequestSerialId(identity_update_request.serial_id))
        {
            tracing::warn!(
                "Received multiple modification operations in batch on serial id: {}. Skipping {:?}",
                identity_update_request.serial_id,
                identity_update_request,
            );
            metrics::counter!("request.skipped_duplicate", "type" => request_type.to_string())
                .increment(1);
            let error_result = IdentityUpdateAckResult::new_error_result(
                identity_update_request.request_id,
                self.config.party_id,
                identity_update_request.serial_id,
                "duplicate serial_id in batch",
            );
            let message = serde_json::to_string(&error_result).unwrap();
            let error_attrs = if request_type == RESET_UPDATE_MESSAGE_TYPE {
                self.reset_update_error_result_attributes
            } else {
                self.recovery_update_error_result_attributes
            };
            send_error_results_to_sns(
                message,
                &batch_metadata,
                self.sns_client,
                self.config,
                error_attrs,
                request_type,
            )
            .await?;
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
            identity_update_request.s3_key.clone(),
        )?;
        // Preserve the old GPU-local behavior: skip only this identity update if share
        // fetching/parsing fails, and keep receiving the rest of the batch.
        let (left_shares, right_shares) = match task_handle.await {
            Ok(result) => match result {
                Ok(shares) => shares,
                Err(e) => {
                    tracing::error!(
                        "Failed to process iris shares for {}: {:?}",
                        request_type,
                        e
                    );
                    metrics::counter!(
                        "request.skipped",
                        "type" => request_type.to_string(),
                        "reason" => "failed_to_process_iris_shares"
                    )
                    .increment(1);
                    return Ok(());
                }
            },
            Err(e) => {
                tracing::error!("Failed to join task handle for {}: {:?}", request_type, e);
                metrics::counter!(
                    "request.skipped",
                    "type" => request_type.to_string(),
                    "reason" => "failed_to_join_handle"
                )
                .increment(1);
                return Ok(());
            }
        };

        let modification = persist_modification(
            self.config.disable_persistence,
            self.iris_store,
            Some(identity_update_request.serial_id as i64),
            request_type,
            Some(identity_update_request.s3_key.as_ref()),
        )
        .await?;

        self.batch_query.modifications.insert(
            RequestSerialId(identity_update_request.serial_id),
            modification,
        );

        self.batch_query.push_identity_update_request(
            sns_message_id,
            identity_update_request.request_id,
            request_type,
            identity_update_request.serial_id - 1,
            GaloisSharesBothSides {
                code_left: left_shares.code,
                mask_left: left_shares.mask,
                code_right: right_shares.code,
                mask_right: right_shares.mask,
            },
        );
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
                    self.batch_query
                        .push_matching_request_shares(share_left, share_right, true);
                }
                Err(e) => {
                    tracing::error!("Failed to process iris shares: {:?}", e);
                    self.handle_share_processing_error(index).await?;

                    // Create dummy shares for invalid entry
                    let (dummy_left, dummy_right) = self.create_dummy_shares();
                    self.batch_query
                        .push_matching_request_shares(dummy_left, dummy_right, false);
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
                let message = IdentityMatchCheckResult::new_error_result(
                    request_id.clone(),
                    self.config.party_id,
                    ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                );
                (
                    self.reset_check_error_result_attributes,
                    serde_json::to_string(&message).unwrap(),
                )
            }
            RECOVERY_CHECK_MESSAGE_TYPE => {
                let message = IdentityMatchCheckResult::new_error_result(
                    request_id.clone(),
                    self.config.party_id,
                    ERROR_FAILED_TO_PROCESS_IRIS_SHARES,
                );
                (
                    self.recovery_check_error_result_attributes,
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
        let receipt_handle = sqs_message.receipt_handle.as_deref().ok_or_else(|| {
            ReceiveRequestError::FailedToMarkRequestAsDeleted(eyre::eyre!(
                "SQS message missing receipt handle: {:?}",
                sqs_message.message_id
            ))
        })?;
        self.client
            .delete_message()
            .queue_url(&self.config.requests_queue_url)
            .receipt_handle(receipt_handle)
            .send()
            .await
            .map_err(ReceiveRequestError::from)?;
        tracing::debug!("Deleted message: {:?}", sqs_message.message_id);
        Ok(())
    }

    fn update_luc_config_if_needed(&mut self, uniqueness_request: &UniquenessRequest) -> Vec<u32> {
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
        or_rule_indices
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

    fn create_dummy_shares(&self) -> (GaloisShares, GaloisShares) {
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

        (dummy.clone(), dummy)
    }
}

async fn persist_modification(
    disable_persistence: bool,
    iris_store: &Store,
    serial_id: Option<i64>,
    request_type: &str,
    s3_url: Option<&str>,
) -> Result<Modification, ReceiveRequestError> {
    if disable_persistence {
        tracing::debug!("Persistence is disabled, skipping modification persistence");
        return Ok(Modification::default());
    }
    let modification = iris_store
        .insert_modification(serial_id, request_type, s3_url)
        .await
        .map_err(ReceiveRequestError::from)?;
    Ok(modification)
}

pub async fn get_own_batch_sync_state(
    config: &Config,
    sqs_client: &Client,
    iris_store: &Store,
    current_batch_id: u64,
) -> Result<BatchSyncState> {
    let available_messages = if config.db_backed_ingest {
        let pending = iris_store.count_pending_ingested_requests().await?;
        std::cmp::min(pending, u32::MAX as i64) as u32
    } else {
        get_approximate_number_of_messages(&sqs_client.clone(), &config.requests_queue_url).await?
    };

    let index = current_batch_id.saturating_sub(1) as usize;
    if config.predefined_batch_sizes.len() > index {
        // predefined_batch_sizes are only used in test environments to reproduce specific scenarios
        tracing::info!(
            "Using predefined batch size {} for batch ID {}",
            config.predefined_batch_sizes[index],
            current_batch_id
        );
    };

    let messages_to_poll = if config.db_backed_ingest {
        messages_to_poll_for_available(
            available_messages,
            config.max_batch_size,
            &config.predefined_batch_sizes,
            current_batch_id,
        )
    } else if config.predefined_batch_sizes.len() > index {
        std::cmp::min(config.predefined_batch_sizes[index], config.max_batch_size) as u32
    } else {
        std::cmp::min(available_messages, config.max_batch_size as u32)
    };

    let count_source = if config.db_backed_ingest {
        "pending ingested DB rows"
    } else {
        "approximate visible SQS messages"
    };
    let log_msg = format!("fetching {}: {}", count_source, available_messages);
    match messages_to_poll {
        0 => tracing::debug!(log_msg),
        _ => tracing::info!(log_msg),
    };

    let batch_sync_state = BatchSyncState {
        messages_to_poll,
        batch_id: current_batch_id,
    };
    Ok(batch_sync_state)
}

#[cfg(test)]
mod tests {
    use super::{messages_to_poll_for_available, DbIngestBackoff};
    use iris_mpc_store::{normalize_sns_sequence_number, SNS_SEQUENCE_NUMBER_WIDTH};

    #[test]
    fn messages_to_poll_for_available_caps_by_available_messages() {
        assert_eq!(messages_to_poll_for_available(3, 10, &[], 2), 3);
    }

    #[test]
    fn messages_to_poll_for_available_caps_by_max_batch_size() {
        assert_eq!(messages_to_poll_for_available(20, 7, &[], 2), 7);
    }

    #[test]
    fn messages_to_poll_for_available_honors_predefined_batch_size() {
        assert_eq!(messages_to_poll_for_available(20, 10, &[6], 1), 6);
        assert_eq!(messages_to_poll_for_available(4, 10, &[6], 1), 4);
        assert_eq!(messages_to_poll_for_available(20, 5, &[6], 1), 5);
    }

    #[test]
    fn messages_to_poll_for_available_uses_saturating_batch_index() {
        let predefined_batch_sizes = [3, 9];

        assert_eq!(
            messages_to_poll_for_available(20, 10, &predefined_batch_sizes, 0),
            3
        );
        assert_eq!(
            messages_to_poll_for_available(20, 10, &predefined_batch_sizes, 1),
            3
        );
        assert_eq!(
            messages_to_poll_for_available(20, 10, &predefined_batch_sizes, 2),
            9
        );
    }

    #[test]
    fn db_ingest_backoff_starts_grows_caps_and_resets() {
        let mut backoff = DbIngestBackoff::new(8, 32);
        assert_eq!(backoff.current_ms, 8);

        let first_sleep = backoff.next_sleep();
        assert!(first_sleep.as_millis() >= 8);
        assert!(first_sleep.as_millis() <= 16);
        assert_eq!(backoff.current_ms, 16);

        let second_sleep = backoff.next_sleep();
        assert!(second_sleep >= first_sleep);
        assert!(second_sleep.as_millis() <= 32);
        assert_eq!(backoff.current_ms, 32);

        for _ in 0..10 {
            assert!(backoff.next_sleep().as_millis() <= 32);
            assert_eq!(backoff.current_ms, 32);
        }

        backoff.reset();
        assert_eq!(backoff.current_ms, 8);
    }

    #[test]
    fn db_ingest_backoff_clamps_initial_and_max() {
        let backoff = DbIngestBackoff::new(0, 0);
        assert_eq!(backoff.initial_ms, 1);
        assert_eq!(backoff.max_ms, 1);
        assert_eq!(backoff.current_ms, 1);

        let backoff = DbIngestBackoff::new(10, 5);
        assert_eq!(backoff.initial_ms, 10);
        assert_eq!(backoff.max_ms, 10);
        assert_eq!(backoff.current_ms, 10);
    }

    #[test]
    fn normalize_sns_sequence_number_pads_to_fixed_width() {
        let normalized = normalize_sns_sequence_number("123456789012345678901234567890").unwrap();

        assert_eq!(normalized.len(), SNS_SEQUENCE_NUMBER_WIDTH);
        assert_eq!(normalized, "0000000000123456789012345678901234567890");
    }

    #[test]
    fn normalize_sns_sequence_number_preserves_numeric_order_lexically() {
        let normalized_9 = normalize_sns_sequence_number("9").unwrap();
        let normalized_10 = normalize_sns_sequence_number("10").unwrap();
        let normalized_large =
            normalize_sns_sequence_number("123456789012345678901234567890").unwrap();

        assert!(normalized_9 < normalized_10);
        assert!(normalized_10 < normalized_large);
    }

    #[test]
    fn normalize_sns_sequence_number_rejects_non_numeric_input() {
        assert!(normalize_sns_sequence_number("not-a-number").is_err());
    }
}
