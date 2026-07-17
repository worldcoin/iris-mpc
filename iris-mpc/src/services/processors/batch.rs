use crate::coordinator::{batch_digest, CoordinatedRequest, CoordinatorBatchReceiver};
use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_error_results_to_sns;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sns::Client as SNSClient;
use eyre::{eyre, Result};
use iris_mpc_common::config::Config;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare, GaloisShares,
};
use iris_mpc_common::helpers::aws::{
    SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
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
use iris_mpc_store::Store;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

/// Request types the batch processor knows how to handle.
const KNOWN_MESSAGE_TYPES: &[&str] = &[
    IDENTITY_DELETION_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE,
    RECOVERY_CHECK_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE,
    RECOVERY_UPDATE_MESSAGE_TYPE,
];

/// True iff the failure is fully determined by the message CONTENT — identical
/// bytes on all three parties, so quarantining is symmetric by construction.
/// Everything environment-dependent (S3, DB, joins, coordinator I/O) must stay fatal:
/// quarantining one of those on a single party would skip a row only there and
/// permanently diverge the batches. Exhaustive on purpose — a new variant must
/// be classified here explicitly, at compile time.
fn is_content_poison(err: &ReceiveRequestError) -> bool {
    match err {
        ReceiveRequestError::JsonParseError { .. }
        | ReceiveRequestError::NoMessageTypeAttribute
        | ReceiveRequestError::NoStringMessageTypeAttribute
        | ReceiveRequestError::InvalidMessageType => true,
        ReceiveRequestError::FailedToPersistModification(_)
        | ReceiveRequestError::FailedToJoinHandle(_)
        | ReceiveRequestError::CoordinatorError(_)
        | ReceiveRequestError::FailedToProcessIrisShares(_) => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn receive_batch_stream(
    party_id: usize,
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
    iris_store: Store,
    mut coordinator_receiver: CoordinatorBatchReceiver,
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
                    &iris_store,
                    &mut coordinator_receiver,
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
    iris_store: &Store,
    coordinator_receiver: &mut CoordinatorBatchReceiver,
) -> Result<Option<BatchQuery>, ReceiveRequestError> {
    let mut processor = BatchProcessor::new(
        party_id,
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
        iris_store,
    );

    processor.receive_batch(coordinator_receiver).await
}

pub struct BatchProcessor<'a> {
    party_id: usize,
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
    coordinator_rejected_request_ids: Vec<String>,
    iris_store: &'a Store,
}

impl<'a> BatchProcessor<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        party_id: usize,
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
        iris_store: &'a Store,
    ) -> Self {
        Self {
            party_id,
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
            coordinator_rejected_request_ids: vec![],
            iris_store,
        }
    }

    pub async fn receive_batch(
        &mut self,
        coordinator_receiver: &mut CoordinatorBatchReceiver,
    ) -> Result<Option<BatchQuery>, ReceiveRequestError> {
        if self.shutdown_handler.is_shutting_down() {
            tracing::info!("Stopping batch receive due to shutdown signal...");
            return Ok(None);
        }

        self.receive_coordinated_batch(coordinator_receiver).await
    }

    async fn receive_coordinated_batch(
        &mut self,
        coordinator_receiver: &mut CoordinatorBatchReceiver,
    ) -> Result<Option<BatchQuery>, ReceiveRequestError> {
        let Some(batch) = coordinator_receiver
            .next_batch(self.config.max_batch_size)
            .await
            .map_err(ReceiveRequestError::CoordinatorError)?
        else {
            return Ok(None);
        };

        let process_result = async {
            for request in &batch.requests {
                self.process_coordinated_request(request).await?;
            }
            self.process_parse_tasks().await
        }
        .await;

        if let Err(error) = process_result {
            let _ = coordinator_receiver.reject(error.to_string()).await;
            return Err(error);
        }

        if self.batch_query.coordinator_request_ids.len() != self.batch_query.valid_entries.len() {
            let error = ReceiveRequestError::CoordinatorError(eyre!(
                "coordinator request ids and validity flags differ in length: {} != {}",
                self.batch_query.coordinator_request_ids.len(),
                self.batch_query.valid_entries.len()
            ));
            let _ = coordinator_receiver.reject(error.to_string()).await;
            return Err(error);
        }
        let mut locally_rejected = std::mem::take(&mut self.coordinator_rejected_request_ids);
        locally_rejected.extend(
            self.batch_query
                .coordinator_request_ids
                .iter()
                .zip(&self.batch_query.valid_entries)
                .filter_map(|(request_id, valid)| (!valid).then_some(request_id.clone())),
        );
        let rejected = coordinator_receiver
            .prepared(batch_digest(&batch.requests), locally_rejected)
            .await
            .map_err(ReceiveRequestError::CoordinatorError)?;
        self.batch_query.reject_coordinator_requests(&rejected);

        tracing::info!(
            "Coordinator batch {} prepared with {} request(s), {} rejected",
            batch.batch_id,
            batch.requests.len(),
            rejected.len(),
        );
        Ok(Some(self.batch_query.clone()))
    }

    async fn process_coordinated_request(
        &mut self,
        coordinated_request: &CoordinatedRequest,
    ) -> Result<(), ReceiveRequestError> {
        let executable_requests_before = self.batch_query.requests_order.len();
        let result = match self
            .try_process_message_body(
                &coordinated_request.message_body,
                Some(&coordinated_request.request_id),
            )
            .await
        {
            Err(error) if is_content_poison(&error) => {
                tracing::error!(
                    request_id = coordinated_request.request_id,
                    "Coordinator quarantined invalid request: {error}"
                );
                Ok(())
            }
            other => other,
        };
        result?;

        if self.batch_query.requests_order.len() == executable_requests_before {
            self.coordinator_rejected_request_ids
                .push(coordinated_request.request_id.clone());
        }

        Ok(())
    }

    async fn try_process_message_body(
        &mut self,
        message_body: &str,
        expected_request_id: Option<&str>,
    ) -> Result<(), ReceiveRequestError> {
        let message: SQSMessage = serde_json::from_str(message_body)
            .map_err(|e| ReceiveRequestError::json_parse_error("request envelope", e))?;

        if let Some(expected_request_id) = expected_request_id {
            if message.message_id != expected_request_id {
                return Err(ReceiveRequestError::CoordinatorError(eyre!(
                    "coordinator request id mismatch: row={}, envelope={}",
                    expected_request_id,
                    message.message_id
                )));
            }
        }

        let message_attributes = message.message_attributes.clone();
        let batch_metadata = self.extract_batch_metadata(&message_attributes);

        let request_type = message_attributes
            .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            .string_value()
            .ok_or(ReceiveRequestError::NoMessageTypeAttribute)?
            .to_string();

        if !KNOWN_MESSAGE_TYPES.contains(&request_type.as_str()) {
            return Err(ReceiveRequestError::InvalidMessageType);
        }

        self.process_message_(&message, &request_type, batch_metadata)
            .await?;
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

    async fn process_identity_deletion(
        &mut self,
        message: &SQSMessage,
        batch_metadata: BatchMetadata,
    ) -> Result<(), ReceiveRequestError> {
        metrics::counter!("request.received", "type" => "identity_deletion").increment(1);
        let coordinator_request_id = message.message_id.clone();
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
                coordinator_request_id,
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
        let coordinator_request_id = message.message_id.clone();
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
            coordinator_request_id,
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
        let coordinator_request_id = message.message_id.clone();
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
            coordinator_request_id,
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
        let coordinator_request_id = message.message_id.clone();
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
            coordinator_request_id,
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
        let coordinator_request_id = message.message_id.clone();
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
            coordinator_request_id,
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

#[cfg(test)]
mod tests {
    use super::{is_content_poison, ReceiveRequestError, SQSMessage, KNOWN_MESSAGE_TYPES};
    use iris_mpc_common::helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RECOVERY_CHECK_MESSAGE_TYPE,
        RECOVERY_UPDATE_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
        UNIQUENESS_MESSAGE_TYPE,
    };
    #[test]
    fn is_content_poison_classifies_every_constructible_variant() {
        // Content-determined -> poison (quarantine is symmetric across parties).
        let json_err = serde_json::from_str::<SQSMessage>("{{{").unwrap_err();
        assert!(is_content_poison(&ReceiveRequestError::json_parse_error(
            "test", json_err
        )));
        assert!(is_content_poison(
            &ReceiveRequestError::NoMessageTypeAttribute
        ));
        assert!(is_content_poison(
            &ReceiveRequestError::NoStringMessageTypeAttribute
        ));
        assert!(is_content_poison(&ReceiveRequestError::InvalidMessageType));

        // Environment-dependent -> MUST stay fatal. Quarantining any of these
        // on one party would skip a row only there and diverge the batches.
        assert!(!is_content_poison(
            &ReceiveRequestError::FailedToPersistModification(eyre::eyre!("db down"))
        ));
        assert!(!is_content_poison(&ReceiveRequestError::CoordinatorError(
            eyre::eyre!("peer timeout")
        )));
        assert!(!is_content_poison(
            &ReceiveRequestError::FailedToProcessIrisShares(eyre::eyre!("s3"))
        ));
        // FailedToJoinHandle is not constructible without runtime machinery;
        // the exhaustive match in is_content_poison (no wildcard arm) is the
        // compile-time guard that any new variant gets an explicit classification.
    }

    #[test]
    fn known_message_types_cover_all_processor_arms() {
        for t in [
            IDENTITY_DELETION_MESSAGE_TYPE,
            UNIQUENESS_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE,
            RECOVERY_CHECK_MESSAGE_TYPE,
            RESET_CHECK_MESSAGE_TYPE,
            RESET_UPDATE_MESSAGE_TYPE,
            RECOVERY_UPDATE_MESSAGE_TYPE,
        ] {
            assert!(KNOWN_MESSAGE_TYPES.contains(&t));
        }
        assert!(!KNOWN_MESSAGE_TYPES.contains(&"bogus-type"));
    }
}
