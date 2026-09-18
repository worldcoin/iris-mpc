use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use async_from::{self, AsyncFrom};
use futures::StreamExt;
use iris_mpc_cpu::execution::hawk_main::BothEyes;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::Serialize;

use iris_mpc_common::{helpers::smpc_request, SerialId};
use tokio::time::{sleep, timeout, Instant};
use uuid::Uuid;

use crate::{
    aws::types::SnsMessageInfo,
    client::options::{RequestBatchOptions, SharesGeneratorOptions},
    constants::N_PARTIES,
    irises::GaloisRingSharedIrisForUpload,
};

use super::aws::AwsClient;
use components::SharesGenerator;

pub use options::{AwsOptions, ServiceClientOptions};
pub use typeset::*;

mod components;
mod options;
mod typeset;

/// Delay (seconds) after S3 uploads to allow share propagation before publishing SNS messages.
const S3_PROPAGATION_DELAY_SECS: u64 = 1;

/// Maximum time (seconds) without receiving a response before declaring a batch timed out.
const RESPONSE_TIMEOUT_SECS: u64 = 180;

/// Given a RequestBatchOptions, do the following:
/// - turn them into requests
/// - upload the corresponding iris shares (on failure exit early)
/// - upload the requests (on failure, process the uploaded requests and then exit early)
/// - receive the results (on failure, process the outstanding requests and then exit early)
///
/// Due to the need to continue processing to clean up requests when possible, error handling uses
/// a ErrorBits struct rather than returning a Result immediately.
pub struct ServiceClient {
    aws_client: AwsClient,
    request_batch: Option<RequestBatchOptions>,
    shares_generator: SharesGenerator<StdRng>,
    request_id_rng: Option<StdRng>,
    results_output_path: Option<PathBuf>,
    cleanup_on_exit: bool,
    state: ExecState,
}

impl ServiceClient {
    pub async fn new(
        opts_aws: AwsOptions,
        request_batch: RequestBatchOptions,
        shares_generator: SharesGeneratorOptions,
    ) -> Result<Self, ServiceClientError> {
        let aws_client = AwsClient::async_from(opts_aws).await;
        let request_id_rng = request_batch
            .deterministic_seed()
            .map(|seed| StdRng::seed_from_u64(seed ^ 0x6a09_e667_f3bc_c909));
        let shares_generator = SharesGenerator::<StdRng>::from_options(shares_generator);

        Ok(Self {
            aws_client,
            request_batch: Some(request_batch),
            shares_generator,
            request_id_rng,
            results_output_path: None,
            cleanup_on_exit: true,
            state: ExecState::new(),
        })
    }

    pub fn with_results_output_path(mut self, path: Option<&Path>) -> Self {
        self.results_output_path = path.map(Path::to_path_buf);
        self
    }

    pub fn with_cleanup_on_exit(mut self, cleanup_on_exit: bool) -> Self {
        self.cleanup_on_exit = cleanup_on_exit;
        self
    }

    async fn init(&mut self) -> Result<(), ServiceClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ServiceClientError::AwsServiceError)?;

        self.aws_client
            .sqs_purge_response_queue()
            .await
            .map_err(ServiceClientError::AwsServiceError)?;

        Ok(())
    }

    // consumes the service client intentionally
    pub async fn run(mut self) -> Result<(), ServiceClientError> {
        self.init().await?;

        // Run main batch loop, interruptible by Ctrl+C.
        let execution_result = tokio::select! {
            result = self.exec() => {
                tracing::info!("service client finished");
                result
            },
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("\nCtrl+C received. Stopping service client...");
                Err(ServiceClientError::ResponseError(
                    "service client interrupted".to_string(),
                ))
            }
        };

        if let Some(path) = self.results_output_path.as_deref() {
            self.state.write_results(path)?;
        }

        if self.cleanup_on_exit {
            self.cleanup_live_serial_ids().await;
        } else {
            tracing::info!(
                "Skipping cleanup of {} live serial IDs for ground-truth capture",
                self.state.live_serial_ids.len()
            );
        }

        execution_result
    }

    async fn cleanup_live_serial_ids(&mut self) {
        tracing::info!(
            "Cleaning up {} serial IDs",
            self.state.live_serial_ids.len()
        );

        // Send deletion requests for all live serial IDs.
        let live_serial_ids = std::mem::take(&mut self.state.live_serial_ids)
            .into_iter()
            .collect::<Vec<_>>();
        let deletion_messages: Vec<SnsMessageInfo> = live_serial_ids
            .iter()
            .map(|&serial_id| {
                let payload =
                    RequestPayload::IdentityDeletion(smpc_request::IdentityDeletionRequest {
                        serial_id,
                    });
                SnsMessageInfo::from(payload)
            })
            .collect();

        let idxs = self
            .aws_client
            .sns_publish_json_batch(&deletion_messages)
            .await;

        for idx in &idxs {
            tracing::info!("publishing Deletion for {}", live_serial_ids[*idx]);
        }

        if idxs.len() != live_serial_ids.len() {
            tracing::error!(
                "Failed to send {} deletions",
                live_serial_ids.len() - idxs.len()
            );
        } else {
            tracing::info!("Cleanup complete. Deletions have been submitted.");
        }
    }

    async fn exec(&mut self) -> Result<(), ServiceClientError> {
        let request_batch = self
            .request_batch
            .take()
            .expect("exec() called more than once");

        for (batch_idx, batch) in request_batch.into_iter().enumerate() {
            self.handle_batch(batch_idx, batch).await?;
        }

        Ok(())
    }

    async fn handle_batch(
        &mut self,
        batch_idx: usize,
        batch: Vec<options::RequestOptions>,
    ) -> Result<(), ServiceClientError> {
        // Phase 1: Prepare requests and generate shares.
        let (batch_requests, batch_shares) = self.prepare_batch_requests(batch_idx, &batch)?;

        // Phase 2: Upload shares to S3.
        self.upload_shares(batch_shares).await?;

        // Phase 3: Wait for S3 propagation.
        sleep(Duration::from_secs(S3_PROPAGATION_DELAY_SECS)).await;

        // Phase 4: Publish requests to SNS.
        // From this point forward, continue on error in an attempt to clean up requests
        // which have already been published
        let published_idxs = self.publish_requests(&batch_requests).await;

        // Phase 5: Track published requests and wait for responses.
        self.state
            .track_batch_requests(&published_idxs, &batch_requests);
        self.state.process_responses(&self.aws_client).await;

        // Check all error conditions and return consolidated error
        if self.state.error_bits.has_errors() {
            return Err(ServiceClientError::ResponseError(format!(
                "batch {} failed: {}",
                batch_idx, self.state.error_bits
            )));
        }

        tracing::info!(
            "Batch {} finished. Responses to non-deletion requests have been received",
            batch_idx
        );

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn prepare_batch_requests(
        &mut self,
        batch_idx: usize,
        batch: &[options::RequestOptions],
    ) -> Result<
        (
            Vec<typeset::Request>,
            Vec<Option<(Uuid, BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>)>>,
        ),
        ServiceClientError,
    > {
        use crate::client::options::Parent;

        let mut batch_requests: Vec<typeset::Request> = Vec::new();
        let mut batch_shares = Vec::new();

        for (item_idx, opts) in batch.iter().enumerate() {
            // Resolve parent serial ID from labels or use provided ID.
            let parent_serial_id: Option<SerialId> = match opts.get_parent() {
                Some(Parent::Id(id)) => Some(id),
                Some(Parent::Label(label)) => {
                    if let Some(&serial_id) = self.state.uniqueness_labels.get(label.as_str()) {
                        Some(serial_id)
                    } else {
                        tracing::error!(
                            "batch {}.{}: dropping request — parent label '{}' unresolved",
                            batch_idx,
                            item_idx,
                            label,
                        );
                        self.state.error_bits.set_request_dropped_error();
                        continue;
                    }
                }
                _ => None,
            };

            let info = typeset::RequestInfo::with_indices(
                batch_idx,
                item_idx,
                opts.label(),
                opts.expected().cloned(),
            );
            let correlation_id = if let Some(rng) = self.request_id_rng.as_mut() {
                let mut bytes = [0u8; 16];
                rng.fill_bytes(&mut bytes);
                // Set RFC 4122 variant and v4 bits while keeping the UUID fully
                // deterministic for a given ground-truth seed.
                bytes[6] = (bytes[6] & 0x0f) | 0x40;
                bytes[8] = (bytes[8] & 0x3f) | 0x80;
                Uuid::from_bytes(bytes)
            } else {
                Uuid::new_v4()
            };
            let request = match opts.make_request_with_uuid(info, parent_serial_id, correlation_id)
            {
                Ok(r) => r,
                Err(e) => {
                    // may as well see all the failed requests in the batch. continuing won't result
                    // in any more requests getting sent to iris-mpc-hawk
                    tracing::error!("batch {}.{}: dropping request — {}", batch_idx, item_idx, e,);
                    self.state.error_bits.set_request_dropped_error();
                    continue;
                }
            };

            // Pre-generate shares for request types that require them.
            let shares_info = if let Some((op_uuid, iris_pair)) = request.get_shares_info() {
                let shares = if opts.is_mirrored() {
                    self.shares_generator.generate_mirrored(iris_pair.as_ref())
                } else {
                    self.shares_generator.generate(iris_pair.as_ref())
                };
                Some((op_uuid, shares))
            } else {
                None
            };

            batch_requests.push(request);
            batch_shares.push(shares_info);
        }

        if self.state.error_bits.get_request_dropped_error() {
            Err(ServiceClientError::RequestPreparationError)
        } else {
            Ok((batch_requests, batch_shares))
        }
    }

    async fn upload_shares(
        &mut self,
        batch_shares: Vec<Option<(Uuid, BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>)>>,
    ) -> Result<(), ServiceClientError> {
        tracing::info!("uploading iris shares");
        for shares_info in batch_shares.iter().filter_map(|opt| opt.as_ref()) {
            let (op_uuid, shares) = shares_info;
            if let Err(e) = self.aws_client.s3_upload_iris_shares(op_uuid, shares).await {
                tracing::error!("S3 shares upload failed: {:?}", e);
                return Err(ServiceClientError::SharesUploadError);
            }
        }
        tracing::info!("upload finished");
        Ok(())
    }

    #[cfg(not(feature = "explicit-sns-batching"))]
    async fn publish_requests(&mut self, batch_requests: &[typeset::Request]) -> Vec<usize> {
        use crate::aws::types::SnsMessageInfo;

        // Collect all messages for batch publishing
        let messages: Vec<SnsMessageInfo> =
            batch_requests.iter().map(SnsMessageInfo::from).collect();

        let idxs = self.aws_client.sns_publish_json_batch(&messages).await;
        if idxs.len() != messages.len() {
            self.state.error_bits.set_sns_publish_error();
        }

        for idx in &idxs {
            let request = &batch_requests[*idx];
            tracing::info!("publishing {}", request.log_tag());
        }

        idxs
    }

    #[cfg(feature = "explicit-sns-batching")]
    async fn publish_requests(&mut self, batch_requests: &[typeset::Request]) -> Vec<usize> {
        use crate::aws::types::SnsMessageInfo;
        use iris_mpc_common::helpers::smpc_request::{CompactBatchRequest, CompressedBatchPayload};

        // Convert requests to RequestPayload items
        let items: Vec<smpc_request::RequestPayload> = batch_requests
            .iter()
            .map(|request| {
                let payload = typeset::RequestPayload::from(request);
                payload.to_smpc_request()
            })
            .collect();

        let compact_batch = CompactBatchRequest { items };

        // Compress the batch
        let compressed_data = match compact_batch.compress() {
            Ok(data) => data,
            Err(e) => {
                tracing::error!("Failed to compress batch: {}", e);
                self.state.error_bits.set_sns_publish_error();
                return Vec::new();
            }
        };

        let payload = CompressedBatchPayload {
            data: compressed_data,
        };

        // Publish the compressed batch as a single SNS message
        let batch_sns_info =
            SnsMessageInfo::new("enrollment", smpc_request::BATCH_MESSAGE_TYPE, &payload);

        let res = self.aws_client.sns_publish_json(batch_sns_info).await;
        let published_idxs: Vec<usize> = if res.is_ok() {
            (0..batch_requests.len()).collect()
        } else {
            self.state.error_bits.set_sns_publish_error();
            Vec::new()
        };

        for &idx in &published_idxs {
            let request = &batch_requests[idx];
            tracing::info!("publishing {}", request.log_tag());
        }

        published_idxs
    }
}

/// Bitmask for tracking various error conditions during batch processing.
struct ErrorBits(u32);

impl ErrorBits {
    const SNS_PUBLISH: u32 = 1 << 0;
    const SQS_RECEIVE: u32 = 1 << 1;
    const VALIDATION: u32 = 1 << 2;
    const REQUEST_DROPPED: u32 = 1 << 3;
    const RESPONSE_TIMEOUT: u32 = 1 << 4;

    fn new() -> Self {
        Self(0)
    }

    fn set_sns_publish_error(&mut self) {
        self.0 |= Self::SNS_PUBLISH;
    }

    fn set_sqs_receive_error(&mut self) {
        self.0 |= Self::SQS_RECEIVE;
    }

    fn set_validation_error(&mut self) {
        self.0 |= Self::VALIDATION;
    }

    fn set_request_dropped_error(&mut self) {
        self.0 |= Self::REQUEST_DROPPED;
    }

    fn set_response_timeout_error(&mut self) {
        self.0 |= Self::RESPONSE_TIMEOUT;
    }

    fn get_request_dropped_error(&self) -> bool {
        self.0 & Self::REQUEST_DROPPED != 0
    }

    fn has_errors(&self) -> bool {
        self.0 != 0
    }
}

impl std::fmt::Display for ErrorBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            return write!(f, "no errors");
        }

        let mut parts = Vec::new();
        if self.0 & Self::SNS_PUBLISH != 0 {
            parts.push("sns_publish");
        }
        if self.0 & Self::SQS_RECEIVE != 0 {
            parts.push("sqs_receive");
        }
        if self.0 & Self::VALIDATION != 0 {
            parts.push("validation");
        }
        if self.0 & Self::REQUEST_DROPPED != 0 {
            parts.push("request_dropped");
        }
        if self.0 & Self::RESPONSE_TIMEOUT != 0 {
            parts.push("response_timeout");
        }

        write!(f, "{}", parts.join(" | "))
    }
}

#[derive(Debug, Serialize)]
struct CanonicalResultsFile<'a> {
    schema_version: u32,
    records: &'a [CanonicalResultRecord],
}

#[derive(Debug, Serialize)]
struct CanonicalResultRecord {
    batch_index: usize,
    batch_item_index: usize,
    label: Option<String>,
    /// Responses are always stored in node-id order, regardless of SQS
    /// delivery order.
    responses: Vec<typeset::ResponsePayload>,
}

// Holds the cross-batch state needed while processing requests and responses.
struct ExecState {
    uniqueness_labels: HashMap<String, SerialId>,
    signup_id_to_labels: HashMap<Uuid, String>,
    outstanding_requests: HashMap<Uuid, typeset::RequestInfo>,
    outstanding_deletions: HashMap<SerialId, typeset::RequestInfo>,
    live_serial_ids: HashSet<SerialId>,
    captured_results: Vec<CanonicalResultRecord>,
    error_bits: ErrorBits,
}

/// Records a response result against its tracked request info and handles completion.
/// Returns the completed `RequestInfo` if all parties have responded.
/// Validation errors are logged but requests are still completed normally.
fn handle_completion<K: std::fmt::Display + std::hash::Hash + Eq>(
    key: &K,
    response: &typeset::ResponsePayload,
    map: &mut HashMap<K, typeset::RequestInfo>,
    error_bits: &mut ErrorBits,
) -> Option<typeset::RequestInfo> {
    let is_complete = map.get_mut(key).map(|info| info.record_response(response));
    match is_complete {
        Some(true) => {
            let info = map.remove(key);
            if let Some(ref info) = info {
                // Check for error responses from servers
                if info.has_error_response() {
                    error_bits.set_validation_error();
                    let details = info.get_error_msgs();
                    tracing::error!("request {} completed with errors: {}", info, details);
                }
                // Check if responses match expected values
                if let Err(err_msg) = info.validate_expected() {
                    error_bits.set_validation_error();
                    tracing::error!("request {} failed expected validation: {}", info, err_msg);
                }
            }
            info
        }
        Some(false) => None,
        None => {
            tracing::warn!(
                "Received response not tracked in outstanding requests: {}",
                key
            );
            None
        }
    }
}

impl ExecState {
    fn new() -> Self {
        Self {
            uniqueness_labels: HashMap::new(),
            signup_id_to_labels: HashMap::new(),
            outstanding_requests: HashMap::new(),
            outstanding_deletions: HashMap::new(),
            live_serial_ids: HashSet::new(),
            captured_results: Vec::new(),
            error_bits: ErrorBits::new(),
        }
    }

    fn capture_completed(&mut self, info: typeset::RequestInfo) {
        let responses = info
            .responses()
            .iter()
            .cloned()
            .collect::<Option<Vec<_>>>()
            .expect("capture_completed requires one response from every party");

        let normalized = responses
            .iter()
            .map(|response| {
                let mut value = serde_json::to_value(response)
                    .expect("serializing an SMPC response should not fail");
                if let Some(payload) = value.as_object_mut().and_then(|outer| {
                    outer
                        .values_mut()
                        .next()
                        .and_then(serde_json::Value::as_object_mut)
                }) {
                    payload.remove("node_id");
                }
                value
            })
            .collect::<Vec<_>>();
        if normalized.windows(2).any(|pair| pair[0] != pair[1]) {
            self.error_bits.set_validation_error();
            tracing::error!(
                "request {} produced disagreeing party responses: {:?}",
                info,
                normalized
            );
        }

        self.captured_results.push(CanonicalResultRecord {
            batch_index: info.batch_idx(),
            batch_item_index: info.batch_item_idx(),
            label: info.label().clone(),
            responses,
        });
    }

    fn write_results(&mut self, path: &Path) -> Result<(), ServiceClientError> {
        self.captured_results
            .sort_by_key(|record| (record.batch_index, record.batch_item_index));
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                ServiceClientError::ResponseError(format!(
                    "failed to create result directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let contents = serde_json::to_vec_pretty(&CanonicalResultsFile {
            schema_version: 1,
            records: &self.captured_results,
        })
        .map_err(|error| {
            ServiceClientError::ResponseError(format!(
                "failed to serialize canonical results: {error}"
            ))
        })?;
        fs::write(path, contents).map_err(|error| {
            ServiceClientError::ResponseError(format!(
                "failed to write canonical results to {}: {error}",
                path.display()
            ))
        })?;
        tracing::info!(
            "Wrote {} canonical results to {}",
            self.captured_results.len(),
            path.display()
        );
        Ok(())
    }

    /// Correlates a single response to its outstanding request/deletion, updating state.
    fn correlate_response(&mut self, response: &typeset::ResponsePayload) {
        // Extract correlation UUID from response (IdentityDeletion has none).
        let corr_uuid: Option<Uuid> = match response {
            typeset::ResponsePayload::Uniqueness(r) => r.signup_id.parse().ok(),
            typeset::ResponsePayload::Reauthorization(r) => r.reauth_id.parse().ok(),
            typeset::ResponsePayload::RecoveryCheck(r) => r.request_id.parse().ok(),
            typeset::ResponsePayload::ResetCheck(r) => r.request_id.parse().ok(),
            typeset::ResponsePayload::ResetUpdate(r) => r.request_id.parse().ok(),
            typeset::ResponsePayload::RecoveryUpdate(r) => r.request_id.parse().ok(),
            typeset::ResponsePayload::IdentityDeletion(r) => {
                if let Some(info) = handle_completion(
                    &r.serial_id,
                    response,
                    &mut self.outstanding_deletions,
                    &mut self.error_bits,
                ) {
                    self.live_serial_ids.remove(&r.serial_id);
                    self.capture_completed(info);
                }
                return;
            }
        };

        if let Some(uuid) = corr_uuid {
            if let Some(info) = handle_completion(
                &uuid,
                response,
                &mut self.outstanding_requests,
                &mut self.error_bits,
            ) {
                if !info.has_error_response() {
                    // For uniqueness: search all node responses for a serial_id
                    // and record it against the request's label.
                    let maybe_serial_id = info.responses().iter().find_map(|opt| {
                        if let Some(typeset::ResponsePayload::Uniqueness(result)) = opt {
                            result.get_serial_id()
                        } else {
                            None
                        }
                    });
                    if let Some(serial_id) = maybe_serial_id {
                        if let Some(label) = self.signup_id_to_labels.remove(&uuid) {
                            self.uniqueness_labels.insert(label, serial_id);
                        }
                        // track these to clean them up later
                        self.live_serial_ids.insert(serial_id);
                    }
                }
                self.capture_completed(info);
            }
        }
    }

    // Phase 5: Register published requests so responses can be correlated. IdentityDeletion
    // correlates by serial_id rather than UUID, so it goes into outstanding_deletions.
    fn track_batch_requests(&mut self, idxs: &[usize], batch_requests: &[typeset::Request]) {
        for idx in idxs {
            let request = &batch_requests[*idx];
            let opt_tracking_uuid: Option<Uuid> = match request {
                typeset::Request::Uniqueness { signup_id, .. } => {
                    if let Some(label) = request.info().label() {
                        self.signup_id_to_labels.insert(*signup_id, label.clone());
                    }
                    Some(*signup_id)
                }
                typeset::Request::Reauthorization { reauth_id, .. } => Some(*reauth_id),
                typeset::Request::RecoveryCheck { request_id, .. } => Some(*request_id),
                typeset::Request::ResetCheck { reset_id, .. }
                | typeset::Request::ResetUpdate { reset_id, .. } => Some(*reset_id),
                typeset::Request::RecoveryUpdate { recovery_id, .. } => Some(*recovery_id),
                typeset::Request::IdentityDeletion { parent, .. } => {
                    self.outstanding_deletions
                        .insert(*parent, request.info().clone());
                    None
                }
            };
            if let Some(tracking_uuid) = opt_tracking_uuid {
                self.outstanding_requests
                    .insert(tracking_uuid, request.info().clone());
            }
        }
    }

    // Drains outstanding_requests and outstanding_deletions by consuming from the SQS response stream.
    // Times out if no response is received within RESPONSE_TIMEOUT_SECS.
    async fn process_responses(&mut self, aws_client: &AwsClient) {
        let max_poll_time = aws_client.config().sqs_long_poll_wait_time() as i32;
        let mut response_stream = aws_client.sqs_response_stream(max_poll_time).fuse();

        let mut last_response_time = Instant::now();
        let timeout_duration = Duration::from_secs(RESPONSE_TIMEOUT_SECS);

        while !self.outstanding_requests.is_empty() || !self.outstanding_deletions.is_empty() {
            let remaining_time = timeout_duration
                .checked_sub(last_response_time.elapsed())
                .unwrap_or(Duration::ZERO);

            if remaining_time.is_zero() {
                self.error_bits.set_response_timeout_error();
                tracing::error!(
                    "Response timeout: no response received for {} seconds",
                    RESPONSE_TIMEOUT_SECS
                );
                break;
            }

            // Wait for next message with timeout
            match timeout(remaining_time, response_stream.next()).await {
                Ok(Some(Ok(sqs_msg))) => {
                    last_response_time = Instant::now();

                    let response = match typeset::ResponsePayload::try_from(&sqs_msg) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!(
                                "failed to parse response payload for sqs msg {}: {}",
                                sqs_msg.kind(),
                                e
                            );
                            continue;
                        }
                    };
                    tracing::info!("received {}", response.log_tag());
                    self.correlate_response(&response);
                }
                Ok(Some(Err(e))) => {
                    self.error_bits.set_sqs_receive_error();
                    tracing::error!("SQS stream error: {:?}", e);
                    break;
                }
                Ok(None) => {
                    // Stream ended unexpectedly
                    self.error_bits.set_sqs_receive_error();
                    tracing::error!("SQS response stream ended unexpectedly");
                    break;
                }
                Err(_) => {
                    // Timeout elapsed
                    self.error_bits.set_response_timeout_error();
                    tracing::error!(
                        "Response timeout: no response received for {} seconds",
                        RESPONSE_TIMEOUT_SECS
                    );
                    break;
                }
            }
        }

        for (_, info) in self.outstanding_requests.iter() {
            tracing::warn!("Request still pending: {}", info);
        }
        for (_, info) in self.outstanding_deletions.iter() {
            tracing::warn!("Deletion still pending: {}", info);
        }
    }
}
