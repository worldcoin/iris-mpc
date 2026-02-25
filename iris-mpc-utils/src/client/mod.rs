use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use async_from::{self, AsyncFrom};
use iris_mpc_cpu::execution::hawk_main::BothEyes;
use rand::rngs::StdRng;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};
use tokio::time::{sleep, Instant};
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

/// Maximum time (seconds) to wait for all responses before declaring a batch timed out.
const RESPONSE_TIMEOUT_SECS: u64 = 360;

/// Interval (seconds) between progress log messages while waiting for responses.
const PROGRESS_LOG_INTERVAL_SECS: u64 = 60;

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
    state: ExecState,
    server_config_urls: Option<Vec<String>>,
    http_client: reqwest::Client,
}

impl ServiceClient {
    pub async fn new(
        opts_aws: AwsOptions,
        request_batch: RequestBatchOptions,
        shares_generator: SharesGeneratorOptions,
        server_config_urls: Option<Vec<String>>,
    ) -> Result<Self, ServiceClientError> {
        let aws_client = AwsClient::async_from(opts_aws).await;
        let shares_generator = SharesGenerator::<StdRng>::from_options(shares_generator);

        Ok(Self {
            aws_client,
            request_batch: Some(request_batch),
            shares_generator,
            state: ExecState::new(),
            server_config_urls,
            http_client: reqwest::Client::new(),
        })
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
        tokio::select! {
            _ = self.exec() => {
                tracing::info!("service client finished");
            },
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("\nCtrl+C received. Initiating cleanup...");
            }
        };

        tracing::info!(
            "Cleaning up {} serial IDs",
            self.state.live_serial_ids.len()
        );

        // Send deletion requests for all live serial IDs.
        let live_serial_ids = std::mem::take(&mut self.state.live_serial_ids);
        for serial_id in live_serial_ids.into_iter() {
            let payload = RequestPayload::IdentityDeletion(smpc_request::IdentityDeletionRequest {
                serial_id,
            });
            let sns_msg_info = SnsMessageInfo::from(payload);
            if let Err(e) = self
                .aws_client
                .sns_publish_json(sns_msg_info)
                .await
                .map_err(ServiceClientError::AwsServiceError)
            {
                tracing::error!("Failed to send deletion for serial_id {}: {}", serial_id, e);
            } else {
                tracing::info!("publishing Deletion for {}", serial_id);
            }
        }
        // iris-mpc-hawk will not send responses for a batch of just deletions
        tracing::info!("Cleanup complete. Deletions have been submitted.");

        Ok(())
    }

    async fn exec(&mut self) {
        let request_batch = self
            .request_batch
            .take()
            .expect("exec() called more than once");

        for (batch_idx, batch) in request_batch.into_iter().enumerate() {
            let batch_size = batch.len();

            if self.server_config_urls.is_some() {
                if let Err(e) = self.set_fixed_batch_size(batch_size).await {
                    tracing::error!("Failed to set fixed batch size: {}", e);
                    break;
                }
            }

            if let Err(e) = self.handle_batch(batch_idx, batch).await {
                tracing::error!("{}", e);
                break;
            }
        }

        // Clear fixed_batch_size on all parties so it doesn't affect subsequent runs.
        if self.server_config_urls.is_some() {
            self.clear_fixed_batch_size().await;
        }
    }

    /// Sets the fixed_batch_size on all server parties via their /config endpoint.
    /// Posts to all parties concurrently to minimize the window where parties disagree.
    async fn set_fixed_batch_size(&self, batch_size: usize) -> Result<(), ServiceClientError> {
        let urls = self
            .server_config_urls
            .as_ref()
            .expect("set_fixed_batch_size called without server_config_urls");

        let body = serde_json::json!({ "fixed_batch_size": batch_size });

        let futs = urls.iter().enumerate().map(|(party_idx, base_url)| {
            let url = format!("{}/config", base_url);
            let body = body.clone();
            let client = &self.http_client;
            async move {
                let resp = client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| {
                        ServiceClientError::RuntimeConfigError(format!(
                            "party {}: POST {} failed: {}",
                            party_idx, url, e
                        ))
                    })?;

                if !resp.status().is_success() {
                    return Err(ServiceClientError::RuntimeConfigError(format!(
                        "party {}: POST {} returned status {}",
                        party_idx,
                        url,
                        resp.status()
                    )));
                }

                let config_resp: serde_json::Value =
                    resp.json().await.map_err(|e| {
                        ServiceClientError::RuntimeConfigError(format!(
                            "party {}: failed to parse /config response: {}",
                            party_idx, e
                        ))
                    })?;

                let confirmed_size = config_resp
                    .get("fixed_batch_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);

                if confirmed_size != Some(batch_size) {
                    return Err(ServiceClientError::RuntimeConfigError(format!(
                        "party {}: batch size not confirmed (expected {}, got {:?})",
                        party_idx, batch_size, confirmed_size
                    )));
                }

                tracing::info!(
                    "Party {} confirmed fixed_batch_size = {}",
                    party_idx,
                    batch_size
                );
                Ok(())
            }
        });

        futures::future::try_join_all(futs).await?;
        Ok(())
    }

    /// Clears fixed_batch_size on all parties. Best-effort — logs errors but doesn't fail.
    async fn clear_fixed_batch_size(&self) {
        let urls = self
            .server_config_urls
            .as_ref()
            .expect("clear_fixed_batch_size called without server_config_urls");

        let body = serde_json::json!({ "fixed_batch_size": null });

        let futs = urls.iter().enumerate().map(|(party_idx, base_url)| {
            let url = format!("{}/config", base_url);
            let body = body.clone();
            let client = &self.http_client;
            async move {
                match client.post(&url).json(&body).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        tracing::info!("Party {}: cleared fixed_batch_size", party_idx);
                    }
                    Ok(resp) => {
                        tracing::error!(
                            "Party {}: clearing fixed_batch_size returned status {}",
                            party_idx,
                            resp.status()
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "Party {}: failed to clear fixed_batch_size: {}",
                            party_idx,
                            e
                        );
                    }
                }
            }
        });

        futures::future::join_all(futs).await;
    }

    async fn handle_batch(
        &mut self,
        batch_idx: usize,
        batch: Vec<options::RequestOptions>,
    ) -> Result<(), ServiceClientError> {
        // Phase 1: Prepare requests and generate shares.
        let (batch_requests, batch_shares) = self.prepare_batch_requests(batch_idx, &batch)?;

        // Phase 2: Upload shares to S3.
        self.upload_shares(batch_shares).await;

        // Phase 3: Wait for S3 propagation.
        sleep(Duration::from_secs(S3_PROPAGATION_DELAY_SECS)).await;

        // Phase 4: Publish requests to SNS.
        let published_count = self.publish_requests(batch_idx, &batch_requests).await;

        // Phase 5: Track published requests and wait for responses.
        self.state
            .track_batch_requests(&batch_requests[..published_count]);
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
            let parent_serial_id: Option<IrisSerialId> = match opts.get_parent() {
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
            let request = match opts.make_request(info, parent_serial_id) {
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
    ) {
        for shares_info in batch_shares.iter().filter_map(|opt| opt.as_ref()) {
            let (op_uuid, shares) = shares_info;
            if let Err(e) = self.aws_client.s3_upload_iris_shares(op_uuid, shares).await {
                self.state.error_bits.set_upload_shares_error();
                tracing::error!("S3 shares upload failed: {:?}", e);
                // Stop uploading further shares
                break;
            }
        }
    }

    async fn publish_requests(
        &mut self,
        batch_idx: usize,
        batch_requests: &[typeset::Request],
    ) -> usize {
        use crate::aws::types::SnsMessageInfo;

        let mut published_count = 0;
        for request in batch_requests.iter() {
            let log_tag = request.log_tag();
            let sns_msg_info = SnsMessageInfo::from(request);

            match self.aws_client.sns_publish_json(sns_msg_info).await {
                Ok(_) => {
                    tracing::info!("publishing {}", log_tag);
                    published_count += 1;
                }
                Err(e) => {
                    self.state.error_bits.set_sns_publish_error();
                    tracing::error!(
                        "batch {}: SNS publish failed for request {}: {:?}",
                        batch_idx,
                        log_tag,
                        e,
                    );
                    // Stop publishing further requests, but return count of already published
                    break;
                }
            }
        }
        published_count
    }
}

/// Bitmask for tracking various error conditions during batch processing.
struct ErrorBits(u32);

impl ErrorBits {
    const SNS_PUBLISH: u32 = 1 << 0;
    const SQS_RECEIVE: u32 = 1 << 1;
    const VALIDATION: u32 = 1 << 2;
    const REQUEST_DROPPED: u32 = 1 << 3;
    const UPLOAD_SHARES: u32 = 1 << 4;

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

    fn set_upload_shares_error(&mut self) {
        self.0 |= Self::UPLOAD_SHARES;
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
            parts.push("sns_publish_error");
        }
        if self.0 & Self::SQS_RECEIVE != 0 {
            parts.push("sqs_receive_error");
        }
        if self.0 & Self::VALIDATION != 0 {
            parts.push("validation_error");
        }
        if self.0 & Self::REQUEST_DROPPED != 0 {
            parts.push("request_dropped_error");
        }
        if self.0 & Self::UPLOAD_SHARES != 0 {
            parts.push("upload_shares_error");
        }

        write!(f, "{}", parts.join(" | "))
    }
}

// Holds the cross-batch state needed while processing requests and responses.
struct ExecState {
    uniqueness_labels: HashMap<String, IrisSerialId>,
    signup_id_to_labels: HashMap<Uuid, String>,
    outstanding_requests: HashMap<Uuid, typeset::RequestInfo>,
    outstanding_deletions: HashMap<IrisSerialId, typeset::RequestInfo>,
    live_serial_ids: HashSet<IrisSerialId>,
    error_bits: ErrorBits,
}

/// Records a response result against its tracked request info and handles completion.
/// Returns the completed `RequestInfo` if all parties have responded successfully.
/// Validation errors are logged and removed, returning `None`.
fn handle_completion<K: std::fmt::Display + std::hash::Hash + Eq>(
    key: &K,
    response: &typeset::ResponsePayload,
    map: &mut HashMap<K, typeset::RequestInfo>,
    error_bits: &mut ErrorBits,
) -> Option<typeset::RequestInfo> {
    let opt = map.get_mut(key).map(|info| info.record_response(response));
    let is_complete = match opt {
        None => {
            tracing::warn!(
                "Received response not tracked in outstanding requests: {}",
                key
            );
            None
        }
        Some(x) => x,
    };

    match is_complete {
        Some(true) => {
            let info = map.remove(key);
            if let Some(ref info) = info {
                if info.has_error_response() {
                    error_bits.set_validation_error();
                    let details = info.get_error_msgs();
                    tracing::error!("request {} completed with errors: {}", info, details);
                }
            }
            info
        }
        Some(false) => None,
        None => {
            error_bits.set_validation_error();
            // Remove the request from outstanding map since it failed validation
            if let Some(info) = map.remove(key) {
                tracing::error!("request {} failed validation", info);
            }
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
            error_bits: ErrorBits::new(),
        }
    }

    /// Correlates a single response to its outstanding request/deletion, updating state.
    fn correlate_response(&mut self, response: &typeset::ResponsePayload) {
        // Extract correlation UUID from response (IdentityDeletion has none).
        let corr_uuid: Option<Uuid> = match response {
            typeset::ResponsePayload::Uniqueness(r) => r.signup_id.parse().ok(),
            typeset::ResponsePayload::Reauthorization(r) => r.reauth_id.parse().ok(),
            typeset::ResponsePayload::ResetCheck(r) => r.reset_id.parse().ok(),
            typeset::ResponsePayload::ResetUpdate(r) => r.reset_id.parse().ok(),
            typeset::ResponsePayload::IdentityDeletion(r) => {
                if handle_completion(
                    &r.serial_id,
                    response,
                    &mut self.outstanding_deletions,
                    &mut self.error_bits,
                )
                .is_some()
                {
                    self.live_serial_ids.remove(&r.serial_id);
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
            }
        }
    }

    // Phase 5: Register published requests so responses can be correlated. IdentityDeletion
    // correlates by serial_id rather than UUID, so it goes into outstanding_deletions.
    fn track_batch_requests(&mut self, batch_requests: &[typeset::Request]) {
        for request in batch_requests {
            let opt_tracking_uuid: Option<Uuid> = match request {
                typeset::Request::Uniqueness { signup_id, .. } => {
                    if let Some(label) = request.info().label() {
                        self.signup_id_to_labels.insert(*signup_id, label.clone());
                    }
                    Some(*signup_id)
                }
                typeset::Request::Reauthorization { reauth_id, .. } => Some(*reauth_id),
                typeset::Request::ResetCheck { reset_id, .. }
                | typeset::Request::ResetUpdate { reset_id, .. } => Some(*reset_id),
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

    // Drains outstanding_requests and outstanding_deletions by polling SQS until both are empty.
    // Returns an error if the batch times out with outstanding requests remaining.
    async fn process_responses(&mut self, aws_client: &AwsClient) {
        use crate::constants::N_PARTIES;

        let start = Instant::now();
        let deadline = start + Duration::from_secs(RESPONSE_TIMEOUT_SECS);
        let mut next_progress = start + Duration::from_secs(PROGRESS_LOG_INTERVAL_SECS);

        'OUTER: while !self.outstanding_requests.is_empty()
            || !self.outstanding_deletions.is_empty()
        {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            if Instant::now() >= next_progress {
                let elapsed = start.elapsed().as_secs();
                tracing::info!(
                    "Waiting for responses... {}s elapsed, {} requests and {} deletions still pending",
                    elapsed,
                    self.outstanding_requests.len(),
                    self.outstanding_deletions.len()
                );
                next_progress += Duration::from_secs(PROGRESS_LOG_INTERVAL_SECS);
            }

            // SQS long polling: cap at 20s (SQS maximum) and the remaining deadline.
            let long_poll_secs = remaining.as_secs().min(1) as i32;
            let messages = match aws_client
                .sqs_receive_messages(Some(N_PARTIES), Some(long_poll_secs))
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    self.error_bits.set_sqs_receive_error();
                    tracing::error!("SQS receive failed: {:?}", e);
                    break 'OUTER;
                }
            };

            for sqs_msg in messages {
                if let Err(e) = aws_client.sqs_purge_response_queue_message(&sqs_msg).await {
                    self.error_bits.set_sqs_receive_error();
                    tracing::error!("SQS message purge failed: {:?}", e);
                    break 'OUTER;
                }

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
        }

        for (_, info) in self.outstanding_requests.iter() {
            tracing::warn!("Request still pending: {}", info);
        }
        for (_, info) in self.outstanding_deletions.iter() {
            tracing::warn!("Deletion still pending: {}", info);
        }
    }
}
