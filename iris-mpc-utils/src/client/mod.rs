use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use async_from::{self, AsyncFrom};
use rand::rngs::StdRng;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};
use tokio::time::{sleep, timeout, Instant};
use uuid::Uuid;

use crate::{
    aws::types::SnsMessageInfo,
    client::options::{RequestBatchOptions, SharesGeneratorOptions},
};

use super::aws::AwsClient;
use components::SharesGenerator;

pub use options::{AwsOptions, ServiceClientOptions};
pub use typeset::*;

mod components;
mod options;
mod typeset;

pub struct ServiceClient2 {
    aws_client: AwsClient,
    request_batch: RequestBatchOptions,
    shares_generator: SharesGenerator<StdRng>,
}

impl ServiceClient2 {
    pub async fn new(
        opts_aws: AwsOptions,
        request_batch: RequestBatchOptions,
        shares_generator: SharesGeneratorOptions,
    ) -> Result<Self, ServiceClientError> {
        let aws_client = AwsClient::async_from(opts_aws).await;
        let shares_generator = SharesGenerator::<StdRng>::from_options(shares_generator);

        // todo: better validation code
        if matches!(request_batch, RequestBatchOptions::Complex { .. }) {
            for result in [
                request_batch.validate_iris_pairs(),
                request_batch.find_duplicate_label().map_or(Ok(()), |dup| {
                    Err(format!("contains duplicate label '{}'", dup))
                }),
                request_batch.validate_parents(),
                request_batch.validate_batch_ordering(),
            ] {
                if let Err(msg) = result {
                    return Err(ServiceClientError::InvalidOptions(format!(
                        "RequestBatchOptions::Complex {}",
                        msg
                    )));
                }
            }
        }

        Ok(Self {
            aws_client,
            request_batch,
            shares_generator,
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

    pub async fn run(mut sc: Self) {
        sc.init().await.expect("service client init failed");

        let aws_client = sc.aws_client.clone();
        let mut live_serial_ids: HashSet<IrisSerialId> = HashSet::new();

        // Run main batch loop, interruptible by Ctrl+C.
        tokio::select! {
            _ = sc.exec(&mut live_serial_ids) => {
                tracing::info!("service client finished");
            },
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("\nCtrl+C received. Initiating cleanup...");
            }
        };

        tracing::info!("Cleaning up {} serial IDs", live_serial_ids.len());

        // Send deletion requests for all live serial IDs.
        for serial_id in live_serial_ids.into_iter() {
            let payload = RequestPayload::IdentityDeletion(smpc_request::IdentityDeletionRequest {
                serial_id,
            });
            let sns_msg_info = SnsMessageInfo::from(payload);
            if let Err(e) = aws_client
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
    }

    async fn exec(mut self, live_serial_ids: &mut HashSet<IrisSerialId>) {
        use crate::aws::types::SnsMessageInfo;
        use crate::client::options::Parent;

        let mut state = ExecState::new();

        for (batch_idx, batch) in self.request_batch.into_iter().enumerate() {
            // Phase 1: Gather all requests and pre-generate shares.
            let mut batch_requests: Vec<typeset::Request> = Vec::new();
            let mut batch_shares = Vec::new();

            for (item_idx, opts) in batch.iter().enumerate() {
                // Look up the iris serial id for uniquess_labels for anything in the batch that
                // needs it; if not there, drop the item and emit a warning.
                let parent_serial_id: Option<IrisSerialId> = match opts.get_parent() {
                    Some(Parent::Id(id)) => Some(id),
                    Some(Parent::Label(label)) => {
                        if let Some(&serial_id) = state.uniqueness_labels.get(label.as_str()) {
                            Some(serial_id)
                        } else {
                            tracing::warn!(
                                "batch {}.{}: dropping request — parent label '{}' unresolved",
                                batch_idx,
                                item_idx,
                                label,
                            );
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
                let request = opts.make_request(info, parent_serial_id);

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

            // Phase 2: Upload all shares in parallel.
            futures::future::join_all(
                batch_shares
                    .iter()
                    .filter_map(|opt| opt.as_ref())
                    .map(|(op_uuid, shares)| {
                        self.aws_client.s3_upload_iris_shares(op_uuid, shares)
                    }),
            )
            .await
            .into_iter()
            .for_each(|r| {
                r.expect("S3 shares upload failed");
            });
            drop(batch_shares);

            // Phase 3: Wait before publishing to allow shares to propagate.
            sleep(Duration::from_secs(1)).await;

            // Phase 4: Publish all requests
            for request in batch_requests.iter() {
                let log_tag = request.log_tag();
                let sns_msg_info = SnsMessageInfo::from(request);
                if let Err(e) = self.aws_client.sns_publish_json(sns_msg_info).await {
                    tracing::error!("SNS publish failed: for request {}: {:?}", log_tag, e);
                    break;
                } else {
                    tracing::info!("publishing {}", log_tag);
                }
            }

            // Phase 5: Track published requests so we can match incoming responses.
            state.track_batch_requests(&batch_requests);

            // Wait for all responses for this batch.
            state
                .process_responses(&self.aws_client, live_serial_ids)
                .await;

            tracing::info!(
                "Batch {} finished. Responses to non-deletion requests have been received",
                batch_idx
            );
        }

        state.report_errors();
    }
}

// Holds the cross-batch state needed while processing requests and responses.
struct ExecState {
    uniqueness_labels: HashMap<String, IrisSerialId>,
    signup_id_to_labels: HashMap<Uuid, String>,
    outstanding_requests: HashMap<Uuid, typeset::RequestInfo>,
    outstanding_deletions: HashMap<IrisSerialId, typeset::RequestInfo>,
    error_log: Vec<typeset::RequestInfo>,
    had_validation_errors: bool,
}

impl ExecState {
    fn new() -> Self {
        Self {
            uniqueness_labels: HashMap::new(),
            signup_id_to_labels: HashMap::new(),
            outstanding_requests: HashMap::new(),
            outstanding_deletions: HashMap::new(),
            error_log: Vec::new(),
            had_validation_errors: false,
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
    async fn process_responses(
        &mut self,
        aws_client: &AwsClient,
        live_serial_ids: &mut HashSet<IrisSerialId>,
    ) {
        use crate::constants::N_PARTIES;

        let timeout_secs: u64 = 360;
        let start = Instant::now();
        let deadline = start + Duration::from_secs(timeout_secs);
        let mut next_progress = start + Duration::from_secs(60);

        while !self.outstanding_requests.is_empty() || !self.outstanding_deletions.is_empty() {
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
                next_progress += Duration::from_secs(60);
            }

            let messages =
                match timeout(remaining, aws_client.sqs_receive_messages(Some(N_PARTIES))).await {
                    Ok(result) => result.expect("SQS receive failed"),
                    Err(_) => {
                        break;
                    }
                };

            for sqs_msg in messages {
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

                // Extract correlation UUID from response (IdentityDeletion has none).
                let corr_uuid: Option<Uuid> = match &response {
                    typeset::ResponsePayload::Uniqueness(r) => r.signup_id.parse().ok(),
                    typeset::ResponsePayload::Reauthorization(r) => r.reauth_id.parse().ok(),
                    typeset::ResponsePayload::ResetCheck(r) => r.reset_id.parse().ok(),
                    typeset::ResponsePayload::ResetUpdate(r) => r.reset_id.parse().ok(),
                    typeset::ResponsePayload::IdentityDeletion(r) => {
                        let is_complete = self
                            .outstanding_deletions
                            .get_mut(&r.serial_id)
                            .map(|info| info.record_response(&response));
                        match is_complete {
                            None => {
                                tracing::warn!(
                                    "Received IdentityDeletion response: not tracked in outstanding_requests"
                                );
                            }
                            Some(Ok(true)) => {
                                if let Some(info) = self.outstanding_deletions.remove(&r.serial_id)
                                {
                                    if info.has_error_response() {
                                        let details = info.get_error_msgs();
                                        tracing::warn!(
                                            "Deletion request {} completed with errors: {}",
                                            info,
                                            details
                                        );
                                        self.error_log.push(info);
                                    } else {
                                        live_serial_ids.remove(&r.serial_id);
                                    }
                                }
                            }
                            Some(Ok(false)) => {}
                            Some(Err(_)) => {
                                self.had_validation_errors = true;
                            }
                        }
                        None
                    }
                };

                if let Some(uuid) = corr_uuid {
                    let is_complete = self
                        .outstanding_requests
                        .get_mut(&uuid)
                        .map(|info| info.record_response(&response));

                    match is_complete {
                        None => {
                            tracing::warn!(
                                "Orphan response: no matching request for UUID {}",
                                uuid
                            );
                        }
                        Some(Ok(true)) => {
                            if let Some(info) = self.outstanding_requests.remove(&uuid) {
                                if info.has_error_response() {
                                    let details = info.get_error_msgs();
                                    tracing::warn!(
                                        "request {} completed with errors: {}",
                                        info,
                                        details
                                    );
                                    self.signup_id_to_labels.remove(&uuid);
                                    self.error_log.push(info);
                                } else {
                                    // For uniqueness: search all node responses for a serial_id
                                    // and record it against the request's label.
                                    let maybe_serial_id = info.responses().iter().find_map(|opt| {
                                        if let Some(typeset::ResponsePayload::Uniqueness(result)) =
                                            opt
                                        {
                                            result.get_serial_id()
                                        } else {
                                            None
                                        }
                                    });
                                    if let Some(serial_id) = maybe_serial_id {
                                        if let Some(label) = self.signup_id_to_labels.remove(&uuid)
                                        {
                                            self.uniqueness_labels.insert(label, serial_id);
                                        }
                                        // track these to clean them up later
                                        live_serial_ids.insert(serial_id);
                                    }
                                }
                            }
                        }
                        Some(Ok(false)) => {}
                        Some(Err(_)) => {
                            self.had_validation_errors = true;
                        }
                    }
                }

                aws_client
                    .sqs_purge_response_queue_message(&sqs_msg)
                    .await
                    .expect("SQS message purge failed");
            }
        }

        if !self.outstanding_requests.is_empty() || !self.outstanding_deletions.is_empty() {
            tracing::warn!(
                "Batch timed out after {}s: {} requests, {} deletions still pending",
                timeout_secs,
                self.outstanding_requests.len(),
                self.outstanding_deletions.len()
            );
            self.error_log
                .extend(self.outstanding_requests.drain().map(|(_, info)| info));
            self.error_log
                .extend(self.outstanding_deletions.drain().map(|(_, info)| info));
        }
    }

    fn report_errors(&self) {
        if !self.error_log.is_empty() {
            tracing::warn!(
                "=== {} request(s) completed with errors ===",
                self.error_log.len()
            );
            for info in &self.error_log {
                let details = info.get_error_msgs();
                tracing::warn!("  {}: {}", info, details);
            }
        }
        if self.had_validation_errors {
            tracing::warn!("=== Validation errors occurred - check logs for details ===");
        }
    }
}
