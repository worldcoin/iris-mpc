use std::collections::{HashMap, HashSet};

use async_from::{self, AsyncFrom};
use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

use iris_mpc_common::IrisSerialId;
use uuid::Uuid;

use crate::client::options::{RequestBatchOptions, SharesGeneratorOptions};

use super::aws::AwsClient;
use components::{
    RequestEnqueuer, RequestGenerator, ResponseDequeuer, SharesGenerator, SharesUploader,
};
use typeset::{Initialize, ProcessRequestBatch};

pub use options::{AwsOptions, ServiceClientOptions};
pub use typeset::ServiceClientError;

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
        let shares_generator = SharesGenerator::<StdRng>::from_options2(shares_generator);

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

    pub async fn exec(mut self) {
        use crate::aws::types::SnsMessageInfo;
        use crate::client::options::{Parent, RequestPayloadOptions};
        use crate::constants::N_PARTIES;

        self.init().await.expect("init failed");

        // need to track all the uniqueness requests sent, and their serial ids. also need to track which requests are dependent on a parent.
        let mut uniquess_labels: HashMap<String, IrisSerialId> = HashMap::new();
        let mut uuid_to_labels: HashMap<Uuid, String> = HashMap::new();
        let mut outstanding_requests: HashMap<Uuid, typeset::RequestInfo> = HashMap::new();

        for (batch_idx, batch) in self.request_batch.into_iter().enumerate() {
            // Phase 1: Gather all requests and pre-generate shares.
            let mut batch_requests: Vec<typeset::Request> = Vec::new();
            let mut batch_shares = Vec::new();

            for (item_idx, opts) in batch.iter().enumerate() {
                // Look up the iris serial id for uniquess_labels for anything in the batch that
                // needs it; if not there, drop the item and emit a warning.
                let parent_serial_id: Option<IrisSerialId> = match opts.payload() {
                    RequestPayloadOptions::IdentityDeletion { parent }
                    | RequestPayloadOptions::Reauthorisation { parent, .. }
                    | RequestPayloadOptions::ResetUpdate { parent, .. } => match parent {
                        Parent::Id(id) => Some(*id),
                        Parent::Label(label) => {
                            if let Some(&serial_id) = uniquess_labels.get(label.as_str()) {
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
                    },
                    _ => None,
                };

                // Generate a fresh correlation UUID and create RequestInfo.
                let corr_uuid = Uuid::new_v4();
                let info = typeset::RequestInfo::with_indices(batch_idx, item_idx, opts.label());

                // Build a Request from the options and resolved parent data.
                let request = match opts.payload() {
                    RequestPayloadOptions::Uniqueness { iris_pair, .. } => {
                        typeset::Request::Uniqueness {
                            info,
                            iris_pair: Some(*iris_pair),
                            signup_id: corr_uuid,
                        }
                    }
                    RequestPayloadOptions::Reauthorisation { iris_pair, .. } => {
                        typeset::Request::Reauthorization {
                            info,
                            iris_pair: *iris_pair,
                            parent: parent_serial_id.unwrap(),
                            reauth_id: corr_uuid,
                        }
                    }
                    RequestPayloadOptions::ResetCheck { iris_pair } => {
                        typeset::Request::ResetCheck {
                            info,
                            iris_pair: *iris_pair,
                            reset_id: corr_uuid,
                        }
                    }
                    RequestPayloadOptions::ResetUpdate { iris_pair, .. } => {
                        typeset::Request::ResetUpdate {
                            info,
                            iris_pair: *iris_pair,
                            parent: parent_serial_id.unwrap(),
                            reset_id: corr_uuid,
                        }
                    }
                    RequestPayloadOptions::IdentityDeletion { .. } => {
                        typeset::Request::IdentityDeletion {
                            info,
                            parent: parent_serial_id.unwrap(),
                        }
                    }
                };

                // Pre-generate shares for request types that require them.
                let shares_info = if let Some((op_uuid, iris_pair)) = request.get_shares_info() {
                    let shares = self.shares_generator.generate(iris_pair.as_ref());
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

            // Phase 3: Wait before publishing to allow shares to propagate.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;

            // Phase 4: Publish all requests in parallel.
            futures::future::join_all(batch_requests.iter().map(|request| {
                self.aws_client
                    .sns_publish_json(SnsMessageInfo::from(request))
            }))
            .await
            .into_iter()
            .for_each(|r| r.expect("SNS publish failed"));

            // Phase 5: Track in outstanding_requests keyed by correlation UUID. IdentityDeletion
            // correlates by serial_id rather than UUID, so it is not tracked here.
            for request in &batch_requests {
                let opt_tracking_uuid: Option<Uuid> = match request {
                    typeset::Request::Uniqueness { signup_id, .. } => {
                        if let Some(label) = request.info().label() {
                            uuid_to_labels.insert(*signup_id, label.clone());
                        }
                        Some(*signup_id)
                    }
                    typeset::Request::Reauthorization { reauth_id, .. } => Some(*reauth_id),
                    typeset::Request::ResetCheck { reset_id, .. }
                    | typeset::Request::ResetUpdate { reset_id, .. } => Some(*reset_id),
                    typeset::Request::IdentityDeletion { .. } => None,
                };
                if let Some(tracking_uuid) = opt_tracking_uuid {
                    outstanding_requests.insert(tracking_uuid, request.info().clone());
                }
            }

            // After doing this for everything in the batch, wait for responses, updating
            // outstanding_requests. For uniqueness results, get the label from uuid_to_labels
            // and then add the iris serial id to uniquess_labels.
            while !outstanding_requests.is_empty() {
                for sqs_msg in self
                    .aws_client
                    .sqs_receive_messages(Some(N_PARTIES))
                    .await
                    .expect("SQS receive failed")
                {
                    let response = typeset::ResponsePayload::from(&sqs_msg);

                    // Extract correlation UUID from response (IdentityDeletion has none).
                    let corr_uuid: Option<Uuid> = match &response {
                        typeset::ResponsePayload::Uniqueness(r) => r.signup_id.parse().ok(),
                        typeset::ResponsePayload::Reauthorization(r) => r.reauth_id.parse().ok(),
                        typeset::ResponsePayload::ResetCheck(r) => r.reset_id.parse().ok(),
                        typeset::ResponsePayload::ResetUpdate(r) => r.reset_id.parse().ok(),
                        typeset::ResponsePayload::IdentityDeletion(_) => None,
                    };

                    if let Some(uuid) = corr_uuid {
                        let is_complete = if let Some(info) = outstanding_requests.get_mut(&uuid) {
                            Some(info.record_response(&response))
                        } else {
                            None
                        };

                        match is_complete {
                            None => {
                                tracing::warn!(
                                    "Orphan response: no matching request for UUID {}",
                                    uuid
                                );
                            }
                            Some(true) => {
                                if let Some(info) = outstanding_requests.remove(&uuid) {
                                    // For uniqueness: search all node responses for a serial_id
                                    // and record it against the request's label.
                                    let maybe_serial_id = info.responses().iter().find_map(|opt| {
                                        if let Some(typeset::ResponsePayload::Uniqueness(result)) =
                                            opt
                                        {
                                            result.serial_id.or_else(|| {
                                                result.matched_serial_ids.as_ref()?.first().copied()
                                            })
                                        } else {
                                            None
                                        }
                                    });
                                    if let (Some(serial_id), Some(label)) =
                                        (maybe_serial_id, uuid_to_labels.remove(&uuid))
                                    {
                                        uniquess_labels.insert(label, serial_id);
                                    }
                                }
                            }
                            Some(false) => {}
                        }
                    } else {
                        tracing::warn!(
                            "Received IdentityDeletion response: not tracked in outstanding_requests"
                        );
                    }

                    self.aws_client
                        .sqs_purge_response_queue_message(&sqs_msg)
                        .await
                        .expect("SQS message purge failed");
                }
            }
            tracing::info!(
                "Batch {} finished. Responses to non-deletion requests have been received",
                batch_idx
            );
        }
        tracing::info!("Client finished.");
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
}

/// A utility for enqueuing system requests & correlating with system responses.
pub struct ServiceClient<R: Rng + CryptoRng + SeedableRng + Send> {
    // Component that enqueues system requests upon system ingress queues.
    request_enqueuer: RequestEnqueuer,

    // Component that generates system requests.
    request_generator: RequestGenerator,

    // Component that dequeues system responses from system egress queues.
    response_dequeuer: ResponseDequeuer,

    // Component that uploads iris shares to services prior to request processing.
    shares_uploader: SharesUploader<R>,
}

impl<R: Rng + CryptoRng + SeedableRng + Send> ServiceClient<R> {
    pub async fn new(
        opts: ServiceClientOptions,
        opts_aws: AwsOptions,
    ) -> Result<Self, ServiceClientError> {
        // Ensure options & 2nd order config are validated.
        opts.validate()?;

        let aws_client = AwsClient::async_from(opts_aws).await;

        Ok(Self {
            shares_uploader: SharesUploader::new(
                aws_client.clone(),
                SharesGenerator::<R>::from_options(&opts),
            ),
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::from_options(&opts)?,
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        })
    }

    pub async fn init(&mut self) -> Result<(), ServiceClientError> {
        for initializer in [self.shares_uploader.init(), self.response_dequeuer.init()] {
            initializer.await.map_err(|e| {
                tracing::error!("Service client: component initialisation failed: {}", e);
                ServiceClientError::InitialisationError(e.to_string())
            })?;
        }

        Ok(())
    }

    pub async fn exec(&mut self) -> Result<(), ServiceClientError> {
        let mut live_serial_ids: HashSet<IrisSerialId> = HashSet::new();

        // Run main batch loop, interruptible by Ctrl+C.
        tokio::select! {
            result = self._exec(&mut live_serial_ids) => {
                result?;
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nCtrl+C received. Initiating cleanup...");
            }
        };

        self.cleanup(live_serial_ids).await;
        Ok(())
    }

    async fn _exec(
        &mut self,
        live_serial_ids: &mut HashSet<IrisSerialId>,
    ) -> Result<(), ServiceClientError> {
        // Maps Uniqueness request labels → resolved serial IDs for cross-batch parent resolution.
        let mut label_resolutions: std::collections::HashMap<String, IrisSerialId> =
            std::collections::HashMap::new();

        while let Some(mut batch) = self.request_generator.next().await.unwrap() {
            println!("------------------------------------------------------------------------");
            println!(
                "Processing Batch {}: size={}",
                batch.batch_idx(),
                batch.requests().len()
            );
            println!("------------------------------------------------------------------------");

            // Upload shares for all active requests (and pending items with iris_pairs).
            self.shares_uploader.process_batch(&mut batch).await?;

            // Activate pending items whose parent label was resolved in a previous batch.
            batch.activate_cross_batch_pending(&label_resolutions);

            // Enqueue and dequeue until all enqueueable and enqueued items are processed.
            while batch.is_enqueueable() || batch.has_enqueued_items() {
                self.request_enqueuer.process_batch(&mut batch).await?;
                self.response_dequeuer
                    .process_batch(&mut batch, live_serial_ids, &mut label_resolutions)
                    .await?;
            }
        }

        Ok(())
    }

    async fn cleanup(&self, live_serial_ids: HashSet<IrisSerialId>) {
        println!("Cleaning up {} serial IDs", live_serial_ids.len());

        // Send deletion requests for all live serial IDs.
        for serial_id in live_serial_ids.into_iter() {
            if let Err(e) = self.request_enqueuer.publish_deletion(serial_id).await {
                eprintln!("Failed to send deletion for serial_id {}: {}", serial_id, e);
            }
        }

        // iris-mpc-hawk will not send responses for a batch of just deletions
        println!("Cleanup complete. Deletions have been submitted.");
    }
}
