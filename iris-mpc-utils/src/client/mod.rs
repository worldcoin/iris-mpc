use std::collections::{HashMap, HashSet};

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

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
    shares_generator: SharesGeneratorOptions,
}

impl ServiceClient2 {
    pub async fn new(
        opts_aws: AwsOptions,
        request_batch: RequestBatchOptions,
        shares_generator: SharesGeneratorOptions,
    ) -> Result<Self, ServiceClientError> {
        let aws_client = AwsClient::async_from(opts_aws).await;

        // todo: better validation code
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

        Ok(Self {
            aws_client,
            request_batch,
            shares_generator,
        })
    }

    pub async fn exec(mut self) {
        self.init().await.expect("init failed");

        // need to track all the uniqueness requests sent, and their serial ids. also need to track which requests are dependent on a parent.
        let mut uniquess_labels: HashMap<String, IrisSerialId> = HashMap::new();
        let mut uuid_to_labels: HashMap<Uuid, String> = HashMap::new();
        let mut outstanding_requests: HashMap<Uuid, typeset::RequestInfo> = HashMap::new();

        for batch in self.request_batch.into_iter() {
            // for each item in the batch

            // look up the iris serial id for uniqueness_labels for anything in the batch that needs it if not there, drop the item i the batch and emit a warning

            // send a request and add it to outstanding requests. if it is a uniqueness request, add the label and uuid to uuid_to_labels

            // after doing this for everything in the batch, do a new loop, where you do the following:

            // wait for responses, updating outstanding_requests. for uniqueness results, get the label from uuid_to_labels and then add the iris serial id to uniqueness_labels.
            todo!();
        }

        todo!()
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
