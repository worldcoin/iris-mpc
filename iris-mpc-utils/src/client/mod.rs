use std::collections::HashSet;

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_common::IrisSerialId;

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
