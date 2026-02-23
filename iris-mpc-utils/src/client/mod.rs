use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

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

    pub async fn exec(&mut self) -> Result<(), ServiceClientError> {
        let mut cross_batch_resolutions = std::collections::HashMap::new();

        while let Some(mut batch) = self.request_generator.next().await.unwrap() {
            println!("------------------------------------------------------------------------");
            println!(
                "Processing Batch {}: size={}",
                batch.batch_idx(),
                batch.requests().len()
            );
            println!("------------------------------------------------------------------------");

            self.shares_uploader.process_batch(&mut batch).await?;
            batch.resolve_cross_batch_parents(&cross_batch_resolutions);
            while batch.is_enqueueable() {
                self.request_enqueuer.process_batch(&mut batch).await?;
                self.response_dequeuer.process_batch(&mut batch).await?;
            }

            // Collect uniqueness resolutions for future batches.
            for request in batch.requests() {
                if let Some((signup_id, serial_id)) = request.uniqueness_resolution() {
                    cross_batch_resolutions.insert(signup_id, serial_id);
                }
            }
        }

        Ok(())
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
}
