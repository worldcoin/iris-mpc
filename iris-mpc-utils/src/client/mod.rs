use rand::{CryptoRng, Rng};

use crate::aws::{AwsClient, AwsClientConfig};

use components::DataUploader;
use components::RequestEnqueuer;
use components::RequestGenerator;
use components::ResponseCorrelator;
use components::ResponseDequeuer;
pub use typeset::{
    ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch, RequestBatchKind,
    RequestBatchSize, RequestData,
};

mod components;
mod typeset;

/// A utility for correlating enqueued system requests with system responses.
#[derive(Debug)]
pub struct ServiceClient<R: Rng + CryptoRng + Send> {
    // Component that uploads data to services prior to request processing.
    data_uploader: DataUploader<R>,

    // Component that enqueues system requests upon system ingress queues.
    request_enqueuer: RequestEnqueuer,

    // Component that generates system requests.
    request_generator: RequestGenerator,

    // Component that correlates system requests & responses.
    #[allow(dead_code)]
    response_correlator: ResponseCorrelator,

    // Component that dequeues system responses from system egress queues.
    #[allow(dead_code)]
    response_dequeuer: ResponseDequeuer,
}

impl<R: Rng + CryptoRng> ServiceClient<R> {
    /// Constructor.
    pub async fn new(
        aws_client_config: AwsClientConfig,
        batch_count: usize,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        rng_seed: R,
    ) -> Self {
        let aws_client = AwsClient::new(aws_client_config);

        Self {
            data_uploader: DataUploader::new(aws_client.clone(), rng_seed),
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::new(batch_kind, batch_size, batch_count),
            response_correlator: ResponseCorrelator::new(aws_client.clone()),
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        }
    }

    /// Initializer.
    pub async fn init(&mut self) -> Result<(), ClientError> {
        tracing::info!("Initializing ...");
        for initializer in [self.data_uploader.init(), self.response_correlator.init()] {
            match initializer.await {
                Ok(()) => (),
                Err(e) => {
                    tracing::error!("Service client: component initialisation failed: {}", e);
                    return Err(ClientError::InitialisationError(e.to_string()));
                }
            }
        }

        Ok(())
    }

    /// Executor.
    pub async fn exec(&mut self) -> Result<(), ClientError> {
        tracing::info!("Executing ...");
        while let Some(batch) = self.request_generator.next().await.unwrap() {
            for batch_processor in [
                self.data_uploader.process_batch(&batch),
                self.request_enqueuer.process_batch(&batch),
                self.response_dequeuer.process_batch(&batch),
            ] {
                batch_processor.await?
            }
        }

        Ok(())
    }
}
