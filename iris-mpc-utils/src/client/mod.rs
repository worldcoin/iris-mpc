use rand::{CryptoRng, Rng};

use crate::aws::{AwsClient, AwsClientConfig};

use components::{
    DataUploader, RequestEnqueuer, RequestGenerator, ResponseCorrelator, ResponseDequeuer,
};
pub use typeset::{
    ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch, RequestBatchKind,
    RequestBatchSize,
};

mod components;
mod typeset;

/// A utility for enqueuing system requests & correlating with system responses.
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

impl<R: Rng + CryptoRng + Send> ServiceClient<R> {
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
            request_generator: RequestGenerator::new(batch_count, batch_kind, batch_size),
            response_correlator: ResponseCorrelator::new(aws_client.clone()),
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        }
    }

    pub async fn exec(&mut self) -> Result<(), ClientError> {
        while let Some(batch) = self.request_generator.next().await.unwrap() {
            self.data_uploader.process_batch(&batch).await?;
            self.request_enqueuer.process_batch(&batch).await?;
            self.response_dequeuer.process_batch(&batch).await?;
        }

        Ok(())
    }

    pub async fn init(&mut self) -> Result<(), ClientError> {
        for initializer in [self.data_uploader.init(), self.response_correlator.init()] {
            initializer.await.map_err(|e| {
                tracing::error!("Service client: component initialisation failed: {}", e);
                ClientError::InitialisationError(e.to_string())
            })?;
        }

        Ok(())
    }
}
