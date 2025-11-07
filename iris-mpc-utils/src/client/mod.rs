use rand::rngs::StdRng;

use crate::aws::{NetAwsClientConfig, NodeAwsClient};

pub use components::RequestEnqueuer;
pub use components::RequestGenerator;
pub use components::ResponseCorrelator;
pub use components::ResponseDequeuer;
pub use types::{
    Request, RequestBatch, RequestBatchKind, RequestBatchSize, RequestData, RequestDataUniqueness,
};

mod components;
mod types;

#[derive(Debug)]
pub struct Client {
    request_enqueuer: RequestEnqueuer,
    request_generator: RequestGenerator,
    #[allow(dead_code)]
    response_correlator: ResponseCorrelator,
    #[allow(dead_code)]
    response_dequeuer: ResponseDequeuer,
}

impl Client {
    pub fn new(
        net_aws_client_config: NetAwsClientConfig,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        n_batches: usize,
        rng_seed: StdRng,
    ) -> Self {
        Self {
            request_enqueuer: RequestEnqueuer::new(net_aws_client_config.map(NodeAwsClient::new)),
            request_generator: RequestGenerator::new(batch_kind, batch_size, n_batches, rng_seed),
            response_correlator: ResponseCorrelator::default(),
            response_dequeuer: ResponseDequeuer::default(),
        }
    }

    pub async fn run(&mut self) {
        while let Some(batch) = self.request_generator.next().await {
            self.request_enqueuer.enqueue(&batch).await;
            self.response_dequeuer.dequeue(&batch).await;
            self.response_correlator.correlate(&batch).await;
        }
    }
}
