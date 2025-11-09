use rand::{CryptoRng, Rng};

use crate::aws::{AwsClient, AwsClientConfig};

use components::RequestEnqueuer;
use components::RequestGenerator;
use components::ResponseCorrelator;
use components::ResponseDequeuer;
pub use types::{
    Request, RequestBatch, RequestBatchKind, RequestBatchSize, RequestData, RequestDataUniqueness,
};

mod components;
mod types;

#[derive(Debug)]
pub struct Client<R: Rng + CryptoRng> {
    // Component that enqueues system requests upon system ingress queues.
    request_enqueuer: RequestEnqueuer,

    // Component that generates system requests.
    request_generator: RequestGenerator<R>,

    // Component that correlates system requests & responses.
    #[allow(dead_code)]
    response_correlator: ResponseCorrelator,

    // Component that dequeues system responses from system egress queues.
    #[allow(dead_code)]
    response_dequeuer: ResponseDequeuer,
}

impl<R: Rng + CryptoRng> Client<R> {
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
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::new(batch_kind, batch_size, batch_count, rng_seed),
            response_correlator: ResponseCorrelator::new(aws_client.clone()),
            response_dequeuer: ResponseDequeuer::default(),
        }
    }

    /// Initializer.
    pub async fn init(&mut self, public_key_base_url: String) {
        self.request_enqueuer.init(public_key_base_url).await;
        self.response_correlator.init().await;
    }

    /// Executor.
    pub async fn exec(&mut self) {
        println!("Executing ...");
        while let Some(batch) = self.request_generator.next().await {
            self.request_enqueuer.enqueue(&batch).await;
            // TODO await responses & correlate.
        }
    }
}
