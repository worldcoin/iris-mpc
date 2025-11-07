use rand::{CryptoRng, Rng};

use crate::aws::{NetAwsClientConfig, NodeAwsClient};

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
    pub async fn new(
        net_aws_client_config: NetAwsClientConfig,
        net_public_key_base_url: &str,
        batch_count: usize,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        rng_seed: R,
    ) -> Self {
        let net_encryption_public_keys =
            NodeAwsClient::download_net_encryption_public_keys(net_public_key_base_url)
                .await
                .unwrap();
        let net_aws_clients = [
            NodeAwsClient::new(
                net_aws_client_config[0].to_owned(),
                net_encryption_public_keys,
            ),
            NodeAwsClient::new(
                net_aws_client_config[1].to_owned(),
                net_encryption_public_keys,
            ),
            NodeAwsClient::new(
                net_aws_client_config[2].to_owned(),
                net_encryption_public_keys,
            ),
        ];

        Self {
            request_enqueuer: RequestEnqueuer::new(net_aws_clients),
            request_generator: RequestGenerator::new(batch_kind, batch_size, batch_count, rng_seed),
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
