use rand::{CryptoRng, Rng};

use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

use super::super::{
    errors::ServiceClientError,
    types::{Request, RequestBatch, RequestBatchKind, RequestBatchSize, RequestData},
};
use crate::irises::generate_iris_code_and_mask_shares_both_eyes;

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator<R: Rng + CryptoRng> {
    // Count of generated batches.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: RequestBatchKind,

    /// Size of each batch.
    batch_size: RequestBatchSize,

    /// Number of request batches to generate.
    n_batches: usize,

    /// Entropy source.
    rng_seed: R,
}

impl<R: Rng + CryptoRng> RequestGenerator<R> {
    fn rng_seed_mut(&mut self) -> &mut R {
        &mut self.rng_seed
    }

    fn batch_size(&self) -> usize {
        match self.batch_size {
            RequestBatchSize::Static(size) => size,
        }
    }

    pub fn new(
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        n_batches: usize,
        rng_seed: R,
    ) -> Self {
        Self {
            batch_count: 0,
            batch_kind,
            batch_size,
            n_batches,
            rng_seed,
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ServiceClientError> {
        if self.batch_count == self.n_batches {
            return Ok(None);
        }

        let batch_idx = self.batch_count + 1;
        let batch_size = self.batch_size();
        let mut batch = RequestBatch::new(batch_idx, batch_size);
        tracing::info!("----------------------------------------------------------------------");
        tracing::info!("{} :: Instantiated", batch);
        tracing::info!("----------------------------------------------------------------------");

        for item_idx in 1..(batch_size + 1) {
            let request = self.generate_request(batch_idx, item_idx);
            tracing::info!("{} :: Generated", request);
            batch.requests_mut().push(request);
        }

        self.batch_count += 1;

        Ok(Some(batch))
    }

    fn generate_request(&mut self, batch_idx: usize, item_idx: usize) -> Request {
        Request::new(
            batch_idx,
            item_idx,
            match self.batch_kind {
                RequestBatchKind::Simple(kind) => match kind {
                    UNIQUENESS_MESSAGE_TYPE => RequestData::Uniqueness {
                        shares: generate_iris_code_and_mask_shares_both_eyes(self.rng_seed_mut()),
                    },
                    _ => panic!("Unsupported request kind: {}", kind),
                },
            },
        )
    }
}
