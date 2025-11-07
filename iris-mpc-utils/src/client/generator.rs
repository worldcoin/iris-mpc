use rand::rngs::StdRng;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::types::{
    Request, RequestBatch, RequestBatchKind, RequestBatchSize, RequestData, RequestDataUniqueness,
};
use crate::irises::generate_iris_code_and_mask_shares_both_eyes;

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator {
    // Count of generated batches.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: RequestBatchKind,

    /// Size of each batch.
    batch_size: RequestBatchSize,

    /// Number of request batches to generate.
    n_batches: usize,

    /// Entropy source.
    rng_seed: StdRng,
}

impl RequestGenerator {
    fn rng_seed_mut(&mut self) -> &mut StdRng {
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
        rng_seed: StdRng,
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
    pub async fn next(&mut self) -> Option<RequestBatch> {
        if self.batch_count == self.n_batches {
            return None;
        }

        let batch_idx = self.batch_count + 1;
        let batch_size = self.batch_size();
        let mut batch = RequestBatch::new(batch_idx, batch_size);
        for item_idx in 1..(batch_size + 1) {
            batch
                .requests_mut()
                .push(self.generate_request(batch_idx, item_idx));
        }
        self.batch_count += 1;

        Some(batch)
    }

    fn generate_request(&mut self, batch_idx: usize, item_idx: usize) -> Request {
        // Assume 1 based ordinal identifiers.
        assert!(batch_idx >= 1 && item_idx >= 1);

        Request::new(
            batch_idx,
            item_idx,
            match self.batch_kind {
                RequestBatchKind::Simple(kind) => match kind {
                    IDENTITY_DELETION_MESSAGE_TYPE => RequestData::IdentityDeletion,
                    REAUTH_MESSAGE_TYPE => RequestData::Reauthorisation,
                    RESET_CHECK_MESSAGE_TYPE => RequestData::ResetCheck,
                    RESET_UPDATE_MESSAGE_TYPE => RequestData::ResetUpdate,
                    UNIQUENESS_MESSAGE_TYPE => RequestData::Uniqueness(RequestDataUniqueness {
                        iris_shares: generate_iris_code_and_mask_shares_both_eyes(
                            self.rng_seed_mut(),
                        ),
                    }),
                    _ => panic!("Unsupported request kind: {}", kind),
                },
            },
        )
    }
}
