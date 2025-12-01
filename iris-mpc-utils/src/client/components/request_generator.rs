use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, Request, RequestBatch, RequestBatchKind, RequestBatchSize, RequestData,
};

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
}

impl RequestGenerator {
    fn batch_size(&self) -> usize {
        match self.batch_size {
            RequestBatchSize::Static(size) => size,
        }
    }

    pub fn new(
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        n_batches: usize,
    ) -> Self {
        Self {
            batch_count: 0,
            batch_kind,
            batch_size,
            n_batches,
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ClientError> {
        if self.batch_count == self.n_batches {
            return Ok(None);
        }

        let batch_idx = self.batch_count + 1;
        let mut batch = RequestBatch::new(batch_idx, self.batch_size());
        tracing::info!("{} :: Instantiated", batch);

        for item_idx in 1..(self.batch_size() + 1) {
            batch.requests_mut().push(Request::new(
                batch_idx,
                item_idx,
                match self.batch_kind {
                    RequestBatchKind::Simple(kind) => self.get_request_data_from_batch_kind(kind),
                },
            ));
        }

        self.batch_count += 1;

        Ok(Some(batch))
    }

    fn get_request_data_from_batch_kind(&mut self, batch_kind: &'static str) -> RequestData {
        match batch_kind {
            IDENTITY_DELETION_MESSAGE_TYPE => RequestData::IdentityDeletion {
                signup_id: uuid::Uuid::new_v4(),
            },
            REAUTH_MESSAGE_TYPE => RequestData::Reauthorization {
                reauthorisation_id: uuid::Uuid::new_v4(),
                signup_id: uuid::Uuid::new_v4(),
            },
            RESET_CHECK_MESSAGE_TYPE => RequestData::ResetCheck {
                reset_id: uuid::Uuid::new_v4(),
            },
            RESET_UPDATE_MESSAGE_TYPE => RequestData::ResetUpdate {
                reset_id: uuid::Uuid::new_v4(),
            },
            UNIQUENESS_MESSAGE_TYPE => RequestData::Uniqueness {
                signup_id: uuid::Uuid::new_v4(),
            },
            _ => unreachable!(),
        }
    }
}
