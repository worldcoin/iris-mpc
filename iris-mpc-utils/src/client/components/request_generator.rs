use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator {
    /// Number of request batches to generate.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: RequestBatchKind,

    /// Size of each batch.
    batch_size: RequestBatchSize,

    // Count of generated batches.
    generated_batch_count: usize,
}

impl RequestGenerator {
    fn batch_size(&self) -> usize {
        match self.batch_size {
            RequestBatchSize::Static(size) => size,
        }
    }

    pub fn new(
        batch_count: usize,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
    ) -> Self {
        Self {
            generated_batch_count: 0,
            batch_count,
            batch_kind,
            batch_size,
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ClientError> {
        if self.generated_batch_count == self.batch_count {
            return Ok(None);
        }

        let batch_idx = self.generated_batch_count + 1;
        let mut batch = RequestBatch::new(batch_idx, self.batch_size());
        for _ in 0..self.batch_size() {
            match self.batch_kind {
                RequestBatchKind::Simple(batch_kind) => match batch_kind {
                    IDENTITY_DELETION_MESSAGE_TYPE => {
                        let r1 = Request::new_uniqueness(&batch);
                        let r2 = Request::new_identity_deletion(&batch, &r1);
                        batch.requests_mut().push(r1);
                        batch.requests_mut().push(r2);
                    }
                    REAUTH_MESSAGE_TYPE => {
                        let r1 = Request::new_uniqueness(&batch);
                        let r2 = Request::new_reauthorisation(&batch, &r1);
                        batch.requests_mut().push(r1);
                        batch.requests_mut().push(r2);
                    }
                    RESET_CHECK_MESSAGE_TYPE => {
                        let r1 = Request::new_reset_check(&batch);
                        batch.requests_mut().push(r1);
                    }
                    RESET_UPDATE_MESSAGE_TYPE => {
                        let r1 = Request::new_uniqueness(&batch);
                        let r2 = Request::new_reset_update(&batch, &r1);
                        batch.requests_mut().push(r1);
                        batch.requests_mut().push(r2);
                    }
                    UNIQUENESS_MESSAGE_TYPE => {
                        let r1 = Request::new_uniqueness(&batch);
                        batch.requests_mut().push(r1);
                    }
                    _ => unreachable!(),
                },
            }
        }
        self.generated_batch_count += 1;

        Ok(Some(batch))
    }
}
