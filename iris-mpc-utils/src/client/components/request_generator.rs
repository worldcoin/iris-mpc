use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use super::super::typeset::{
    ClientError, RequestBatch, RequestBatchKind, RequestBatchSize, RequestFactory,
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

    // A known serial identifier that allows response correlation to be bypassed.
    known_iris_serial_id: Option<IrisSerialId>,
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
        known_iris_serial_id: Option<IrisSerialId>,
    ) -> Self {
        Self {
            generated_batch_count: 0,
            batch_count,
            batch_kind,
            batch_size,
            known_iris_serial_id,
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
                RequestBatchKind::Simple(kind) => match kind {
                    IDENTITY_DELETION_MESSAGE_TYPE => {
                        if let Some(known_iris_serial_id) = self.known_iris_serial_id {
                            batch.push_request(RequestFactory::new_identity_deletion_2(
                                &batch,
                                known_iris_serial_id,
                            ));
                        } else {
                            let r1 = RequestFactory::new_uniqueness(&batch);
                            let r2 = RequestFactory::new_identity_deletion_1(&batch, &r1);
                            batch.push_request(r1);
                            batch.push_request(r2);
                        }
                    }
                    RESET_CHECK_MESSAGE_TYPE => {
                        batch.push_request(RequestFactory::new_reset_check(&batch));
                    }
                    REAUTH_MESSAGE_TYPE => {
                        if let Some(known_iris_serial_id) = self.known_iris_serial_id {
                            batch.push_request(RequestFactory::new_reauthorisation_2(
                                &batch,
                                known_iris_serial_id,
                            ));
                        } else {
                            let r1 = RequestFactory::new_uniqueness(&batch);
                            let r2 = RequestFactory::new_reauthorisation_1(&batch, &r1);
                            batch.push_request(r1);
                            batch.push_request(r2);
                        }
                    }
                    RESET_UPDATE_MESSAGE_TYPE => {
                        if let Some(known_iris_serial_id) = self.known_iris_serial_id {
                            batch.push_request(RequestFactory::new_reset_update_2(
                                &batch,
                                known_iris_serial_id,
                            ));
                        } else {
                            let r1 = RequestFactory::new_uniqueness(&batch);
                            let r2 = RequestFactory::new_reset_update_1(&batch, &r1);
                            batch.push_request(r1);
                            batch.push_request(r2);
                        }
                    }
                    UNIQUENESS_MESSAGE_TYPE => {
                        batch.push_request(RequestFactory::new_uniqueness(&batch));
                    }
                    _ => panic!("Invalid batch kind"),
                },
            }
        }
        self.generated_batch_count += 1;

        Ok(Some(batch))
    }
}
