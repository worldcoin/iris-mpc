use std::collections::HashSet;

use iris_mpc_common::helpers::smpc_request;

use super::{
    super::typeset::ServiceClientError,
    types::{RequestBatchOptions, SharesGeneratorOptions},
    ServiceClientOptions,
};

impl ServiceClientOptions {
    pub(crate) fn validate(&self) -> Result<(), ServiceClientError> {
        match self.request_batch() {
            RequestBatchOptions::Complex { .. } => {
                // Error if used alongside compute shares generation.
                if matches!(
                    self.shares_generator(),
                    SharesGeneratorOptions::FromCompute { .. }
                ) {
                    return Err(ServiceClientError::InvalidOptions("RequestBatchOptions::Complex can only be used with SharesGeneratorOptions::FromFile".to_string()));
                }

                // Error if there are duplicate Iris descriptors.
                let indexes = self.request_batch().iris_code_indexes();
                if !indexes.is_empty() {
                    let mut set = HashSet::with_capacity(indexes.len());
                    if !indexes.iter().all(|i| set.insert(i)) {
                        return Err(ServiceClientError::InvalidOptions(
                            "RequestBatchOptions::Complex contains duplicate Iris descriptors "
                                .to_string(),
                        ));
                    }
                }

                // Error if there are duplicate labels.
                let labels = self.request_batch().labels();
                if !labels.is_empty() {
                    let mut set = HashSet::with_capacity(labels.len());
                    if !labels.iter().all(|l| set.insert(l)) {
                        return Err(ServiceClientError::InvalidOptions(
                            "RequestBatchOptions::Complex contains duplicate labels".to_string(),
                        ));
                    }
                }

                // Error if there are invalid parent labels.
                let labels_of_parents = self.request_batch().labels_of_parents();
                if !labels_of_parents.is_empty() {
                    let labels_set: HashSet<_> = labels.iter().collect();
                    for label_of_parent in &labels_of_parents {
                        if !labels_set.contains(label_of_parent) {
                            return Err(ServiceClientError::InvalidOptions(
                                format!(
                                    "RequestBatchOptions::Complex contains a parent label '{}' that is not found in labels",
                                    label_of_parent
                                ),
                            ));
                        }
                    }
                }
            }
            RequestBatchOptions::Simple {
                batch_count,
                batch_kind,
                batch_size,
                ..
            } => {
                // Error if total requests exceed arbitrary limit.
                if batch_count * batch_size > 1_000_000 {
                    return Err(ServiceClientError::InvalidOptions(
                        "RequestBatchOptions::Simple total requests will exceed limit of 1_000_000"
                            .to_string(),
                    ));
                }

                // Error it batch kind cannot be mapped to a supported SMPC request type.
                if ![
                    smpc_request::IDENTITY_DELETION_MESSAGE_TYPE,
                    smpc_request::REAUTH_MESSAGE_TYPE,
                    smpc_request::RESET_CHECK_MESSAGE_TYPE,
                    smpc_request::RESET_UPDATE_MESSAGE_TYPE,
                    smpc_request::UNIQUENESS_MESSAGE_TYPE,
                ]
                .contains(&batch_kind.as_str())
                {
                    return Err(ServiceClientError::InvalidOptions(format!(
                        "RequestBatchOptions::Simple batch_kind ({}) is unsupported",
                        batch_kind
                    )));
                };
            }
        }

        Ok(())
    }
}
