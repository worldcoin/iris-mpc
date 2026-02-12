use std::collections::HashSet;

use super::{
    super::typeset::ServiceClientError,
    types::{RequestBatchOptions, SharesGeneratorOptions},
    ServiceClientOptions,
};

impl ServiceClientOptions {
    pub(crate) fn validate(&self) -> Result<(), ServiceClientError> {
        // Validate complex request options.
        if let RequestBatchOptions::Complex { .. } = self.request_batch() {
            // Error if used alongside compute shares generation.
            match self.shares_generator() {
                    SharesGeneratorOptions::FromCompute { .. } => {
                        return Err(ServiceClientError::InvalidOptions("RequestBatchOptions::Complex can only be used with SharesGeneratorOptions::FromFile".to_string()))
                    }
                    _ => {},
                };

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

        Ok(())
    }
}
