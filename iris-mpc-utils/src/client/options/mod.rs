use std::collections::HashSet;

use serde::{Deserialize, Serialize};

mod aws;
mod descriptors;
pub(crate) mod mapper;
mod requests;
mod shares;

use super::typeset::ServiceClientError;
pub use aws::AwsOptions;
pub(crate) use descriptors::{
    IrisDescriptorOptions, IrisPairDescriptorOptions, UniquenessRequestDescriptorOptions,
};
pub(crate) use requests::{RequestBatchOptions, RequestOptions, RequestPayloadOptions};
pub(crate) use shares::{IrisCodeSelectionStrategyOptions, SharesGeneratorOptions};

/// Service client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClientOptions {
    // A representation of remote system state prior to execution. E.G. a hex encoded hash value.
    prestate: Option<String>,

    // Associated request batch generation configuration.
    request_batch: RequestBatchOptions,

    // Associated Iris shares generator configuration.
    shares_generator: SharesGeneratorOptions,
}

impl ServiceClientOptions {
    pub fn request_batch(&self) -> &RequestBatchOptions {
        &self.request_batch
    }

    pub fn shares_generator(&self) -> &SharesGeneratorOptions {
        &self.shares_generator
    }

    pub(crate) fn validate_and_parse(&self) -> Result<(), ServiceClientError> {
        self.validate()?;

        Ok(())
    }

    fn validate(&self) -> Result<(), ServiceClientError> {
        // Error if complex request batch is being used alongside compute shares generation.
        match self.request_batch() {
            RequestBatchOptions::Complex { .. } => match self.shares_generator() {
                SharesGeneratorOptions::FromCompute { .. } => {
                    return Err(ServiceClientError::InvalidOptions("RequestBatchOptions::Complex can only be used with SharesGeneratorOptions::FromFile".to_string()))
                }
                _ => {},
            },
            _ => {},
        }

        // Error if there are Iris descriptor duplicates.
        if let RequestBatchOptions::Complex { .. } = self.request_batch() {
            let indexes = self.request_batch().iris_code_indexes();
            if !indexes.is_empty() {
                let set: HashSet<usize> = indexes.iter().copied().collect();
                if set.len() != indexes.len() {
                    return Err(ServiceClientError::InvalidOptions(
                        "RequestBatchOptions Iris descriptor set contains duplicates".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ServiceClientOptions;
    use crate::fsys::{local::get_path_to_service_client_exec_opts, reader::read_toml};

    #[test]
    fn test_exec_opts_deserialization() {
        (1..=2).for_each(move |opts_idx| {
            let path_to_opts = get_path_to_service_client_exec_opts(opts_idx);
            let _ = read_toml::<ServiceClientOptions>(path_to_opts.as_path())
                .expect("Failed to deserialize service client exec options file");
        });
    }
}
