use serde::{Deserialize, Serialize};

mod types;
mod validator;

pub use types::AwsOptions;
pub(crate) use types::{
    RequestBatchOptions, RequestOptions, RequestPayloadOptions, SharesGeneratorOptions,
};

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
