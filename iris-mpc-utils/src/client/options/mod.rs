mod aws;
mod client;
mod request_batch;

pub use aws::AwsOptions;
pub use client::ServiceClientOptions;
pub(crate) use client::{IrisCodeSelectionStrategy, RequestBatchOptions, SharesGeneratorOptions};
