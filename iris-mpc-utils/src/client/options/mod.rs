mod aws;
mod client;

pub use aws::AwsOptions;
pub use client::ServiceClientOptions;
pub(crate) use client::{IrisCodeSelectionStrategy, RequestBatchOptions, SharesGeneratorOptions};
