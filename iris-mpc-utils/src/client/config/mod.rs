mod aws;
mod client;

pub use aws::AwsConfiguration;
pub use client::ServiceClientConfiguration;
pub(crate) use client::{
    IrisCodeSelectionStrategy, RequestBatchConfiguration, SharesGeneratorConfiguration,
};
