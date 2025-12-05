mod client;
mod config;
mod errors;
mod factory;
mod keys;
mod ops;
pub mod types;

pub use client::AwsClient;
pub use config::AwsClientConfig;
pub use errors::AwsClientError;
pub use factory::{create_iris_code_party_shares, create_iris_party_shares_for_s3};
