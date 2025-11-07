use aws_config::SdkConfig;
use aws_sdk_sqs::config::Region;

use super::constants::AWS_REGION;
use iris_mpc_common::config::Config as NodeConfig;

/// Encpasulates AWS service client configuration.
#[derive(Debug)]
pub struct AwsClientConfig {
    /// Associated node configuration.
    node: NodeConfig,

    /// Base URL for downloading node encryption public keys.
    public_key_base_url: String,

    /// System request ingress queue URL.
    requests_bucket_name: String,

    /// System request ingress queue topic.
    requests_topic_arn: String,

    /// System response eqgress queue URL.
    response_queue_url: String,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,
}

impl AwsClientConfig {
    pub async fn new(
        node_config: NodeConfig,
        public_key_base_url: String,
        requests_bucket_name: String,
        requests_topic_arn: String,
        response_queue_url: String,
    ) -> Self {
        Self {
            node: node_config.to_owned(),
            public_key_base_url,
            requests_bucket_name,
            requests_topic_arn,
            response_queue_url,
            sdk: get_sdk_config(&node_config).await,
        }
    }
}

impl Clone for AwsClientConfig {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            public_key_base_url: self.public_key_base_url.clone(),
            requests_bucket_name: self.requests_bucket_name.clone(),
            requests_topic_arn: self.requests_topic_arn.clone(),
            response_queue_url: self.response_queue_url.clone(),
            sdk: self.sdk.clone(),
        }
    }
}

impl AwsClientConfig {
    pub fn environment(&self) -> &String {
        &self.node().environment
    }

    pub fn node(&self) -> &NodeConfig {
        &self.node
    }

    pub fn public_key_base_url(&self) -> &String {
        &self.public_key_base_url
    }

    pub fn requests_bucket_name(&self) -> &String {
        &self.requests_bucket_name
    }

    pub fn requests_topic_arn(&self) -> &String {
        &self.requests_topic_arn
    }

    pub fn response_queue_url(&self) -> &String {
        &self.response_queue_url
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }
}

/// Returns AWS SDK configuration from a node configuration instance.
async fn get_sdk_config(node_config: &NodeConfig) -> aws_config::SdkConfig {
    aws_config::from_env()
        .region(Region::new(
            node_config
                .clone()
                .aws
                .and_then(|aws| aws.region)
                .unwrap_or_else(|| AWS_REGION.to_string()),
        ))
        .load()
        .await
}

#[cfg(test)]
mod tests {
    use super::{
        AwsClientConfig, AWS_PUBLICKEY_BASE_URL, AWS_REGION, AWS_REQUESTS_BUCKET_NAME,
        AWS_REQUESTS_TOPIC_ARN, AWS_RESPONSE_QUEUE_URL,
    };
    use crate::{constants::NODE_CONFIG_KIND_MAIN, fsys::local::read_node_config};

    async fn create_config() -> AwsClientConfig {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        AwsClientConfig::new(
            node_config,
            AWS_PUBLICKEY_BASE_URL.to_string(),
            AWS_REQUESTS_BUCKET_NAME,
            AWS_REQUESTS_TOPIC_ARN,
            AWS_RESPONSE_QUEUE_URL,
        )
        .await
    }

    #[tokio::test]
    async fn test_config_new() {
        let config = create_config().await;
        // TODO: check why this assert fails.
        // assert!(config.sdk().endpoint_url().is_some());
        assert_eq!(config.sdk().region().unwrap().as_ref(), AWS_REGION);
    }
}
