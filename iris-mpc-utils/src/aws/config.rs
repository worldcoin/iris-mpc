use aws_config::SdkConfig;
use aws_sdk_sqs::config::Region;

use super::constants::AWS_REGION;
use crate::constants::N_PARTIES;
use iris_mpc_common::config::Config as NodeConfig;

/// Encpasulates AWS service client configuration.
#[derive(Debug)]
pub struct NodeAwsClientConfig {
    /// Associated node configuration.
    node: NodeConfig,

    /// System request ingress queue URL.
    request_bucket_name: String,

    /// System request ingress queue topic.
    request_topic_arn: String,

    /// System response eqgress queue URL.
    response_queue_url: String,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,
}

// Network wide configuration set.
pub type NetAwsClientConfig = [NodeAwsClientConfig; N_PARTIES];

impl NodeAwsClientConfig {
    pub fn environment(&self) -> &String {
        &self.node().environment
    }

    pub fn node(&self) -> &NodeConfig {
        &self.node
    }

    pub fn request_bucket_name(&self) -> &String {
        &self.request_bucket_name
    }

    pub fn request_topic_arn(&self) -> &String {
        &self.request_topic_arn
    }

    pub fn response_queue_url(&self) -> &String {
        &self.response_queue_url
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }

    pub async fn new(
        node_config: NodeConfig,
        request_bucket_name: String,
        request_topic_arn: String,
        response_queue_url: String,
    ) -> Self {
        Self {
            node: node_config.to_owned(),
            request_bucket_name,
            request_topic_arn,
            response_queue_url,
            sdk: get_sdk_config(&node_config).await,
        }
    }
}

impl Clone for NodeAwsClientConfig {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            request_bucket_name: self.request_bucket_name.clone(),
            request_topic_arn: self.request_topic_arn.clone(),
            response_queue_url: self.response_queue_url.clone(),
            sdk: self.sdk.clone(),
        }
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
    use super::super::constants::{
        AWS_REGION, AWS_REQUEST_BUCKET_NAME, AWS_REQUEST_TOPIC_ARN, AWS_RESPONSE_QUEUE_URL,
    };
    use super::NodeAwsClientConfig;
    use crate::{constants::NODE_CONFIG_KIND_MAIN, fsys::local::read_node_config};

    async fn create_config() -> NodeAwsClientConfig {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        NodeAwsClientConfig::new(
            node_config,
            AWS_REQUEST_BUCKET_NAME.to_string(),
            AWS_REQUEST_TOPIC_ARN.to_string(),
            AWS_RESPONSE_QUEUE_URL.to_string(),
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
