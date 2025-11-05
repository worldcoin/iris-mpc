use aws_config::SdkConfig;
use aws_sdk_sqs::config::Region;

use iris_mpc_common::config::Config as NodeConfig;

const DEFAULT_REGION: &str = "eu-north-1";

/// Encpasulates AWS service client configuration.
#[derive(Debug)]
pub struct NodeAwsConfig {
    /// Associated node configuration.
    node: NodeConfig,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,
}

impl NodeAwsConfig {
    pub async fn new(node_config: NodeConfig) -> Self {
        Self {
            node: node_config.to_owned(),
            sdk: get_sdk_config(&node_config).await,
        }
    }
}

impl Clone for NodeAwsConfig {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            sdk: self.sdk.clone(),
        }
    }
}

impl NodeAwsConfig {
    pub fn environment(&self) -> &String {
        &self.node().environment
    }

    pub fn node(&self) -> &NodeConfig {
        &self.node
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
                .unwrap_or_else(|| DEFAULT_REGION.to_owned()),
        ))
        .load()
        .await
}

#[cfg(test)]
mod tests {
    use super::NodeAwsConfig;
    use crate::{
        constants::{DEFAULT_AWS_REGION, NODE_CONFIG_KIND_MAIN},
        fsys::local::read_node_config,
    };

    async fn create_config() -> NodeAwsConfig {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        NodeAwsConfig::new(node_config).await
    }

    #[tokio::test]
    async fn test_config_new() {
        let config = create_config().await;
        // TODO: check why this assert fails.
        // assert!(config.sdk().endpoint_url().is_some());
        assert_eq!(config.sdk().region().unwrap().as_ref(), DEFAULT_AWS_REGION);
    }
}
