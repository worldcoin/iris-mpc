use aws_config::{from_env, retry::RetryConfig};
use aws_sdk_s3::{
    config::{Builder as S3_ConfigBuilder, Region as AWS_Region},
    Client as S3_Client,
};
use eyre::Result;
use iris_mpc_common::config::{Config as NodeConfig, NetConfig, ENV_PROD, ENV_STAGE};

/// Default AWS region.
const DEFAULT_AWS_REGION: &str = "eu-north-1";

/// Returns an S3 client with retry configuration.
pub(crate) async fn get_s3_client(config: &NodeConfig) -> Result<S3_Client> {
    let region = config
        .to_owned()
        .aws
        .and_then(|aws| aws.region)
        .unwrap_or_else(|| DEFAULT_AWS_REGION.to_owned());
    let region_provider = AWS_Region::new(region);
    let shared_config = from_env().region(region_provider).load().await;
    let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
    let retry_config = RetryConfig::standard().with_max_attempts(5);
    let s3_config = S3_ConfigBuilder::from(&shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    Ok(S3_Client::from_conf(s3_config))
}

#[cfg(test)]
mod test {
    use super::{get_s3_client, NetConfig};
    use crate::resources::{self, NODE_CONFIG_KIND_GENESIS};

    fn get_net_config() -> NetConfig {
        resources::read_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap()
    }

    #[tokio::test]
    async fn test_get_s3_client() {
        for (party_idx, cfg) in get_net_config().iter().enumerate() {
            match get_s3_client(cfg).await {
                Ok(_) => (),
                Err(err) => panic!(
                    "Failed to get S3 client: Party IDX={} :: Err={}",
                    party_idx, err
                ),
            };
        }
    }
}
