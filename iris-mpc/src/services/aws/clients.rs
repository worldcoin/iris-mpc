use crate::services::aws::s3::create_s3_client;
use crate::services::aws::sns::create_sns_client;
use crate::services::aws::sqs::create_sqs_client;
use aws_sdk_s3::config::Region as S3Region;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Region, Client as SQSClient};
use eyre::Result;
use iris_mpc_common::config::{Config, ENV_PROD, ENV_STAGE};

const DEFAULT_REGION: &str = "eu-north-1";

pub struct AwsClients {
    pub sqs_client: SQSClient,
    pub sns_client: SNSClient,
    pub s3_client: S3Client,
    pub shares_s3_client: S3Client,
    // used to obtain graph checkpoints. this bucket could be in a different region
    pub checkpoint_s3_client: S3Client,
    pub secrets_manager_client: SecretsManagerClient,
}

impl AwsClients {
    pub async fn new(config: &Config) -> Result<Self> {
        // Get region from config or use default
        let region = config
            .clone()
            .aws
            .and_then(|aws| aws.region)
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());

        let region_provider = Region::new(region);
        let shared_config = aws_config::from_env().region(region_provider).load().await;

        Ok(Self::from_sdk_config(config, &shared_config))
    }

    fn from_sdk_config(config: &Config, shared_config: &aws_config::SdkConfig) -> Self {
        let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;

        let sns_client = create_sns_client(
            shared_config,
            config.sns_retry_max_attempts,
            config.results_topic_region.clone().map(Region::new),
        );
        let sqs_client = create_sqs_client(
            shared_config,
            config.sqs_long_poll_wait_time,
            config.requests_queue_region.clone().map(Region::new),
        );
        let s3_client = create_s3_client(shared_config, force_path_style, None);
        let shares_s3_client = match &config.shares_bucket_region {
            Some(region) => create_s3_client(
                shared_config,
                force_path_style,
                Some(S3Region::new(region.clone())),
            ),
            None => s3_client.clone(),
        };
        let secrets_manager_client = SecretsManagerClient::new(shared_config);

        let checkpoint_region = S3Region::new(config.graph_checkpoint_bucket_region.clone());
        let checkpoint_s3_client =
            create_s3_client(shared_config, force_path_style, Some(checkpoint_region));

        Self {
            sqs_client,
            sns_client,
            s3_client,
            shares_s3_client,
            checkpoint_s3_client,
            secrets_manager_client,
        }
    }
}

// implement clone for AwsClients
impl Clone for AwsClients {
    fn clone(&self) -> Self {
        Self {
            sqs_client: self.sqs_client.clone(),
            sns_client: self.sns_client.clone(),
            s3_client: self.s3_client.clone(),
            shares_s3_client: self.shares_s3_client.clone(),
            checkpoint_s3_client: self.checkpoint_s3_client.clone(),
            secrets_manager_client: self.secrets_manager_client.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_region_overrides_only_shares_s3_client() {
        let shared_config = aws_config::SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .region(Region::new("eu-north-1"))
            .build();

        for (settings, expected_region) in [
            (serde_json::json!({}), "eu-north-1"),
            (
                serde_json::json!({ "shares_bucket_region": "eu-west-1" }),
                "eu-west-1",
            ),
        ] {
            let config: Config = serde_json::from_value(settings).unwrap();
            let clients = AwsClients::from_sdk_config(&config, &shared_config);

            assert_eq!(
                clients.shares_s3_client.config().region().unwrap().as_ref(),
                expected_region
            );
            assert_eq!(
                clients.s3_client.config().region().unwrap().as_ref(),
                "eu-north-1"
            );
            assert_eq!(
                clients
                    .secrets_manager_client
                    .config()
                    .region()
                    .unwrap()
                    .as_ref(),
                "eu-north-1"
            );
            assert_eq!(
                clients
                    .checkpoint_s3_client
                    .config()
                    .region()
                    .unwrap()
                    .as_ref(),
                "eu-central-1"
            );
        }
    }
}
