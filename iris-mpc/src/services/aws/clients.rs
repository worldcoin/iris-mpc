use crate::services::aws::sns::create_sns_client;
use crate::services::aws::sqs::create_sqs_client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Region, Client as SQSClient};
use eyre::Result;
use iris_mpc_common::config::{Config, ENV_PROD, ENV_STAGE};
use iris_mpc_common::object_store::ObjectStoreClient;

const DEFAULT_REGION: &str = "eu-north-1";

pub struct AwsClients {
    pub sqs_client: SQSClient,
    pub sns_client: SNSClient,
    pub object_store_client: ObjectStoreClient,
    // Graph checkpoints may use a store in a different region.
    pub checkpoint_object_store_client: ObjectStoreClient,
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

        let region_provider = Region::new(region.clone());
        let shared_config = aws_config::from_env().region(region_provider).load().await;

        let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;

        let sns_client = create_sns_client(&shared_config, config.sns_retry_max_attempts);
        let sqs_client = create_sqs_client(&shared_config, config.sqs_long_poll_wait_time);
        let object_store_client = ObjectStoreClient::new(Some(region), force_path_style)
            .with_aws_sdk_config(&shared_config);
        let secrets_manager_client = SecretsManagerClient::new(&shared_config);

        let checkpoint_object_store_client = ObjectStoreClient::new(
            Some(config.graph_checkpoint_bucket_region.clone()),
            force_path_style,
        )
        .with_aws_sdk_config(&shared_config);

        Ok(Self {
            sqs_client,
            sns_client,
            object_store_client,
            checkpoint_object_store_client,
            secrets_manager_client,
        })
    }
}

// implement clone for AwsClients
impl Clone for AwsClients {
    fn clone(&self) -> Self {
        Self {
            sqs_client: self.sqs_client.clone(),
            sns_client: self.sns_client.clone(),
            object_store_client: self.object_store_client.clone(),
            checkpoint_object_store_client: self.checkpoint_object_store_client.clone(),
            secrets_manager_client: self.secrets_manager_client.clone(),
        }
    }
}
