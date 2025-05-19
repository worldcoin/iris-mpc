use crate::services::aws::s3::{create_s3_client};
use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Region, Client as SQSClient, Client};
use eyre::Result;
use iris_mpc_common::config::Config;

const DEFAULT_REGION: &str = "eu-north-1";

pub struct AwsClients {
    pub sqs_client: SQSClient,
    pub sns_client: SNSClient,
    pub s3_client: S3Client,
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
        let sqs_client = Client::new(&shared_config);
        let sns_client = SNSClient::new(&shared_config);

        let force_path_style = config.environment != "prod" && config.environment != "stage";

        let s3_client = create_s3_client(&shared_config, force_path_style);
        let secrets_manager_client = SecretsManagerClient::new(&shared_config);

        Ok(Self {
            sqs_client,
            sns_client,
            s3_client,
            secrets_manager_client,
        })
    }
}
