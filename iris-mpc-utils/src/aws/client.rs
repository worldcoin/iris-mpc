use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use thiserror::Error;

use iris_mpc_common::helpers::sqs_s3_helper;

use super::config::NodeAwsClientConfig;
use crate::{
    constants::N_PARTIES,
    misc::{log_error, log_info},
    types::NetEncryptionPublicKeys,
};

/// Component name for logging purposes.
const COMPONENT: &str = "State-AWS";

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct NodeAwsClient {
    /// Associated configuration.
    config: NodeAwsClientConfig,

    /// Encryption public key set ... one per MPC node.
    encryption_keys: NetEncryptionPublicKeys,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    secrets_manager: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs: SQSClient,
}

// Network wide node AWS service clients.
pub type NetAwsClient = [NodeAwsClient; N_PARTIES];

impl NodeAwsClient {
    pub fn config(&self) -> &NodeAwsClientConfig {
        &self.config
    }

    pub fn encryption_keys(&self) -> &NetEncryptionPublicKeys {
        &self.encryption_keys
    }

    pub fn s3(&self) -> &S3Client {
        &self.s3
    }

    pub fn secrets_manager(&self) -> &SecretsManagerClient {
        &self.secrets_manager
    }

    pub fn sns(&self) -> &SNSClient {
        &self.sns
    }

    pub fn sqs(&self) -> &SQSClient {
        &self.sqs
    }

    pub fn new(config: NodeAwsClientConfig, encryption_keys: NetEncryptionPublicKeys) -> Self {
        Self {
            config: config.to_owned(),
            encryption_keys,
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }

    pub(super) fn log_error(&self, msg: &str) {
        log_error(COMPONENT, msg);
    }

    pub(super) fn log_info(&self, msg: &str) {
        log_info(COMPONENT, msg);
    }

    pub async fn upload_to_s3(
        &self,
        bucket: &str,
        key: &str,
        payload: &[u8],
    ) -> Result<(), NodeAwsClientError> {
        match sqs_s3_helper::upload_file_to_s3(bucket, key, self.s3().clone(), payload).await {
            Err(_) => Err(NodeAwsClientError::S3UploadFailed),
            Ok(_) => Ok(()),
        }
    }
}

impl Clone for NodeAwsClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            encryption_keys: self.encryption_keys,
            sqs: self.sqs.clone(),
            sns: self.sns.clone(),
            s3: self.s3.clone(),
            secrets_manager: self.secrets_manager.clone(),
        }
    }
}

#[derive(Error, Debug)]
pub enum NodeAwsClientError {
    #[error("AWS S3 file upload error")]
    S3UploadFailed,
}

#[cfg(test)]
mod tests {
    use super::super::{NodeAwsClient, NodeAwsClientConfig};
    use crate::constants::{self, N_PARTIES};
    use sodiumoxide::crypto::box_::{gen_keypair, PublicKey};

    fn assert_clients(clients: &NodeAwsClient) {
        assert!(clients.s3().config().region().is_some());
        assert!(clients.secrets_manager().config().region().is_some());
        assert!(clients.sns().config().region().is_some());
        assert!(clients.sqs().config().region().is_some());
    }

    async fn create_client() -> NodeAwsClient {
        NodeAwsClient::new(create_config().await, create_public_keys_for_encryption())
    }

    fn create_public_keys_for_encryption() -> [PublicKey; N_PARTIES] {
        std::array::from_fn(|_| gen_keypair().0)
    }

    async fn create_config() -> NodeAwsClientConfig {
        NodeAwsClientConfig::new(
            constants::DEFAULT_ENV.to_string(),
            constants::AWS_REGION.to_string(),
            constants::AWS_REQUEST_BUCKET_NAME.to_string(),
            constants::AWS_REQUEST_TOPIC_ARN.to_string(),
            constants::AWS_RESPONSE_QUEUE_URL.to_string(),
            constants::AWS_SQS_LONG_POLL_WAIT_TIME,
        )
        .await
    }

    #[tokio::test]
    async fn test_client_new() {
        let clients = create_client().await;
        assert_clients(&clients);
    }

    #[tokio::test]
    async fn test_client_clone() {
        let clients = create_client().await.clone();
        assert_clients(&clients);
    }
}
