use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use serde::ser::Serialize;
use serde_json;
use thiserror::Error;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::config::AwsClientConfig;
use crate::{
    constants::N_PARTIES,
    misc::{log_error, log_info},
    types::NetworkEncryptionPublicKeys,
};

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct AwsClient {
    /// Associated configuration.
    config: AwsClientConfig,

    /// Encryption public key set ... one per MPC node.
    encryption_keys: NetworkEncryptionPublicKeys,

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
pub type NetworkAwsClient = [AwsClient; N_PARTIES];

impl AwsClient {
    pub fn config(&self) -> &AwsClientConfig {
        &self.config
    }

    pub fn encryption_keys(&self) -> &NetworkEncryptionPublicKeys {
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

    pub fn new(config: AwsClientConfig, encryption_keys: NetworkEncryptionPublicKeys) -> Self {
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
        log_error("Utils-AWS", msg);
    }

    pub(super) fn log_info(&self, msg: &str) {
        log_info("Utils-AWS", msg);
    }

    /// Enqueues a message upon an AWS SNS service topic.
    pub async fn publish_to_sns<T>(
        &self,
        message_type: &str,
        message_group_id: &str,
        message_payload: T,
    ) -> Result<(), AwsClientError>
    where
        T: Sized + Serialize,
    {
        match self
            .sns()
            .clone()
            .publish()
            .topic_arn(self.config().request_topic_arn())
            .message_group_id(message_group_id)
            .message(serde_json::to_string(&message_payload).unwrap())
            .set_message_attributes(Some(create_sns_message_attributes(message_type)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Failed to publish data to SNS: {:?}", e);
                return Err(AwsClientError::SnsPublishError);
            }
        }
    }

    /// Enqueues data to an S3 bucket.
    pub async fn upload_to_s3(
        &self,
        s3_bucket: &str,
        s3_key: &str,
        payload: &[u8],
    ) -> Result<(), AwsClientError> {
        match self
            .s3()
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(ByteStream::new(SdkBody::from(payload)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                return Err(AwsClientError::S3UploadError {
                    key: s3_key.to_string(),
                    error: e.to_string(),
                });
            }
        }
    }
}

impl Clone for AwsClient {
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
pub enum AwsClientError {
    #[error("AWS SNS publish error")]
    SnsPublishError,

    #[error("AWS S3 upload error: key={}: error={}", .key, .error)]
    S3UploadError { key: String, error: String },
}

#[cfg(test)]
mod tests {
    use super::super::{AwsClient, AwsClientConfig};
    use crate::constants::{self, N_PARTIES};
    use sodiumoxide::crypto::box_::{gen_keypair, PublicKey};

    fn assert_clients(clients: &AwsClient) {
        assert!(clients.s3().config().region().is_some());
        assert!(clients.secrets_manager().config().region().is_some());
        assert!(clients.sns().config().region().is_some());
        assert!(clients.sqs().config().region().is_some());
    }

    async fn create_client() -> AwsClient {
        AwsClient::new(create_config().await, create_public_keys_for_encryption())
    }

    fn create_public_keys_for_encryption() -> [PublicKey; N_PARTIES] {
        std::array::from_fn(|_| gen_keypair().0)
    }

    async fn create_config() -> AwsClientConfig {
        AwsClientConfig::new(
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
