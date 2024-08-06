use crate::config::Config;
use aws_config::Region;
use aws_sdk_secretsmanager::{
    error::SdkError, operation::get_secret_value::GetSecretValueError,
    Client as SecretsManagerClient,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use sodiumoxide::crypto::box_::{PublicKey, SecretKey};
use thiserror::Error;

const REGION: &str = "eu-north-1";
const CURRENT_SECRET_LABEL: &str = "AWSCURRENT";

#[derive(Error, Debug)]
pub enum SecretError {
    #[error("Secrets Manager error: {0}")]
    SecretsManagerError(#[from] SdkError<GetSecretValueError>),
    #[error("Secret string not found")]
    SecretStringNotFound,
    #[error("Public key not found")]
    PublicKeyNotFound(#[from] reqwest::Error),
    #[error("Decoding error: {0}")]
    DecodingError(#[from] base64::DecodeError),
}

#[derive(Clone, Debug)]
pub struct SharesEncryptionKeyPair {
    pub pk: PublicKey,
    pub sk: SecretKey,
}

impl SharesEncryptionKeyPair {
    pub async fn initialize_from_storage(config: Config) -> Self {
        let region_provider = Region::new(REGION);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let client = SecretsManagerClient::new(&shared_config);

        let pk_b64_string = match download_public_key_from_s3(
            config.public_key_bucket_name,
            config.party_id.to_string(),
        )
        .await
        {
            Ok(pk) => pk,
            Err(e) => panic!(
                "Failed to download public key from AWS Secrets Manager: {}",
                e
            ),
        };

        let sk_b64_string = match download_private_key_from_asm(
            &client,
            &config.environment,
            &config.party_id.to_string(),
            CURRENT_SECRET_LABEL,
        )
        .await
        {
            Ok(sk) => sk,
            Err(e) => panic!(
                "Failed to download private key from AWS Secrets Manager: {}",
                e
            ),
        };

        SharesEncryptionKeyPair::from_b64_strings(pk_b64_string, sk_b64_string)
    }

    pub fn from_b64_strings(pk: String, sk: String) -> Self {
        let pk_bytes = match STANDARD.decode(pk) {
            Ok(bytes) => bytes,
            Err(_) => panic!("Invalid base64 string for public key"),
        };
        let sk_bytes = match STANDARD.decode(sk) {
            Ok(bytes) => bytes,
            Err(_) => panic!("Invalid base64 string for secret key"),
        };

        let pk = match PublicKey::from_slice(&pk_bytes) {
            Some(pk) => pk,
            None => panic!("Invalid public key"),
        };
        let sk = match SecretKey::from_slice(&sk_bytes) {
            Some(sk) => sk,
            None => panic!("Invalid secret key"),
        };

        Self { pk, sk }
    }
}

pub async fn download_private_key_from_asm(
    client: &SecretsManagerClient,
    env: &str,
    node_id: &str,
    version_stage: &str,
) -> Result<String, SecretError> {
    let private_key_secret_id: String =
        format!("{}/gpu-iris-mpc/ecdh-private-key-{}", env, node_id);
    match client
        .get_secret_value()
        .secret_id(private_key_secret_id)
        .version_stage(version_stage)
        .send()
        .await
    {
        Ok(secret_key_output) => match secret_key_output.secret_string {
            Some(data) => Ok(data),
            None => Err(SecretError::SecretStringNotFound),
        },
        Err(e) => Err(e.into()),
    }
}

pub async fn download_public_key_from_s3(
    bucket_name: String,
    node_id: String,
) -> Result<String, SecretError> {
    let client = reqwest::Client::new();
    // TODO: remove coupling to S3
    let url: String = format!(
        "https://{}.s3.amazonaws.com/public-key-{}",
        bucket_name, node_id
    );
    let response = client.get(url).send().await;
    match response {
        Ok(response) => {
            let body = response.text().await;
            match body {
                Ok(body) => Ok(body),
                Err(e) => Err(SecretError::PublicKeyNotFound(e)),
            }
        }
        Err(e) => Err(SecretError::PublicKeyNotFound(e)),
    }
}
