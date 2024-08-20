use crate::config::Config;
use aws_config::Region;
use aws_sdk_secretsmanager::{
    error::SdkError, operation::get_secret_value::GetSecretValueError,
    Client as SecretsManagerClient,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use sodiumoxide::crypto::{
    box_::{PublicKey, SecretKey},
    sealedbox,
};
use std::string::FromUtf8Error;
use thiserror::Error;
use zeroize::Zeroize;

const REGION: &str = "eu-north-1";
const CURRENT_SECRET_LABEL: &str = "AWSCURRENT";

#[derive(Error, Debug)]
pub enum SharesDecodingError {
    #[error("Secrets Manager error: {0}")]
    SecretsManagerError(#[from] SdkError<GetSecretValueError>),
    #[error("Secret string not found")]
    SecretStringNotFound,
    #[error(transparent)]
    RequestError(#[from] reqwest::Error),
    #[error("Decoding error: {0}")]
    DecodingError(#[from] base64::DecodeError),
    #[error("Parsing bytes to UTF8 error")]
    DecodedShareParsingToUTF8Error(#[from] FromUtf8Error),
    #[error("Parsing key error")]
    ParsingKeyError,
    #[error("Sealed box open error")]
    SealedBoxOpenError,
    #[error("Public key not found error")]
    PublicKeyNotFound,
    #[error("Private key not found error")]
    PrivateKeyNotFound,
    #[error("Base64 decoding error")]
    Base64DecodeError,
    #[error("Received error message from server: [{}] {}", .status, .message)]
    ResponseContent {
        status:  reqwest::StatusCode,
        url:     String,
        message: String,
    },
    #[error(transparent)]
    SerdeError(#[from] serde_json::error::Error),
    #[error(transparent)]
    PresigningConfigError(#[from] aws_sdk_s3::presigning::PresigningConfigError),
    #[error(transparent)]
    PresignedRequestError(
        #[from] aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>,
    ),
}

#[derive(Clone, Debug)]
pub struct SharesEncryptionKeyPair {
    pk: PublicKey,
    sk: SecretKey,
}

impl Zeroize for SharesEncryptionKeyPair {
    fn zeroize(&mut self) {
        self.pk.0.zeroize();
        self.sk.0.zeroize();
    }
}

impl Drop for SharesEncryptionKeyPair {
    fn drop(&mut self) {
        self.pk.0.zeroize();
        self.sk.0.zeroize();
    }
}

impl SharesEncryptionKeyPair {
    pub async fn from_storage(config: Config) -> Result<Self, SharesDecodingError> {
        let region_provider = Region::new(REGION);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let client = SecretsManagerClient::new(&shared_config);

        let pk_b64_string = match download_public_key(
            config.public_key_base_url,
            config.party_id.to_string(),
        )
        .await
        {
            Ok(pk) => pk,
            Err(e) => return Err(e),
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
            Err(e) => return Err(e),
        };

        match SharesEncryptionKeyPair::from_b64_strings(pk_b64_string, sk_b64_string) {
            Ok(key_pair) => Ok(key_pair),
            Err(e) => Err(e),
        }
    }

    pub fn from_b64_strings(pk: String, sk: String) -> Result<Self, SharesDecodingError> {
        let pk_bytes = match STANDARD.decode(pk) {
            Ok(bytes) => bytes,
            Err(e) => return Err(SharesDecodingError::DecodingError(e)),
        };
        let sk_bytes = match STANDARD.decode(sk) {
            Ok(bytes) => bytes,
            Err(e) => return Err(SharesDecodingError::DecodingError(e)),
        };

        let pk = match PublicKey::from_slice(&pk_bytes) {
            Some(pk) => pk,
            None => return Err(SharesDecodingError::ParsingKeyError),
        };
        let sk = match SecretKey::from_slice(&sk_bytes) {
            Some(sk) => sk,
            None => return Err(SharesDecodingError::ParsingKeyError),
        };

        Ok(Self { pk, sk })
    }

    pub fn open_sealed_box(&self, code: Vec<u8>) -> Result<Vec<u8>, SharesDecodingError> {
        let decrypted = sealedbox::open(&code, &self.pk, &self.sk);
        match decrypted {
            Ok(bytes) => Ok(bytes),
            Err(_) => Err(SharesDecodingError::SealedBoxOpenError),
        }
    }
}

async fn download_private_key_from_asm(
    client: &SecretsManagerClient,
    env: &str,
    node_id: &str,
    version_stage: &str,
) -> Result<String, SharesDecodingError> {
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
            None => Err(SharesDecodingError::SecretStringNotFound),
        },
        Err(e) => Err(e.into()),
    }
}

pub async fn download_public_key(
    base_url: String,
    node_id: String,
) -> Result<String, SharesDecodingError> {
    let client = reqwest::Client::new();
    let url: String = format!("{}/public-key-{}", base_url, node_id);
    let response = client.get(url.clone()).send().await;
    match response {
        Ok(response) => {
            if response.status().is_success() {
                let body = response.text().await;
                match body {
                    Ok(body) => Ok(body),
                    Err(e) => Err(SharesDecodingError::RequestError(e)),
                }
            } else {
                Err(SharesDecodingError::ResponseContent {
                    status: response.status(),
                    message: response.text().await.unwrap_or_default(),
                    url,
                })
            }
        }
        Err(e) => Err(SharesDecodingError::RequestError(e)),
    }
}
