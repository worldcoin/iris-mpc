use alkali::{
    asymmetric::seal::{curve25519xsalsa20poly1305 as sealedbox, SealError},
    AlkaliError,
};
use aws_sdk_secretsmanager::{
    error::SdkError, operation::get_secret_value::GetSecretValueError,
    Client as SecretsManagerClient,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::{fmt, string::FromUtf8Error};
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

const CURRENT_SECRET_LABEL: &str = "AWSCURRENT";
const PREVIOUS_SECRET_LABEL: &str = "AWSPREVIOUS";

#[derive(Error, Debug)]
pub enum SharesDecodingError {
    #[error("Secrets Manager error: {0}")]
    SecretsManagerError(#[from] Box<SdkError<GetSecretValueError>>),
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
    #[error("Sealed box library error")]
    CryptoError(#[from] AlkaliError),
    #[error("Sealed box open error")]
    SealedBoxOpenError,
    #[error("Public key not found error")]
    PreviousKeyNotFound,
    #[error("Previous key not found error")]
    PublicKeyNotFound,
    #[error("Private key not found error")]
    PrivateKeyNotFound,
    #[error("Base64 decoding error")]
    Base64DecodeError,
    #[error("Received error message from server: [{}] {}", .status, .message)]
    ResponseContent {
        status: reqwest::StatusCode,
        url: String,
        message: String,
    },
    #[error("Received error message from S3 for key {}: {}", .key, .message)]
    S3ResponseContent { key: String, message: String },
    #[error(transparent)]
    SerdeError(#[from] serde_json::error::Error),
    #[error(transparent)]
    PresigningConfigError(#[from] aws_sdk_s3::presigning::PresigningConfigError),
    #[error(transparent)]
    PresignedRequestError(
        #[from] Box<aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>>,
    ),
    #[error("Upload share file error")]
    UploadS3Error,
}

impl From<SdkError<GetSecretValueError>> for SharesDecodingError {
    fn from(value: SdkError<GetSecretValueError>) -> Self {
        Self::SecretsManagerError(Box::new(value))
    }
}

impl From<aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>>
    for SharesDecodingError
{
    fn from(
        value: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>,
    ) -> Self {
        Self::PresignedRequestError(Box::new(value))
    }
}

/// Loaded once and shared by server tasks through `Arc`; secret bytes are not cloned.
/// Explicit zeroization requires exclusive access. Drop wipes the last owner's keys.
#[derive(Debug)]
pub struct SharesEncryptionKeyPairs {
    pub current_key_pair: SharesEncryptionKeyPair,
    pub previous_key_pair: Option<SharesEncryptionKeyPair>,
}

impl Zeroize for SharesEncryptionKeyPairs {
    fn zeroize(&mut self) {
        self.current_key_pair.zeroize();
        self.previous_key_pair.zeroize();
    }
}

impl Drop for SharesEncryptionKeyPairs {
    fn drop(&mut self) {
        self.current_key_pair.zeroize();
        self.previous_key_pair.zeroize();
    }
}

impl SharesEncryptionKeyPairs {
    pub async fn from_storage(
        client: SecretsManagerClient,
        environment: &str,
        party_id: &usize,
    ) -> Result<Self, SharesDecodingError> {
        let current_sk_b64_string = match download_private_key_from_asm(
            &client,
            environment,
            &party_id.to_string(),
            CURRENT_SECRET_LABEL,
        )
        .await
        {
            Ok(sk) => sk,
            Err(e) => return Err(e),
        };

        let previous_sk_b64_string = match download_private_key_from_asm(
            &client,
            environment,
            &party_id.to_string(),
            PREVIOUS_SECRET_LABEL,
        )
        .await
        {
            Ok(sk) => sk,
            Err(e) => return Err(e),
        };

        match SharesEncryptionKeyPairs::from_b64_private_key_strings(
            current_sk_b64_string,
            previous_sk_b64_string,
        ) {
            Ok(key_pairs) => Ok(key_pairs),
            Err(e) => Err(e),
        }
    }

    pub fn from_b64_private_key_strings(
        current_sk_b64_string: String,
        previous_sk_b64_string: String,
    ) -> Result<Self, SharesDecodingError> {
        let current_key_pair =
            SharesEncryptionKeyPair::from_b64_private_key_string(current_sk_b64_string)?;
        if previous_sk_b64_string.is_empty() {
            return Ok(SharesEncryptionKeyPairs {
                current_key_pair,
                previous_key_pair: None,
            });
        }

        let previous_key_pair =
            SharesEncryptionKeyPair::from_b64_private_key_string(previous_sk_b64_string)?;
        Ok(SharesEncryptionKeyPairs {
            current_key_pair,
            previous_key_pair: Some(previous_key_pair),
        })
    }
}

pub struct SharesEncryptionKeyPair {
    keypair: sealedbox::Keypair,
}

impl fmt::Debug for SharesEncryptionKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharesEncryptionKeyPair")
            .field("pk", &self.keypair.public_key)
            .finish_non_exhaustive()
    }
}

impl Zeroize for SharesEncryptionKeyPair {
    fn zeroize(&mut self) {
        self.keypair.public_key.zeroize();
        self.keypair.private_key.as_mut().zeroize();
    }
}

impl Drop for SharesEncryptionKeyPair {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl SharesEncryptionKeyPair {
    pub fn from_b64_private_key_string(sk: String) -> Result<Self, SharesDecodingError> {
        let sk = Zeroizing::new(sk);
        let sk_bytes = Zeroizing::new(STANDARD.decode(sk.as_bytes())?);
        if sk_bytes.len() != sealedbox::PRIVATE_KEY_LENGTH {
            return Err(SharesDecodingError::ParsingKeyError);
        }
        let private_key = sealedbox::PrivateKey::try_from(sk_bytes.as_slice())?;
        let public_key = private_key.public_key()?;
        Ok(Self {
            keypair: sealedbox::Keypair {
                public_key,
                private_key,
            },
        })
    }

    pub fn open_sealed_box(&self, code: Vec<u8>) -> Result<Vec<u8>, SharesDecodingError> {
        let plaintext_length = code
            .len()
            .checked_sub(sealedbox::OVERHEAD_LENGTH)
            .ok_or(SharesDecodingError::SealedBoxOpenError)?;
        let mut plaintext = Zeroizing::new(vec![0; plaintext_length]);
        sealedbox::decrypt(&code, &self.keypair, &mut plaintext).map_err(|error| {
            if error == AlkaliError::SealError(SealError::DecryptionFailed) {
                SharesDecodingError::SealedBoxOpenError
            } else {
                SharesDecodingError::CryptoError(error)
            }
        })?;
        Ok(std::mem::take(&mut *plaintext))
    }
}

async fn download_private_key_from_asm(
    client: &SecretsManagerClient,
    env: &str,
    node_id: &str,
    version_stage: &str,
) -> Result<String, SharesDecodingError> {
    let private_key_secret_id: String = format!("{}/iris-mpc/ecdh-private-key-{}", env, node_id);
    tracing::info!(
        "Downloading private key from Secrets Manager: {}",
        private_key_secret_id
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic native-libsodium fixture, generated with crypto_box_seed_keypair
    // from seed bytes 0..31 and crypto_box_seal (the sodiumoxide construction).
    const SECRET: &str = "PZTupJxYCu+BaTV2K+BJVZ1tFEDe3hLmoSXxhB//jm8=";
    const PUBLIC: &str = "RwHQhIhFH1RaQJ+1iuPlhYHKQKw/fxFGmM1x3qxzygE=";
    const CIPHERTEXT: &str = "tYslsdK1fan3ZTcJfhqRpv1McqHvOrkpkyVgxvbHc13vpTUfSy+nCvqcHZl/E8kJu40kKpk6Q5eWeMjpEwDbeFS8zvuJ/r6g4d4CT/shyp3l5R+2F2A/HyjZusSnwQ==";
    const MESSAGE: &[u8] = b"synthetic sodiumoxide-compatible share fixture";

    #[test]
    fn imports_existing_private_key_and_opens_native_sealed_box() {
        let key = SharesEncryptionKeyPair::from_b64_private_key_string(SECRET.into()).unwrap();
        assert_eq!(STANDARD.encode(key.keypair.public_key), PUBLIC);
        assert_eq!(
            key.open_sealed_box(STANDARD.decode(CIPHERTEXT).unwrap())
                .unwrap(),
            MESSAGE
        );
    }

    #[test]
    fn rejects_invalid_private_key_encoding_and_lengths() {
        assert!(matches!(
            SharesEncryptionKeyPair::from_b64_private_key_string("!".into()),
            Err(SharesDecodingError::DecodingError(_))
        ));
        for len in [0, 31, 33, 64] {
            assert!(matches!(
                SharesEncryptionKeyPair::from_b64_private_key_string(STANDARD.encode(vec![1; len])),
                Err(SharesDecodingError::ParsingKeyError)
            ));
        }
    }

    #[test]
    fn rejects_wrong_key_tampering_and_truncated_ciphertexts() {
        let key = SharesEncryptionKeyPair::from_b64_private_key_string(SECRET.into()).unwrap();
        let ciphertext = STANDARD.decode(CIPHERTEXT).unwrap();
        let wrong = SharesEncryptionKeyPair::from_b64_private_key_string(STANDARD.encode([42; 32]))
            .unwrap();
        assert!(matches!(
            wrong.open_sealed_box(ciphertext.clone()),
            Err(SharesDecodingError::SealedBoxOpenError)
        ));
        for i in 0..ciphertext.len() {
            let mut changed = ciphertext.clone();
            changed[i] ^= 1;
            assert!(matches!(
                key.open_sealed_box(changed),
                Err(SharesDecodingError::SealedBoxOpenError)
            ));
            assert!(matches!(
                key.open_sealed_box(ciphertext[..i].to_vec()),
                Err(SharesDecodingError::SealedBoxOpenError)
            ));
        }
    }

    #[test]
    fn explicit_zeroize_clears_hardened_key_and_debug_omits_secret() {
        let mut key = SharesEncryptionKeyPair::from_b64_private_key_string(SECRET.into()).unwrap();
        assert_eq!(
            format!("{key:?}"),
            format!(
                "SharesEncryptionKeyPair {{ pk: {:?}, .. }}",
                key.keypair.public_key
            )
        );
        key.zeroize();
        assert_eq!(key.keypair.public_key, [0; 32]);
        assert_eq!(&key.keypair.private_key[..], &[0; 32]);
    }

    #[tokio::test]
    async fn tasks_share_hardened_storage_until_the_last_owner_drops() {
        use std::sync::Arc;
        use tokio::sync::oneshot;

        let keys = Arc::new(
            SharesEncryptionKeyPairs::from_b64_private_key_strings(SECRET.into(), String::new())
                .unwrap(),
        );
        let weak = Arc::downgrade(&keys);
        let storage = keys.current_key_pair.keypair.private_key.as_ptr() as usize;
        let worker_keys = Arc::clone(&keys);
        let (start, wait) = oneshot::channel();
        let task = tokio::spawn(async move {
            wait.await.unwrap();
            assert_eq!(
                worker_keys.current_key_pair.keypair.private_key.as_ptr() as usize,
                storage
            );
            worker_keys
                .current_key_pair
                .open_sealed_box(STANDARD.decode(CIPHERTEXT).unwrap())
                .unwrap()
        });
        assert_eq!(Arc::strong_count(&keys), 2);
        drop(keys);
        assert!(weak.upgrade().is_some());
        start.send(()).unwrap();
        assert_eq!(task.await.unwrap(), MESSAGE);
        // The final task owner releases the key set; Drop clears its private buffers.
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn cancelling_a_task_releases_its_key_ownership() {
        use std::sync::Arc;
        use tokio::sync::oneshot;

        let keys = Arc::new(
            SharesEncryptionKeyPairs::from_b64_private_key_strings(SECRET.into(), String::new())
                .unwrap(),
        );
        let weak = Arc::downgrade(&keys);
        let worker_keys = Arc::clone(&keys);
        let (ready, started) = oneshot::channel();
        let task = tokio::spawn(async move {
            ready.send(()).unwrap();
            std::future::pending::<()>().await;
            drop(worker_keys);
        });
        started.await.unwrap();
        drop(keys);
        assert!(weak.upgrade().is_some());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn empty_plaintext_round_trips() {
        let key = SharesEncryptionKeyPair::from_b64_private_key_string(SECRET.into()).unwrap();
        let mut ciphertext = vec![0; sealedbox::OVERHEAD_LENGTH];
        sealedbox::encrypt(b"", &key.keypair.public_key, &mut ciphertext).unwrap();
        assert_eq!(key.open_sealed_box(ciphertext).unwrap(), b"");
    }
}
