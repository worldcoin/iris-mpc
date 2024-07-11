use aws_sdk_kms::{types::KeyAgreementAlgorithmSpec, Client};
use base64::{prelude::BASE64_STANDARD, Engine};
use eyre::ContextCompat;

/// Derive a shared secret from two KMS keys
pub async fn derive_shared_secret(own_key_id: &str, other_key_id: &str) -> eyre::Result<[u8; 32]> {
    let shared_config = aws_config::from_env().load().await;

    let client = Client::new(&shared_config);
    let other_public_key = client.get_public_key().key_id(other_key_id).send().await?;
    let public_key = other_public_key.public_key.unwrap();

    let res = client
        .derive_shared_secret()
        .key_id(own_key_id)
        .public_key(public_key)
        .key_agreement_algorithm(KeyAgreementAlgorithmSpec::Ecdh)
        .send()
        .await?;

    let derived_shared_secret = res.shared_secret();
    let unwrapped_secret = derived_shared_secret
        .expect("Expected derived shared secret from KMS")
        .clone()
        .into_inner();

    let mut array = [0u8; 32];
    array.copy_from_slice(&unwrapped_secret);

    Ok(array)
}

async fn get_public_key(key_id: &str) -> eyre::Result<String> {
    let shared_config = aws_config::from_env().load().await;

    let client = Client::new(&shared_config);

    let res = client.get_public_key().key_id(key_id).send().await?;

    Ok(BASE64_STANDARD.encode(res.public_key().context("No public key found")?))
}
