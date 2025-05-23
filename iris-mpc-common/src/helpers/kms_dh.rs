use aws_sdk_kms::{types::KeyAgreementAlgorithmSpec, Client};
use eyre::Result;

/// Derive a shared secret from two KMS keys
pub async fn derive_shared_secret(own_key_arn: &str, other_key_arn: &str) -> Result<[u8; 32]> {
    let shared_config = aws_config::from_env().load().await;

    let client = Client::new(&shared_config);
    let other_public_key = client.get_public_key().key_id(other_key_arn).send().await?;
    let public_key = other_public_key.public_key.unwrap();

    let res = client
        .derive_shared_secret()
        .key_id(own_key_arn)
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
