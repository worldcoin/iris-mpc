use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use eyre::Result;
use iris_mpc_common::{
    config::Config,
    helpers::key_pair::{SharesDecodingError, SharesEncryptionKeyPairs},
};

pub(crate) async fn fetch_shares_encryption_key_pair(
    config: &Config,
    sm_client: SecretsManagerClient,
) -> Result<SharesEncryptionKeyPairs, SharesDecodingError> {
    match SharesEncryptionKeyPairs::from_storage(sm_client, &config.environment, &config.party_id)
        .await
    {
        Ok(key_pair) => Ok(key_pair),
        Err(e) => {
            tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
            Err(e)
        }
    }
}

pub(crate) fn get_check_addresses(
    hostnames: Vec<String>,
    ports: Vec<String>,
    endpoint: &str,
) -> Vec<String> {
    hostnames
        .iter()
        .zip(ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host, port, endpoint))
        .collect::<Vec<String>>()
}
