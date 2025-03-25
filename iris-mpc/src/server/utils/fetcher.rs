use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sqs::Client as SQSClient;
use eyre::{Result, WrapErr};
use iris_mpc_common::helpers::sqs::get_next_sns_seq_num;
use iris_mpc_common::{
    config::Config,
    helpers::{
        key_pair::{SharesDecodingError, SharesEncryptionKeyPairs},
        sync::SyncState,
    },
};
use iris_mpc_store::Store;

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

pub(crate) async fn fetch_sync_state(
    config: &Config,
    iris_pg_store: &Store,
    sqs_client: &SQSClient,
    store_len: usize,
    max_sync_lookback: usize,
) -> Result<SyncState, String> {
    Ok(SyncState {
        db_len: store_len as u64,
        deleted_request_ids: iris_pg_store
            .last_deleted_requests(max_sync_lookback)
            .await
            .wrap_err("Failed to fetch last deleted requests")
            .unwrap(),
        modifications: iris_pg_store
            .last_modifications(max_sync_lookback)
            .await
            .wrap_err("Failed to fetch last set of modifications")
            .unwrap(),
        next_sns_sequence_num: get_next_sns_seq_num(&config, &sqs_client)
            .await
            .wrap_err("Failed to fetch next SQS sequence number")
            .unwrap(),
    })
}
