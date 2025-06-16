use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::aws::clients::AwsClients;
use crate::services::processors::get_iris_shares_parse_task;
use eyre::{eyre, Report};
use iris_mpc_common::config::Config;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::sync::{Modification, SyncResult};
use iris_mpc_store::{Store, StoredIrisRef};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub async fn sync_modifications(
    config: &Config,
    store: &Store,
    aws_clients: &AwsClients,
    shares_encryption_key_pair: &SharesEncryptionKeyPairs,
    sync_result: SyncResult,
    dummy_shares_for_deletions: (GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare),
) -> eyre::Result<(), Report> {
    let (mut to_update, to_delete) = sync_result.compare_modifications();
    tracing::info!(
        "Modifications to update: {:?}, to delete: {:?}",
        to_update,
        to_delete
    );
    // Update node_id in each modification because they are coming from another more advanced node
    let to_update: Vec<&Modification> = to_update
        .iter_mut()
        .map(|modification| {
            if let Err(e) = modification.update_result_message_node_id(config.party_id) {
                tracing::error!("Failed to update modification node_id: {:?}", e);
            }
            &*modification
        })
        .collect();

    let mut tx = store.tx().await?;
    store.update_modifications(&mut tx, &to_update).await?;
    store.delete_modifications(&mut tx, &to_delete).await?;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    // update irises table for persisted modifications which are missing in local
    for modification in to_update {
        if !modification.persisted {
            tracing::debug!(
                "Skip writing non-persisted modification to iris table: {:?}",
                modification
            );
            continue;
        }
        tracing::warn!("Applying modification to local node: {:?}", modification);
        metrics::counter!("db.modifications.rollforward").increment(1);
        let (lc, lm, rc, rm) = match modification.request_type.as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => (
                dummy_shares_for_deletions.clone().0,
                dummy_shares_for_deletions.clone().1,
                dummy_shares_for_deletions.clone().0,
                dummy_shares_for_deletions.clone().1,
            ),
            REAUTH_MESSAGE_TYPE | RESET_UPDATE_MESSAGE_TYPE | UNIQUENESS_MESSAGE_TYPE => {
                let (left_shares, right_shares) = get_iris_shares_parse_task(
                    config.party_id,
                    shares_encryption_key_pair.clone(),
                    Arc::clone(&semaphore),
                    aws_clients.s3_client.clone(),
                    config.shares_bucket_name.clone(),
                    modification.clone().s3_url.unwrap(),
                )?
                .await?
                .unwrap();
                (
                    left_shares.code,
                    left_shares.mask,
                    right_shares.code,
                    right_shares.mask,
                )
            }
            _ => {
                panic!("Unknown modification type: {:?}", modification);
            }
        };
        let iris_ref = StoredIrisRef {
            id: modification
                .serial_id
                .ok_or_else(|| eyre!("Modification has no serial_id: {:?}", modification))?,
            left_code: &lc.coefs,
            left_mask: &lm.coefs,
            right_code: &rc.coefs,
            right_mask: &rm.coefs,
        };
        store.insert_irises_overriding(&mut tx, &[iris_ref]).await?;
    }
    tx.commit().await?;
    Ok(())
}
