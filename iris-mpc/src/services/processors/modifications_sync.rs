use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::aws::clients::AwsClients;
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_results_to_sns;
use ampc_server_utils::modifications::{recovery_plan, replay_modification_results};
use aws_sdk_sns::Client as SNSClient;
use eyre::{eyre, Report};
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RECOVERY_CHECK_MESSAGE_TYPE,
    RECOVERY_UPDATE_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::create_message_type_attribute_map;
use iris_mpc_common::helpers::sync::{Modification, SyncResult};
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_store::{ExplicitVersionToken, Store, StoredIrisRef};
use sqlx::{Postgres, Transaction};
use std::sync::Arc;
use tokio::sync::Semaphore;

// Graph mutations are already in the WAL (hawk_graph_mutations table) from when
// they were originally committed, keyed by modification_id. Actual mutation application
// to GraphMem happens during startup delta replay (checkpoint load → WAL replay), not here.
/// Rolls modifications forward; the caller must commit the returned transaction.
pub async fn sync_modifications<'a>(
    config: &Config,
    store: &'a Store,
    aws_clients: &AwsClients,
    shares_encryption_key_pair: &Arc<SharesEncryptionKeyPairs>,
    sync_result: SyncResult,
) -> eyre::Result<Transaction<'a, Postgres>, Report> {
    let all = sync_result
        .all_states
        .iter()
        .map(|state| state.modifications.clone())
        .collect::<Vec<_>>();
    let mut plan = recovery_plan(
        &sync_result.my_state.modifications,
        &all,
        sync_result
            .my_state
            .common_config
            .get_max_modifications_lookback(),
    )
    .expect("Inconsistent modification snapshots or lookback");
    tracing::info!(
        "Modifications to update: {:?}, to delete: {:?}",
        plan.updates,
        plan.deletes
    );

    let dummy_shares_for_deletions = get_dummy_shares_for_deletion(config.party_id);

    // Update node_id for each modification and collect &refs
    let to_update_refs: Vec<&Modification> = plan
        .updates
        .iter_mut()
        .map(|update| {
            let modification = &mut update.modification;
            if let Err(e) = modification.update_result_message_node_id(config.party_id) {
                tracing::error!("Failed to update modification node_id: {:?}", e);
            }
            &*modification
        })
        .collect();

    let mut iris_tx = store.tx().await?;

    // Persist changes into modifications table
    store
        .update_modifications(&mut iris_tx, &to_update_refs)
        .await?;
    store
        .delete_modifications(&mut iris_tx, &plan.deletes)
        .await?;

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    // Live result persistence (`process_job_result`) advances `version_id`
    // explicitly for every accepted reauth, identity update, and deletion, so
    // that idempotent writes (dummy -> dummy deletions, identical reset shares)
    // keep Postgres in step with the actor registry. Replay must advance the
    // version the same way; otherwise a node that rolls a modification forward
    // at startup ends up one version behind the nodes that applied it live.
    // `SET LOCAL` scopes the mode to this transaction; the only other `irises`
    // write below is the uniqueness insert, which the trigger never sees.
    let mut version_tx = ExplicitVersionToken::enable(&mut iris_tx).await?;

    // Persist changes into iris and graph tables
    for update in &plan.updates {
        let modification = &update.modification;
        if !update.apply_mutation {
            tracing::debug!(
                "Skip writing already applied or non-persisted modification to iris table: {:?}",
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
            REAUTH_MESSAGE_TYPE
            | RESET_UPDATE_MESSAGE_TYPE
            | RECOVERY_UPDATE_MESSAGE_TYPE
            | UNIQUENESS_MESSAGE_TYPE => {
                let s3_url = modification.s3_url.clone().ok_or_else(|| {
                    eyre!("Persisted modification missing s3_url: {:?}", modification)
                })?;
                let (left_shares, right_shares) = get_iris_shares_parse_task(
                    config.party_id,
                    shares_encryption_key_pair.clone(),
                    Arc::clone(&semaphore),
                    aws_clients.s3_client.clone(),
                    config.shares_bucket_name.clone(),
                    s3_url,
                )?
                .await??;
                (
                    left_shares.code,
                    left_shares.mask,
                    right_shares.code,
                    right_shares.mask,
                )
            }
            _ => {
                return Err(eyre!("Unknown modification type: {:?}", modification));
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

        if modification.request_type == UNIQUENESS_MESSAGE_TYPE {
            // A new identity: insert the row (version 0), or overwrite an
            // identical partial insert.
            store
                .insert_irises_overriding(version_tx.tx(), &[iris_ref])
                .await?;
        } else {
            // A mutation of an existing identity: advance the version exactly
            // like the live path did on the other nodes.
            let updated = store
                .update_iris_ref_and_increment_version(&mut version_tx, &iris_ref)
                .await?;
            if !updated {
                tracing::warn!(
                    "Modification {:?} targets serial id {} without a Postgres row; inserting it",
                    modification,
                    iris_ref.id
                );
                store
                    .insert_irises_overriding(version_tx.tx(), &[iris_ref])
                    .await?;
            }
        }
    }
    drop(version_tx);

    Ok(iris_tx)
}

pub async fn send_last_modifications_to_sns(
    store: &Store,
    sns_client: &SNSClient,
    config: &Config,
    lookback: usize,
) -> eyre::Result<()> {
    let last_modifications = store.last_modifications(lookback).await?;
    tracing::info!(
        "Replaying last {} modification results to SNS",
        last_modifications.len()
    );
    let order = [
        UNIQUENESS_MESSAGE_TYPE,
        IDENTITY_DELETION_MESSAGE_TYPE,
        REAUTH_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE,
        RESET_CHECK_MESSAGE_TYPE,
        RECOVERY_CHECK_MESSAGE_TYPE,
        RECOVERY_UPDATE_MESSAGE_TYPE,
    ];
    replay_modification_results(
        &last_modifications,
        &order,
        |request_type, bodies| async move {
            let attributes = create_message_type_attribute_map(request_type);
            send_results_to_sns(
                bodies,
                &Vec::new(),
                sns_client,
                config,
                &attributes,
                request_type,
            )
            .await
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ampc_server_utils::modifications::{
        compare_modifications, requires_modification_apply as requires_iris_update,
    };
    use iris_mpc_common::helpers::sync::MOD_STATUS_COMPLETED;
    use iris_mpc_common::helpers::sync::MOD_STATUS_IN_PROGRESS;

    #[test]
    fn completed_metadata_repair_does_not_replay_older_uniqueness_shares() {
        let older = Modification {
            id: 1,
            request_type: UNIQUENESS_MESSAGE_TYPE.into(),
            status: MOD_STATUS_COMPLETED.into(),
            persisted: true,
            ..Default::default()
        };
        let newer = Modification {
            id: 2,
            serial_id: Some(100),
            request_type: REAUTH_MESSAGE_TYPE.into(),
            ..older.clone()
        };
        let repaired = Modification {
            serial_id: Some(100),
            ..older.clone()
        };
        let local = vec![older.clone(), newer.clone()];
        let (updates, _) =
            compare_modifications(&local, &[local.clone(), vec![repaired, newer]]).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].serial_id, Some(100));
        assert!(!requires_iris_update(&updates[0], Some(&older)));
    }

    #[test]
    fn applies_gallery_only_when_the_persisted_mutation_is_missing_locally() {
        let modification = Modification {
            persisted: true,
            ..Default::default()
        };
        for status in [MOD_STATUS_IN_PROGRESS, MOD_STATUS_COMPLETED] {
            let local = Modification {
                status: status.into(),
                persisted: false,
                ..Default::default()
            };
            assert!(requires_iris_update(&modification, Some(&local)));
        }
        assert!(requires_iris_update(&modification, None));
        assert!(!requires_iris_update(&Modification::default(), None));
    }
}
