use crate::server::MAX_CONCURRENT_REQUESTS;
use crate::services::aws::clients::AwsClients;
use crate::services::processors::get_iris_shares_parse_task;
use crate::services::processors::result_message::send_results_to_sns;
use aws_sdk_sns::Client as SNSClient;
use bincode;
use eyre::{eyre, Report};
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RECOVERY_CHECK_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::create_message_type_attribute_map;
use iris_mpc_common::helpers::sync::{Modification, SyncResult};
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_cpu::execution::hawk_main::{GraphStore, HawkMutation, SingleHawkMutation};
use iris_mpc_store::{Store, StoredIrisRef};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub async fn sync_modifications(
    config: &Config,
    store: &Store,
    graph_store: Option<&GraphStore>,
    aws_clients: &AwsClients,
    shares_encryption_key_pair: &SharesEncryptionKeyPairs,
    sync_result: SyncResult,
) -> eyre::Result<(), Report> {
    let (mut to_update, to_delete) = sync_result.compare_modifications();
    tracing::info!(
        "Modifications to update: {:?}, to delete: {:?}",
        to_update,
        to_delete
    );

    let dummy_shares_for_deletions = get_dummy_shares_for_deletion(config.party_id);

    // Sort modifications in id order
    to_update.sort_by_key(|m| m.id);

    // Update node_id for each modification (mutable pass)
    for modification in &mut to_update {
        if let Err(e) = modification.update_result_message_node_id(config.party_id) {
            tracing::error!("Failed to update modification node_id: {:?}", e);
        }
    }

    // Prefetch shares in bounded batches before taking the modification lock.
    // This avoids holding `RERAND_MODIFY_LOCK` across remote I/O while also
    // preventing unbounded memory growth if `to_update` is large.
    const PREFETCH_BATCH_SIZE: usize = 128;

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    // Apply deletions even if there are no updates.
    if !to_delete.is_empty() {
        let mut iris_tx = store.tx().await?;
        store.delete_modifications(&mut iris_tx, &to_delete).await?;
        iris_tx.commit().await?;
    }

    if to_update.is_empty() {
        return Ok(());
    }

    // Ensure all modification rows exist locally before the update loop.
    // Recovered modifications (completed on peers but missing locally) need
    // to be inserted. We set persisted=false so the batch loop below fetches
    // shares from S3 and writes them to iris; only then does
    // update_modifications mark persisted=true after the iris write succeeds.
    {
        let mut tx = store.tx().await?;
        for m in &to_update {
            let mut staging = m.clone();
            staging.persisted = false;
            store
                .upsert_recovered_modification(&mut tx, &staging)
                .await?;
        }
        tx.commit().await?;
    }

    for batch in to_update.chunks(PREFETCH_BATCH_SIZE) {
        struct PrefetchedShareData {
            serial_id: i64,
            left_code: Vec<u16>,
            left_mask: Vec<u16>,
            right_code: Vec<u16>,
            right_mask: Vec<u16>,
        }

        // Kick off S3 fetches concurrently for this batch.
        let fetched: Vec<Option<PrefetchedShareData>> = futures::future::try_join_all(
            batch.iter().map(|modification| {
                let semaphore = Arc::clone(&semaphore);
                let s3_client = aws_clients.s3_client.clone();
                let bucket_name = config.shares_bucket_name.clone();
                let party_id = config.party_id;
                let shares_encryption_key_pair = shares_encryption_key_pair.clone();
                let dummy_shares_for_deletions = dummy_shares_for_deletions.clone();
                async move {
                    if !modification.persisted {
                        return Ok::<_, Report>(None);
                    }

                    tracing::warn!("Applying modification to local node: {:?}", modification);
                    metrics::counter!("db.modifications.rollforward").increment(1);

                    let serial_id = modification.serial_id.ok_or_else(|| {
                        eyre!("Modification has no serial_id: {:?}", modification)
                    })?;

                    let (left_code, left_mask, right_code, right_mask) =
                        match modification.request_type.as_str() {
                            IDENTITY_DELETION_MESSAGE_TYPE => (
                                dummy_shares_for_deletions.0.coefs.to_vec(),
                                dummy_shares_for_deletions.1.coefs.to_vec(),
                                dummy_shares_for_deletions.0.coefs.to_vec(),
                                dummy_shares_for_deletions.1.coefs.to_vec(),
                            ),
                        REAUTH_MESSAGE_TYPE | RESET_UPDATE_MESSAGE_TYPE | UNIQUENESS_MESSAGE_TYPE => {
                            let s3_url = modification.s3_url.clone().ok_or_else(|| {
                                eyre!("Persisted modification missing s3_url: {:?}", modification)
                            })?;
                                let (left_shares, right_shares) = get_iris_shares_parse_task(
                                    party_id,
                                    shares_encryption_key_pair.clone(),
                                    Arc::clone(&semaphore),
                                    s3_client.clone(),
                                    bucket_name.clone(),
                                    s3_url,
                                )?
                                .await??;
                                (
                                    left_shares.code.coefs.to_vec(),
                                    left_shares.mask.coefs.to_vec(),
                                    right_shares.code.coefs.to_vec(),
                                    right_shares.mask.coefs.to_vec(),
                                )
                            }
                            _ => {
                            return Err(eyre!("Unknown modification type: {:?}", modification));
                            }
                        };

                    Ok(Some(PrefetchedShareData {
                        serial_id,
                        left_code,
                        left_mask,
                        right_code,
                        right_mask,
                    }))
                }
            }),
        )
        .await?;

        // Decode graph mutations (small) outside the DB lock window.
        let mut batch_graph_mutations = Vec::new();
        for modification in batch.iter().filter(|m| m.persisted) {
            if let Some(serialized) = &modification.graph_mutation {
                let single_mutation: SingleHawkMutation = bincode::deserialize::<SingleHawkMutation>(serialized)
                    .map_err(|e| eyre!("Failed to deserialize SingleHawkMutation: {}", e))?;
                batch_graph_mutations.push(single_mutation.clone());
            }
        }

        // Now acquire the modification lock and write this batch atomically.
        let mut iris_tx = store.tx().await?;
        iris_mpc_store::rerand::acquire_modify_lock(&mut iris_tx).await?;

        let batch_refs: Vec<&Modification> = batch.iter().collect();
        store.update_modifications(&mut iris_tx, &batch_refs).await?;

        let prefetched_rows: Vec<PrefetchedShareData> = fetched.into_iter().flatten().collect();
        if !prefetched_rows.is_empty() {
            let iris_refs: Vec<StoredIrisRef<'_>> = prefetched_rows
                .iter()
                .map(|row| StoredIrisRef {
                    id: row.serial_id,
                    left_code: &row.left_code,
                    left_mask: &row.left_mask,
                    right_code: &row.right_code,
                    right_mask: &row.right_mask,
                })
                .collect();
            store
                .insert_irises_overriding(&mut iris_tx, &iris_refs)
                .await?;
        }

        if let Some(graph_store) = graph_store {
            let mut graph_tx = graph_store.tx_wrap(iris_tx);
            if !batch_graph_mutations.is_empty() {
                let hawk_mutation = HawkMutation(batch_graph_mutations);
                hawk_mutation.persist(&mut graph_tx).await?;
            }
            graph_tx.tx.commit().await?;
        } else {
            iris_tx.commit().await?;
        }
    }

    Ok(())
}

pub async fn send_last_modifications_to_sns(
    store: &Store,
    sns_client: &SNSClient,
    config: &Config,
    lookback: usize,
) -> eyre::Result<()> {
    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_message_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_update_message_attributes =
        create_message_type_attribute_map(RESET_UPDATE_MESSAGE_TYPE);
    let deletion_message_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);
    let reset_check_message_attributes =
        create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let recovery_check_message_attributes =
        create_message_type_attribute_map(RECOVERY_CHECK_MESSAGE_TYPE);

    // Fetch the last modifications from the database
    let last_modifications = store.last_modifications(lookback).await?;
    tracing::info!(
        "Replaying last {} modification results to SNS",
        last_modifications.len()
    );

    if last_modifications.is_empty() {
        tracing::info!("No last modifications found to send to SNS");
        return Ok(());
    }

    // Collect messages by type
    let mut deletion_messages = Vec::new();
    let mut reauth_messages = Vec::new();
    let mut reset_update_messages = Vec::new();
    let mut reset_check_messages = Vec::new();
    let mut uniqueness_messages = Vec::new();
    let mut recovery_check_messages = Vec::new();
    for modification in &last_modifications {
        if modification.result_message_body.is_none() {
            tracing::error!("Missing modification result message body");
            continue;
        }

        let body = modification
            .result_message_body
            .as_ref()
            .expect("Missing SNS message body")
            .clone();

        match modification.request_type.as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                deletion_messages.push(body);
            }
            REAUTH_MESSAGE_TYPE => {
                reauth_messages.push(body);
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                reset_update_messages.push(body);
            }
            RESET_CHECK_MESSAGE_TYPE => {
                reset_check_messages.push(body);
            }
            RECOVERY_CHECK_MESSAGE_TYPE => {
                recovery_check_messages.push(body);
            }
            UNIQUENESS_MESSAGE_TYPE => {
                uniqueness_messages.push(body);
            }
            other => {
                tracing::error!("Unknown message type: {}", other);
            }
        }
    }

    tracing::info!(
        "Sending {} last modifications to SNS. {} uniqueness, {} deletion, {} reauth, {} reset update, {} reset check, {} recovery check",
        last_modifications.len(),
        uniqueness_messages.len(),
        deletion_messages.len(),
        reauth_messages.len(),
        reset_update_messages.len(),
        reset_check_messages.len(),
        recovery_check_messages.len(),
    );

    if !uniqueness_messages.is_empty() {
        send_results_to_sns(
            uniqueness_messages,
            &Vec::new(),
            sns_client,
            config,
            &uniqueness_result_attributes,
            UNIQUENESS_MESSAGE_TYPE,
        )
        .await?;
    }

    if !deletion_messages.is_empty() {
        send_results_to_sns(
            deletion_messages,
            &Vec::new(),
            sns_client,
            config,
            &deletion_message_attributes,
            IDENTITY_DELETION_MESSAGE_TYPE,
        )
        .await?;
    }

    if !reauth_messages.is_empty() {
        send_results_to_sns(
            reauth_messages,
            &Vec::new(),
            sns_client,
            config,
            &reauth_message_attributes,
            REAUTH_MESSAGE_TYPE,
        )
        .await?;
    }

    if !reset_update_messages.is_empty() {
        send_results_to_sns(
            reset_update_messages,
            &Vec::new(),
            sns_client,
            config,
            &reset_update_message_attributes,
            RESET_UPDATE_MESSAGE_TYPE,
        )
        .await?;
    }

    if !reset_check_messages.is_empty() {
        send_results_to_sns(
            reset_check_messages,
            &Vec::new(),
            sns_client,
            config,
            &reset_check_message_attributes,
            RESET_CHECK_MESSAGE_TYPE,
        )
        .await?;
    }
    if !recovery_check_messages.is_empty() {
        send_results_to_sns(
            recovery_check_messages,
            &Vec::new(),
            sns_client,
            config,
            &recovery_check_message_attributes,
            RECOVERY_CHECK_MESSAGE_TYPE,
        )
        .await?;
    }

    Ok(())
}
