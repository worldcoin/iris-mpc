use crate::services::processors::result_message::send_results_to_sns;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use eyre::{bail, Result, WrapErr};
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    IdentityDeletionResult, ReAuthResult, ResetCheckResult, ResetUpdateAckResult, UniquenessResult,
};
use iris_mpc_common::helpers::sync::ModificationKey::{RequestId, RequestSerialId};
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_common::job::ServerJobResult;
use iris_mpc_cpu::execution::hawk_main::{GraphStore, HawkMutation};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::izip;
use sqlx::{Postgres, Transaction};
use std::{collections::HashMap, time::Instant};

/// Processes a ServerJobResult, storing data in the database and sending result messages
/// through SNS.
#[allow(clippy::too_many_arguments)]
pub async fn process_job_result(
    job_result: ServerJobResult<HawkMutation>,
    party_id: usize,
    store: &Store,
    graph_store: &GraphStore,
    sns_client: &SNSClient,
    config: &Config,
    uniqueness_result_attributes: &HashMap<String, MessageAttributeValue>,
    reauth_result_attributes: &HashMap<String, MessageAttributeValue>,
    identity_deletion_result_attributes: &HashMap<String, MessageAttributeValue>,
    reset_check_result_attributes: &HashMap<String, MessageAttributeValue>,
    reset_update_result_attributes: &HashMap<String, MessageAttributeValue>,
    shutdown_handler: &ShutdownHandler,
) -> Result<()> {
    let ServerJobResult {
        merged_results,
        request_ids,
        request_types,
        metadata,
        matches,
        matches_with_skip_persistence,
        skip_persistence,
        match_ids,
        partial_match_ids_left,
        partial_match_ids_right,
        partial_match_counters_left,
        partial_match_counters_right,
        left_iris_requests,
        right_iris_requests,
        deleted_ids,
        matched_batch_request_ids,
        successful_reauths,
        reauth_target_indices,
        reauth_or_rule_used,
        mut modifications,
        actor_data: hawk_mutation,
        full_face_mirror_attack_detected,
        reset_update_request_ids,
        reset_update_indices,
        reset_update_shares,
        ..
    } = job_result;
    let now = Instant::now();
    let dummy_deletion_shares = get_dummy_shares_for_deletion(party_id);

    // returned serial_ids are 0 indexed, but we want them to be 1 indexed
    let uniqueness_results = merged_results
        .iter()
        .enumerate()
        .filter(|(i, _)| request_types[*i] == UNIQUENESS_MESSAGE_TYPE)
        .map(|(i, &idx_result)| {
            let result_event = UniquenessResult::new(
                party_id,
                match matches[i] {
                    true => None,
                    false => Some(idx_result + 1),
                },
                matches_with_skip_persistence[i],
                request_ids[i].clone(),
                match matches[i] {
                    true => Some(match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>()),
                    false => None,
                },
                match partial_match_ids_left[i].is_empty() {
                    false => Some(
                        partial_match_ids_left[i]
                            .iter()
                            .map(|x| x + 1)
                            .collect::<Vec<_>>(),
                    ),
                    true => None,
                },
                match partial_match_ids_right[i].is_empty() {
                    false => Some(
                        partial_match_ids_right[i]
                            .iter()
                            .map(|x| x + 1)
                            .collect::<Vec<_>>(),
                    ),
                    true => None,
                },
                Some(matched_batch_request_ids[i].clone()),
                match partial_match_counters_left.is_empty() {
                    false => Some(partial_match_counters_left[i]),
                    true => None,
                },
                match partial_match_counters_right.is_empty() {
                    false => Some(partial_match_counters_right[i]),
                    true => None,
                },
                None, // partial_match_rotation_indices_left - not applicable for CPU
                None, // partial_match_rotation_indices_right - not applicable for CPU
                None, // not applicable for hnsw
                None, // not applicable for hnsw
                None, // not applicable for hnsw
                None, // not applicable for hnsw
                None, // not applicable for hnsw
                match full_face_mirror_attack_detected.is_empty() {
                    false => full_face_mirror_attack_detected[i],
                    true => false,
                },
            );
            let result_string = serde_json::to_string(&result_event)
                .wrap_err("failed to serialize uniqueness result")?;

            let modification_key = RequestId(result_event.signup_id);
            let graph_mutation = hawk_mutation.get_serialized_mutation_by_key(&modification_key);
            modifications
                .get_mut(&modification_key)
                .unwrap()
                .mark_completed(
                    !result_event.is_match,
                    &result_string,
                    result_event.serial_id,
                    graph_mutation,
                );

            Ok(result_string)
        })
        .collect::<Result<Vec<_>>>()?;

    // Insert non-matching uniqueness queries into the persistent store.
    let (memory_serial_ids, codes_and_masks): (Vec<i64>, Vec<StoredIrisRef>) = matches
        .iter()
        .enumerate()
        .filter_map(
            // Find the indices of non-matching queries in the batch.
            |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
        )
        .map(|query_idx| {
            let serial_id = (merged_results[query_idx] + 1) as i64;
            // Get the original vectors from `receive_batch`.
            (
                serial_id,
                StoredIrisRef {
                    id: serial_id,
                    left_code: &left_iris_requests.code[query_idx].coefs[..],
                    left_mask: &left_iris_requests.mask[query_idx].coefs[..],
                    right_code: &right_iris_requests.code[query_idx].coefs[..],
                    right_mask: &right_iris_requests.mask[query_idx].coefs[..],
                },
            )
        })
        .unzip();

    let reauth_results = request_types
        .iter()
        .enumerate()
        .filter(|(_, request_type)| *request_type == REAUTH_MESSAGE_TYPE)
        .map(|(i, _)| {
            let reauth_id = request_ids[i].clone();
            let serial_id = reauth_target_indices.get(&reauth_id).unwrap() + 1;
            let success = successful_reauths[i];
            let result_event = ReAuthResult::new(
                reauth_id.clone(),
                party_id,
                serial_id,
                success,
                match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>(),
                *reauth_or_rule_used.get(&reauth_id).unwrap(),
            );
            let result_string = serde_json::to_string(&result_event)
                .wrap_err("failed to serialize reauth result")?;

            let modification_key = RequestSerialId(serial_id);
            let graph_mutation = hawk_mutation.get_serialized_mutation_by_key(&modification_key);
            modifications
                .get_mut(&modification_key)
                .unwrap()
                .mark_completed(success, &result_string, None, graph_mutation);

            Ok(result_string)
        })
        .collect::<Result<Vec<_>>>()?;

    // handling identity deletion results
    let identity_deletion_results = deleted_ids
        .iter()
        .map(|&idx| {
            let serial_id = idx + 1;
            let result_event = IdentityDeletionResult::new(party_id, serial_id, true);
            let result_string = serde_json::to_string(&result_event)
                .wrap_err("failed to serialize identity deletion result")?;

            let modification_key = RequestSerialId(serial_id);
            let graph_mutation = hawk_mutation.get_serialized_mutation_by_key(&modification_key);
            modifications
                .get_mut(&modification_key)
                .unwrap()
                .mark_completed(true, &result_string, None, graph_mutation);

            Ok(result_string)
        })
        .collect::<Result<Vec<_>>>()?;

    // handling reset check results
    let reset_check_results = request_types
        .iter()
        .enumerate()
        .filter(|(_, request_type)| *request_type == RESET_CHECK_MESSAGE_TYPE)
        .map(|(i, _)| {
            let reset_id = request_ids[i].clone();
            let result_event = ResetCheckResult::new(
                reset_id.clone(),
                party_id,
                Some(match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>()),
                Some(
                    partial_match_ids_left[i]
                        .iter()
                        .map(|x| x + 1)
                        .collect::<Vec<_>>(),
                ),
                Some(
                    partial_match_ids_right[i]
                        .iter()
                        .map(|x| x + 1)
                        .collect::<Vec<_>>(),
                ),
                Some(matched_batch_request_ids[i].clone()),
                Some(partial_match_counters_right[i]),
                Some(partial_match_counters_left[i]),
            );
            let result_string = serde_json::to_string(&result_event)
                .wrap_err("failed to serialize reset check result")?;

            // Mark the reset check modification as completed.
            // Note that reset_check is only a query and does not persist anything into the database.
            // We store modification so that the SNS result can be replayed.
            let modification_key = RequestId(reset_id);
            modifications
                .get_mut(&modification_key)
                .unwrap()
                .mark_completed(false, &result_string, None, None);

            Ok(result_string)
        })
        .collect::<Result<Vec<_>>>()?;

    // handling reset update results
    let reset_update_results = reset_update_request_ids
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let reset_id = reset_update_request_ids[i].clone();
            let serial_id = reset_update_indices[i] + 1;
            let result_event = ResetUpdateAckResult::new(reset_id.clone(), party_id, serial_id);
            let result_string = serde_json::to_string(&result_event)
                .wrap_err("failed to serialize reset update result")?;
            modifications
                .get_mut(&RequestSerialId(serial_id))
                .unwrap()
                .mark_completed(true, &result_string, None, None);
            Ok(result_string)
        })
        .collect::<Result<Vec<_>>>()?;

    // Update modification results in a separate transaction to minimize lock
    // duration on the modifications table. The assign_modification_id() trigger
    // takes an EXCLUSIVE table lock on INSERT, which conflicts with the
    // RowExclusiveLock held by UPDATE. Committing early releases the lock before
    // the expensive batch_set_links INSERT runs.
    if !config.disable_persistence {
        let tx1_start = Instant::now();
        let mut mod_tx = store.tx().await?;
        store
            .update_modifications(&mut mod_tx, &modifications.values().collect::<Vec<_>>())
            .await?;
        mod_tx.commit().await?;
        tracing::info!(
            "[TX1] update_modifications committed in {:.2}s ({} modifications)",
            tx1_start.elapsed().as_secs_f64(),
            modifications.len()
        );
    }

    // Log hawk_graph_links table stats before the expensive TX2
    if !config.disable_persistence {
        if let Ok(rows) = sqlx::query_as::<_, (String, Option<i64>, Option<i64>, Option<String>)>(
            "SELECT relname, n_live_tup, n_dead_tup, last_autovacuum::text \
             FROM pg_stat_user_tables WHERE relname LIKE 'hawk_graph_links%'",
        )
        .fetch_all(&store.pool)
        .await
        {
            for (relname, live, dead, last_vac) in &rows {
                tracing::info!(
                    "[DB stats] {}: live_tup={}, dead_tup={}, last_autovacuum={}",
                    relname,
                    live.unwrap_or(0),
                    dead.unwrap_or(0),
                    last_vac.as_deref().unwrap_or("never")
                );
            }
        }
        if let Ok(rows) = sqlx::query_as::<_, (String, Option<i64>, Option<i64>)>(
            "SELECT s.relname, s.heap_blks_read, s.heap_blks_hit \
             FROM pg_statio_user_tables s WHERE s.relname LIKE 'hawk_graph_links%'",
        )
        .fetch_all(&store.pool)
        .await
        {
            for (relname, reads, hits) in &rows {
                let r = reads.unwrap_or(0) as f64;
                let h = hits.unwrap_or(0) as f64;
                let ratio = if r + h > 0.0 { h / (r + h) } else { 0.0 };
                tracing::info!(
                    "[DB stats] {}: heap_blks_read={}, heap_blks_hit={}, cache_hit_ratio={:.4}",
                    relname,
                    reads.unwrap_or(0),
                    hits.unwrap_or(0),
                    ratio
                );
            }
        }
    }

    let tx2_start = Instant::now();
    let mut iris_tx = store.tx().await?;

    if !codes_and_masks.is_empty() && !config.disable_persistence {
        let db_serial_ids = store.insert_irises(&mut iris_tx, &codes_and_masks).await?;

        // Check if the serial_ids match between memory and db.
        if memory_serial_ids != db_serial_ids {
            tracing::error!(
                "Serial IDs do not match between memory and db: {:?} != {:?}",
                memory_serial_ids,
                db_serial_ids
            );
            bail!(
                "Serial IDs do not match between memory and db: {:?} != {:?}",
                memory_serial_ids,
                db_serial_ids
            );
        }
    }

    if !config.disable_persistence {
        // persist reauth results into db
        for (i, success) in successful_reauths.iter().enumerate() {
            if !success {
                continue;
            }
            if skip_persistence.get(i).copied().unwrap_or(false) {
                tracing::info!(
                    "Skipping reauth persistence for request {} due to skip_persistence",
                    request_ids[i]
                );
                continue;
            }
            let reauth_id = request_ids[i].clone();
            // convert from memory index (0-based) to db index (1-based)
            let serial_id = *reauth_target_indices.get(&reauth_id).unwrap() + 1;
            tracing::info!(
                "Persisting successful reauth update {} into postgres on serial id {} ",
                reauth_id,
                serial_id
            );
            store
                .update_iris(
                    Some(&mut iris_tx),
                    serial_id as i64,
                    &left_iris_requests.code[i],
                    &left_iris_requests.mask[i],
                    &right_iris_requests.code[i],
                    &right_iris_requests.mask[i],
                )
                .await?;
        }

        // persist reset_update results into db
        for (idx, shares) in izip!(reset_update_indices, reset_update_shares) {
            // overwrite postgres db with reset update shares.
            // note that both serial_id and postgres db are 1-indexed.
            let serial_id = idx + 1;
            tracing::info!(
                "Persisting reset update into postgres on serial id {}",
                serial_id
            );
            store
                .update_iris(
                    Some(&mut iris_tx),
                    serial_id as i64,
                    &shares.code_left,
                    &shares.mask_left,
                    &shares.code_right,
                    &shares.mask_right,
                )
                .await?;
        }

        // persist deletion results into db
        for idx in deleted_ids.iter() {
            // overwrite postgres db with dummy shares.
            // note that both serial_id and postgres db are 1-indexed.
            let serial_id = *idx + 1;
            tracing::info!(
                "Persisting identity deletion into postgres on serial id {}",
                serial_id
            );
            store
                .update_iris(
                    Some(&mut iris_tx),
                    serial_id as i64,
                    &dummy_deletion_shares.0,
                    &dummy_deletion_shares.1,
                    &dummy_deletion_shares.0,
                    &dummy_deletion_shares.1,
                )
                .await?;
        }

        persist(iris_tx, graph_store, hawk_mutation, config).await?;
        tracing::info!(
            "[TX2] iris + graph_links committed in {:.2}s",
            tx2_start.elapsed().as_secs_f64()
        );
    }

    for memory_serial_id in memory_serial_ids {
        tracing::info!("Inserted serial_id: {}", memory_serial_id + 1);
        metrics::gauge!("results_inserted.latest_serial_id").set((memory_serial_id + 1) as f64);
    }

    tracing::info!("Sending {} uniqueness results", uniqueness_results.len());
    send_results_to_sns(
        uniqueness_results,
        &metadata,
        sns_client,
        config,
        uniqueness_result_attributes,
        UNIQUENESS_MESSAGE_TYPE,
    )
    .await?;

    tracing::info!("Sending {} reauth results", reauth_results.len());
    send_results_to_sns(
        reauth_results,
        &metadata,
        sns_client,
        config,
        reauth_result_attributes,
        REAUTH_MESSAGE_TYPE,
    )
    .await?;

    tracing::info!(
        "Sending {} identity deletion results",
        identity_deletion_results.len()
    );
    send_results_to_sns(
        identity_deletion_results,
        &metadata,
        sns_client,
        config,
        identity_deletion_result_attributes,
        IDENTITY_DELETION_MESSAGE_TYPE,
    )
    .await?;

    tracing::info!("Sending {} reset check results", reset_check_results.len());
    send_results_to_sns(
        reset_check_results,
        &metadata,
        sns_client,
        config,
        reset_check_result_attributes,
        RESET_CHECK_MESSAGE_TYPE,
    )
    .await?;

    tracing::info!(
        "Sending {} reset update results",
        reset_update_results.len()
    );
    send_results_to_sns(
        reset_update_results,
        &metadata,
        sns_client,
        config,
        reset_update_result_attributes,
        RESET_UPDATE_MESSAGE_TYPE,
    )
    .await?;

    metrics::histogram!("process_job_duration").record(now.elapsed().as_secs_f64());

    shutdown_handler.decrement_batches_pending_completion();

    Ok(())
}

async fn persist(
    iris_tx: Transaction<'_, Postgres>,
    graph_store: &GraphStore,
    hawk_mutation: HawkMutation,
    config: &Config,
) -> Result<()> {
    // simply persist or not both iris and graph changes
    if !config.disable_persistence {
        let mut graph_tx = graph_store.tx_wrap(iris_tx);
        hawk_mutation.persist(&mut graph_tx).await?;
        graph_tx.tx.commit().await?;
    }

    Ok(())
}
