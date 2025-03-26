use crate::services::processors::result_message::send_results_to_sns;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use eyre::{eyre, WrapErr};
use iris_mpc_common::config::{Config, ModeOfDeployment};
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    ANONYMIZED_STATISTICS_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    IdentityDeletionResult, ReAuthResult, UniquenessResult,
};
use iris_mpc_common::job::ServerJobResult;
use iris_mpc_cpu::execution::hawk_main::{GraphStore, HawkMutation};
use iris_mpc_store::{Store, StoredIrisRef};
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
    anonymized_statistics_attributes: &HashMap<String, MessageAttributeValue>,
    shutdown_handler: &ShutdownHandler,
) -> eyre::Result<()> {
    let ServerJobResult {
        merged_results,
        request_ids,
        request_types,
        metadata,
        matches,
        matches_with_skip_persistence,
        match_ids,
        partial_match_ids_left,
        partial_match_ids_right,
        partial_match_counters_left,
        partial_match_counters_right,
        left_iris_requests,
        right_iris_requests,
        deleted_ids,
        matched_batch_request_ids,
        anonymized_bucket_statistics_left,
        anonymized_bucket_statistics_right,
        successful_reauths,
        reauth_target_indices,
        reauth_or_rule_used,
        modifications,
        actor_data: hawk_mutation,
    } = job_result;
    let now = Instant::now();

    let _modifications = modifications;

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
            );

            serde_json::to_string(&result_event).wrap_err("failed to serialize result")
        })
        .collect::<eyre::Result<Vec<_>>>()?;

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
            let result_event = ReAuthResult::new(
                reauth_id.clone(),
                party_id,
                reauth_target_indices.get(&reauth_id).unwrap() + 1,
                successful_reauths[i],
                match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>(),
                *reauth_or_rule_used.get(&reauth_id).unwrap(),
            );
            serde_json::to_string(&result_event).wrap_err("failed to serialize reauth result")
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    let mut iris_tx = store.tx().await?;

    store
        .insert_results(&mut iris_tx, &uniqueness_results)
        .await?;

    // TODO: update modifications table to store reauth and deletion results

    if !codes_and_masks.is_empty() && !config.disable_persistence {
        let db_serial_ids = store.insert_irises(&mut iris_tx, &codes_and_masks).await?;

        // Check if the serial_ids match between memory and db.
        if memory_serial_ids != db_serial_ids {
            tracing::error!(
                "Serial IDs do not match between memory and db: {:?} != {:?}",
                memory_serial_ids,
                db_serial_ids
            );
            return Err(eyre!(
                "Serial IDs do not match between memory and db: {:?} != {:?}",
                memory_serial_ids,
                db_serial_ids
            ));
        }

        for (i, success) in successful_reauths.iter().enumerate() {
            if !success {
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
    }

    // The way we are reusing the transaction between the stores is bound to fail in a multi-DB environment
    // During the ShadowModeReadOnly we will perform two separate transactions
    if config.mode_of_deployment == ModeOfDeployment::ShadowReadOnly {
        iris_tx.commit().await?;
        let mut graph_tx = graph_store.tx().await?;
        if !config.disable_persistence {
            hawk_mutation.persist(&mut graph_tx).await?;
        }
        graph_tx.tx.commit().await?;
    } else {
        let mut graph_tx = graph_store.tx_wrap(iris_tx);

        if !config.disable_persistence {
            hawk_mutation.persist(&mut graph_tx).await?;
        }

        // Because this transaction was built on top of the irises transaction, commiting it persists both tables
        graph_tx.tx.commit().await?;
    };
    
    for memory_serial_id in memory_serial_ids {
        tracing::info!("Inserted serial_id: {}", memory_serial_id);
        metrics::gauge!("results_inserted.latest_serial_id").set(memory_serial_id as f64);
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

    // handling identity deletion results
    let identity_deletion_results = deleted_ids
        .iter()
        .map(|&serial_id| {
            let result_event = IdentityDeletionResult::new(party_id, serial_id + 1, true);
            serde_json::to_string(&result_event)
                .wrap_err("failed to serialize identity deletion result")
        })
        .collect::<eyre::Result<Vec<_>>>()?;

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

    if (config.enable_sending_anonymized_stats_message)
        && (!anonymized_bucket_statistics_left.buckets.is_empty()
            || !anonymized_bucket_statistics_right.buckets.is_empty())
    {
        tracing::info!("Sending anonymized stats results");
        let anonymized_statistics_results = [
            anonymized_bucket_statistics_left,
            anonymized_bucket_statistics_right,
        ];
        // transform to vector of string ands remove None values
        let anonymized_statistics_results = anonymized_statistics_results
            .iter()
            .map(|anonymized_bucket_statistics| {
                serde_json::to_string(anonymized_bucket_statistics)
                    .wrap_err("failed to serialize anonymized statistics result")
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        send_results_to_sns(
            anonymized_statistics_results,
            &metadata,
            sns_client,
            config,
            anonymized_statistics_attributes,
            ANONYMIZED_STATISTICS_MESSAGE_TYPE,
        )
        .await?;
    }
    metrics::histogram!("process_job_duration").record(now.elapsed().as_secs_f64());

    shutdown_handler.decrement_batches_pending_completion();

    Ok(())
}
