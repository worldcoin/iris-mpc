#![allow(clippy::needless_range_loop, unused)]

use ampc_anon_stats::store::postgres::AccessMode as AnonStatsAccessMode;
use ampc_anon_stats::store::postgres::PostgresClient as AnonStatsPgClient;
use ampc_anon_stats::AnonStatsStore;
use ampc_server_utils::batch_sync::{
    BatchSyncSharedState, CURRENT_BATCH_SHA, CURRENT_BATCH_VALID_ENTRIES,
};
use ampc_server_utils::{
    delete_messages_until_sequence_num, get_next_sns_seq_num, get_others_sync_state,
    init_heartbeat_task, set_node_ready, shutdown_handler::ShutdownHandler,
    start_coordination_server, wait_for_others_ready, wait_for_others_unready, TaskMonitor,
};
use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use aws_sdk_sqs::Client;
use clap::Parser;
use eyre::{bail, eyre, Context, Report, Result};
use futures::{stream::BoxStream, StreamExt};
use iris_mpc::services::aws::clients::AwsClients;
use iris_mpc::services::init::initialize_chacha_seeds;
use iris_mpc::services::processors::batch::receive_batch_stream;
use iris_mpc::services::processors::get_iris_shares_parse_task;
use iris_mpc::services::processors::modifications_sync::{
    send_last_modifications_to_sns, sync_modifications,
};
use iris_mpc::services::processors::result_message::{
    send_error_results_to_sns, send_results_to_sns,
};
use iris_mpc_common::config::CommonConfig;
use iris_mpc_common::galois_engine::degree4::GaloisShares;
use iris_mpc_common::helpers::sync::ModificationKey::{RequestId, RequestSerialId};
use iris_mpc_common::job::{GaloisSharesBothSides, RequestIndex};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_common::{
    config::{Config, Opt},
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        aws::{SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME},
        inmemory_store::InMemoryStore,
        key_pair::SharesEncryptionKeyPairs,
        sha256::sha256_bytes,
        smpc_request::{
            decrypt_iris_share, get_iris_data_by_party_id, validate_iris_share,
            CircuitBreakerRequest, IdentityDeletionRequest, IdentityMatchCheckRequest,
            IdentityUpdateRequest, ReAuthRequest, ReceiveRequestError, SQSMessage,
            UniquenessRequest, CIRCUIT_BREAKER_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE, RECOVERY_CHECK_MESSAGE_TYPE, RECOVERY_UPDATE_MESSAGE_TYPE,
            RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
        },
        smpc_response::{
            create_message_type_attribute_map, IdentityDeletionResult, IdentityMatchCheckResult,
            IdentityUpdateAckResult, ReAuthResult, UniquenessResult,
            ERROR_FAILED_TO_PROCESS_IRIS_SHARES, ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH,
            SMPC_MESSAGE_TYPE_ATTRIBUTE,
        },
        sync::{Modification, ModificationKey, SyncResult, SyncState},
    },
    iris_db::get_dummy_shares_for_deletion,
    job::{BatchMetadata, BatchQuery, JobSubmissionHandle, ServerJobResult},
};
use iris_mpc_gpu::server::ServerActor;
use iris_mpc_store::loader::load_iris_db;
use iris_mpc_store::{
    fetch_and_parse_chunks, last_snapshot_timestamp, DbStoredIris, ObjectStore, S3Store,
    S3StoredIris, Store, StoredIrisRef,
};
use itertools::{cloned, izip, Itertools};
use metrics_exporter_statsd::StatsdBuilder;
use serde::{Deserialize, Serialize};
use std::process::exit;
use std::{
    collections::HashMap,
    fmt::Debug,
    mem, panic,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc::Receiver;
use tokio::{
    sync::{mpsc, oneshot, Semaphore},
    task::{spawn_blocking, JoinHandle},
    time::timeout,
};
const RNG_SEED_INIT_DB: u64 = 42;
const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
const MAX_CONCURRENT_REQUESTS: usize = 32;

fn decode_iris_message_shares(
    code_share: String,
    mask_share: String,
) -> Result<(GaloisRingIrisCodeShare, GaloisRingIrisCodeShare)> {
    let iris_share = GaloisRingIrisCodeShare::from_base64(&code_share)
        .context("Failed to base64 parse iris code")?;
    let mask_share = GaloisRingIrisCodeShare::from_base64(&mask_share)
        .context("Failed to base64 parse iris mask")?;

    Ok((iris_share, mask_share))
}

fn trim_mask(mask: GaloisRingIrisCodeShare) -> GaloisRingTrimmedMaskCodeShare {
    mask.into()
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    println!("Init tracing");
    let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    match server_main(config).await {
        Ok(_) => {
            tracing::info!("Server exited normally");
        }
        Err(e) => {
            tracing::error!("Server exited with error: {:?}", e);
            exit(1);
        }
    }
    Ok(())
}

async fn server_main(config: Config) -> Result<()> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.register_signal_handler().await;
    let max_modification_lookback = config.max_modifications_lookback;

    let schema_name = format!(
        "{}{}_{}_{}",
        config.schema_name, config.gpu_schema_name_suffix, config.environment, config.party_id
    );
    let db_config = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?;

    tracing::info!(
        "Creating new iris storage from: {:?} with schema {}",
        db_config,
        schema_name
    );
    let postgres_client =
        PostgresClient::new(&db_config.url, schema_name.as_str(), AccessMode::ReadWrite).await?;
    let store = Store::new(&postgres_client).await?;

    tracing::info!("Initialising AWS services");
    let aws_clients = AwsClients::new(&config.clone()).await?;
    let next_sns_seq_number_future = get_next_sns_seq_num(
        &aws_clients.sqs_client,
        &config.requests_queue_url,
        config.sqs_sync_long_poll_seconds,
    );

    let shares_encryption_key_pair = match SharesEncryptionKeyPairs::from_storage(
        aws_clients.secrets_manager_client.clone(),
        &config.environment,
        &config.party_id,
    )
    .await
    {
        Ok(key_pair) => key_pair,
        Err(e) => {
            tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
            return Ok(());
        }
    };

    let party_id = config.party_id;
    tracing::info!("Deriving shared secrets");
    let chacha_seeds = initialize_chacha_seeds(config.clone()).await?;

    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_result_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_check_result_attributes = create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let recovery_check_result_attributes =
        create_message_type_attribute_map(RECOVERY_CHECK_MESSAGE_TYPE);
    let reset_update_result_attributes =
        create_message_type_attribute_map(RESET_UPDATE_MESSAGE_TYPE);
    let recovery_update_result_attributes =
        create_message_type_attribute_map(RECOVERY_UPDATE_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);

    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database before init: {}", store_len);

    // Seed the persistent storage with random shares if configured and db is still
    // empty.
    if store_len == 0 && config.init_db_size > 0 {
        tracing::info!(
            "Initialize persistent iris DB with {} randomly generated shares",
            config.init_db_size
        );
        tracing::info!("Resetting the db: {}", config.clear_db_before_init);
        store
            .init_db_with_random_shares(
                RNG_SEED_INIT_DB,
                party_id,
                config.init_db_size,
                config.clear_db_before_init,
            )
            .await?;
    }

    // Fetch again in case we've just initialized the DB
    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database after init: {}", store_len);

    // Check if the sequence id is consistent with the number of irises
    let max_serial_id = store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        tracing::error!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );

        eyre::bail!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );
    }

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        eyre::bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

    tracing::info!("Preparing task monitor");
    let mut background_tasks = TaskMonitor::new();
    let current_batch_id_atomic = Arc::new(AtomicU64::new(0));
    let batch_sync_shared_state =
        Arc::new(tokio::sync::Mutex::new(BatchSyncSharedState::default()));

    // --------------------------------------------------------------------------
    // ANCHOR: Starting Healthcheck, Readiness and Sync server
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let my_state = SyncState {
        db_len: store_len as u64,
        modifications: store.last_modifications(max_modification_lookback).await?,
        next_sns_sequence_num: next_sns_seq_number_future.await?,
        common_config: CommonConfig::from(config.clone()),
    };

    tracing::info!("Sync state: {:?}", my_state);

    let server_coord_config = config.server_coordination.clone().unwrap_or_else(|| {
        panic!("Server coordination config must be provided for healthcheck server");
    });
    let (is_ready_flag, verified_peers, uuid) = start_coordination_server(
        &server_coord_config,
        &mut background_tasks,
        &shutdown_handler,
        &my_state,
        Some(batch_sync_shared_state.clone()),
    )
    .await;

    background_tasks.check_tasks();
    let download_shutdown_handler = Arc::clone(&shutdown_handler);

    background_tasks.check_tasks();
    wait_for_others_unready(&server_coord_config, &verified_peers, &uuid).await?;

    // Start the actor in separate task.
    // A bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    let s3_load_parallelism = config.load_chunks_parallelism;
    let s3_chunks_bucket_name = config.db_chunks_bucket_name.clone();
    let s3_chunks_folder_name = config.db_chunks_folder_name.clone();
    let s3_load_max_retries = config.load_chunks_max_retries;
    let s3_load_initial_backoff_ms = config.load_chunks_initial_backoff_ms;

    // --------------------------------------------------------------------------
    // ANCHOR: Syncing latest node state
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");
    tracing::info!("Database store length is: {}", store_len);
    let mut states = vec![my_state.clone()];
    states.extend(get_others_sync_state::<SyncState>(&server_coord_config).await?);
    let sync_result = SyncResult::new(my_state.clone(), states);

    // check if common part of the config is the same across all nodes
    sync_result.check_common_config()?;

    // sync the queues
    let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
    delete_messages_until_sequence_num(
        &aws_clients.sqs_client,
        &config.requests_queue_url,
        my_state.next_sns_sequence_num,
        max_sqs_sequence_num,
        config.sqs_sync_long_poll_seconds,
    )
    .await?;

    let dummy_shares_for_deletions = get_dummy_shares_for_deletion(party_id);

    // Handle modifications sync
    if config.enable_modifications_sync {
        sync_modifications(
            &config,
            &store,
            &aws_clients,
            &shares_encryption_key_pair,
            sync_result,
        )
        .await?;
    }

    if config.enable_modifications_replay {
        // replay last `max_modification_lookback` modifications to SNS
        if let Err(e) = send_last_modifications_to_sns(
            &store,
            &aws_clients.sns_client,
            &config,
            max_modification_lookback,
        )
        .await
        {
            tracing::error!("Failed to replay last modifications: {:?}", e);
        }
    }

    if download_shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    // refetch store_len in case we rolled back
    let store_len = store.count_irises().await?;
    tracing::info!("Database store length after sync: {}", store_len);

    let runtime_handle = tokio::runtime::Handle::current();
    let anon_stats_writer = if let Some(url) = config.get_anon_stats_db_url() {
        let schema = config.get_anon_stats_db_schema();
        let anon_client =
            AnonStatsPgClient::new(&url, &schema, AnonStatsAccessMode::ReadWrite).await?;
        let anon_store = AnonStatsStore::new(&anon_client).await?;
        Some((anon_store, runtime_handle.clone()))
    } else {
        tracing::warn!("No database URL configured for anon stats; skipping DB persistence");
        None
    };
    let anon_stats_writer_for_actor = anon_stats_writer.clone();

    let (tx, rx) = oneshot::channel();
    let config_clone = config.clone();
    background_tasks.spawn_blocking(move || {
        let config = config_clone;
        // --------------------------------------------------------------------------
        // ANCHOR: Load the database
        // --------------------------------------------------------------------------
        tracing::info!("⚓️ ANCHOR: Starting server actor");
        match ServerActor::new(
            config.party_id,
            chacha_seeds,
            8,
            config.max_db_size,
            config.max_batch_size,
            config.match_distances_buffer_size,
            config.match_distances_buffer_size_extra_percent,
            config.return_partial_results,
            config.disable_persistence,
            config.enable_debug_timing,
            config.full_scan_side,
            config.full_scan_side_switching_enabled,
            anon_stats_writer_for_actor,
        ) {
            Ok((mut actor, handle)) => {
                tracing::info!("⚓️ ANCHOR: Load the database");
                let res = if config.fake_db_size > 0 {
                    // TODO: does this even still work, since we do not page-lock the memory here?
                    actor.fake_db(config.fake_db_size);
                    Ok(())
                } else {
                    tracing::info!(
                        "Initialize iris db: Loading from DB (parallelism: {})",
                        parallelism
                    );
                    let download_shutdown_handler = Arc::clone(&download_shutdown_handler);

                    tokio::runtime::Handle::current().block_on(async {
                        load_iris_db(
                            &mut actor,
                            &store,
                            store_len,
                            parallelism,
                            None,
                            &config,
                            download_shutdown_handler,
                        )
                        .await
                    })
                };

                match res {
                    Ok(_) => {
                        tx.send(Ok((handle, store))).unwrap();
                    }
                    Err(e) => {
                        tx.send(Err(e)).unwrap();
                        return Ok(());
                    }
                }

                actor.run(); // forever
            }
            Err(e) => {
                tx.send(Err(e)).unwrap();
                return Ok(());
            }
        };
        Ok(())
    });

    let (mut handle, store) = rx.await??;

    background_tasks.check_tasks();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let sns_client_bg = aws_clients.sns_client.clone();
    let s3_client_bg = aws_clients.s3_client.clone();
    let config_bg = config.clone();
    let store_bg = store.clone();
    let shutdown_handler_bg = Arc::clone(&shutdown_handler);
    let _result_sender_abort = background_tasks.spawn(async move {
        while let Some(ServerJobResult {
            merged_results,
            request_ids,
            request_types,
            metadata,
            matches,
            matches_with_skip_persistence,
            skip_persistence,
            match_ids,
            full_face_mirror_match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_rotation_indices_left,
            partial_match_rotation_indices_right,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_counters_right,
            left_iris_requests,
            right_iris_requests,
            deleted_ids,
            matched_batch_request_ids,
            successful_reauths,
            reauth_target_indices,
            reauth_or_rule_used,
            identity_update_indices,
            identity_update_request_ids,
            identity_update_request_types,
            identity_update_shares,
            mut modifications,
            actor_data: _,
            full_face_mirror_attack_detected,
        }) = rx.recv().await
        {
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
                        match partial_match_counters_right.is_empty() {
                            false => Some(partial_match_counters_right[i]),
                            true => None,
                        },
                        match partial_match_counters_left.is_empty() {
                            false => Some(partial_match_counters_left[i]),
                            true => None,
                        },
                        match partial_match_rotation_indices_left[i].is_empty() {
                            false => Some(partial_match_rotation_indices_left[i].clone()),
                            true => None,
                        },
                        match partial_match_rotation_indices_right[i].is_empty() {
                            false => Some(partial_match_rotation_indices_right[i].clone()),
                            true => None,
                        },
                        match full_face_mirror_match_ids[i].is_empty() {
                            false => Some(
                                full_face_mirror_match_ids[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_ids_left[i].is_empty() {
                            false => Some(
                                full_face_mirror_partial_match_ids_left[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_ids_right[i].is_empty() {
                            false => Some(
                                full_face_mirror_partial_match_ids_right[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match full_face_mirror_partial_match_counters_left.is_empty() {
                            false => Some(full_face_mirror_partial_match_counters_left[i]),
                            true => None,
                        },
                        match full_face_mirror_partial_match_counters_right.is_empty() {
                            false => Some(full_face_mirror_partial_match_counters_right[i]),
                            true => None,
                        },
                        full_face_mirror_attack_detected[i],
                    );
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize reauth result");
                    modifications
                        .get_mut(&RequestId(request_ids[i].clone()))
                        .unwrap()
                        .mark_completed(!result_event.is_match, &result_string, result_event.serial_id);
                    result_string
                })
                .collect::<Vec<String>>();

            // Insert non-matching uniqueness queries into the persistent store.
            let (memory_serial_ids, codes_and_masks): (Vec<i64>, Vec<StoredIrisRef>) = matches
                .iter()
                .enumerate()
                .filter_map(
                    // Find the indices of non-matching queries in the batch.
                    |(query_idx, is_match)| {
                        if !is_match {
                                Some(query_idx)
                        } else {
                            // Check for full face mirror attack (only for UNIQUENESS requests) and log it.
                            if request_types[query_idx] == UNIQUENESS_MESSAGE_TYPE && full_face_mirror_attack_detected[query_idx]
                            {
                                tracing::warn!(
                                    "Mirror attack detected for request_id {} - Not persisting to database",
                                    request_ids[query_idx]
                                );
                                metrics::counter!("mirror.attack.rejected").increment(1);
                            }
                            // It matched, don't include.
                            None
                        }
                    },
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
                    let or_rule_used = reauth_or_rule_used.get(&reauth_id).unwrap();
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
                        .expect("failed to serialize reauth result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(success, &result_string, None);
                    result_string
                })
                .collect::<Vec<String>>();

            // handling identity deletion results
            let identity_deletion_results = deleted_ids
                .iter()
                .map(|&idx| {
                    let serial_id = idx + 1;
                    let result_event = IdentityDeletionResult::new(party_id, serial_id, true);
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize identity deletion result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(true, &result_string, None);
                    result_string
                })
                .collect::<Vec<String>>();

            let identity_checks_by_type = request_types
                .iter()
                .enumerate()
                .filter(|(_, request_type)| matches!(request_type.as_str(), RESET_CHECK_MESSAGE_TYPE | RECOVERY_CHECK_MESSAGE_TYPE))
                .map(|(i, request_type)| {
                    let request_id = request_ids[i].clone();
                    let result_event = IdentityMatchCheckResult::new(
                        request_id.clone(),
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
                        .expect("failed to serialize reset check result");

                    // Mark the reset check modification as completed.
                    // Note that reset_check is only a query and does not persist anything into the database.
                    // We store modification so that the SNS result can be replayed.
                    modifications
                        .get_mut(&RequestId(request_id))
                        .unwrap()
                        .mark_completed(false, &result_string, None);
                    (request_type.clone().to_string(), result_string.clone())
                })
                .collect::<Vec<(String, String)>>()
                .into_iter()
                .into_group_map();

            // identity update results (reset_update and recovery_update)
            let identity_update_results_by_type = identity_update_request_ids
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let request_id = identity_update_request_ids[i].clone();
                    let request_type = identity_update_request_types[i].clone();
                    let serial_id = identity_update_indices[i] + 1;
                    let result_event =
                        IdentityUpdateAckResult::new(request_id.clone(), party_id, serial_id);
                    let result_string = serde_json::to_string(&result_event)
                        .expect("failed to serialize identity update result");
                    modifications
                        .get_mut(&RequestSerialId(serial_id))
                        .unwrap()
                        .mark_completed(true, &result_string, None);
                    (request_type, result_string)
                })
                .collect::<Vec<(String, String)>>()
                .into_iter()
                .into_group_map();

            let mut tx = store_bg.tx().await?;

            store_bg
                .update_modifications(&mut tx, &modifications.values().collect::<Vec<_>>())
                .await?;

            // persist uniqueness results into db
            if !codes_and_masks.is_empty() && !config_bg.disable_persistence {
                let db_serial_ids = store_bg.insert_irises(&mut tx, &codes_and_masks).await?;

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

            if !config_bg.disable_persistence {
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
                    store_bg
                        .update_iris(
                            Some(&mut tx),
                            serial_id as i64,
                            &left_iris_requests.code[i],
                            &left_iris_requests.mask[i],
                            &right_iris_requests.code[i],
                            &right_iris_requests.mask[i],
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
                    store_bg.update_iris(
                        Some(&mut tx),
                        serial_id as i64,
                        &dummy_deletion_shares.0,
                        &dummy_deletion_shares.1,
                        &dummy_deletion_shares.0,
                        &dummy_deletion_shares.1,
                    )
                    .await?;
                }

                // persist identity update (reset_update/recovery_update) results into db
                for (idx, shares) in
                    izip!(identity_update_indices, identity_update_shares)
                {
                    // overwrite postgres db with identity update shares.
                    // note that both serial_id and postgres db are 1-indexed.
                    let serial_id = idx + 1;
                    tracing::info!(
                        "Persisting identity update into postgres on serial id {}",
                        serial_id
                    );
                    store_bg
                        .update_iris(
                            Some(&mut tx),
                            serial_id as i64,
                            &shares.code_left,
                            &shares.mask_left,
                            &shares.code_right,
                            &shares.mask_right,
                        )
                        .await?;
                }
            }

            tx.commit().await?;

            for memory_serial_id in memory_serial_ids {
                tracing::info!("Inserted serial_id: {}", memory_serial_id+1);
                metrics::gauge!("results_inserted.latest_serial_id").set((memory_serial_id +1) as f64);
            }

            tracing::info!("Sending {} uniqueness results", uniqueness_results.len());
            send_results_to_sns(
                uniqueness_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &uniqueness_result_attributes,
                UNIQUENESS_MESSAGE_TYPE,
            )
            .await?;

            tracing::info!("Sending {} reauth results", reauth_results.len());
            send_results_to_sns(
                reauth_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &reauth_result_attributes,
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
                &sns_client_bg,
                &config_bg,
                &identity_deletion_result_attributes,
                IDENTITY_DELETION_MESSAGE_TYPE,
            )
            .await?;

            let reset_check_results = identity_checks_by_type.get(RESET_CHECK_MESSAGE_TYPE).unwrap_or(&Vec::new()).clone();
            if !reset_check_results.is_empty() {
                tracing::info!("Sending {} reset check results", reset_check_results.len());
                send_results_to_sns(
                    reset_check_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &reset_check_result_attributes,
                    RESET_CHECK_MESSAGE_TYPE,
                )
                .await?;
            }
            let recovery_check_results = identity_checks_by_type.get(RECOVERY_CHECK_MESSAGE_TYPE).unwrap_or(&Vec::new()).clone();
            if !recovery_check_results.is_empty() {
                tracing::info!("Sending {} recovery check results", recovery_check_results.len());
                send_results_to_sns(
                    recovery_check_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &recovery_check_result_attributes,
                    RECOVERY_CHECK_MESSAGE_TYPE,
                )
                .await?;
            }

            let reset_update_results = identity_update_results_by_type
                .get(RESET_UPDATE_MESSAGE_TYPE)
                .unwrap_or(&Vec::new())
                .clone();
            if !reset_update_results.is_empty() {
                tracing::info!(
                    "Sending {} reset update results",
                    reset_update_results.len()
                );
                send_results_to_sns(
                    reset_update_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &reset_update_result_attributes,
                    RESET_UPDATE_MESSAGE_TYPE,
                )
                .await?;
            }

            let recovery_update_results = identity_update_results_by_type
                .get(RECOVERY_UPDATE_MESSAGE_TYPE)
                .unwrap_or(&Vec::new())
                .clone();
            if !recovery_update_results.is_empty() {
                tracing::info!(
                    "Sending {} recovery update results",
                    recovery_update_results.len()
                );
                send_results_to_sns(
                    recovery_update_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &recovery_update_result_attributes,
                    RECOVERY_UPDATE_MESSAGE_TYPE,
                )
                .await?;
            }
            shutdown_handler_bg.decrement_batches_pending_completion();
        }

        Ok(())
    });
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Enable readiness and check all nodes
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Enable readiness and check all nodes");

    init_heartbeat_task(
        &server_coord_config,
        &mut background_tasks,
        &shutdown_handler,
    )
    .await?;
    set_node_ready(is_ready_flag);
    wait_for_others_ready(&server_coord_config).await?;
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Start the main loop
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Start the main loop");

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
    let uniqueness_error_result_attribute =
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_error_result_attribute = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_check_error_result_attribute =
        create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let identity_deletion_error_result_attribute =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);
    let reset_update_error_result_attribute =
        create_message_type_attribute_map(RESET_UPDATE_MESSAGE_TYPE);
    let recovery_update_error_result_attribute =
        create_message_type_attribute_map(RECOVERY_UPDATE_MESSAGE_TYPE);
    let recovery_check_error_result_attribute =
        create_message_type_attribute_map(RECOVERY_CHECK_MESSAGE_TYPE);
    let res: Result<()> = async {
        tracing::info!("Entering main loop");
        // **Tensor format of queries**
        //
        // The functions `receive_batch` and `prepare_query_shares` will prepare the
        // _query_ variables as `Vec<Vec<u8>>` formatted as follows:
        //
        // - The inner Vec is a flattening of these dimensions (inner to outer):
        //   - One u8 limb of one iris bit.
        //   - One code: 12800 coefficients.
        //   - One query: all rotated variants of a code.
        //   - One batch: many queries.
        // - The outer Vec is the dimension of the Galois Ring (2):
        //   - A decomposition of each iris bit into two u8 limbs.

        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above
        let (mut batch_stream, sem) = receive_batch_stream(
            party_id,
            aws_clients.sqs_client.clone(),
            aws_clients.sns_client.clone(),
            aws_clients.s3_client.clone(),
            config.clone(),
            shares_encryption_key_pair.clone(),
            shutdown_handler.clone(),
            uniqueness_error_result_attribute,
            reauth_error_result_attribute,
            reset_check_error_result_attribute,
            recovery_check_error_result_attribute,
            identity_deletion_error_result_attribute,
            reset_update_error_result_attribute,
            recovery_update_error_result_attribute,
            current_batch_id_atomic.clone(),
            store.clone(),
            batch_sync_shared_state.clone(),
        );
        current_batch_id_atomic.fetch_add(1, Ordering::SeqCst);

        loop {
            let now = Instant::now();

            let mut batch = match batch_stream.recv().await {
                Some(Ok(None)) | None => {
                    tracing::info!("No more batches to process, exiting main loop");
                    return Ok(());
                }
                Some(Err(e)) => {
                    return Err(e.into());
                }
                Some(Ok(Some(batch))) => batch,
            };

            let batch_hash = sha256_bytes(batch.sns_message_ids.join(""));
            let batch_valid_entries = batch.valid_entries.clone();
            *CURRENT_BATCH_SHA
                .lock()
                .expect("Failed to lock CURRENT_BATCH_SHA") = batch_hash;
            *CURRENT_BATCH_VALID_ENTRIES
                .lock()
                .expect("Failed to lock CURRENT_BATCH_VALID_ENTRIES") = batch_valid_entries;

            loop {
                if shutdown_handler.is_shutting_down() {
                    tracing::info!("Shutdown requested during batch sync retry, exiting");
                    return Ok(());
                }

                match batch.sync_batch_entries(&config).await {
                    Ok(()) => break,
                    Err(e) => {
                        tracing::warn!("Batch sync entries failed: {:?}. Retrying...", e);
                    }
                }
            }

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());

            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id and
            // parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Started processing share",
                );
            }

            background_tasks.check_tasks();
            current_batch_id_atomic.fetch_add(1, Ordering::SeqCst);
            sem.add_permits(1);

            let result_future = handle.submit_batch_query(batch);

            // await the result
            let result = timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("ServerActor processing timeout: {:?}", e))??;

            tx.send(result).await?;

            shutdown_handler.increment_batches_pending_completion()
            // wrap up span context
        }
    }
    .await;

    match res {
        Ok(_) => {
            tracing::info!(
                "Main loop exited normally. Waiting for last batch results to be processed before \
                 shutting down..."
            );

            let _ = shutdown_handler.wait_for_pending_batches_completion().await;
        }
        Err(e) => {
            tracing::error!("ServerActor processing error: {:?}", e);
            // drop actor handle to initiate shutdown
            drop(handle);

            // Clean up server tasks, then wait for them to finish
            background_tasks.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Check for background task hangs and shutdown panics
            background_tasks.check_tasks_finished();
        }
    }
    Ok(())
}

fn is_enabled(request_type: &str, config: &Config) -> bool {
    match request_type {
        RECOVERY_CHECK_MESSAGE_TYPE => config.enable_recovery,
        RESET_CHECK_MESSAGE_TYPE => config.enable_reset,
        _ => true,
    }
}
