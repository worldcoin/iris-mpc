#![allow(clippy::needless_range_loop)]

use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Region, Client};
use axum::{routing::get, Router};
use clap::Parser;
use eyre::{eyre, Context};
use futures::StreamExt;
use gpu_iris_mpc::{
    config::{json_wrapper::JsonStrWrapper, Config, Opt},
    dot::ROTATIONS,
    helpers::{
        aws::{
            NODE_ID_MESSAGE_ATTRIBUTE_NAME, SPAN_ID_MESSAGE_ATTRIBUTE_NAME,
            TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
        },
        device_manager::DeviceManager,
        kms_dh::derive_shared_secret,
        sqs::{ResultEvent, SMPCRequest, SQSMessage},
        task_monitor::TaskMonitor,
    },
    server::{BatchMetadata, BatchQuery, ServerActor, ServerJobResult},
    setup::{galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::db::IrisDB},
    store::{
        sync::{SyncResult, SyncState, Syncer},
        Store, StoredIrisRef,
    },
};
use rand::{rngs::StdRng, SeedableRng};
use static_assertions::const_assert;
use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};
use telemetry_batteries::{
    metrics::statsd::StatsdBattery,
    tracing::{datadog::DatadogBattery, TracingShutdownHandle},
};
use tokio::{
    sync::{mpsc, oneshot},
    task::spawn_blocking,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REGION: &str = "eu-north-1";
const DB_SIZE: usize = 8 * 1_000;
const N_QUERIES: usize = 32;
const N_BATCHES: usize = 100;
const RNG_SEED: u64 = 42;
const SYNC_QUERIES: usize = N_QUERIES * 2;
const_assert!(SYNC_QUERIES <= SyncState::MAX_REQUESTS);
/// The number of batches before a stream is re-used.

const QUERIES: usize = ROTATIONS * N_QUERIES;

async fn receive_batch(
    party_id: usize,
    client: &Client,
    queue_url: &String,
    store: &Store,
    skip_request_ids: Vec<String>,
) -> eyre::Result<BatchQuery> {
    let mut batch_query = BatchQuery::default();

    while batch_query.db.code.len() < QUERIES {
        let rcv_message_output = client
            .receive_message()
            .max_number_of_messages(1)
            .queue_url(queue_url)
            .send()
            .await?;

        if let Some(messages) = rcv_message_output.messages {
            for sqs_message in messages {
                let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())?;
                let message: SMPCRequest = serde_json::from_str(&message.message)?;

                store
                    .mark_requests_deleted(&[message.request_id.clone()])
                    .await?;

                client
                    .delete_message()
                    .queue_url(queue_url)
                    .receipt_handle(sqs_message.receipt_handle.unwrap())
                    .send()
                    .await?;

                if skip_request_ids.contains(&message.request_id) {
                    // Some party (maybe us) already meant to delete this request, so we skip it.
                    continue;
                }

                let message_attributes = sqs_message.message_attributes.unwrap_or_default();

                let mut batch_metadata = BatchMetadata::default();

                if let Some(node_id) = message_attributes.get(NODE_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let node_id = node_id.string_value().unwrap();
                    batch_metadata.node_id = node_id.to_string();
                }
                if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let trace_id = trace_id.string_value().unwrap();
                    batch_metadata.trace_id = trace_id.to_string();
                }
                if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let span_id = span_id.string_value().unwrap();
                    batch_metadata.span_id = span_id.to_string();
                }

                batch_query.request_ids.push(message.request_id.clone());
                batch_query.metadata.push(batch_metadata);

                let (
                    store_iris_shares,
                    store_mask_shares,
                    db_iris_shares,
                    db_mask_shares,
                    iris_shares,
                    mask_shares,
                ) = spawn_blocking(move || {
                    let mut iris_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_iris_shares());
                    let mut mask_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_mask_shares());

                    // Original for storage.
                    let store_iris_shares = iris_share.clone();
                    let store_mask_shares = mask_share.clone();

                    // With rotations for in-memory database.
                    let db_iris_shares = iris_share.all_rotations();
                    let db_mask_shares = mask_share.all_rotations();

                    // With Lagrange interpolation.
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut iris_share);
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut mask_share);

                    (
                        store_iris_shares,
                        store_mask_shares,
                        db_iris_shares,
                        db_mask_shares,
                        iris_share.all_rotations(),
                        mask_share.all_rotations(),
                    )
                })
                .await?;

                batch_query.store.code.push(store_iris_shares);
                batch_query.store.mask.push(store_mask_shares);
                batch_query.db.code.extend(db_iris_shares);
                batch_query.db.mask.extend(db_mask_shares);
                batch_query.query.code.extend(iris_shares);
                batch_query.query.mask.extend(mask_shares);
            }
        }
    }

    Ok(batch_query)
}

fn initialize_tracing(config: &Config) -> eyre::Result<TracingShutdownHandle> {
    if let Some(service) = &config.service {
        let tracing_shutdown_handle = DatadogBattery::init(
            service.traces_endpoint.as_deref(),
            &service.service_name,
            None,
            true,
        );

        if let Some(metrics_config) = &service.metrics {
            StatsdBattery::init(
                &metrics_config.host,
                metrics_config.port,
                metrics_config.queue_size,
                metrics_config.buffer_size,
                Some(&metrics_config.prefix),
            )?;
        }

        Ok(tracing_shutdown_handle)
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().pretty().compact())
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .init();

        Ok(TracingShutdownHandle)
    }
}

async fn initialize_chacha_seeds(
    kms_key_arns: &JsonStrWrapper<Vec<String>>,
    party_id: usize,
) -> eyre::Result<([u32; 8], [u32; 8])> {
    // Init RNGs
    let own_key_arn = kms_key_arns
        .0
        .get(party_id)
        .expect("Expected value not found in kms_key_arns");
    let dh_pairs = match party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let dh_pair_0: &str = kms_key_arns
        .0
        .get(dh_pairs.0)
        .expect("Expected value not found in kms_key_arns");
    let dh_pair_1: &str = kms_key_arns
        .0
        .get(dh_pairs.1)
        .expect("Expected value not found in kms_key_arns");

    let chacha_seeds = (
        bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_0).await?),
        bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_1).await?),
    );

    Ok(chacha_seeds)
}

async fn initialize_iris_dbs(
    party_id: usize,
    store: &Store,
) -> eyre::Result<(Vec<u16>, Vec<u16>, usize)> {
    // Generate or load DB
    let (mut codes_db, mut masks_db) = {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let codes_db = db
            .db
            .iter()
            .flat_map(|iris| {
                GaloisRingIrisCodeShare::encode_iris_code(
                    &iris.code,
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .collect::<Vec<_>>();

        let masks_db = db
            .db
            .iter()
            .flat_map(|iris| {
                GaloisRingIrisCodeShare::encode_mask_code(
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .collect::<Vec<_>>();

        (codes_db, masks_db)
    };

    // Load DB from persistent storage.
    let random_len = codes_db.len();
    while let Some(iris) = store.stream_irises().await.next().await {
        let iris = iris?;
        codes_db.extend(iris.left_code());
        masks_db.extend(iris.left_mask());
    }
    let store_len = codes_db.len() - random_len;

    Ok((codes_db, masks_db, store_len))
}

async fn startup_sync(
    config: &Config,
    device_manager: &DeviceManager,
    store: &Store,
    db_len: usize,
) -> eyre::Result<SyncResult> {
    let mut syncer = Syncer::new(
        config.party_id,
        config.servers.bootstrap_url.clone(),
        config.servers.sync_port,
        device_manager.device(0),
    );

    let my_state = SyncState {
        db_len:              db_len as u64,
        deleted_request_ids: store.last_deleted_requests(SYNC_QUERIES).await?,
    };
    let result = syncer.sync(&my_state)?;

    // Not using the syncer anymore, stop it.
    syncer.stop();
    Ok(result)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    dotenvy::dotenv().ok();
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    let _tracing_shutdown_handle = initialize_tracing(&config)?;
    let store = Store::new_from_config(&config).await?;

    // TODO: probably move into separate function
    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let sqs_client = Client::new(&shared_config);
    let sns_client = SNSClient::new(&shared_config);

    let party_id = config.party_id;
    let chacha_seeds = initialize_chacha_seeds(&config.kms_key_arns, party_id).await?;

    let (codes_db, masks_db, store_len) = initialize_iris_dbs(party_id, &store).await?;

    let device_manager = Arc::new(DeviceManager::init());

    let sync_result = startup_sync(&config, &device_manager, &store, store_len).await?;
    if let Some(db_len) = sync_result.must_rollback_storage() {
        eprintln!("Databases are out-of-sync: {:?}", sync_result);
        store.rollback(db_len).await?;
        return Err(eyre!("Rolled back to common state. Restarting…"));
    }
    // Input queues will be synchronized while consuming them.
    let mut skip_request_ids = sync_result.deleted_request_ids();

    let mut background_tasks = TaskMonitor::new();
    // a bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let (tx, rx) = oneshot::channel();
    background_tasks.spawn_blocking(move || {
        let actor = match ServerActor::new_with_device_manager(
            config.party_id,
            config.servers,
            chacha_seeds,
            &codes_db,
            &masks_db,
            device_manager,
            8,
        ) {
            Ok((actor, handle)) => {
                tx.send(Ok(handle)).unwrap();
                actor
            }
            Err(e) => {
                tx.send(Err(e)).unwrap();
                return Ok(());
            }
        };
        actor.run();
        Ok(())
    });
    background_tasks.check_tasks();
    let mut handle = rx.await??;

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let rx_sns_client = sns_client.clone();

    let store_bg = store.clone();
    let _result_sender_abort = background_tasks.spawn(async move {
        while let Some(ServerJobResult {
            merged_results,
            request_ids,
            matches,
            store: query_store,
        }) = rx.recv().await
        {
            // Insert non-matching queries into the persistent store.
            {
                let codes_and_masks: Vec<StoredIrisRef> = matches
                    .iter()
                    .enumerate()
                    .filter_map(
                        // Find the indices of non-matching queries in the batch.
                        |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
                    )
                    .map(|query_idx| {
                        // Get the original vectors from `receive_batch`.
                        let code = &query_store.code[query_idx].coefs[..];
                        let mask = &query_store.mask[query_idx].coefs[..];
                        StoredIrisRef {
                            left_code:  code,
                            left_mask:  mask,
                            // TODO: second eye.
                            right_code: &[],
                            right_mask: &[],
                        }
                    })
                    .collect();

                store_bg
                    .insert_irises(&codes_and_masks)
                    .await
                    .wrap_err("failed to persist queries")?;
            }

            for (i, &idx_result) in merged_results.iter().enumerate() {
                // Notify consumers about result
                println!("Sending results back to SNS...");
                let result_event =
                    ResultEvent::new(party_id, idx_result, matches[i], request_ids[i].clone());

                rx_sns_client
                    .publish()
                    .topic_arn(&config.results_topic_arn)
                    .message(serde_json::to_string(&result_event).unwrap())
                    .send()
                    .await?;
            }
        }

        Ok(())
    });
    background_tasks.check_tasks();

    println!("All systems ready.");
    println!("Starting healthcheck server.");

    let _health_check_abort = background_tasks.spawn(async move {
        let app = Router::new().route("/health", get(|| async {})); // implicit 200 return
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .wrap_err("healthcheck listener bind error")?;
        axum::serve(listener, app)
            .await
            .wrap_err("healthcheck listener server launch error")?;

        Ok(())
    });
    background_tasks.check_tasks();

    let mut total_time = Instant::now();
    let mut batch_times = Duration::from_secs(0);

    // Main loop
    for request_counter in 0..N_BATCHES {
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

        // Skip first iteration
        if request_counter == 1 {
            total_time = Instant::now();
            batch_times = Duration::from_secs(0);
        }
        let now = Instant::now();

        // Skip requests based on the startup sync, only in the first iteration.
        let skip_request_ids = mem::take(&mut skip_request_ids);

        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above
        let batch = receive_batch(
            party_id,
            &sqs_client,
            &config.requests_queue_url,
            &store,
            skip_request_ids,
        )
        .await?;

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

        // start trace span - with single TraceId and single ParentTraceID
        println!("Received batch in {:?}", now.elapsed());
        batch_times += now.elapsed();
        background_tasks.check_tasks();

        let result_future = handle.submit_batch_query(batch).await;

        // await the result
        let result = result_future.await;
        tx.send(result).await.unwrap();
        println!("CPU time of one iteration {:?}", now.elapsed());

        // wrap up span context
    }
    // drop actor handle to initiate shutdown
    drop(handle);

    println!(
        "Total time for {} iterations: {:?}",
        N_BATCHES - 1,
        total_time.elapsed() - batch_times
    );

    // Clean up server tasks, then wait for them to finish
    background_tasks.abort_all();
    tokio::time::sleep(Duration::from_secs(5)).await;
    // Check for background task hangs and shutdown panics
    background_tasks.check_tasks_finished();

    Ok(())
}
