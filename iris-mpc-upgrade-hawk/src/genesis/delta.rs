//! Version-join delta phase for genesis.
//!
//! Reconciles the in-memory HNSW graph and the HNSW iris store to the source
//! content by a `(serial → version)` comparison against the base checkpoint.
//! The public entry point is [`exec_delta`]; the coordination-server slots
//! ([`DeltaExchangeSlots`], [`delta_slot_route`]) are wired up by the caller
//! before the delta runs.

use ampc_server_utils::{
    shutdown_handler::ShutdownHandler, try_get_endpoint_other_nodes, ServerCoordinationConfig,
    TaskMonitor,
};
use aws_sdk_s3::Client as S3Client;
use axum::routing::get;
use eyre::{bail, eyre, Result};
use iris_mpc_common::{
    config::Config, helpers::sync::Modification, iris_db::get_dummy_shares_for_deletion, SerialId,
    VectorId, VersionId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{
        iris_worker::IrisWorkerPool, state_check::SetHash, BothEyes, GraphRef, HawkOps, LEFT, RIGHT,
    },
    genesis::{
        state_accessor::set_last_indexed_modification_id,
        version_join::{
            compute_version_join, make_repair_plan, versions_per_serial, RepairPlan,
            VersionJoinPlan,
        },
        Handle as GenesisHawkHandle, JobRequest, JobResult,
    },
    hawkers::aby3::aby3_store::{Aby3Store, VectorIdRegistryRef},
    hnsw::graph::graph_store::GraphPg,
    protocol::shared_iris::ArcIris,
};
use iris_mpc_store::{ExplicitVersionToken, Store as IrisStore, StoredIrisRef};
use itertools::izip;
use std::collections::{HashMap, HashSet};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{sync::mpsc::Sender, time::timeout};

use super::{upload_and_sync_genesis_checkpoint, ExecutionContextInfo, PERSIST_DELAY};

// Delta consensus exchange over the coordination server: each node publishes
// its value into a slot served on a GET route and polls the peers' slots.
pub(super) const DELTA_REPAIR_ROUTE: &str = "/delta-repair";
const DELTA_REPAIR_ENDPOINT: &str = "delta-repair";
pub(super) const DELTA_TOMBSTONES_ROUTE: &str = "/delta-tombstones";
const DELTA_TOMBSTONES_ENDPOINT: &str = "delta-tombstones";
const DELTA_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const DELTA_EXCHANGE_POLL: Duration = Duration::from_millis(500);

pub(super) type DeltaExchangeSlot = Arc<tokio::sync::RwLock<Option<Vec<u8>>>>;

/// Server-side slots backing the delta consensus routes.
pub(super) struct DeltaExchangeSlots {
    pub(super) repair: DeltaExchangeSlot,
    pub(super) tombstones: DeltaExchangeSlot,
}

impl DeltaExchangeSlots {
    pub(super) fn new() -> Self {
        Self {
            repair: Arc::new(tokio::sync::RwLock::new(None)),
            tombstones: Arc::new(tokio::sync::RwLock::new(None)),
        }
    }
}

/// GET handler for a delta exchange slot: 503 until the local value is
/// published, then its serialized bytes.
pub(super) fn delta_slot_route(slot: DeltaExchangeSlot) -> axum::routing::MethodRouter {
    get(move || {
        let slot = slot.clone();
        async move {
            match slot.read().await.clone() {
                Some(payload) => (axum::http::StatusCode::OK, payload),
                None => (axum::http::StatusCode::SERVICE_UNAVAILABLE, Vec::new()),
            }
        }
    })
}

/// One exchange round: publish `value` on this node's slot, poll the peers'
/// `endpoint` until both answer, and return their values (peer order as
/// returned by the coordination layer).
///
/// Callers must rendezvous first ([`delta_exchange_barrier`]): the deadline
/// starts at publish, and the work preceding an exchange runs at party-local
/// speed.
async fn exchange_delta_state<T: serde::Serialize + serde::de::DeserializeOwned>(
    server_coord_config: &ServerCoordinationConfig,
    slot: &DeltaExchangeSlot,
    endpoint: &str,
    value: &T,
) -> Result<Vec<T>> {
    *slot.write().await = Some(serde_json::to_vec(value)?);
    let deadline = Instant::now() + DELTA_EXCHANGE_TIMEOUT;
    loop {
        let responses = try_get_endpoint_other_nodes(server_coord_config, endpoint).await?;
        if responses.iter().all(|(status, _)| status.is_success()) {
            return responses
                .into_iter()
                .map(|(_, body)| serde_json::from_slice(&body).map_err(Into::into))
                .collect();
        }
        if Instant::now() > deadline {
            bail!("timed out waiting for peers on {endpoint}");
        }
        tokio::time::sleep(DELTA_EXCHANGE_POLL).await;
    }
}

/// Rendezvous ahead of a timed delta exchange, carrying the shutdown flag; a
/// shutdown on any party aborts the delta (a partial delta must not proceed).
async fn delta_exchange_barrier(
    hawk_handle: &mut GenesisHawkHandle,
    shutdown_handler: &ShutdownHandler,
) -> Result<()> {
    let shutdown = shutdown_handler.is_shutting_down();
    let mismatched = hawk_handle.sync_state(shutdown, None).await?;
    if shutdown || mismatched {
        bail!("shutdown signalled during version-join delta");
    }
    Ok(())
}

/// The planning outcome for a delta: the repair actions to apply, plus the
/// per-eye graph key lists needed to derive per-serial removals during the
/// graph repair.
struct DeltaPlan {
    repair: RepairPlan,
    graph_versions: [HashMap<SerialId, Vec<VersionId>>; 2],
}

/// Apply the version-join delta: reconcile the graph and HNSW iris store to the
/// source by `(serial → version)` comparison against the base checkpoint.
///
/// The base checkpoint has already been loaded into the in-memory graph and the
/// HNSW schema reset to it (see `reset_to_checkpoint`). Computes per-eye plans,
/// unions them across eyes and parties, runs the graph repair via the Hawk
/// handle, and only then writes iris rows.
///
/// Ordering invariant: no HNSW row becomes source-consistent before the
/// repaired graph is durable (the post-delta checkpoint). Row disagreements are
/// repair triggers; writing a row first and crashing before the checkpoint
/// would erase the trigger while the distrusted graph survives. A crash
/// anywhere before the flush leaves rows untouched, so a rerun re-derives the
/// same plan. With no graph change the flush runs immediately (no graph
/// durability at stake).
///
/// The flow is a linear pipeline: [`compute_delta_plan`] derives the plan
/// (steps 1–4), [`apply_graph_repair`] applies it (step 5), and this function
/// finalizes (cursor write, checkpoint, row flush) or tears down on error.
#[allow(clippy::too_many_arguments)]
pub(super) async fn exec_delta(
    config: &Config,
    ctx: &ExecutionContextInfo,
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
    s3_client: &S3Client,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: &IrisStore,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    delta_exchange: &DeltaExchangeSlots,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    task_monitor_bg: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<GenesisHawkHandle> {
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;

    let res: Result<(bool, RepairPlan)> = async {
        let plan = compute_delta_plan(
            config,
            ctx,
            registries,
            worker_pools,
            hnsw_iris_store,
            imem_graph_stores,
            delta_exchange,
            &mut hawk_handle,
            server_coord_config,
            shutdown_handler,
        )
        .await?;
        let graph_changed =
            apply_graph_repair(config, &plan, &mut hawk_handle, tx_results, shutdown_handler)
                .await?;
        Ok((graph_changed, plan.repair))
    }
    .await;

    match res {
        Ok((graph_changed, repair)) => {
            tracing::info!("Waiting for version replays to be processed...");
            let _ = shutdown_handler.wait_for_pending_batches_completion().await;
            tracing::info!("All version replays have been processed");

            // Single end-of-delta cursor write (no per-replay cursor writes):
            // set to the global max persisted+completed source modification id
            // read before the pools loaded (safe under-claim).
            tracing::info!(
                "Setting last indexed modification id to {}",
                ctx.max_modification_persist_id
            );
            let mut graph_tx = graph_store.tx().await?;
            set_last_indexed_modification_id(&mut graph_tx.tx, ctx.max_modification_persist_id)
                .await?;
            graph_tx.tx.commit().await?;

            // S3 checkpoint after delta, only if the graph changed (row writes
            // do not touch the graph, so the base checkpoint still represents it).
            if graph_changed {
                tracing::info!("Creating S3 checkpoint after version-join delta...");
                upload_and_sync_genesis_checkpoint(
                    &config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    ctx.last_indexed_id,
                    ctx.max_modification_persist_id,
                    true, // is_archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                tracing::info!("S3 checkpoint created after version-join delta");
            }

            // Row writes last: the graph repair is durable, so erasing the
            // row-level repair triggers is now safe.
            flush_row_writes(&repair, registries, worker_pools, hnsw_iris_store).await?;

            Ok(hawk_handle)
        }
        Err(err) => {
            tracing::error!(
                "HawkActor processing error while applying version-join delta: {:?}",
                err
            );
            tracing::info!("Initiating shutdown");
            drop(hawk_handle);
            task_monitor_bg.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor_bg.check_tasks_finished();
            Err(err)
        }
    }
}

/// Derive the repair plan (steps 1–4): build the per-eye version state, union
/// the repair set across eyes and parties, classify tombstones by content, and
/// emit the comparison log. Purely computes state and exchanges consensus; no
/// graph mutation or row writes happen here.
#[allow(clippy::too_many_arguments)]
async fn compute_delta_plan(
    config: &Config,
    ctx: &ExecutionContextInfo,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: &IrisStore,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    delta_exchange: &DeltaExchangeSlots,
    hawk_handle: &mut GenesisHawkHandle,
    server_coord_config: &ServerCoordinationConfig,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<DeltaPlan> {
    // S_cp: the checkpoint's last indexed iris id (ctx.last_indexed_id was set
    // to it during reset). Serials are compared over 1..=S_cp.
    let max_serial = ctx.last_indexed_id;
    // Logged cross-check only; tombstones are detected from source content.
    let excluded: HashSet<SerialId> = ctx.excluded_serial_ids.iter().copied().collect();

    // 1. Build the version state per eye and compute per-eye plans.
    //    versions_from_iris_store is a single DB scan (one iris row covers both eyes);
    //    versions_from_source are eye-invariant (one source row covers both
    //    eyes), so the LEFT registry is authoritative.
    let hnsw_versions = get_versions_from_iris_store(hnsw_iris_store, max_serial).await?;
    let source_versions = get_versions_from_source(&registries[LEFT], max_serial).await;

    let mut graph_versions: [HashMap<SerialId, Vec<VersionId>>; 2] =
        [HashMap::new(), HashMap::new()];
    for side in [LEFT, RIGHT] {
        graph_versions[side] = get_versions_from_hnsw_graph(&imem_graph_stores[side]).await;
    }
    let plans: [VersionJoinPlan; 2] = [LEFT, RIGHT].map(|side| {
        compute_version_join(
            &graph_versions[side],
            &source_versions,
            &hnsw_versions,
            max_serial,
        )
    });

    // Local repair set = union across eyes. Only per-eye graph state can
    // differ (the store axis is eye-invariant); asymmetric per-eye damage
    // is repairable — removals stay per-eye, reinsertion is uniform (a
    // harmless refresh for the clean eye).
    log_per_eye_asymmetries(&plans, &graph_versions, &source_versions);
    let mut local_repair: Vec<SerialId> = plans[LEFT]
        .graph_repair
        .iter()
        .chain(plans[RIGHT].graph_repair.iter())
        .copied()
        .collect();
    local_repair.sort_unstable();
    local_repair.dedup();

    // 2. Cross-party repair membership. Every replay is an interactive
    //    MPC job that all parties must submit identically, but the row
    //    axis is party-local — so exchange the local sets and take the
    //    union. A replay on a party whose local state was clean is a
    //    harmless refresh.
    delta_exchange_barrier(hawk_handle, shutdown_handler).await?;
    let peer_sets: Vec<Vec<SerialId>> = exchange_delta_state(
        server_coord_config,
        &delta_exchange.repair,
        DELTA_REPAIR_ENDPOINT,
        &local_repair,
    )
    .await?;
    let repair_serials = union_repair_with_peers(&local_repair, &peer_sets);

    // 3. Tombstones among repair serials, by content check against the
    //    party's dummy shares. The classification gates the reinsert flag,
    //    so parties must agree; a mismatch means source-level content
    //    inconsistency, which the repair must not paper over.
    let deleted =
        gather_tombstones(config.party_id, &repair_serials, registries, worker_pools).await?;
    let deleted_hash = {
        let mut h = SetHash::default();
        for s in &deleted {
            h.add_unordered(*s);
        }
        h.checksum()
    };
    delta_exchange_barrier(hawk_handle, shutdown_handler).await?;
    let peer_hashes: Vec<u64> = exchange_delta_state(
        server_coord_config,
        &delta_exchange.tombstones,
        DELTA_TOMBSTONES_ENDPOINT,
        &deleted_hash,
    )
    .await?;
    if peer_hashes.iter().any(|h| *h != deleted_hash) {
        bail!(
            "tombstone sets differ across parties (mine={deleted_hash}, peers={peer_hashes:?}, \
             count={}); source content is inconsistent",
            deleted.len()
        );
    }

    let serials_in_graph: HashSet<SerialId> = graph_versions[LEFT]
        .keys()
        .chain(graph_versions[RIGHT].keys())
        .copied()
        .collect();
    let repair = make_repair_plan(
        &repair_serials,
        &plans[LEFT].missing_hnsw_rows,
        &deleted,
        &serials_in_graph,
        &source_versions,
        &hnsw_versions,
    );

    // 4. Comparison log (join vs modifications, deletion cross-checks) + metrics.
    log_version_join_comparison(
        &plans,
        &repair_serials,
        &repair,
        &deleted,
        &excluded,
        &ctx.modifications,
        ctx.max_modification_persist_id,
    );

    Ok(DeltaPlan {
        repair,
        graph_versions,
    })
}

/// Apply the graph repair (step 5): replays and removals in serial order, one
/// job per serial, syncing periodically. Removing an entry point is fine (the
/// searcher falls back to a temporary EP). No row writes here. Returns whether
/// the graph was changed (i.e. any job was submitted).
async fn apply_graph_repair(
    config: &Config,
    plan: &DeltaPlan,
    hawk_handle: &mut GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<bool> {
    let repair = &plan.repair;
    let graph_versions = &plan.graph_versions;

    let mut items: Vec<(SerialId, bool)> = repair
        .graph_replay
        .iter()
        .map(|s| (*s, true))
        .chain(repair.graph_remove.iter().map(|s| (*s, false)))
        .collect();
    items.sort_unstable_by_key(|(serial, _)| *serial);

    if !items.is_empty() {
        metrics::gauge!("genesis_number_version_replays").set(items.len() as f64);
        let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
        let end = items.len().saturating_sub(1);
        let mut now = Instant::now();
        for (idx, (serial, reinsert)) in items.iter().enumerate() {
            // Version lists follow hash iteration order; mutation order
            // must match across parties.
            let removals = [LEFT, RIGHT].map(|side| {
                let mut keys: Vec<VectorId> = graph_versions[side]
                    .get(serial)
                    .map(|versions| {
                        versions
                            .iter()
                            .map(|v| VectorId::new(*serial, *v))
                            .collect()
                    })
                    .unwrap_or_default();
                keys.sort_unstable_by_key(|k| k.version_id());
                keys
            });
            let request = JobRequest::new_version_replay(*serial, removals, *reinsert);
            let result_future = hawk_handle.submit_request(request).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    tracing::error!("HawkActor processing timeout: {:?}", err);
                    eyre!("HawkActor processing timeout: {:?}", err)
                })??;

            let (done_rx, result) = result;
            // Increment before send: the results thread decrements on
            // completion, and send-first would let the counter wrap.
            shutdown_handler.increment_batches_pending_completion();
            tx_results.send(result).await?;

            let is_sync_batch = idx % (PERSIST_DELAY - 1) == 0 || idx == end;
            if is_sync_batch {
                let wait_start = Instant::now();
                let shutdown = shutdown_handler.is_shutting_down();
                let mismatched = hawk_handle.sync_state(shutdown, Some(done_rx)).await?;
                if shutdown || mismatched {
                    // Shutdown on any party: a partial delta must not
                    // proceed to the checkpoint or the row flush.
                    bail!("shutdown signalled during version-join delta");
                }
                metrics::histogram!("genesis_persist_wait_duration")
                    .record(wait_start.elapsed().as_secs_f64());
            }
            metrics::histogram!("genesis_version_replay_total_duration",
                "synced" => if is_sync_batch { "true" } else { "false" })
            .record(now.elapsed().as_secs_f64());
            now = Instant::now();
        }
    }

    Ok(!items.is_empty())
}

/// Scan the HNSW iris store's `(serial → version)` index over `1..=max_serial`.
async fn get_versions_from_iris_store(
    hnsw_iris_store: &IrisStore,
    max_serial: SerialId,
) -> Result<HashMap<SerialId, i16>> {
    use futures::TryStreamExt;
    let rows: Vec<(i64, i16)> = hnsw_iris_store
        .stream_iris_ids(max_serial as usize)
        .try_collect()
        .await?;
    Ok(rows
        .into_iter()
        .map(|(id, version)| (id as SerialId, version))
        .collect())
}

/// Build the graph's per-serial key list from layer-0 node keys (layer 0 holds
/// every node, so this enumerates all graph keys).
async fn get_versions_from_hnsw_graph(graph_ref: &GraphRef) -> HashMap<SerialId, Vec<VersionId>> {
    let graph = graph_ref.read().await;
    if graph.layers.is_empty() {
        return HashMap::new();
    }
    versions_per_serial(
        graph.layers[0]
            .links
            .keys()
            .map(|v| (v.serial_id(), v.version_id())),
    )
}

/// Serials whose source content equals the party's deletion dummy on both
/// eyes.
async fn gather_tombstones(
    party_id: usize,
    serials: &[SerialId],
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
) -> Result<HashSet<SerialId>> {
    let mut out = HashSet::new();
    if serials.is_empty() {
        return Ok(out);
    }
    let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(party_id);
    const CHUNK: usize = 1024;

    for chunk in serials.chunks(CHUNK) {
        let maybe_vids = registries[LEFT].get_vector_ids(chunk).await;
        let mut vids: Vec<VectorId> = Vec::with_capacity(chunk.len());
        for (serial, maybe) in chunk.iter().zip(maybe_vids) {
            let vid = maybe.ok_or_else(|| eyre!("repair serial {serial} missing from registry"))?;
            vids.push(vid);
        }

        let (left_data, right_data) = tokio::try_join!(
            worker_pools[LEFT].fetch_irises(vids.clone()),
            worker_pools[RIGHT].fetch_irises(vids.clone()),
        )?;

        for (i, vid) in vids.iter().enumerate() {
            if left_data[i].code.coefs == dummy_code.coefs
                && left_data[i].mask.coefs == dummy_mask.coefs
                && right_data[i].code.coefs == dummy_code.coefs
                && right_data[i].mask.coefs == dummy_mask.coefs
            {
                out.insert(vid.serial_id());
            }
        }
    }
    Ok(out)
}

/// Build the source `(serial → version)` map from the registry (no DB scan).
async fn get_versions_from_source(
    registry: &VectorIdRegistryRef,
    max_serial: SerialId,
) -> HashMap<SerialId, i16> {
    let reg = registry.read().await;
    let mut out = HashMap::new();
    for serial in 1..=max_serial {
        if let Some(version) = reg.get_current_version(serial) {
            out.insert(serial, version);
        }
    }
    out
}

/// Log serials surgeried in one eye but not the other, with the eye-local
/// graph class. Sample capped.
fn log_per_eye_asymmetries(
    plans: &[VersionJoinPlan; 2],
    graph_versions: &[HashMap<SerialId, Vec<VersionId>>; 2],
    source_versions: &HashMap<SerialId, VersionId>,
) {
    const SAMPLE: usize = 50;

    let graph_class = |side: usize, serial: SerialId| -> &'static str {
        let Some(&v_src) = source_versions.get(&serial) else {
            return "source_missing";
        };
        match graph_versions[side].get(&serial) {
            None => "graph_missing",
            Some(versions) => {
                let v_max = *versions.iter().max().expect("non-empty");
                if v_max < v_src {
                    "version_behind"
                } else if v_max > v_src {
                    "version_ahead"
                } else if versions.len() > 1 {
                    "multi_version"
                } else {
                    "clean"
                }
            }
        }
    };

    for (side, name) in [(LEFT, "left"), (RIGHT, "right")] {
        let other: HashSet<SerialId> = plans[1 - side].graph_repair.iter().copied().collect();
        let only: Vec<(SerialId, &'static str)> = plans[side]
            .graph_repair
            .iter()
            .filter(|s| !other.contains(s))
            .map(|&s| (s, graph_class(side, s)))
            .collect();
        if !only.is_empty() {
            tracing::warn!(
                "version-join: {} serials surgeried only in the {} eye, sample={:?}",
                only.len(),
                name,
                &only[..only.len().min(SAMPLE)],
            );
        }
    }
}

/// Log the join-gated vs modification-gated sets, deletion cross-checks, and
/// per-class gauges. The set hashes cross-check the locally derived sets
/// (tombstones, row repairs), which graph checksums do not cover.
fn log_version_join_comparison(
    plans: &[VersionJoinPlan; 2],
    repair_serials: &[SerialId],
    repair: &RepairPlan,
    deleted: &HashSet<SerialId>,
    excluded: &HashSet<SerialId>,
    modifications: &[Modification],
    max_modification_persist_id: i64,
) {
    const SAMPLE: usize = 50;

    let join_set: HashSet<SerialId> = repair_serials.iter().copied().collect();
    let mod_set: HashSet<SerialId> = modifications
        .iter()
        .filter_map(|m| m.serial_id.map(|s| s as SerialId))
        .collect();

    let intersection = join_set.intersection(&mod_set).count();
    let mut mod_only: Vec<SerialId> = mod_set.difference(&join_set).copied().collect();
    let mut join_only: Vec<SerialId> = join_set.difference(&mod_set).copied().collect();
    let mut mod_all: Vec<SerialId> = mod_set.iter().copied().collect();
    mod_only.sort_unstable();
    join_only.sort_unstable();
    mod_all.sort_unstable();

    // Listed ∧ live = reinserted after deletion (replayed); tombstone ∉ list =
    // deletion the list missed.
    let listed_live: Vec<SerialId> = repair
        .graph_replay
        .iter()
        .filter(|s| excluded.contains(s))
        .copied()
        .collect();
    let mut unlisted_tombstones: Vec<SerialId> = deleted
        .iter()
        .filter(|s| !excluded.contains(s))
        .copied()
        .collect();
    unlisted_tombstones.sort_unstable();

    let hash_of = |serials: &[SerialId]| -> u64 {
        let mut h = SetHash::default();
        for s in serials {
            h.add_unordered(*s);
        }
        h.checksum()
    };
    let sample = |v: &[SerialId]| v[..v.len().min(SAMPLE)].to_vec();

    tracing::info!(
        "version-join comparison: repair={} reasons_left={:?} reasons_right={:?} replay={} \
         remove={} row_inserts={} tombstone_overwrites={} mod_set={} intersection={} mod_only={} \
         join_only={} source_missing={} listed_live={} mod_cursor_stamp={} | replay_hash={} \
         remove_hash={} mod_hash={} row_insert_hash={} tombstone_overwrite_hash={} | \
         mod_only_sample={:?} join_only_sample={:?} row_insert_sample={:?} \
         tombstone_overwrite_sample={:?}",
        repair_serials.len(),
        plans[LEFT].repair_reasons,
        plans[RIGHT].repair_reasons,
        repair.graph_replay.len(),
        repair.graph_remove.len(),
        repair.insert_missing_rows.len(),
        repair.stale_tombstone_rows.len(),
        mod_set.len(),
        intersection,
        mod_only.len(),
        join_only.len(),
        plans[LEFT].source_missing.len(),
        listed_live.len(),
        max_modification_persist_id,
        hash_of(&repair.graph_replay),
        hash_of(&repair.graph_remove),
        hash_of(&mod_all),
        hash_of(&repair.insert_missing_rows),
        hash_of(&repair.stale_tombstone_rows),
        sample(&mod_only),
        sample(&join_only),
        sample(&repair.insert_missing_rows),
        sample(&repair.stale_tombstone_rows),
    );

    if !plans[LEFT].source_missing.is_empty() {
        tracing::error!(
            "version-join: {} serials absent from source pool, sample={:?}",
            plans[LEFT].source_missing.len(),
            sample(&plans[LEFT].source_missing),
        );
    }
    if !unlisted_tombstones.is_empty() {
        tracing::warn!(
            "version-join: {} tombstones not on the S3 deletion list, sample={:?}",
            unlisted_tombstones.len(),
            sample(&unlisted_tombstones),
        );
    }

    metrics::gauge!("genesis_version_join_graph_replay_count")
        .set(repair.graph_replay.len() as f64);
    metrics::gauge!("genesis_version_join_graph_remove_count")
        .set(repair.graph_remove.len() as f64);
    metrics::gauge!("genesis_version_join_row_insert_count")
        .set(repair.insert_missing_rows.len() as f64);
    metrics::gauge!("genesis_version_join_tombstone_overwrite_count")
        .set(repair.stale_tombstone_rows.len() as f64);
    metrics::gauge!("genesis_version_join_mod_only_count").set(mod_only.len() as f64);
    metrics::gauge!("genesis_version_join_join_only_count").set(join_only.len() as f64);
    metrics::gauge!("genesis_version_join_anomaly_count")
        .set(plans[LEFT].source_missing.len() as f64);
}

/// Union the local repair set with the peers', logging per-peer differences
/// (a non-empty difference = party-local row damage). Output sorted.
fn union_repair_with_peers(local: &[SerialId], peer_sets: &[Vec<SerialId>]) -> Vec<SerialId> {
    const SAMPLE: usize = 50;
    let local_set: HashSet<SerialId> = local.iter().copied().collect();
    let mut union: Vec<SerialId> = local.to_vec();
    for (i, peers) in peer_sets.iter().enumerate() {
        let peer_set: HashSet<SerialId> = peers.iter().copied().collect();
        let peer_only: Vec<SerialId> = peers
            .iter()
            .filter(|s| !local_set.contains(s))
            .copied()
            .collect();
        let local_only: Vec<SerialId> = local
            .iter()
            .filter(|s| !peer_set.contains(s))
            .copied()
            .collect();
        if !peer_only.is_empty() || !local_only.is_empty() {
            tracing::warn!(
                "version-join: repair set differs from peer {}: peer_only={} local_only={} \
                 peer_only_sample={:?} local_only_sample={:?}",
                i,
                peer_only.len(),
                local_only.len(),
                &peer_only[..peer_only.len().min(SAMPLE)],
                &local_only[..local_only.len().min(SAMPLE)],
            );
        }
        union.extend(peer_only);
    }
    union.sort_unstable();
    union.dedup();
    union
}

/// Resolve current vector ids and fetch both eyes' pool content for `serials`.
async fn fetch_pool_rows(
    serials: &[SerialId],
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
) -> Result<(Vec<VectorId>, Vec<ArcIris>, Vec<ArcIris>)> {
    // Source versions are eye-invariant; the LEFT registry is authoritative.
    let maybe_vids = registries[LEFT].get_vector_ids(serials).await;
    let mut vids: Vec<VectorId> = Vec::with_capacity(serials.len());
    for (serial, maybe) in serials.iter().zip(maybe_vids) {
        let vid = maybe.ok_or_else(|| eyre!("row-write serial {serial} missing from registry"))?;
        vids.push(vid);
    }
    let (left_data, right_data) = tokio::try_join!(
        worker_pools[LEFT].fetch_irises(vids.clone()),
        worker_pools[RIGHT].fetch_irises(vids.clone()),
    )?;
    Ok((vids, left_data, right_data))
}

/// Flush the deferred row writes from the worker pools (source content):
/// missing rows INSERTed first, then the update set (replayed serials and
/// tombstone overwrites) rewritten in place. Must run only once the rebuilt
/// graph is durable — see the ordering invariant on [`exec_delta`]. Chunked to
/// bound transaction size.
async fn flush_row_writes(
    repair: &RepairPlan,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: &IrisStore,
) -> Result<()> {
    let (inserts, updates) = repair.row_writes();
    if inserts.is_empty() && updates.is_empty() {
        return Ok(());
    }
    const CHUNK: usize = 1024;

    for chunk in inserts.chunks(CHUNK) {
        let (vids, left_data, right_data) =
            fetch_pool_rows(chunk, registries, worker_pools).await?;
        let refs: Vec<StoredIrisRef> = izip!(&vids, &left_data, &right_data)
            .map(|(vid, left, right)| StoredIrisRef {
                id: vid.serial_id() as i64,
                left_code: &left.code.coefs,
                left_mask: &left.mask.coefs,
                right_code: &right.code.coefs,
                right_mask: &right.mask.coefs,
            })
            .collect();
        let mut tx = hnsw_iris_store.tx().await?;
        hnsw_iris_store
            .insert_copy_irises(&mut tx, &vids, &refs)
            .await?;
        tx.commit().await?;
    }

    for chunk in updates.chunks(CHUNK) {
        let (vids, left_data, right_data) =
            fetch_pool_rows(chunk, registries, worker_pools).await?;
        let mut tx = hnsw_iris_store.tx().await?;
        {
            let mut ev = ExplicitVersionToken::enable(&mut tx).await?;
            for (vid, left, right) in izip!(&vids, &left_data, &right_data) {
                let iris_ref = StoredIrisRef {
                    id: vid.serial_id() as i64,
                    left_code: &left.code.coefs,
                    left_mask: &left.mask.coefs,
                    right_code: &right.code.coefs,
                    right_mask: &right.mask.coefs,
                };
                hnsw_iris_store
                    .update_iris_with_version_id(&mut ev, vid.version_id(), &iris_ref)
                    .await?;
            }
        }
        tx.commit().await?;
    }

    tracing::info!(
        "version-join flushed {} row writes ({} inserts, {} updates)",
        inserts.len() + updates.len(),
        inserts.len(),
        updates.len()
    );
    Ok(())
}
