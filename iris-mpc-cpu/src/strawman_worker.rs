//! Strawman worker process for the remote `IrisWorkerPool`.
//!
//! Wraps `LocalIrisWorkerPool` with the ampc-actor-utils workpool worker
//! protocol so a leader (Hawk Main) can drive iris computations over TCP
//! using the wire types from `iris-mpc-worker-protocol`. Used as a
//! development stand-in until the production worker binary lands; it is
//! also the conformance baseline that the production worker must match.
//!
//! ## Throwaway scope
//!
//! When the production worker arrives, this module + its bin go away.
//! Until then, this is the only complete, runnable end-to-end remote
//! pool that can drive integration tests.
//!
//! ## v1 limitations (intentional)
//!
//! - **Single shard.** `shard_index` is accepted as a knob but not used
//!   for routing; the worker serves the entire `VectorId` space it sees.
//!   Sharding lives on the leader for the cross-machine case and will be
//!   layered in once `RemoteIrisWorkerPool` exists.
//! - **No TLS.** Plain TCP only. Fine for loopback tests and dev
//!   clusters; production swaps in `TlsConfig`.
//! - **No persistence.** The store is in-memory. Tests pre-seed by
//!   either calling `insert_irises` over the wire or constructing a
//!   `SharedIrisesRef` directly via `run_local_with_store`.
//! - **No readiness signal.** The worker accepts jobs as soon as TCP is
//!   up; tests that require seeded state pre-populate before the leader
//!   dispatches.

use ampc_actor_utils::network::workpool::{
    worker::{build_worker_handle, Job, WorkerArgs},
    Payload,
};
use eyre::{eyre, Result};
use iris_mpc_worker_protocol::{
    decode_request, encode_response, IrisShare, QueryId as WireQueryId, QuerySpec as WireQuerySpec,
    WorkerRequest, WorkerResponse,
};
use std::{collections::HashMap, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::{
    execution::hawk_main::iris_worker::{
        init_workers, IrisWorkerPool, LocalIrisWorkerPool, QueryId, QuerySpec,
    },
    hawkers::{
        aby3::aby3_store::DistanceMode,
        shared_irises::{SharedIrises, SharedIrisesRef},
    },
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};

/// Configuration for spawning a strawman worker.
#[derive(Clone, Debug)]
pub struct StrawmanWorkerArgs {
    pub worker_id: String,
    pub worker_address: String,
    pub leader_id: String,
    pub leader_address: String,
    pub party_id: usize,
    pub distance_mode: DistanceMode,
    pub numa: bool,
    /// Reserved for future shard routing. v1 strawman ignores this.
    pub shard_index: usize,
}

/// Run a strawman worker that delegates each request to the given pool.
///
/// Useful for tests that want to wrap an arbitrary `IrisWorkerPool` impl
/// (e.g. a pre-populated `LocalIrisWorkerPool`) and expose it over the
/// wire. Returns when `shutdown_ct` is cancelled or the worker handle's
/// job channel closes.
pub async fn run_with_pool(
    args: StrawmanWorkerArgs,
    pool: Arc<dyn IrisWorkerPool>,
    shutdown_ct: CancellationToken,
) -> Result<()> {
    let mut handle = build_worker_handle(
        WorkerArgs {
            worker_id: args.worker_id.into(),
            worker_address: args.worker_address,
            leader_id: args.leader_id.into(),
            leader_address: args.leader_address,
            tls: None,
        },
        shutdown_ct.clone(),
    )
    .await
    .map_err(|e| eyre!("worker handle setup failed: {e:?}"))?;

    loop {
        tokio::select! {
            _ = shutdown_ct.cancelled() => break,
            maybe_job = handle.recv() => match maybe_job {
                None => break,
                Some(job) => {
                    let pool = pool.clone();
                    tokio::spawn(handle_job(pool, job));
                }
            }
        }
    }

    Ok(())
}

/// Convenience: build a `LocalIrisWorkerPool` over an empty store and
/// run the strawman against it. Intended for the bin entry point and
/// for tests that don't need to pre-seed.
pub async fn run_local(args: StrawmanWorkerArgs, shutdown_ct: CancellationToken) -> Result<()> {
    let empty_iris: ArcIris = Arc::new(GaloisRingSharedIris::default_for_party(args.party_id));
    let store = SharedIrises::<ArcIris>::new(HashMap::new(), empty_iris).to_arc();
    run_local_with_store(args, store, shutdown_ct).await
}

/// Build a `LocalIrisWorkerPool` over the given (possibly pre-seeded)
/// store and run the strawman against it. Useful for tests that want a
/// hot worker without going through `insert_irises`.
pub async fn run_local_with_store(
    args: StrawmanWorkerArgs,
    store: SharedIrisesRef<ArcIris>,
    shutdown_ct: CancellationToken,
) -> Result<()> {
    let inner = init_workers(args.shard_index, store.clone(), args.numa);
    let pool: Arc<dyn IrisWorkerPool> = Arc::new(LocalIrisWorkerPool::new(
        inner,
        store,
        args.distance_mode,
        args.party_id,
    ));
    run_with_pool(args, pool, shutdown_ct).await
}

async fn handle_job(pool: Arc<dyn IrisWorkerPool>, job: Job) {
    let mut job = job;
    let bytes = job.take_payload().to_bytes();
    let response = match decode_request(&bytes) {
        Ok(req) => dispatch(&pool, req).await,
        Err(e) => WorkerResponse::ProtocolError(format!("decode: {e}")),
    };
    let encoded = match encode_response(&response) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("encode_response failed: {e}");
            return;
        }
    };
    job.send_result(Payload::Bytes(encoded.into()));
}

async fn dispatch(pool: &Arc<dyn IrisWorkerPool>, req: WorkerRequest) -> WorkerResponse {
    match req {
        WorkerRequest::CacheQueries { queries } => {
            let mapped = queries
                .into_iter()
                .map(|(qid, iris)| (to_qid(qid), to_arc_iris(iris)))
                .collect();
            WorkerResponse::CacheQueries(
                pool.cache_queries(mapped).await.map_err(|e| e.to_string()),
            )
        }
        WorkerRequest::ComputeDotProducts { batches } => {
            let mapped = batches
                .into_iter()
                .map(|(spec, vids)| (to_spec(spec), vids))
                .collect();
            WorkerResponse::ComputeDotProducts(
                pool.compute_dot_products(mapped)
                    .await
                    .map(|results| {
                        results
                            .into_iter()
                            .map(|inner| inner.into_iter().map(|r| r.0).collect())
                            .collect()
                    })
                    .map_err(|e| e.to_string()),
            )
        }
        WorkerRequest::FetchIrises { ids } => WorkerResponse::FetchIrises(
            pool.fetch_irises(ids)
                .await
                .map(|irises| irises.into_iter().map(arc_iris_to_share).collect())
                .map_err(|e| e.to_string()),
        ),
        WorkerRequest::InsertIrises { inserts } => {
            let mapped = inserts
                .into_iter()
                .map(|(qid, vid)| (to_qid(qid), vid))
                .collect();
            WorkerResponse::InsertIrises(
                pool.insert_irises(mapped).await.map_err(|e| e.to_string()),
            )
        }
        WorkerRequest::ComputePairwiseDistances { pairs } => {
            let mapped = pairs
                .into_iter()
                .map(|opt| opt.map(|(spec, qid)| (to_spec(spec), to_qid(qid))))
                .collect();
            WorkerResponse::ComputePairwiseDistances(
                pool.compute_pairwise_distances(mapped)
                    .await
                    .map(|res| res.into_iter().map(|r| r.0).collect())
                    .map_err(|e| e.to_string()),
            )
        }
        WorkerRequest::EvictQueries { query_ids } => {
            let mapped = query_ids.into_iter().map(to_qid).collect();
            WorkerResponse::EvictQueries(
                pool.evict_queries(mapped).await.map_err(|e| e.to_string()),
            )
        }
        WorkerRequest::DeleteIrises { ids } => {
            WorkerResponse::DeleteIrises(pool.delete_irises(ids).await.map_err(|e| e.to_string()))
        }
    }
}

fn to_qid(q: WireQueryId) -> QueryId {
    QueryId(q.0)
}

fn to_spec(s: WireQuerySpec) -> QuerySpec {
    QuerySpec {
        query_id: to_qid(s.query_id),
        rotation: s.rotation as usize,
        mirrored: s.mirrored,
    }
}

fn to_arc_iris(iris: IrisShare) -> ArcIris {
    Arc::new(GaloisRingSharedIris {
        code: iris.code,
        mask: iris.mask,
    })
}

fn arc_iris_to_share(arc: ArcIris) -> IrisShare {
    IrisShare {
        code: arc.code.clone(),
        mask: arc.mask.clone(),
    }
}
