#![recursion_limit = "256"]
//! Conformance test: drive the existing `Aby3Store` HNSW flow through
//! `RemoteIrisWorkerPool` instead of `LocalIrisWorkerPool`. Mirrors the
//! existing `test_gr_hnsw` from `aby3_store.rs::tests` end-to-end; if this
//! passes, the wire path (encode → broadcast → strawman → LocalIrisWorkerPool
//! → encode → decode) is observably equivalent to the in-process path for
//! the production-shaped configuration.
//!
//! When the production worker binary lands, the only thing that should
//! change is which binary the strawman spawn is replaced with — the rest
//! of the test stays as-is. That's the conformance guarantee we need.

use ampc_actor_utils::{
    execution::player::Identity,
    network::{
        mpc::NetworkType,
        workpool::leader::{build_leader, LeaderArgs, LeaderHandle},
    },
};
use eyre::Result;
use iris_mpc_common::iris_db::db::IrisDB;
use iris_mpc_cpu::{
    execution::{
        hawk_main::{
            iris_worker::{cache_irises, IrisWorkerPool, RemoteIrisWorkerPool},
            HAWK_DISTANCE_MODE,
        },
        local::LocalRuntime,
    },
    hawkers::aby3::{
        aby3_store::{Aby3Store, FhdOps},
        test_utils::get_owner_index,
    },
    hnsw::{GraphMem, HnswSearcher, SortedNeighborhood},
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};
use std::{sync::Arc, time::Duration};
use tokio::{sync::Mutex, task::JoinHandle, task::JoinSet};
use tokio_util::sync::CancellationToken;

/// Local mirror of the (private) `test_utils::Aby3StoreRef` so this
/// integration test doesn't need to crack open crate internals.
type Aby3StoreRef = Arc<Mutex<Aby3Store>>;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

fn find_free_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect()
}

/// Spin up one strawman worker + one leader per party, plug a
/// `RemoteIrisWorkerPool` into each `Aby3Store`. Returns the stores and a
/// guard that keeps worker tasks alive + cancels them on drop.
async fn setup_remote_store_aby3_players(
    network_t: NetworkType,
) -> Result<(Vec<Aby3StoreRef>, RemoteClusterGuard)> {
    let runtime = LocalRuntime::mock_setup(network_t).await?;
    let n_parties = runtime.sessions.len();

    let ports = find_free_ports(2 * n_parties);
    let shutdown = CancellationToken::new();
    let mut worker_tasks: Vec<JoinHandle<()>> = Vec::with_capacity(n_parties);
    let mut stores: Vec<Aby3StoreRef> = Vec::with_capacity(n_parties);

    for (idx, session) in runtime.sessions.into_iter().enumerate() {
        let party_id = session.network_session.own_role.index();
        let leader_addr = format!("127.0.0.1:{}", ports[idx * 2]);
        let worker_addr = format!("127.0.0.1:{}", ports[idx * 2 + 1]);
        let leader_id_str = format!("party-{idx}");
        // Workpool naming convention: each leader's workers must announce
        // `{leader_id}-w-{worker_index}`.
        let worker_id_str = format!("{leader_id_str}-w-0");

        // Strawman worker for this party — empty in-memory store; tests
        // load through cache_queries / insert_irises like normal.
        let strawman_args = iris_mpc_cpu::strawman_worker::StrawmanWorkerArgs {
            worker_id: worker_id_str.clone(),
            worker_address: worker_addr.clone(),
            leader_id: leader_id_str.clone(),
            leader_address: leader_addr.clone(),
            party_id,
            distance_mode: HAWK_DISTANCE_MODE,
            numa: false,
            shard_index: 0,
        };
        let worker_shutdown = shutdown.clone();
        worker_tasks.push(tokio::spawn(async move {
            iris_mpc_cpu::strawman_worker::run_local(strawman_args, worker_shutdown)
                .await
                .expect("strawman worker exited with error");
        }));

        // Leader for this party.
        let leader: LeaderHandle = {
            let mut l = build_leader(
                LeaderArgs {
                    leader_id: Identity(leader_id_str),
                    leader_address: leader_addr,
                    worker_addresses: vec![worker_addr],
                    tls: None,
                },
                shutdown.clone(),
            )
            .await
            .expect("failed to build leader");
            l.wait_for_all_connections(Some(CONNECT_TIMEOUT))
                .await
                .expect("worker failed to connect");
            l
        };

        let workers: Arc<dyn IrisWorkerPool> =
            Arc::new(RemoteIrisWorkerPool::new(Arc::new(leader), 1, party_id));

        // Empty leader-side registry. The actual iris data lives on the
        // strawman worker; the registry only tracks `VectorId` presence
        // for `only_valid_vectors`, which Aby3Store keeps locally.
        let storage = Aby3Store::<FhdOps>::new_storage(None).to_arc();
        let registry = storage.read().await.to_registry().to_arc();

        stores.push(Arc::new(Mutex::new(Aby3Store::new(
            registry,
            session,
            workers,
            HAWK_DISTANCE_MODE,
        ))));
    }

    Ok((
        stores,
        RemoteClusterGuard {
            worker_tasks,
            shutdown,
        },
    ))
}

/// RAII guard for a remote test cluster. On drop, cancels the shutdown
/// token and joins worker tasks.
struct RemoteClusterGuard {
    worker_tasks: Vec<JoinHandle<()>>,
    shutdown: CancellationToken,
}

impl Drop for RemoteClusterGuard {
    fn drop(&mut self) {
        self.shutdown.cancel();
        for task in self.worker_tasks.drain(..) {
            task.abort();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_gr_hnsw_through_remote_pool() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    use aes_prng::AesRng;
    use rand::SeedableRng;

    let mut rng = AesRng::seed_from_u64(0_u64);
    // Smaller than the in-process equivalent (10) — the wire round-trips
    // make this longer per insert. 5 is enough to cover insert + search +
    // is_match across the trait surface. Bump if we want stress coverage.
    let database_size = 5;
    let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
    let shared_irises: Vec<_> = cleartext_database
        .iter()
        .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
        .collect();

    let (stores, _guard) = setup_remote_store_aby3_players(NetworkType::Local).await?;

    let mut jobs = JoinSet::new();
    for store in stores.iter() {
        let player_index = get_owner_index(store).await?;
        let irises: Vec<ArcIris> = (0..database_size)
            .map(|id| Arc::new(shared_irises[id][player_index].clone()))
            .collect();
        let queries = cache_irises(store.lock().await.workers.as_ref(), irises).await?;
        let mut rng = rng.clone();
        let store = store.clone();
        jobs.spawn(async move {
            let mut store = store.lock().await;
            let mut aby3_graph = GraphMem::new();
            let db = HnswSearcher::new_with_test_parameters();

            let mut inserted = vec![];
            for query in queries.iter() {
                let insertion_layer = db.gen_layer_rng(&mut rng).unwrap();
                let inserted_vector = db
                    .insert::<_, SortedNeighborhood<_>>(
                        &mut *store,
                        &mut aby3_graph,
                        query,
                        insertion_layer,
                    )
                    .await
                    .unwrap();
                inserted.push(inserted_vector)
            }

            let mut matching_results = vec![];
            for v in inserted.into_iter() {
                let query = store.cache_query_from_store(&v).await.unwrap();
                let neighbors = db
                    .search::<_, SortedNeighborhood<_>>(&mut *store, &aby3_graph, &query, 1)
                    .await
                    .unwrap();
                matching_results.push(db.is_match(&mut *store, &[neighbors]).await.unwrap())
            }
            matching_results
        });
    }
    let matching_results = jobs.join_all().await;
    for (party_id, party_results) in matching_results.iter().enumerate() {
        for (index, result) in party_results.iter().enumerate() {
            assert!(
                *result,
                "party {party_id} index {index}: expected match for inserted vector \
                 (this would indicate a wire-path divergence from LocalIrisWorkerPool)"
            );
        }
    }

    Ok(())
}
