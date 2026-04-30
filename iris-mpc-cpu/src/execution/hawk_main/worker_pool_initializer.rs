//! Setup-time abstraction for building worker pools and (optionally)
//! loading iris data into them. The runtime trait `IrisWorkerPool` stays
//! narrow (compute only); loading is a one-shot bootstrap step that does
//! not appear on the wire.
//!
//! See `.claude/plans/worker-pool-initializer.md` for design context.
//!
//! # Concrete impls
//! - [`LocalWorkerPoolInitializer`] — builds in-process worker pools and
//!   either pre-seeds them, fills with `fake_db`, or runs `load_iris_db`.
//! - [`RemoteWorkerPoolInitializer`] — stub. The wire variant
//!   (`WorkerRequest::LoadDb`) and `LoadDirective` shape are unblocked by
//!   coordination with the production worker author; until then, calling
//!   `initialize` panics.

use crate::execution::hawk_main::iris_worker::{
    init_workers, IrisPoolHandle, IrisWorkerPool, LocalIrisWorkerPool,
};
use crate::execution::hawk_main::{BothEyes, HawkOps, LEFT, RIGHT};
use crate::hawkers::aby3::aby3_store::{
    Aby3SharedIrises, Aby3SharedIrisesRef, Aby3Store, DistanceMode,
};
use crate::hawkers::shared_irises::SharedIrises;
use crate::protocol::shared_iris::GaloisRingSharedIris;
use ampc_actor_utils::network::workpool::leader::LeaderHandle;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use async_trait::async_trait;
use eyre::Result;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::vector_id::VectorId;
use iris_mpc_store::loader::load_iris_db;
use iris_mpc_store::Store;
use itertools::izip;
use std::sync::Arc;
use tokio::try_join;

/// Result of `WorkerPoolInitializer::initialize`. Contains ready-to-serve
/// worker pools (one per eye), the underlying iris stores (so the actor
/// can build registries / expose snapshots), and per-eye load checksums.
pub struct InitializedWorkers {
    pub pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    /// Iris stores backing each pool. The actor uses these to build the
    /// registry and to read iris snapshots; for `RemoteIrisWorkerPool`
    /// they are not the canonical store (the worker holds that).
    pub iris_stores: BothEyes<Aby3SharedIrisesRef>,
    /// `set_hash` checksum per eye after load. Logged by the actor for
    /// ops cross-check; future work may compare against an expected value.
    pub post_load_checksums: BothEyes<u64>,
    /// Total irises loaded across both eyes' loaders. Mirrors what
    /// `IrisLoader::increment_db_size` accumulated under the old API.
    pub db_size: usize,
    /// Local worker handles, retained for paths that load iris data after
    /// actor construction (genesis: iris load runs after `sync_peers` and
    /// optional DB rollback). `None` for remote pools — those load via
    /// their own bootstrap protocol on the wire and don't accept
    /// post-construction loads.
    pub local_pool_handles: Option<BothEyes<IrisPoolHandle>>,
}

/// One-shot setup for the per-eye worker pools.
#[async_trait]
pub trait WorkerPoolInitializer: Send {
    /// Build both eyes' worker pools, populate their stores, and return
    /// the ready handles. Consumes the initializer.
    async fn initialize(self: Box<Self>) -> Result<InitializedWorkers>;
}

/// Inputs for `iris_mpc_store::loader::load_iris_db`. Lifted into a
/// struct so the initializer can carry a single owned bundle.
pub struct DbLoadParams {
    pub store: Store,
    pub config: Arc<Config>,
    pub max_serial_id: usize,
    pub parallelism: usize,
    pub s3_max_serial_id: Option<usize>,
    pub shutdown_handler: Arc<ShutdownHandler>,
}

/// Strategy for populating the local pools' iris stores at startup.
pub enum LocalInitMode {
    /// Build empty stores; do not load. Used when persistence is
    /// disabled or by tests that don't care about the data.
    Empty,
    /// Install pre-built iris stores (e.g. tests that seed irises via
    /// `Aby3Store::new_storage(Some(...))` before construction).
    Seeded(BothEyes<Aby3SharedIrises>),
    /// Run `load_iris_db` against the empty stores.
    LoadFromDb(DbLoadParams),
    /// Fill each store with `n` copies of the party's default iris —
    /// the production `fake_db_size > 0` path.
    FakeDb(usize),
}

pub struct LocalWorkerPoolInitializer {
    pub party_id: usize,
    pub distance_mode: DistanceMode,
    pub numa: bool,
    pub mode: LocalInitMode,
}

impl LocalWorkerPoolInitializer {
    pub fn new_empty(party_id: usize, distance_mode: DistanceMode, numa: bool) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::Empty,
        }
    }

    pub fn new_seeded(
        party_id: usize,
        distance_mode: DistanceMode,
        numa: bool,
        seed_stores: BothEyes<Aby3SharedIrises>,
    ) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::Seeded(seed_stores),
        }
    }

    pub fn new_load_from_db(
        party_id: usize,
        distance_mode: DistanceMode,
        numa: bool,
        params: DbLoadParams,
    ) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::LoadFromDb(params),
        }
    }

    pub fn new_fake_db(
        party_id: usize,
        distance_mode: DistanceMode,
        numa: bool,
        size: usize,
    ) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::FakeDb(size),
        }
    }
}

#[async_trait]
impl WorkerPoolInitializer for LocalWorkerPoolInitializer {
    async fn initialize(self: Box<Self>) -> Result<InitializedWorkers> {
        let LocalWorkerPoolInitializer {
            party_id,
            distance_mode,
            numa,
            mode,
        } = *self;

        // Materialize the iris stores. `Empty`, `LoadFromDb`, and `FakeDb`
        // all start from the same blank-slate `new_storage(None)` shape;
        // `Seeded` installs caller-provided stores wholesale.
        let iris_stores: BothEyes<Aby3SharedIrisesRef> = match &mode {
            LocalInitMode::Seeded(seeds) => {
                let [left, right] = seeds.clone();
                [SharedIrises::to_arc(left), SharedIrises::to_arc(right)]
            }
            _ => [
                Aby3Store::<HawkOps>::new_storage(None).to_arc(),
                Aby3Store::<HawkOps>::new_storage(None).to_arc(),
            ],
        };

        // Build worker thread pools backed by these stores. The same
        // `IrisPoolHandle` is used both for loading (via the fan-out
        // adapter) and for serving — there is no separate "loader pool"
        // anymore.
        let workers_handle: BothEyes<IrisPoolHandle> =
            [LEFT, RIGHT].map(|side| init_workers(side, iris_stores[side].clone(), numa));

        let mut db_size: usize = 0;

        match mode {
            LocalInitMode::Empty | LocalInitMode::Seeded(_) => {}
            LocalInitMode::LoadFromDb(params) => {
                let DbLoadParams {
                    store,
                    config,
                    max_serial_id,
                    parallelism,
                    s3_max_serial_id,
                    shutdown_handler,
                } = params;
                let mut adapter = FanoutLoader {
                    party_id,
                    iris_pools: workers_handle.clone(),
                    db_size: 0,
                };
                load_iris_db(
                    &mut adapter,
                    &store,
                    max_serial_id,
                    parallelism,
                    s3_max_serial_id,
                    &config,
                    shutdown_handler,
                )
                .await?;
                // Drain the worker channels so every fire-and-forget
                // `Insert` lands in the store before we read `set_hash`.
                try_join!(
                    workers_handle[LEFT].wait_completion(),
                    workers_handle[RIGHT].wait_completion(),
                )?;
                db_size = adapter.db_size;
            }
            LocalInitMode::FakeDb(size) => {
                let dummy = Arc::new(GaloisRingSharedIris::default_for_party(party_id));
                for side in [LEFT, RIGHT] {
                    for i in 0..size {
                        workers_handle[side]
                            .insert(VectorId::from_serial_id(i as u32), dummy.clone())?;
                    }
                }
                try_join!(
                    workers_handle[LEFT].wait_completion(),
                    workers_handle[RIGHT].wait_completion(),
                )?;
                db_size = size;
            }
        }

        let pools: BothEyes<Arc<dyn IrisWorkerPool>> = [LEFT, RIGHT].map(|side| {
            Arc::new(LocalIrisWorkerPool::new(
                workers_handle[side].clone(),
                iris_stores[side].clone(),
                distance_mode,
                party_id,
            )) as Arc<dyn IrisWorkerPool>
        });

        let post_load_checksums = [
            iris_stores[LEFT].data.read().await.set_hash.checksum(),
            iris_stores[RIGHT].data.read().await.set_hash.checksum(),
        ];

        Ok(InitializedWorkers {
            pools,
            iris_stores,
            post_load_checksums,
            db_size,
            local_pool_handles: Some(workers_handle),
        })
    }
}

/// Run `load_iris_db` against an already-constructed pair of local worker
/// handles, fanning each PG record into both eyes. Used by genesis, where
/// iris loading must happen after the actor has been built (so `sync_peers`
/// and a possible iris-DB rollback can run first).
///
/// Returns the number of records loaded.
pub async fn load_iris_db_through_pools(
    party_id: usize,
    iris_pools: BothEyes<IrisPoolHandle>,
    params: DbLoadParams,
) -> Result<usize> {
    let DbLoadParams {
        store,
        config,
        max_serial_id,
        parallelism,
        s3_max_serial_id,
        shutdown_handler,
    } = params;
    let mut adapter = FanoutLoader {
        party_id,
        iris_pools: iris_pools.clone(),
        db_size: 0,
    };
    load_iris_db(
        &mut adapter,
        &store,
        max_serial_id,
        parallelism,
        s3_max_serial_id,
        &config,
        shutdown_handler,
    )
    .await?;
    try_join!(
        iris_pools[LEFT].wait_completion(),
        iris_pools[RIGHT].wait_completion(),
    )?;
    Ok(adapter.db_size)
}

/// `InMemoryStore` adapter that fans a single PG read into both eyes'
/// worker pools. Mirrors the old `IrisLoader` body, kept private to the
/// initializer.
struct FanoutLoader {
    party_id: usize,
    iris_pools: BothEyes<IrisPoolHandle>,
    db_size: usize,
}

const IRIS_STORE_RESERVE_EXTRA: f64 = 0.2;

impl InMemoryStore for FanoutLoader {
    fn load_single_record_from_db(
        &mut self,
        _index: usize,
        vector_id: VectorId,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        for (pool, code, mask) in izip!(
            &self.iris_pools,
            [left_code, right_code],
            [left_mask, right_mask]
        ) {
            let iris = GaloisRingSharedIris::try_from_buffers(self.party_id, code, mask)
                .expect("Wrong code or mask size");
            pool.insert(vector_id, iris).unwrap();
        }
    }

    fn increment_db_size(&mut self, _index: usize) {
        self.db_size += 1;
    }

    fn reserve(&mut self, additional: usize) {
        let additional = additional + (additional as f64 * IRIS_STORE_RESERVE_EXTRA) as usize;
        for side in &self.iris_pools {
            side.reserve(additional).unwrap();
        }
    }

    fn current_db_sizes(&self) -> impl std::fmt::Debug {
        self.db_size
    }

    fn fake_db(&mut self, size: usize) {
        self.db_size = size;
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(self.party_id));
        for side in &self.iris_pools {
            for i in 0..size {
                side.insert(VectorId::from_serial_id(i as u32), iris.clone())
                    .unwrap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Remote initializer — stubbed. Lands once `LoadDirective` / `LoadDb` wire
// types are settled with the production worker author.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct RemoteWorkerPoolInitializer {
    pub leaders: BothEyes<Arc<LeaderHandle>>,
    pub num_shards: usize,
    pub party_id: usize,
    pub distance_mode: DistanceMode,
    // TODO: `pub directive: LoadDirective` once the wire variant ships.
}

#[async_trait]
impl WorkerPoolInitializer for RemoteWorkerPoolInitializer {
    async fn initialize(self: Box<Self>) -> Result<InitializedWorkers> {
        // TODO: Send `WorkerRequest::LoadDb { directive }` to every worker
        // across both eyes' leaders, await `LoadAck`s, combine per-shard
        // checksums, and wrap each leader as `RemoteIrisWorkerPool`. The
        // wire variant + `LoadDirective` shape are open questions tracked
        // in `.claude/plans/worker-pool-initializer.md` (Wire surface).
        unimplemented!(
            "RemoteWorkerPoolInitializer is stubbed pending LoadDirective \
             coordination with the production worker author"
        )
    }
}
