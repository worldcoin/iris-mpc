//! Setup-time abstraction for building worker pools and (optionally)
//! loading iris data into them. The runtime trait `IrisWorkerPool` stays
//! narrow (compute only); loading is a one-shot bootstrap step that does
//! not appear on the wire.
//!
//! The initializer also produces the metadata-only `VectorIdRegistry` that
//! sessions read from. Registries are derived from the same source as the
//! iris-data load (PG id+version scan for `LoadFromDb`, the seed map for
//! `Seeded`, the known size for `FakeDb`), never by reading back from the
//! iris store. This keeps `HawkActor` decoupled from iris storage — the
//! actor only needs the registry, the worker pool trait, and the graph.
//!
//! # Concrete impls
//! - [`LocalWorkerPoolInitializer`] — builds in-process worker pools and
//!   either pre-seeds them, fills with `fake_db`, or runs `load_iris_db`.

use crate::execution::hawk_main::iris_worker::{
    init_workers, IrisPoolHandle, IrisWorkerPool, LocalIrisWorkerPool,
};
use crate::execution::hawk_main::{BothEyes, HawkOps, LEFT, RIGHT};
use crate::hawkers::aby3::aby3_store::{
    Aby3SharedIrises, Aby3SharedIrisesRef, Aby3Store, DistanceMode, VectorIdRegistryRef,
};
use crate::hawkers::shared_irises::SharedIrises;
use crate::protocol::shared_iris::GaloisRingSharedIris;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use async_trait::async_trait;
use eyre::{eyre, Result};
use futures::TryStreamExt;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::vector_id::VectorId;
use iris_mpc_store::loader::load_iris_db;
use iris_mpc_store::Store;
use itertools::izip;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::try_join;

/// Result of `WorkerPoolInitializer::initialize`. Contains ready-to-serve
/// worker pools (one per eye), the metadata-only registries the actor
/// hands to sessions, and per-eye load checksums.
///
/// Notably **does not** expose the underlying iris stores or any
/// pool-specific handles: all iris reads must go through the
/// [`IrisWorkerPool`] trait. Local and remote pool impls share the
/// same external surface, and the actor that consumes this struct does
/// not need to know which one it has.
pub struct InitializedWorkers {
    pub pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    /// Metadata-only registries reflecting the loaded VectorId set, with
    /// `next_id` and `set_hash` already populated. Sessions read from
    /// these; the actor never needs to derive them from an iris store.
    pub registries: BothEyes<VectorIdRegistryRef>,
    /// `set_hash` checksum reported by each pool after load. For local
    /// pools this is the in-process `SharedIrises::set_hash`; for a
    /// future remote pool it would be an aggregate across shards.
    /// Compared against `registries[side].set_hash` to catch a loader
    /// that silently dropped rows.
    pub post_load_checksums: BothEyes<u64>,
    /// Total irises loaded. Mirrors what `IrisLoader::increment_db_size`
    /// accumulated under the old API.
    pub db_size: usize,
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

        // Per-eye registries derived from the same source as the iris
        // load (seed map / known size / DB id+version scan). Built
        // alongside the iris load and never read back from the pool.
        //
        // Both eyes start from identical (id, version) sets but must
        // hold **independent** `Arc<RwLock>` instances — runtime inserts
        // allocate per-eye via `Aby3Store::insert`, and a shared Arc
        // would burn two `next_id`s per logical insert.
        let registries: BothEyes<VectorIdRegistryRef> = match mode {
            LocalInitMode::Empty => [
                SharedIrises::<()>::default().to_arc(),
                SharedIrises::<()>::default().to_arc(),
            ],
            LocalInitMode::Seeded(ref seeds) => [
                seeds[LEFT].to_registry().to_arc(),
                seeds[RIGHT].to_registry().to_arc(),
            ],
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
                // Run the full iris load and the id+version scan in
                // parallel. The id+version scan is index-only on PG and
                // produces the registry + expected checksum without
                // reading the code/mask blobs.
                let (_, template) = try_join!(
                    async {
                        load_iris_db(
                            &mut adapter,
                            &store,
                            max_serial_id,
                            parallelism,
                            s3_max_serial_id,
                            &config,
                            shutdown_handler,
                        )
                        .await
                    },
                    build_registry_from_db(&store, max_serial_id),
                )?;
                // Drain the worker channels so every fire-and-forget
                // `Insert` lands in the store before we read `set_hash`.
                try_join!(
                    workers_handle[LEFT].wait_completion(),
                    workers_handle[RIGHT].wait_completion(),
                )?;
                db_size = adapter.db_size;
                [template.clone().to_arc(), template.to_arc()]
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
                let template = build_registry_from_serial_range(0..size as u32);
                [template.clone().to_arc(), template.to_arc()]
            }
        };

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

        // Cross-check: registry-derived checksum (from the load source)
        // must match the pool's reported `set_hash` checksum (from the
        // loaded data). A mismatch means the loader silently dropped or
        // duplicated rows relative to what the registry-side scan saw.
        for side in [LEFT, RIGHT] {
            let expected = registries[side].data.read().await.set_hash.checksum();
            let got = post_load_checksums[side];
            if expected != got {
                return Err(eyre!(
                    "worker pool checksum mismatch on side {side}: \
                     registry expected {expected:#x}, pool reported {got:#x} \
                     — loader dropped or duplicated rows vs the registry scan"
                ));
            }
        }

        // `workers_handle` is kept alive only via `pools` from here on:
        // each `LocalIrisWorkerPool` owns its own `IrisPoolHandle` clone.
        // Nothing outside the pool needs it.
        Ok(InitializedWorkers {
            pools,
            registries,
            post_load_checksums,
            db_size,
        })
    }
}

/// Stream `(serial_id, version_id)` from PG and build a metadata-only
/// registry template (with `next_id` and `set_hash` populated). Reads
/// no iris bytes — uses the `stream_iris_ids` index-only query.
///
/// Returns the inner `SharedIrises<()>` rather than an `Arc<RwLock>`
/// because callers want **independent** per-eye `Arc<RwLock>` instances
/// — runtime inserts allocate `next_id` per eye, and a shared Arc would
/// have one logical insert burn two ids.
pub async fn build_registry_from_db(
    store: &Store,
    max_serial_id: usize,
) -> Result<SharedIrises<()>> {
    let mut points: HashMap<VectorId, ()> = HashMap::with_capacity(max_serial_id);
    let mut stream = store.stream_iris_ids(max_serial_id);
    while let Some((id, version_id)) = stream.try_next().await? {
        points.insert(VectorId::new(id as u32, version_id), ());
    }
    Ok(SharedIrises::<()>::new(points, ()))
}

/// Build a registry template covering serial ids in `range`, all with
/// version 0. Used by `FakeDb` initializers and equivalent shortcut
/// paths. Returned as `SharedIrises<()>` so callers can wrap each eye
/// in its own `Arc<RwLock>` — see `build_registry_from_db`.
fn build_registry_from_serial_range(range: std::ops::Range<u32>) -> SharedIrises<()> {
    let mut points: HashMap<VectorId, ()> = HashMap::with_capacity(range.len());
    for serial in range {
        points.insert(VectorId::from_serial_id(serial), ());
    }
    SharedIrises::<()>::new(points, ())
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
