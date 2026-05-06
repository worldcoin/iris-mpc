//! Setup-time construction of per-eye worker pools and the metadata-only
//! `VectorIdRegistry`. Registries are derived from the load source
//! (PG id+version scan, seed map, or known size) — independent of
//! the worker pool's iris store.

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

/// Output of `WorkerPoolInitializer::initialize`.
pub struct InitializedWorkers {
    pub pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    /// Metadata-only registries with `next_id` and `set_hash` populated.
    pub registries: BothEyes<VectorIdRegistryRef>,
    /// Per-eye `set_hash` reported by each pool after load. For remote
    /// pools this is an aggregate across shards. Compared against the
    /// registry-side checksum to catch a loader that dropped rows.
    pub post_load_checksums: BothEyes<u64>,
    pub db_size: usize,
}

/// One-shot setup for the per-eye worker pools.
#[async_trait]
pub trait WorkerPoolInitializer: Send {
    async fn initialize(self: Box<Self>) -> Result<InitializedWorkers>;
}

/// Inputs for `iris_mpc_store::loader::load_iris_db`.
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
    /// Empty stores; no load.
    Empty,
    /// Install pre-built iris stores (test seeded path).
    Seeded(BothEyes<Aby3SharedIrises>),
    /// Run `load_iris_db` against empty stores.
    LoadFromDb(DbLoadParams),
    /// Fill each store with `n` copies of the party's default iris.
    /// Backs the production `fake_db_size > 0` path.
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

        // Materialize the iris stores. `Seeded` installs caller-provided
        // stores; the rest start blank.
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

        let workers_handle: BothEyes<IrisPoolHandle> =
            [LEFT, RIGHT].map(|side| init_workers(side, iris_stores[side].clone(), numa));

        let mut db_size: usize = 0;

        // INVARIANT: each eye gets its own `Arc<RwLock>`. `Aby3Store::insert`
        // allocates `next_id` per eye, so sharing one Arc would advance
        // both eyes' ids on every insert.
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
                // Iris load (full rows) and id+version scan (index-only)
                // run in parallel against the same PG.
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
                // Drain the channels so every fire-and-forget `Insert`
                // lands in the store before we read `set_hash`.
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

        // Cross-check: registry-side checksum (load source) must match
        // pool-side `set_hash` (loaded data). Mismatch = loader dropped
        // or duplicated rows.
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

        Ok(InitializedWorkers {
            pools,
            registries,
            post_load_checksums,
            db_size,
        })
    }
}

/// Build a registry template from a `(serial_id, version_id)` index-only
/// scan of PG (no iris bytes). Returns the inner `SharedIrises<()>`;
/// caller wraps each eye in its own `Arc<RwLock>`.
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

/// Build a registry template for serial ids in `range`, all version 0.
fn build_registry_from_serial_range(range: std::ops::Range<u32>) -> SharedIrises<()> {
    let mut points: HashMap<VectorId, ()> = HashMap::with_capacity(range.len());
    for serial in range {
        points.insert(VectorId::from_serial_id(serial), ());
    }
    SharedIrises::<()>::new(points, ())
}

/// `InMemoryStore` adapter that fans a single PG read into both eyes'
/// worker pools.
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
