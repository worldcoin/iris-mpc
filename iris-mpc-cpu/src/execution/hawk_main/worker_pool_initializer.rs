//! Setup-time construction of per-eye worker pools and the metadata-only
//! `VectorIdRegistry`. Each registry is derived from its eye's iris store
//! via `to_registry`, so it mirrors exactly what was loaded.

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
use eyre::Result;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::VectorId;
use iris_mpc_store::loader::{load_iris_db, LoadIrisDbOptions};
use iris_mpc_store::rerand::RerandContext;
use iris_mpc_store::Store;
use itertools::izip;
use std::sync::Arc;
use tokio::try_join;

/// Output of `WorkerPoolInitializer::initialize`.
pub struct InitializedWorkers {
    pub pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    /// Metadata-only registries with `next_id` and `set_hash` populated.
    pub registries: BothEyes<VectorIdRegistryRef>,
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
    pub rerand: Option<RerandContext>,
}

/// Strategy for populating the local pools' iris stores at startup.
pub enum LocalInitMode {
    /// Empty stores; no load.
    Empty,
    /// Install pre-built iris stores (test seeded path).
    Seeded(BothEyes<Aby3SharedIrises>),
    /// Run `load_iris_db` against empty stores.
    LoadFromDb(DbLoadParams),
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
                    rerand,
                } = params;
                let mut adapter = FanoutLoader {
                    party_id,
                    iris_pools: workers_handle.clone(),
                    db_size: 0,
                };
                load_iris_db(
                    &mut adapter,
                    &store,
                    LoadIrisDbOptions {
                        max_serial_id_to_load: max_serial_id,
                        store_load_parallelism: parallelism,
                        s3_max_serial_id_to_load: s3_max_serial_id,
                        config: &config,
                        download_shutdown_handler: shutdown_handler,
                        rerand: rerand.as_ref(),
                    },
                )
                .await?;
                // Drain the channels so every fire-and-forget `Insert` lands
                // in the store before we read it back via `to_registry`.
                try_join!(
                    workers_handle[LEFT].wait_completion(),
                    workers_handle[RIGHT].wait_completion(),
                )?;
                db_size = adapter.db_size;
                [
                    iris_stores[LEFT].data.read().await.to_registry().to_arc(),
                    iris_stores[RIGHT].data.read().await.to_registry().to_arc(),
                ]
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

        tracing::info!(
            "Workers initialized. Checksums: L={:#x} R={:#x}, db_size={}",
            post_load_checksums[LEFT],
            post_load_checksums[RIGHT],
            db_size,
        );

        Ok(InitializedWorkers { pools, registries })
    }
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

    fn fake_db(&mut self, _size: usize) {
        unreachable!(
            "FanoutLoader is only used for LoadFromDb; load_iris_db never invokes fake_db"
        );
    }
}
