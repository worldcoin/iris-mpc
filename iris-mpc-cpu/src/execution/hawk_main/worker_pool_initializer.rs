//! Setup-time construction of per-eye worker pools and the metadata-only
//! `VectorIdRegistry`. Each registry is derived from its eye's iris store
//! via `to_registry`, so it mirrors exactly what was loaded.

use crate::execution::hawk_main::iris_worker::{
    init_workers, ColdStorageInit, IrisPoolHandle, IrisWorkerPool, LocalIrisWorkerPool,
};
use crate::execution::hawk_main::{BothEyes, HawkOps, LEFT, RIGHT};
use crate::hawkers::aby3::aby3_store::{
    Aby3SharedIrises, Aby3Store, DistanceMode, VectorIdRegistryRef,
};
use crate::hawkers::shared_irises::{SharedIrises, SharedIrisesRef};
use crate::protocol::shared_iris::{GaloisRingSharedIris, ResidentIris, ResidentLayout};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use async_trait::async_trait;
use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::VectorId;
use iris_mpc_store::loader::load_iris_db;
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
    /// If set, materialize only this eye. The other eye remains database-backed,
    /// retaining its LUC lookback window while fetching older candidates sparsely.
    pub resident_side: Option<usize>,
}

/// Strategy for populating the local pools' iris stores at startup.
pub enum LocalInitMode {
    /// Empty stores; no load.
    Empty,
    /// Install pre-built iris stores (test seeded path).
    Seeded(BothEyes<Aby3SharedIrises>),
    /// Run `load_iris_db` against empty stores.
    LoadFromDb(DbLoadParams),
    /// Metadata and bounded raw caches only; the spectral eye is populated by
    /// the coordinated migration after networking is initialized.
    LoadSpectralFromDb(DbLoadParams),
}

pub struct LocalWorkerPoolInitializer {
    pub party_id: usize,
    pub distance_mode: DistanceMode,
    pub numa: bool,
    pub mode: LocalInitMode,
    /// Resident representation of the pools' iris stores. `U16` (default)
    /// keeps plain `ArcIris` values as required by the HNSW hot paths;
    /// exact-scan actors opt into `preferred_scan_layout()`.
    pub layout: ResidentLayout,
}

impl LocalWorkerPoolInitializer {
    pub fn new_spectral_from_db(
        party_id: usize,
        distance_mode: DistanceMode,
        numa: bool,
        params: DbLoadParams,
    ) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::LoadSpectralFromDb(params),
            layout: ResidentLayout::U16,
        }
    }
    pub fn new_empty(party_id: usize, distance_mode: DistanceMode, numa: bool) -> Self {
        Self {
            party_id,
            distance_mode,
            numa,
            mode: LocalInitMode::Empty,
            layout: ResidentLayout::U16,
        }
    }

    /// Choose the resident representation of the pools' iris stores.
    pub fn with_resident_layout(mut self, layout: ResidentLayout) -> Self {
        self.layout = layout;
        self
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
            layout: ResidentLayout::U16,
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
            layout: ResidentLayout::U16,
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
            layout,
        } = *self;

        let mode = match mode {
            LocalInitMode::LoadSpectralFromDb(params) => {
                return initialize_spectral_backing(party_id, distance_mode, numa, params).await;
            }
            other => other,
        };

        // Materialize the iris stores. `Seeded` installs caller-provided
        // stores; the rest start blank.
        let iris_stores: BothEyes<SharedIrisesRef<ResidentIris>> = match &mode {
            LocalInitMode::Seeded(seeds) => seeds.clone().map(|seed| {
                seed.map_values(|iris| ResidentIris::from_arc(iris, layout))
                    .to_arc()
            }),
            _ => [LEFT, RIGHT].map(|_| {
                Aby3Store::<HawkOps>::new_storage(None)
                    .map_values(|iris| ResidentIris::from_arc(iris, layout))
                    .to_arc()
            }),
        };

        let workers_handle: BothEyes<IrisPoolHandle> =
            [LEFT, RIGHT].map(|side| init_workers(side, iris_stores[side].clone(), numa, layout));

        let mut db_size: usize = 0;
        let mut cold_storage: Option<(Store, usize, usize, usize)> = None;

        // INVARIANT: each eye gets its own `Arc<RwLock>`. `Aby3Store::insert`
        // allocates `next_id` per eye, so sharing one Arc would advance
        // both eyes' ids on every insert.
        let registries: BothEyes<VectorIdRegistryRef> = match mode {
            LocalInitMode::LoadSpectralFromDb(_) => unreachable!("handled before materialization"),
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
                    resident_side,
                } = params;
                let mut adapter = FanoutLoader {
                    party_id,
                    iris_pools: workers_handle.clone(),
                    db_size: 0,
                    resident_side,
                };
                if let Some(side) = resident_side {
                    let luc_window_capacity =
                        if config.luc_enabled && config.luc_lookback_records > 0 {
                            config.luc_lookback_records + 1
                        } else {
                            0
                        };
                    cold_storage = Some((
                        store.clone(),
                        side,
                        luc_window_capacity,
                        config.cold_eye_lfu_cache_records,
                    ));
                }
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
                // Drain the channels so every fire-and-forget `Insert` lands
                // in the store before we read it back via `to_registry`.
                try_join!(
                    workers_handle[LEFT].wait_completion(),
                    workers_handle[RIGHT].wait_completion(),
                )?;
                db_size = adapter.db_size;
                if let Some(side) = resident_side {
                    let registry = iris_stores[side].data.read().await.to_registry();
                    [registry.clone().to_arc(), registry.to_arc()]
                } else {
                    [
                        iris_stores[LEFT].data.read().await.to_registry().to_arc(),
                        iris_stores[RIGHT].data.read().await.to_registry().to_arc(),
                    ]
                }
            }
        };

        let resident_side = cold_storage.as_ref().map(|(_, side, _, _)| *side);
        let mut cold_worker =
            if let Some((store, resident_side, luc_window_capacity, lfu_cache_capacity)) =
                cold_storage
            {
                let cold_side = 1 - resident_side;
                let registry = registries[cold_side].read().await;
                let luc_window_ids = registry.last_vector_ids(luc_window_capacity);
                drop(registry);
                Some((
                    cold_side,
                    LocalIrisWorkerPool::new_cold(
                        workers_handle[cold_side].clone(),
                        iris_stores[cold_side].clone(),
                        layout,
                        distance_mode,
                        party_id,
                        ColdStorageInit {
                            store,
                            side: cold_side,
                            luc_window_ids,
                            luc_window_capacity,
                            lfu_cache_capacity,
                        },
                    )
                    .await?,
                ))
            } else {
                None
            };

        let pools: BothEyes<Arc<dyn IrisWorkerPool>> = [LEFT, RIGHT].map(|side| {
            let worker = if cold_worker
                .as_ref()
                .is_some_and(|(cold_side, _)| *cold_side == side)
            {
                cold_worker.take().expect("cold worker exists").1
            } else {
                LocalIrisWorkerPool::new(
                    workers_handle[side].clone(),
                    iris_stores[side].clone(),
                    layout,
                    distance_mode,
                    party_id,
                )
            };
            Arc::new(worker) as Arc<dyn IrisWorkerPool>
        });

        let post_load_checksums = [
            registries[LEFT].read().await.set_hash.checksum(),
            registries[RIGHT].read().await.set_hash.checksum(),
        ];

        match resident_side {
            Some(resident_side) => {
                // Only the resident eye was loaded; the other registry is a
                // copy of it, so a second checksum would not be an independent
                // check of anything.
                let resident = if resident_side == LEFT { "L" } else { "R" };
                tracing::info!(
                    "Workers initialized. Resident {resident} checksum={:#x} (the other eye is \
                     database-backed and shares this registry), db_size={}",
                    post_load_checksums[resident_side],
                    db_size,
                );
            }
            None => tracing::info!(
                "Workers initialized. Checksums: L={:#x} R={:#x}, db_size={}",
                post_load_checksums[LEFT],
                post_load_checksums[RIGHT],
                db_size,
            ),
        }

        Ok(InitializedWorkers { pools, registries })
    }
}

/// `InMemoryStore` adapter that fans a single PG read into both eyes'
/// worker pools.
struct FanoutLoader {
    party_id: usize,
    iris_pools: BothEyes<IrisPoolHandle>,
    db_size: usize,
    resident_side: Option<usize>,
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
        for (side, (pool, code, mask)) in izip!(
            &self.iris_pools,
            [left_code, right_code],
            [left_mask, right_mask]
        )
        .enumerate()
        {
            if self.resident_side.is_some_and(|resident| resident != side) {
                continue;
            }
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
        for (side, pool) in self.iris_pools.iter().enumerate() {
            if self.resident_side.is_none_or(|resident| resident == side) {
                pool.reserve(additional).unwrap();
            }
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

/// Read only IDs at startup. Both raw eyes use the same bounded cache machinery;
/// the selected eye additionally receives a resident spectral store.
async fn initialize_spectral_backing(
    party_id: usize,
    distance_mode: DistanceMode,
    numa: bool,
    params: DbLoadParams,
) -> Result<InitializedWorkers> {
    use super::spectral::SpectralState;
    let side = params
        .resident_side
        .ok_or_else(|| eyre::eyre!("NTT mode requires a fixed full-scan eye"))?;
    eyre::ensure!(side < 2, "invalid spectral full-scan side");
    let mut registry = SharedIrises::<()>::default();
    let mut rows = sqlx::query_as::<_, (i64, i16)>("SELECT id, version_id FROM irises ORDER BY id")
        .fetch(&params.store.pool);
    while let Some((id, version)) = rows.try_next().await? {
        registry.insert(VectorId::new(u32::try_from(id)?, version), ());
    }
    let luc_capacity = if params.config.luc_enabled && params.config.luc_lookback_records > 0 {
        params.config.luc_lookback_records + 1
    } else {
        0
    };
    let luc_ids = registry.last_vector_ids(luc_capacity);
    let registries = [registry.clone().to_arc(), registry.to_arc()];
    let mut pools = Vec::with_capacity(2);
    for eye in [LEFT, RIGHT] {
        let raw = Aby3Store::<HawkOps>::new_storage(None)
            .map_values(|iris| ResidentIris::from_arc(iris, ResidentLayout::U16))
            .to_arc();
        let workers = init_workers(eye, raw.clone(), numa, ResidentLayout::U16);
        let mut pool = LocalIrisWorkerPool::new_cold(
            workers.clone(),
            raw,
            ResidentLayout::U16,
            distance_mode,
            party_id,
            ColdStorageInit {
                store: params.store.clone(),
                side: eye,
                luc_window_ids: luc_ids.clone(),
                luc_window_capacity: luc_capacity,
                lfu_cache_capacity: params.config.cold_eye_lfu_cache_records,
            },
        )
        .await?;
        if eye == side {
            pool = pool.with_spectral(Arc::new(SpectralState::new(
                params.store.clone(),
                eye,
                party_id,
                workers,
            )))?;
        }
        pools.push(Arc::new(pool) as Arc<dyn IrisWorkerPool>);
    }
    let pools = pools
        .try_into()
        .map_err(|_| eyre::eyre!("expected two iris worker pools"))?;
    Ok(InitializedWorkers { pools, registries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_eye_loader_does_not_materialize_cold_eye() -> Result<()> {
        let stores: BothEyes<SharedIrisesRef<ResidentIris>> = [LEFT, RIGHT].map(|_| {
            Aby3Store::<HawkOps>::new_storage(None)
                .map_values(|iris| ResidentIris::from_arc(iris, ResidentLayout::U16))
                .to_arc()
        });
        let handles = [LEFT, RIGHT]
            .map(|side| init_workers(side, stores[side].clone(), false, ResidentLayout::U16));
        let iris = GaloisRingSharedIris::default_for_party(0);
        let id = VectorId::from_0_index(7);
        let mut loader = FanoutLoader {
            party_id: 0,
            iris_pools: handles.clone(),
            db_size: 0,
            resident_side: Some(RIGHT),
        };

        loader.reserve(1);
        loader.load_single_record_from_db(
            7,
            id,
            &iris.code.coefs,
            &iris.mask.coefs,
            &iris.code.coefs,
            &iris.mask.coefs,
        );
        try_join!(
            handles[LEFT].wait_completion(),
            handles[RIGHT].wait_completion(),
        )?;

        assert_eq!(stores[LEFT].data.read().await.db_size(), 0);
        assert_eq!(stores[RIGHT].data.read().await.db_size(), 1);
        assert!(stores[RIGHT].data.read().await.get_vector(&id).is_some());
        Ok(())
    }
}
