//! Optional resident spectral eye. Raw shares remain database-backed for the
//! existing sparse, intra-batch, LUC, and identity-update protocols.
use super::iris_worker::{IrisPoolHandle, QueryId, QuerySpec, CENTER_ROTATION};
use crate::{
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    protocol::{
        ntt::{
            convert_irises,
            persistence::{load_or_convert_batch, new_generation, persist_batch},
            FieldIris, SpectralIris, SpectralQuery,
        },
        shared_iris::ArcIris,
    },
};
use ampc_actor_utils::execution::session::Session;
use eyre::{ensure, eyre, Result};
use iris_mpc_common::VectorId;
use iris_mpc_store::Store;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
    time::Instant,
};

struct CachedSpectralQuery {
    original: Arc<SpectralIris>,
    normal: Arc<SpectralQuery>,
    mirrored: Arc<SpectralQuery>,
    generation: [u8; 16],
}

pub struct SpectralState {
    store: Store,
    side: usize,
    party: usize,
    workers: IrisPoolHandle,
    resident: SharedIrisesRef<Option<Arc<SpectralIris>>>,
    queries: RwLock<HashMap<QueryId, CachedSpectralQuery>>,
}

impl std::fmt::Debug for SpectralState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralState")
            .field("side", &self.side)
            .field("party", &self.party)
            .finish_non_exhaustive()
    }
}

impl SpectralState {
    pub fn new(store: Store, side: usize, party: usize, workers: IrisPoolHandle) -> Self {
        Self {
            store,
            side,
            party,
            workers,
            resident: SharedIrises::default().to_arc(),
            queries: RwLock::new(HashMap::new()),
        }
    }

    /// Session ownership fixes batch ordering independently of task scheduling.
    /// Every role must supply the same number of sessions and ordered IDs.
    pub async fn initialize(
        self: &Arc<Self>,
        sessions: Vec<Session>,
        ids: Arc<[VectorId]>,
    ) -> Result<()> {
        ensure!(
            !sessions.is_empty(),
            "spectral migration needs at least one session"
        );
        const BATCH: usize = 64;
        let parallelism = sessions.len();
        let start = Instant::now();
        let completed = Arc::new(AtomicUsize::new(0));
        let mut tasks = tokio::task::JoinSet::new();
        for (lane, mut session) in sessions.into_iter().enumerate() {
            let ids = ids.clone();
            let completed = completed.clone();
            let state = self.clone();
            // Separate tasks let CPU-heavy MPC conversion use the runtime's
            // worker cores. Joining unspawned futures serializes that arithmetic
            // on the single task polling startup. JoinSet aborts peers on error.
            tasks.spawn(async move {
                for batch in (lane..ids.len().div_ceil(BATCH)).step_by(parallelism) {
                    let batch_start = batch * BATCH;
                    let records = load_or_convert_batch(
                        &mut session,
                        &state.store,
                        state.side,
                        &ids[batch_start..(batch_start + BATCH).min(ids.len())],
                    )
                    .await?;
                    let count = records.len();
                    let mut resident = state.resident.write().await;
                    for (id, iris) in records {
                        resident.insert(id, Some(iris));
                    }
                    drop(resident);
                    let before = completed.fetch_add(count, Ordering::Relaxed);
                    if before / 4096 != (before + count) / 4096 {
                        tracing::info!(
                            completed = before + count,
                            total = ids.len(),
                            seconds = start.elapsed().as_secs_f64(),
                            "Spectral database migration progress"
                        );
                    }
                }
                Ok::<_, eyre::Report>(())
            });
        }
        while let Some(result) = tasks.join_next().await {
            result??;
        }
        ensure!(
            self.resident.read().await.size == ids.len(),
            "incomplete spectral database"
        );
        metrics::histogram!("ntt_migration_duration").record(start.elapsed().as_secs_f64());
        tracing::info!(
            records = ids.len(),
            seconds = start.elapsed().as_secs_f64(),
            "Spectral database ready"
        );
        Ok(())
    }

    /// Called exactly once before the request's concurrent scans. Both eyes'
    /// query IDs are supplied because mirror matching swaps the query eyes.
    pub async fn cache_queries(
        &self,
        session: &mut Session,
        queries: &[(QueryId, ArcIris)],
    ) -> Result<()> {
        let start = Instant::now();
        let missing: Vec<_> = {
            let cached = self.queries.read().unwrap();
            queries
                .iter()
                .filter(|(id, _)| !cached.contains_key(id))
                .cloned()
                .collect()
        };
        if missing.is_empty() {
            return Ok(());
        }
        let generation = new_generation(session).await?;
        let refs: Vec<_> = missing.iter().map(|(_, iris)| iris.as_ref()).collect();
        let field = convert_irises(session, &refs).await?;
        let party = self.party;
        let prepared = tokio::task::spawn_blocking(move || {
            field
                .into_iter()
                .map(|iris| {
                    let normal = Arc::new(SpectralQuery::prepare(&[&iris], party));
                    let mirrored = Arc::new(SpectralQuery::prepare(&[&iris.mirrored()], party));
                    CachedSpectralQuery {
                        original: Arc::new(SpectralIris::prepare(&iris)),
                        normal,
                        mirrored,
                        generation,
                    }
                })
                .collect::<Vec<_>>()
        })
        .await?;
        let mut cached = self.queries.write().unwrap();
        for ((id, _), query) in missing.into_iter().zip(prepared) {
            cached.insert(id, query);
        }
        metrics::histogram!("ntt_query_conversion_duration").record(start.elapsed().as_secs_f64());
        Ok(())
    }

    pub async fn score(&self, queries: &[QuerySpec], ids: &[VectorId]) -> Result<Vec<u16>> {
        ensure!(
            (1..=2).contains(&queries.len()),
            "spectral scan needs one or two orientations"
        );
        ensure!(
            queries.iter().all(|q| q.rotation == CENTER_ROTATION),
            "spectral scan must start at the center rotation"
        );
        let query = {
            let cached = self.queries.read().unwrap();
            let get = |q: &QuerySpec| -> Result<Arc<SpectralQuery>> {
                let entry = cached
                    .get(&q.query_id)
                    .ok_or_else(|| eyre!("query missing secure spectral preprocessing"))?;
                Ok(if q.mirrored {
                    entry.mirrored.clone()
                } else {
                    entry.normal.clone()
                })
            };
            let first = get(&queries[0])?;
            if queries.len() == 1 {
                first
            } else {
                Arc::new(SpectralQuery::pair(&first, get(&queries[1])?.as_ref()))
            }
        };
        let targets = {
            let resident = self.resident.read().await;
            ids.iter()
                .map(|id| {
                    resident
                        .get_vector(id)
                        .and_then(Option::as_ref)
                        .cloned()
                        .ok_or_else(|| eyre!("missing exact spectral iris version {id}"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        self.workers.spectral_dot_products(query, targets).await
    }

    pub async fn insert(&self, inserts: &[(QueryId, VectorId)]) -> Result<()> {
        let records = {
            let cached = self.queries.read().unwrap();
            inserts
                .iter()
                .map(|(query_id, id)| {
                    let query = cached
                        .get(query_id)
                        .ok_or_else(|| eyre!("insert missing secure spectral query conversion"))?;
                    Ok((*id, query.original.clone(), query.generation))
                })
                .collect::<Result<Vec<_>>>()?
        };
        for (id, iris, generation) in &records {
            persist_batch(
                &self.store,
                self.side,
                self.party,
                generation,
                &[(*id, iris.clone())],
            )
            .await?;
        }
        let mut resident = self.resident.write().await;
        for (id, iris, _) in records {
            resident.insert(id, Some(iris));
        }
        Ok(())
    }

    pub fn evict(&self, ids: &[QueryId]) {
        let mut queries = self.queries.write().unwrap();
        for id in ids {
            queries.remove(id);
        }
    }

    pub async fn delete(&self, ids: &[VectorId]) -> Result<()> {
        // The production deletion dummy is the public all-zero code/all-one
        // mask iris, hence decoded code=mask=1. Its degree-zero field sharing
        // needs no secret conversion or shared randomness.
        let dummy = Arc::new(SpectralIris::prepare(&FieldIris {
            code: vec![1; 12_800],
            mask: vec![1; 6_400],
        }));
        let records: Vec<_> = ids
            .iter()
            .map(|id| (id.next_version(), dummy.clone()))
            .collect();
        persist_batch(&self.store, self.side, self.party, &[0; 16], &records).await?;
        let mut resident = self.resident.write().await;
        for (id, iris) in records {
            resident.insert(id, Some(iris));
        }
        Ok(())
    }
}
