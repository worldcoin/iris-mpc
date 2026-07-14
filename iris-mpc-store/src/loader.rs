use crate::rerand::RerandContext;
use crate::s3_importer::create_db_chunks_s3_client;
use crate::{
    fetch_and_parse_chunks, fetch_and_parse_safe_snapshot, last_snapshot_timestamp,
    latest_safe_snapshot, DbStoredIris, S3Store, S3StoredIris, Store,
};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use aws_config::Region;
use eyre::{bail, ensure, Result, WrapErr};
use futures::stream::BoxStream;
use futures::StreamExt;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use sqlx::Connection;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const METADATA_PAGE_SIZE: i64 = 2_000;

fn validate_authoritative_count(expected: usize, authoritative: usize) -> Result<()> {
    ensure!(
        authoritative == expected,
        "Aurora snapshot inventory contains {authoritative} rows, but the loader allocated for {expected}"
    );
    Ok(())
}

#[derive(Debug, Default)]
struct RawLoadRerandState {
    store_id: Option<String>,
    last_completed_epoch: i32,
    active_epoch: Option<i32>,
    has_positive_rows: bool,
}

fn validate_raw_load_rerand_state(state: Option<&RawLoadRerandState>) -> Result<()> {
    if let Some(state) = state {
        ensure!(
            state.store_id.is_none()
                && state.last_completed_epoch == 0
                && state.active_epoch.is_none()
                && !state.has_positive_rows,
            "raw database loading is forbidden after rerandomization initialization or positive state"
        );
    }
    Ok(())
}

async fn raw_load_rerand_state_on_connection(
    connection: &mut sqlx::PgConnection,
) -> Result<Option<RawLoadRerandState>> {
    let has_metadata: bool = sqlx::query_scalar(
        "SELECT EXISTS (\
             SELECT 1 FROM information_schema.columns \
              WHERE table_schema = current_schema() AND table_name = 'irises' \
                AND column_name = 'rerand_epoch'\
         )",
    )
    .fetch_one(&mut *connection)
    .await?;
    if !has_metadata {
        return Ok(None);
    }

    let (store_id, last_completed_epoch, active_epoch, has_positive_rows) =
        sqlx::query_as::<_, (Option<String>, i32, Option<i32>, bool)>(
            "SELECT state.store_id, state.last_completed_epoch, state.active_epoch, \
                    CASE WHEN state.store_id IS NULL \
                              AND state.last_completed_epoch = 0 \
                              AND state.active_epoch IS NULL \
                         THEN EXISTS (SELECT 1 FROM irises WHERE rerand_epoch <> 0) \
                         ELSE FALSE \
                    END \
               FROM get_rerand_store_state() AS state",
        )
        .fetch_one(&mut *connection)
        .await?;
    Ok(Some(RawLoadRerandState {
        store_id,
        last_completed_epoch,
        active_epoch,
        has_positive_rows,
    }))
}

/// Refuse legacy code paths that consume or rewrite raw shares once a store
/// has been initialized for continuous rerandomization.
pub async fn ensure_legacy_raw_access_allowed(store: &Store) -> Result<()> {
    let mut connection = store.pool.acquire().await?;
    let state = raw_load_rerand_state_on_connection(&mut connection).await?;
    validate_raw_load_rerand_state(state.as_ref())
}

/// Session guard for a long-running legacy raw-share operation. It uses the
/// same schema-scoped advisory lock as rerandomization initialization and
/// passes, closing the underlying connection on drop so the lock cannot leak
/// back into the pool.
pub struct LegacyRawAccessGuard {
    connection: sqlx::pool::PoolConnection<sqlx::Postgres>,
}

impl LegacyRawAccessGuard {
    /// Stream legacy raw shares on the same physical session that owns the
    /// exclusion lock. If that session is lost, the stream fails instead of
    /// reconnecting without the lock.
    pub fn stream_irises_in_range(
        &mut self,
        id_range: Range<u64>,
    ) -> impl futures::Stream<Item = sqlx::Result<DbStoredIris>> + '_ {
        sqlx::query_as(
            r#"
            SELECT *
            FROM irises
            WHERE id >= $1 AND id < $2
            ORDER BY id ASC
            "#,
        )
        .bind(i64::try_from(id_range.start).expect("id fits into i64"))
        .bind(i64::try_from(id_range.end).expect("id fits into i64"))
        .fetch(&mut *self.connection)
    }

    /// Begin a legacy raw-share write on the lock-owning session. Rechecking
    /// the state on that session also proves it is still live before the write.
    pub async fn transaction(&mut self) -> Result<sqlx::Transaction<'_, sqlx::Postgres>> {
        let state = raw_load_rerand_state_on_connection(&mut self.connection).await?;
        ensure!(
            state.is_some(),
            "guarded legacy raw access requires the rerandomization migration"
        );
        validate_raw_load_rerand_state(state.as_ref())?;
        Ok(self.connection.begin().await?)
    }
}

pub async fn acquire_legacy_raw_access_guard(store: &Store) -> Result<LegacyRawAccessGuard> {
    let mut connection = store.pool.acquire().await?;
    connection.close_on_drop();
    let acquired: bool = sqlx::query_scalar(
        "SELECT pg_catalog.pg_try_advisory_lock(\
             1381126734, \
             (SELECT relnamespace::integer FROM pg_catalog.pg_class \
               WHERE oid = 'irises'::regclass)\
         )",
    )
    .fetch_one(&mut *connection)
    .await?;
    ensure!(
        acquired,
        "legacy raw database access is blocked by rerandomization initialization or an active pass"
    );
    let state = raw_load_rerand_state_on_connection(&mut connection).await?;
    ensure!(
        state.is_some(),
        "guarded legacy raw access requires the rerandomization migration"
    );
    validate_raw_load_rerand_state(state.as_ref())?;
    Ok(LegacyRawAccessGuard { connection })
}

#[derive(Clone, Copy)]
struct CachedRow {
    version_id: i16,
    semantic_id: [u8; 16],
    usable: bool,
}

struct SafeLoadState {
    cache: Vec<Option<CachedRow>>,
    counted: Vec<bool>,
    authoritative: Vec<bool>,
    loaded_count: usize,
}

impl SafeLoadState {
    fn new(rows: usize) -> Self {
        Self {
            cache: vec![None; rows],
            counted: vec![false; rows],
            authoritative: vec![false; rows],
            loaded_count: 0,
        }
    }

    fn index(&self, id: usize) -> Result<usize> {
        ensure!(
            id > 0 && id <= self.cache.len(),
            "iris id {id} is outside 1..={}",
            self.cache.len()
        );
        Ok(id - 1)
    }

    fn record_cache(
        &mut self,
        id: usize,
        version_id: i16,
        semantic_id: [u8; 16],
        usable: bool,
    ) -> Result<bool> {
        let index = self.index(id)?;
        ensure!(self.cache[index].is_none(), "S3 cache repeats iris id {id}");
        self.cache[index] = Some(CachedRow {
            version_id,
            semantic_id,
            usable,
        });
        if usable && !self.counted[index] {
            self.counted[index] = true;
            self.loaded_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn discard_cache(&mut self) {
        // Already materialized slots are overwritten from Aurora, but remain
        // counted so actor sizes are not incremented twice.
        self.cache.fill(None);
    }

    fn needs_aurora(&self, id: usize, version_id: i16, semantic_id: Option<[u8; 16]>) -> bool {
        !matches!(
            (self.cache[id - 1], semantic_id),
            (
                Some(CachedRow {
                    version_id: cached_version,
                    semantic_id: cached_semantic_id,
                    usable: true,
                }),
                Some(authoritative_semantic_id),
            ) if cached_version == version_id
                && cached_semantic_id == authoritative_semantic_id
        )
    }

    fn record_authoritative(&mut self, id: usize) -> Result<bool> {
        let index = self.index(id)?;
        ensure!(!self.authoritative[index], "Aurora repeats iris id {id}");
        self.authoritative[index] = true;
        if self.counted[index] {
            Ok(false)
        } else {
            self.counted[index] = true;
            self.loaded_count += 1;
            Ok(true)
        }
    }

    fn finish(&self) -> Result<()> {
        if let Some(index) = self.authoritative.iter().position(|seen| !seen) {
            bail!("Aurora inventory is missing iris id {}", index + 1);
        }
        Ok(())
    }
}

fn load_s3_record(actor: &mut impl InMemoryStore, iris: &S3StoredIris) {
    actor.load_single_record_from_s3(
        iris.serial_id() - 1,
        iris.vector_id(),
        iris.left_code_odd(),
        iris.left_code_even(),
        iris.right_code_odd(),
        iris.right_code_even(),
        iris.left_mask_odd(),
        iris.left_mask_even(),
        iris.right_mask_odd(),
        iris.right_mask_even(),
    );
}

fn load_db_record(actor: &mut impl InMemoryStore, iris: &DbStoredIris) {
    actor.load_single_record_from_db(
        iris.serial_id() - 1,
        iris.vector_id(),
        iris.left_code(),
        iris.left_mask(),
        iris.right_code(),
        iris.right_mask(),
    );
}

/// Discover the cache candidate before seeds are loaded. Callers use its exact
/// epoch inventory to build the normalization context. Failure is a cache miss,
/// not a database-authority failure.
pub async fn safe_snapshot_manifest(config: &Config) -> Result<crate::SafeSnapshotManifest> {
    ensure!(config.enable_s3_importer, "S3 importer is disabled");
    let region_provider = Region::new(config.db_chunks_bucket_region.clone());
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let s3 = S3Store::new(
        create_db_chunks_s3_client(&shared_config, true),
        config.db_chunks_bucket_name.clone(),
    );
    latest_safe_snapshot(&s3, &config.db_chunks_folder_name, &config.rerand_store_id).await
}

#[allow(clippy::too_many_arguments)]
async fn try_load_safe_cache(
    actor: &mut impl InMemoryStore,
    state: &mut SafeLoadState,
    max_serial_id_to_load: usize,
    s3_max_serial_id_to_load: Option<usize>,
    config: &Config,
    shutdown: Arc<ShutdownHandler>,
    rerand: &RerandContext,
) -> Result<()> {
    let region_provider = Region::new(config.db_chunks_bucket_region.clone());
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let s3 = Arc::new(S3Store::new(
        create_db_chunks_s3_client(&shared_config, true),
        config.db_chunks_bucket_name.clone(),
    ));
    let manifest = latest_safe_snapshot(
        s3.as_ref(),
        &config.db_chunks_folder_name,
        &config.rerand_store_id,
    )
    .await?;
    let cache_rows = manifest
        .row_count
        .min(max_serial_id_to_load)
        .min(s3_max_serial_id_to_load.unwrap_or(max_serial_id_to_load));
    let (tx, mut rx) = mpsc::channel(config.load_chunks_buffer_size);
    let fetch = fetch_and_parse_safe_snapshot(
        s3,
        config.load_chunks_parallelism,
        manifest,
        cache_rows,
        tx,
        config.load_chunks_max_retries,
        config.load_chunks_initial_backoff_ms,
        shutdown,
    );
    tokio::pin!(fetch);
    let mut fetch_done = false;
    loop {
        let mut iris = tokio::select! {
            biased;
            result = &mut fetch, if !fetch_done => {
                fetch_done = true;
                result?;
                continue;
            }
            row = rx.recv() => match row {
                Some(row) => row,
                None => break,
            }
        };
        let id = iris.serial_id();
        let version_id = iris.version_id();
        let semantic_id = iris
            .semantic_id()
            .ok_or_else(|| eyre::eyre!("safe S3 row {id} has no semantic id"))?;
        let usable = match rerand.normalize_s3_iris(&mut iris) {
            Ok(()) => true,
            Err(error) => {
                tracing::info!(id, ?error, "S3 row requires authoritative Aurora fallback");
                false
            }
        };
        let increment = state.record_cache(id, version_id, semantic_id, usable)?;
        if usable {
            load_s3_record(actor, &iris);
            if increment {
                actor.increment_db_size(id - 1);
            }
        }
    }
    if !fetch_done {
        fetch.await?;
    }
    if let Some(index) = state.cache[..cache_rows].iter().position(Option::is_none) {
        bail!("S3 cache is missing iris id {}", index + 1);
    }
    Ok(())
}

async fn reconcile_with_aurora(
    actor: &mut impl InMemoryStore,
    state: &mut SafeLoadState,
    snapshot: &mut crate::snapshot_reconcile::AuroraSnapshot,
    max_serial_id_to_load: usize,
    rerand: &RerandContext,
) -> Result<()> {
    let max_id = i64::try_from(max_serial_id_to_load).wrap_err("iris count does not fit i64")?;
    let mut after_id = 0i64;
    while after_id < max_id {
        let metadata = snapshot
            .metadata_page(after_id, max_id, METADATA_PAGE_SIZE)
            .await?;
        ensure!(
            !metadata.is_empty(),
            "Aurora inventory ends after iris id {after_id}, expected {max_id}"
        );

        let mut missing_ids = Vec::new();
        for (expected_id, row) in (after_id + 1..).zip(&metadata) {
            ensure!(
                row.id == expected_id,
                "Aurora inventory is not contiguous: expected iris id {expected_id}, got {}",
                row.id
            );
            let id = usize::try_from(row.id).wrap_err("negative Aurora iris id")?;
            if state.needs_aurora(id, row.version_id, row.semantic_id) {
                missing_ids.push(row.id);
            } else {
                state.record_authoritative(id)?;
            }
        }

        let mut rows = snapshot.irises(&missing_ids).await?.into_iter();
        for metadata in metadata
            .iter()
            .filter(|row| missing_ids.binary_search(&row.id).is_ok())
        {
            let mut iris = rows
                .next()
                .ok_or_else(|| eyre::eyre!("Aurora did not return iris {}", metadata.id))?;
            ensure!(
                iris.id() == metadata.id
                    && iris.version_id() == metadata.version_id
                    && iris.rerand_epoch() == metadata.rerand_epoch,
                "Aurora metadata/blob mismatch for iris {}",
                metadata.id
            );
            rerand.normalize_db_iris(&mut iris)?;
            let id = iris.serial_id();
            load_db_record(actor, &iris);
            if state.record_authoritative(id)? {
                actor.increment_db_size(id - 1);
            }
        }
        ensure!(
            rows.next().is_none(),
            "Aurora returned an unexpected iris row"
        );
        after_id = metadata.last().unwrap().id;
    }
    state.finish()
}

#[allow(clippy::too_many_arguments)]
async fn load_rerand_db(
    actor: &mut impl InMemoryStore,
    store: &Store,
    max_serial_id_to_load: usize,
    s3_max_serial_id_to_load: Option<usize>,
    config: &Config,
    shutdown: Arc<ShutdownHandler>,
    rerand: &RerandContext,
) -> Result<()> {
    let started = Instant::now();
    ensure!(
        max_serial_id_to_load <= config.max_db_size,
        "requested iris inventory {max_serial_id_to_load} exceeds max_db_size {}",
        config.max_db_size
    );
    let mut state = SafeLoadState::new(max_serial_id_to_load);
    actor.reserve(max_serial_id_to_load);
    if config.enable_s3_importer {
        if let Err(error) = try_load_safe_cache(
            actor,
            &mut state,
            max_serial_id_to_load,
            s3_max_serial_id_to_load,
            config,
            shutdown,
            rerand,
        )
        .await
        {
            tracing::warn!(?error, "Safe S3 cache is unusable; loading from Aurora");
            state.discard_cache();
        }
    }
    // Open the authoritative view as late as possible so a potentially long
    // S3 download neither pins an old database view nor retains MVCC history.
    // A count change since the caller allocated the actor is a hard startup
    // failure; silently adapting it could leave already loaded cache slots or
    // serving capacity inconsistent with Aurora.
    let mut snapshot = store.begin_aurora_snapshot().await?;
    let authoritative_rows = snapshot.authoritative_row_count().await?;
    validate_authoritative_count(max_serial_id_to_load, authoritative_rows)?;
    reconcile_with_aurora(
        actor,
        &mut state,
        &mut snapshot,
        max_serial_id_to_load,
        rerand,
    )
    .await?;
    snapshot.finish().await?;
    actor.preprocess_db();
    tracing::info!(rows = state.loaded_count, elapsed = ?started.elapsed(), "Loaded rerandomized database");
    Ok(())
}

/// Helper function to load Aurora db records from the stream into memory
#[allow(clippy::needless_lifetimes)]
async fn load_db_records_from_aurora<'a>(
    actor: &mut impl InMemoryStore,
    record_counter: &mut i32,
    all_serial_ids: &mut HashSet<i64>,
    mut stream_db: BoxStream<'a, Result<DbStoredIris>>,
) -> Result<()> {
    let mut load_summary_ts = Instant::now();
    let mut time_waiting_for_stream = Duration::from_secs(0);
    let mut time_loading_into_memory = Duration::from_secs(0);
    let n_loaded_via_s3 = *record_counter;
    while let Some(iris) = stream_db.next().await {
        // Update time waiting for the stream
        time_waiting_for_stream += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();

        let iris = iris?;

        actor.load_single_record_from_db(
            iris.serial_id() - 1,
            iris.vector_id(),
            iris.left_code(),
            iris.left_mask(),
            iris.right_code(),
            iris.right_mask(),
        );

        // Only increment db size if record has not been loaded via s3 before
        if all_serial_ids.contains(&(iris.serial_id() as i64)) {
            actor.increment_db_size(iris.serial_id() - 1);
            all_serial_ids.remove(&(iris.serial_id() as i64));
            *record_counter += 1;
        }

        // Update time spent loading into memory
        time_loading_into_memory += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();
    }

    tracing::info!(
        "Aurora Loading summary => Loaded {:?} items. Waited for stream: {:?}, Loaded into \
         memory: {:?}",
        *record_counter - n_loaded_via_s3,
        time_waiting_for_stream,
        time_loading_into_memory,
    );

    Ok(())
}

/// Options controlling an iris database load.
pub struct LoadIrisDbOptions<'a> {
    pub max_serial_id_to_load: usize,
    pub store_load_parallelism: usize,
    pub s3_max_serial_id_to_load: Option<usize>,
    pub config: &'a Config,
    pub download_shutdown_handler: Arc<ShutdownHandler>,
    pub rerand: Option<&'a RerandContext>,
}

/// Main iris loader method into memory. Load from either S3 + Aurora or only Aurora based on the config.
pub async fn load_iris_db(
    actor: &mut impl InMemoryStore,
    store: &Store,
    options: LoadIrisDbOptions<'_>,
) -> Result<()> {
    let shutdown = options.download_shutdown_handler.clone();
    tokio::select! {
        r = load_iris_db_internal(actor, store, options) => r,
        _ = shutdown.wait_for_shutdown() => {
            tracing::warn!("Shutdown requested by shutdown_handler.");
            Err(eyre::eyre!("Shutdown requested"))
        },
    }
}

async fn load_iris_db_internal(
    actor: &mut impl InMemoryStore,
    store: &Store,
    options: LoadIrisDbOptions<'_>,
) -> Result<()> {
    let LoadIrisDbOptions {
        max_serial_id_to_load,
        store_load_parallelism,
        s3_max_serial_id_to_load,
        config,
        download_shutdown_handler,
        rerand,
    } = options;
    if let Some(rerand) = rerand {
        return load_rerand_db(
            actor,
            store,
            max_serial_id_to_load,
            s3_max_serial_id_to_load,
            config,
            download_shutdown_handler,
            rerand,
        )
        .await;
    }
    ensure!(
        !config.rerand_enabled,
        "rerandomization is enabled but no normalization context was provided"
    );
    ensure_legacy_raw_access_allowed(store).await?;
    let total_load_time = Instant::now();
    let now = Instant::now();

    let mut record_counter = 0;
    let mut all_serial_ids: HashSet<i64> = HashSet::from_iter(1..=(max_serial_id_to_load as i64));
    actor.reserve(max_serial_id_to_load);

    if config.enable_s3_importer {
        tracing::info!("S3 importer enabled. Fetching from s3 + db");
        let region = config.clone().db_chunks_bucket_region;

        // Get s3 loading parameters from config
        let s3_load_parallelism = config.load_chunks_parallelism;
        let s3_load_max_retries = config.load_chunks_max_retries;
        let s3_load_initial_backoff_ms = config.load_chunks_initial_backoff_ms;
        let s3_chunks_folder_name = config.db_chunks_folder_name.clone();
        let s3_chunks_bucket_name = config.db_chunks_bucket_name.clone();
        let s3_load_safety_overlap_seconds = config.db_load_safety_overlap_seconds;

        // Construct s3 client and store
        let region_provider = Region::new(region);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let s3_client = create_db_chunks_s3_client(&shared_config, true);
        let s3_store = S3Store::new(s3_client, s3_chunks_bucket_name.clone());
        let s3_arc = Arc::new(s3_store);

        // First fetch last snapshot from S3
        let last_snapshot_details =
            last_snapshot_timestamp(s3_arc.as_ref(), s3_chunks_folder_name.clone()).await?;

        let min_last_modified_at = last_snapshot_details.timestamp - s3_load_safety_overlap_seconds;
        tracing::info!(
            "Last snapshot timestamp: {}, min_last_modified_at: {}",
            last_snapshot_details.timestamp,
            min_last_modified_at
        );

        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(config.load_chunks_buffer_size);
        let import_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(s3_load_parallelism)
            .thread_name("s3-importer")
            .enable_all()
            .build()?;
        let shutdown = download_shutdown_handler.clone();
        let fetch_handle = import_runtime.spawn(async move {
            fetch_and_parse_chunks(
                s3_arc,
                s3_load_parallelism,
                s3_chunks_folder_name.clone(),
                last_snapshot_details,
                s3_max_serial_id_to_load,
                tx,
                s3_load_max_retries,
                s3_load_initial_backoff_ms,
                shutdown,
            )
            .await
        });
        // Guard that calls shutdown_background() on drop so the runtime threads
        // are always released non-blockingly, even on early returns / errors.
        // in the event that the shutdown handler causes this function to terminate before the RuntimeShutdownGuard
        // is created, fetch_and_parse_chunks will terminate anyway because it also listens for the shutdown signal.
        struct RuntimeShutdownGuard(Option<tokio::runtime::Runtime>);
        impl Drop for RuntimeShutdownGuard {
            fn drop(&mut self) {
                if let Some(rt) = self.0.take() {
                    rt.shutdown_background();
                }
            }
        }
        let _drop_guard = RuntimeShutdownGuard(Some(import_runtime));

        // Consume parsed irises while watching the fetch task. `biased` polls the
        // task before the channel, so a fetch failure aborts immediately instead of
        // draining the backlog (and the in-flight retry-with-backoff subtasks).
        let mut time_waiting_for_stream = Duration::from_secs(0);
        let mut time_loading_into_memory = Duration::from_secs(0);
        let mut load_summary_ts = Instant::now();
        let mut fetch_done = false;
        tokio::pin!(fetch_handle);
        loop {
            let iris = tokio::select! {
                biased;
                res = &mut fetch_handle, if !fetch_done => {
                    fetch_done = true;
                    match res {
                        Ok(Ok(())) => continue,
                        Ok(Err(e)) => return Err(e),
                        Err(join_err) => {
                            return Err(eyre::eyre!("S3 fetch task panicked: {}", join_err))
                        }
                    }
                },
                item = rx.recv() => match item {
                    Some(iris) => iris,
                    None => break,
                },
            };

            time_waiting_for_stream += load_summary_ts.elapsed();
            load_summary_ts = Instant::now();

            let serial_id = iris.serial_id();
            if serial_id == 0 {
                tracing::error!("Invalid iris serial_id {}", serial_id);
                bail!("Invalid iris serial_id {}", serial_id);
            } else if serial_id > max_serial_id_to_load {
                tracing::warn!(
                    "Skip loading item: serial_id {} > max_serial_id_to_load {}. This can happen if the max in the case of roll-backs or where the max_serial_id_to_load is specified",
                    serial_id,
                    max_serial_id_to_load,
                );
                continue;
            } else if !all_serial_ids.contains(&(serial_id as i64)) {
                tracing::debug!("Skip loading s3 retried item: serial_id {}", serial_id);
                continue;
            }

            actor.load_single_record_from_s3(
                iris.serial_id() - 1,
                iris.vector_id(),
                iris.left_code_odd(),
                iris.left_code_even(),
                iris.right_code_odd(),
                iris.right_code_even(),
                iris.left_mask_odd(),
                iris.left_mask_even(),
                iris.right_mask_odd(),
                iris.right_mask_even(),
            );
            actor.increment_db_size(serial_id - 1);

            time_loading_into_memory += load_summary_ts.elapsed();
            load_summary_ts = Instant::now();

            all_serial_ids.remove(&(serial_id as i64));
            record_counter += 1;

            if record_counter % 500_000 == 0 {
                let elapsed = now.elapsed();
                tracing::info!(
                    "Loaded {} records into memory in {:?} ({:.2} entries/s)",
                    record_counter,
                    elapsed,
                    record_counter as f64 / elapsed.as_secs_f64()
                );
            }
        }

        // Reached only by breaking on a closed channel. If the task's result wasn't
        // observed in the loop, join it now to surface a late failure.
        if !fetch_done {
            match fetch_handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(join_err) => return Err(eyre::eyre!("S3 fetch task failed: {}", join_err)),
            }
        }

        tracing::info!(
            "S3 Loading summary => Loaded {:?} items. Waited for stream: {:?}, Loaded into \
             memory: {:?}.",
            record_counter,
            time_waiting_for_stream,
            time_loading_into_memory,
        );

        let stream_db = store
            .stream_irises_par(
                Some(min_last_modified_at),
                store_load_parallelism,
                Some(max_serial_id_to_load),
            )
            .await
            .boxed();
        load_db_records_from_aurora(actor, &mut record_counter, &mut all_serial_ids, stream_db)
            .await?;
    } else {
        tracing::info!("S3 importer disabled. Fetching only from Aurora db");
        let stream_db = store
            .stream_irises_par(None, store_load_parallelism, Some(max_serial_id_to_load))
            .await
            .boxed();
        load_db_records_from_aurora(actor, &mut record_counter, &mut all_serial_ids, stream_db)
            .await?;
    }

    if !all_serial_ids.is_empty() {
        tracing::error!("Not all serial_ids were loaded: {:?}", all_serial_ids);
        bail!("Not all serial_ids were loaded: {:?}", all_serial_ids);
    }

    // Recheck after the potentially long S3/Aurora load. If initialization or
    // a positive pass raced this legacy load, the actor is discarded before it
    // can become ready.
    ensure_legacy_raw_access_allowed(store).await?;
    tracing::info!("Preprocessing db");
    actor.preprocess_db();

    tracing::info!(
        "Loaded {} records from db into memory in {:?} [DB sizes: {:?}]",
        record_counter,
        total_load_time.elapsed(),
        actor.current_db_sizes()
    );

    eyre::Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_cache_accepts_only_exact_usable_row_identity() -> Result<()> {
        let semantic_id = [1; 16];
        let replaced_semantic_id = [2; 16];
        let mut state = SafeLoadState::new(3);
        assert!(state.record_cache(1, 4, semantic_id, true)?);
        assert!(!state.record_cache(2, 5, semantic_id, false)?);
        assert!(!state.needs_aurora(1, 4, Some(semantic_id)));
        assert!(state.needs_aurora(1, 5, Some(semantic_id)));
        assert!(state.needs_aurora(1, 4, Some(replaced_semantic_id)));
        assert!(state.needs_aurora(1, 4, None));
        assert!(state.needs_aurora(2, 5, Some(semantic_id)));
        assert!(state.record_cache(1, 4, semantic_id, true).is_err());

        state.discard_cache();
        assert!(state.needs_aurora(1, 4, Some(semantic_id)));
        assert!(!state.record_authoritative(1)?);
        assert!(state.record_authoritative(2)?);
        assert!(state.record_authoritative(3)?);
        state.finish()?;
        assert_eq!(state.loaded_count, 3);
        Ok(())
    }

    #[test]
    fn empty_earlier_count_cannot_hide_authoritative_rows() {
        assert!(validate_authoritative_count(0, 1).is_err());
        assert!(validate_authoritative_count(1, 0).is_err());
        assert!(validate_authoritative_count(0, 0).is_ok());
    }

    #[test]
    fn raw_loader_refuses_initialized_or_positive_rerandomization_state() {
        assert!(validate_raw_load_rerand_state(None).is_ok());
        assert!(validate_raw_load_rerand_state(Some(&RawLoadRerandState::default())).is_ok());

        for state in [
            RawLoadRerandState {
                store_id: Some("store-1".to_owned()),
                ..Default::default()
            },
            RawLoadRerandState {
                last_completed_epoch: 1,
                ..Default::default()
            },
            RawLoadRerandState {
                active_epoch: Some(1),
                ..Default::default()
            },
            RawLoadRerandState {
                has_positive_rows: true,
                ..Default::default()
            },
        ] {
            assert!(validate_raw_load_rerand_state(Some(&state)).is_err());
        }
    }
}
