use aes_prng::AesRng;
use ampc_actor_utils::network::mpc::{
    build_network_handle, handle::NetworkHandleArgs, NetworkValue, Networking,
};
use ampc_actor_utils::network::tcp::TlsConfig;
use async_trait::async_trait;
use eyre::{ensure, Result};
use iris_mpc_common::{iris_db::iris::IrisCode, VectorId};
use iris_mpc_cpu::{
    execution::hawk_main::iris_worker::{
        cache_iris, init_workers, IrisWorkerPool, LocalIrisWorkerPool,
        DEFAULT_FULL_ROTATION_TASK_SIZE,
    },
    hawkers::{
        aby3::aby3_store::{Aby3Store, DistanceMode, FhdOps},
        shared_irises::SharedIrises,
    },
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};
use rand::SeedableRng;
use std::{
    collections::HashMap,
    env,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{
    fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

const DEFAULT_COMPARISONS: usize = 262_144;
const DEFAULT_WARMUP_COMPARISONS: usize = 4_096;
const DEFAULT_CONNECTIONS: usize = 16;
const DEFAULT_SESSIONS: usize = 1;
const DEFAULT_CHUNK_SIZE: usize = 0;
const DEFAULT_REPETITIONS: usize = 1;
const DEFAULT_TOKIO_CORES: usize = 0;
const TCP_SESSION_ID_BYTES: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationMode {
    /// Existing benchmark behavior: scan only the normal orientation.
    Single,
    /// Scan normal and mirror one after the other, using disjoint equal-sized
    /// session groups which share the resident iris worker pool.
    Sequential,
    /// Scan normal and mirror at the same time, otherwise identical to
    /// `Sequential`. This exposes contention and TLS prerotation-cache churn.
    Concurrent,
}

impl OrientationMode {
    fn from_env() -> Result<Self> {
        let value = env::var("IRIS_MPC_NETWORK_BENCH_ORIENTATION_MODE")
            .unwrap_or_else(|_| "single".to_owned());
        match value.to_ascii_lowercase().as_str() {
            "single" => Ok(Self::Single),
            "sequential" => Ok(Self::Sequential),
            "concurrent" => Ok(Self::Concurrent),
            _ => Err(eyre::eyre!(
                "invalid IRIS_MPC_NETWORK_BENCH_ORIENTATION_MODE {value:?}; expected single, sequential, or concurrent"
            )),
        }
    }

    const fn orientations(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Sequential | Self::Concurrent => 2,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Sequential => "sequential",
            Self::Concurrent => "concurrent",
        }
    }
}

#[derive(Debug, Default)]
struct TrafficCounter {
    sent_payload_bytes: AtomicU64,
    sent_messages: AtomicU64,
    received_payload_bytes: AtomicU64,
    received_messages: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrafficSnapshot {
    sent_payload_bytes: u64,
    sent_messages: u64,
    received_payload_bytes: u64,
    received_messages: u64,
}

impl TrafficCounter {
    fn snapshot(&self) -> TrafficSnapshot {
        TrafficSnapshot {
            sent_payload_bytes: self.sent_payload_bytes.load(Ordering::Relaxed),
            sent_messages: self.sent_messages.load(Ordering::Relaxed),
            received_payload_bytes: self.received_payload_bytes.load(Ordering::Relaxed),
            received_messages: self.received_messages.load(Ordering::Relaxed),
        }
    }
}

impl std::ops::Sub for TrafficSnapshot {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            sent_payload_bytes: self.sent_payload_bytes - rhs.sent_payload_bytes,
            sent_messages: self.sent_messages - rhs.sent_messages,
            received_payload_bytes: self.received_payload_bytes - rhs.received_payload_bytes,
            received_messages: self.received_messages - rhs.received_messages,
        }
    }
}

impl TrafficSnapshot {
    fn sent_framed_bytes(self) -> u64 {
        self.sent_payload_bytes + TCP_SESSION_ID_BYTES * self.sent_messages
    }

    fn received_framed_bytes(self) -> u64 {
        self.received_payload_bytes + TCP_SESSION_ID_BYTES * self.received_messages
    }
}

struct CountingNetworking {
    inner: Box<dyn Networking + Send + Sync>,
    counter: Arc<TrafficCounter>,
}

#[async_trait]
impl Networking for CountingNetworking {
    async fn send(
        &mut self,
        value: NetworkValue,
        receiver: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<()> {
        self.counter
            .sent_payload_bytes
            .fetch_add(value.byte_len() as u64, Ordering::Relaxed);
        self.counter.sent_messages.fetch_add(1, Ordering::Relaxed);
        self.inner.send(value, receiver).await
    }

    async fn receive(
        &mut self,
        sender: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<NetworkValue> {
        let value = self.inner.receive(sender).await?;
        self.counter
            .received_payload_bytes
            .fetch_add(value.byte_len() as u64, Ordering::Relaxed);
        self.counter
            .received_messages
            .fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }
}

struct UnusedNetworking;

#[async_trait]
impl Networking for UnusedNetworking {
    async fn send(
        &mut self,
        _value: NetworkValue,
        _receiver: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<()> {
        unreachable!("placeholder networking must never be used")
    }

    async fn receive(
        &mut self,
        _sender: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<NetworkValue> {
        unreachable!("placeholder networking must never be used")
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn addresses() -> Result<Vec<String>> {
    let value = env::var("IRIS_MPC_NETWORK_BENCH_ADDRESSES")?;
    let addresses = value
        .split(',')
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        addresses.len() == 3,
        "expected exactly three party addresses"
    );
    Ok(addresses)
}

fn tls_config() -> Result<Option<TlsConfig>> {
    const PRIVATE_KEY: &str = "IRIS_MPC_NETWORK_BENCH_TLS_PRIVATE_KEY";
    const LEAF_CERT: &str = "IRIS_MPC_NETWORK_BENCH_TLS_LEAF_CERT";
    const ROOT_CERT: &str = "IRIS_MPC_NETWORK_BENCH_TLS_ROOT_CERT";

    let private_key = env::var(PRIVATE_KEY).ok();
    let leaf_cert = env::var(LEAF_CERT).ok();
    let root_cert = env::var(ROOT_CERT).ok();
    let configured = [
        private_key.is_some(),
        leaf_cert.is_some(),
        root_cert.is_some(),
    ];
    ensure!(
        configured.iter().all(|&present| present) || configured.iter().all(|&present| !present),
        "TLS benchmark requires all or none of {PRIVATE_KEY}, {LEAF_CERT}, and {ROOT_CERT}"
    );
    Ok(private_key.map(|private_key| TlsConfig {
        private_key: Some(private_key),
        leaf_cert,
        root_certs: root_cert
            .expect("validated TLS root certificate")
            .split(',')
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(str::to_owned)
            .collect(),
    }))
}

async fn build_stores(
    party: usize,
    comparisons: usize,
    all_matches: bool,
    unique_records: bool,
    numa_realloc: bool,
    full_rotation_task_size: NonZeroUsize,
    sessions: Vec<ampc_actor_utils::execution::session::Session>,
) -> Result<(
    Vec<Aby3Store<FhdOps>>,
    iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Query,
    Vec<VectorId>,
)> {
    let mut rng = AesRng::seed_from_u64(0x0074_6370_5f62_656e_u64);
    let matching_iris = IrisCode::random_rng(&mut rng);
    let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, matching_iris.clone());
    let template = Arc::new(shares[party].clone()) as ArcIris;

    // Route inserts through the NUMA-pinned worker, exactly like production DB
    // loading. This first-touches the complete resident eye on the same NUMA
    // node used by the scan workers rather than benchmarking remote memory.
    let storage = SharedIrises::new(HashMap::new(), template.clone()).to_arc();
    let worker_handle = init_workers(0, storage.clone(), numa_realloc);
    worker_handle.reserve(comparisons)?;
    let vector_ids = (0..comparisons)
        .map(|index| VectorId::from_0_index(index as u32))
        .collect::<Vec<_>>();
    let mut previous_unique_iris: Option<ArcIris> = None;
    for &vector_id in &vector_ids {
        let iris = if unique_records {
            // Every party starts from the same seed and executes the same RNG
            // calls. Generate all three shares before selecting the local one,
            // so separately launched processes still store a valid sharing of
            // each record. For the all-match workload, re-share the matching
            // plaintext with fresh randomness; this makes the resident bytes
            // distinct without changing which records pass either threshold.
            let plaintext = if all_matches {
                matching_iris.clone()
            } else {
                IrisCode::random_rng(&mut rng)
            };
            let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, plaintext);
            let iris = Arc::new(shares[party].clone()) as ArcIris;
            if let Some(previous) = &previous_unique_iris {
                ensure!(
                    previous.as_ref() != iris.as_ref(),
                    "unique-record generation repeated a local share"
                );
            }
            previous_unique_iris = Some(iris.clone());
            iris
        } else {
            template.clone()
        };
        worker_handle.insert(vector_id, iris)?;
    }
    worker_handle.wait_completion().await?;

    let workers: Arc<dyn IrisWorkerPool> = Arc::new(
        LocalIrisWorkerPool::new(
            worker_handle,
            storage.clone(),
            DistanceMode::MinRotation,
            party,
        )
        .with_full_rotation_task_size(full_rotation_task_size),
    );
    let registry = storage.read().await.to_registry().to_arc();
    let stores = sessions
        .into_iter()
        .map(|session| {
            Aby3Store::new(
                registry.clone(),
                session,
                workers.clone(),
                DistanceMode::MinRotation,
            )
        })
        .collect::<Vec<_>>();
    let query = if all_matches {
        stores[0].cache_query_from_store(&vector_ids[0]).await?
    } else {
        let query_iris = IrisCode::random_rng(&mut rng);
        let query_shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, query_iris);
        cache_iris(workers.as_ref(), Arc::new(query_shares[party].clone())).await?
    };
    Ok((stores, query, vector_ids))
}

async fn run_protocol(
    store: &mut Aby3Store<FhdOps>,
    query: &iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Query,
    vector_ids: &[VectorId],
    all_matches: bool,
) -> Result<(usize, usize, usize)> {
    // This is the exact GPU-style production sequence for a full-scan chunk:
    // threshold every rotation once at the wider prefilter, then run the strict
    // threshold only on surviving records. No oblivious minimum is materialized.
    let thresholds = store
        .eval_distance_batch_full_rotation_thresholds(query, vector_ids)
        .await?;

    if all_matches {
        ensure!(
            thresholds.anon_stats_matches.len() >= vector_ids.len(),
            "identical records must pass the anonymous-statistics threshold"
        );
        ensure!(
            thresholds.matches.iter().all(Option::is_some),
            "identical records must pass the match threshold"
        );
    } else {
        ensure!(
            thresholds.anon_stats_matches.is_empty(),
            "independent random records unexpectedly passed the anonymous threshold"
        );
        ensure!(
            thresholds.matches.iter().all(Option::is_none),
            "independent random records unexpectedly passed the match threshold"
        );
    }
    Ok((
        thresholds.matches.len(),
        thresholds.match_rotations.iter().map(Vec::len).sum(),
        thresholds.matches.iter().flatten().count(),
    ))
}

async fn run_protocol_parallel(
    mut stores: Vec<Aby3Store<FhdOps>>,
    query: iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Query,
    vector_ids: Arc<[VectorId]>,
    all_matches: bool,
    configured_chunk_size: usize,
) -> Result<(Vec<Aby3Store<FhdOps>>, usize, usize, usize)> {
    let n_sessions = stores.len();
    if n_sessions == 1 && configured_chunk_size == 0 {
        let mut store = stores.pop().expect("one session must have one store");
        let counts = run_protocol(&mut store, &query, &vector_ids, all_matches).await?;
        return Ok((vec![store], counts.0, counts.1, counts.2));
    }
    let chunk_size = if configured_chunk_size == 0 {
        vector_ids.len().div_ceil(n_sessions).max(1)
    } else {
        configured_chunk_size
    };
    let n_chunks = vector_ids.len().div_ceil(chunk_size);
    let n_workers = n_sessions.min(n_chunks).max(1);
    let mut jobs = tokio::task::JoinSet::new();
    for (index, mut store) in stores.into_iter().enumerate() {
        let vector_ids = vector_ids.clone();
        jobs.spawn(async move {
            let mut counts = (0, 0, 0);
            if index < n_workers {
                for i_chunk in (index..n_chunks).step_by(n_workers) {
                    let start = i_chunk * chunk_size;
                    let end = (start + chunk_size).min(vector_ids.len());
                    let chunk_counts =
                        run_protocol(&mut store, &query, &vector_ids[start..end], all_matches)
                            .await?;
                    counts.0 += chunk_counts.0;
                    counts.1 += chunk_counts.1;
                    counts.2 += chunk_counts.2;
                }
            }
            Ok::<_, eyre::Error>((index, store, counts))
        });
    }

    let mut completed = Vec::with_capacity(n_sessions);
    while let Some(result) = jobs.join_next().await {
        completed.push(result??);
    }
    completed.sort_unstable_by_key(|(index, _, _)| *index);
    let mut distance_count = 0;
    let mut rotation_match_count = 0;
    let mut match_count = 0;
    let stores = completed
        .into_iter()
        .map(|(_, store, counts)| {
            distance_count += counts.0;
            rotation_match_count += counts.1;
            match_count += counts.2;
            store
        })
        .collect();
    Ok((stores, distance_count, rotation_match_count, match_count))
}

struct OrientationRun {
    normal_stores: Vec<Aby3Store<FhdOps>>,
    mirror_stores: Option<Vec<Aby3Store<FhdOps>>>,
    distance_count: usize,
    rotation_match_count: usize,
    match_count: usize,
}

async fn run_orientation_protocol(
    mode: OrientationMode,
    normal_stores: Vec<Aby3Store<FhdOps>>,
    mirror_stores: Option<Vec<Aby3Store<FhdOps>>>,
    query: iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Query,
    vector_ids: Arc<[VectorId]>,
    all_matches: bool,
    chunk_size: usize,
) -> Result<OrientationRun> {
    let mut mirror_query = query;
    mirror_query.mirrored = true;

    let (normal, mirror) = match mode {
        OrientationMode::Single => {
            ensure!(
                mirror_stores.is_none(),
                "single-orientation run unexpectedly received mirror sessions"
            );
            let normal =
                run_protocol_parallel(normal_stores, query, vector_ids, all_matches, chunk_size)
                    .await?;
            (normal, None)
        }
        OrientationMode::Sequential => {
            let mirror_stores = mirror_stores
                .ok_or_else(|| eyre::eyre!("sequential run requires mirror sessions"))?;
            let normal = run_protocol_parallel(
                normal_stores,
                query,
                vector_ids.clone(),
                all_matches,
                chunk_size,
            )
            .await?;
            let mirror = run_protocol_parallel(
                mirror_stores,
                mirror_query,
                vector_ids,
                all_matches,
                chunk_size,
            )
            .await?;
            (normal, Some(mirror))
        }
        OrientationMode::Concurrent => {
            let mirror_stores = mirror_stores
                .ok_or_else(|| eyre::eyre!("concurrent run requires mirror sessions"))?;
            let (normal, mirror) = tokio::try_join!(
                run_protocol_parallel(
                    normal_stores,
                    query,
                    vector_ids.clone(),
                    all_matches,
                    chunk_size,
                ),
                run_protocol_parallel(
                    mirror_stores,
                    mirror_query,
                    vector_ids,
                    all_matches,
                    chunk_size,
                ),
            )?;
            (normal, Some(mirror))
        }
    };

    let mut distance_count = normal.1;
    let mut rotation_match_count = normal.2;
    let mut match_count = normal.3;
    let mirror_stores = mirror.map(|mirror| {
        distance_count += mirror.1;
        rotation_match_count += mirror.2;
        match_count += mirror.3;
        mirror.0
    });

    Ok(OrientationRun {
        normal_stores: normal.0,
        mirror_stores,
        distance_count,
        rotation_match_count,
        match_count,
    })
}

fn main() -> Result<()> {
    let tokio_cores = env_usize("IRIS_MPC_NETWORK_BENCH_TOKIO_CORES", DEFAULT_TOKIO_CORES)?;
    let tokio_cores = (tokio_cores > 0).then_some(tokio_cores);
    iris_mpc_common::helpers::numactl::init(tokio_cores);
    iris_mpc_common::helpers::numactl::restrict_tokio_runtime();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(iris_mpc_common::helpers::numactl::get_tokio_worker_threads())
        .on_thread_start(iris_mpc_common::helpers::numactl::restrict_tokio_runtime)
        .enable_all()
        .build()?;
    runtime.block_on(run_benchmark(tokio_cores))
}

async fn run_benchmark(tokio_cores: Option<usize>) -> Result<()> {
    if env_bool("IRIS_MPC_NETWORK_BENCH_TRACE", false)? {
        let _ = tracing_subscriber::registry()
            .with(EnvFilter::new("searcher::network=trace"))
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_span_events(FmtSpan::CLOSE),
            )
            .try_init();
    }
    let party = env_usize("IRIS_MPC_PARTY_INDEX", usize::MAX)?;
    ensure!(party < 3, "IRIS_MPC_PARTY_INDEX must be 0, 1, or 2");
    let comparisons = env_usize("IRIS_MPC_NETWORK_BENCH_COMPARISONS", DEFAULT_COMPARISONS)?;
    let warmup_comparisons = env_usize(
        "IRIS_MPC_NETWORK_BENCH_WARMUP_COMPARISONS",
        DEFAULT_WARMUP_COMPARISONS,
    )?;
    let connections = env_usize("IRIS_MPC_NETWORK_BENCH_CONNECTIONS", DEFAULT_CONNECTIONS)?;
    let n_sessions = env_usize("IRIS_MPC_NETWORK_BENCH_SESSIONS", DEFAULT_SESSIONS)?;
    let chunk_size = env_usize("IRIS_MPC_NETWORK_BENCH_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)?;
    let repetitions = env_usize("IRIS_MPC_NETWORK_BENCH_REPETITIONS", DEFAULT_REPETITIONS)?;
    let all_matches = env_bool("IRIS_MPC_NETWORK_BENCH_ALL_MATCHES", true)?;
    let unique_records = env_bool("IRIS_MPC_NETWORK_BENCH_UNIQUE_RECORDS", false)?;
    let numa_realloc = env_bool("IRIS_MPC_NETWORK_BENCH_NUMA_REALLOC", true)?;
    let full_rotation_task_size = env_usize(
        "IRIS_MPC_NETWORK_BENCH_FULL_ROTATION_TASK_SIZE",
        DEFAULT_FULL_ROTATION_TASK_SIZE,
    )?;
    let full_rotation_task_size = NonZeroUsize::new(full_rotation_task_size).ok_or_else(|| {
        eyre::eyre!("IRIS_MPC_NETWORK_BENCH_FULL_ROTATION_TASK_SIZE must be positive")
    })?;
    let orientation_mode = OrientationMode::from_env()?;
    let tls = tls_config()?;
    ensure!(comparisons > 0, "comparison count must be positive");
    ensure!(
        comparisons <= u32::MAX as usize,
        "comparison count is too large"
    );
    ensure!(
        warmup_comparisons <= comparisons,
        "warmup comparison count exceeds DB size"
    );
    ensure!(connections > 0, "connection count must be positive");
    ensure!(n_sessions > 0, "session count must be positive");
    ensure!(repetitions > 0, "repetition count must be positive");
    if orientation_mode != OrientationMode::Single {
        ensure!(
            n_sessions >= 2 && n_sessions.is_multiple_of(2),
            "dual-orientation runs require a positive, even session count"
        );
        // A random iris does not generally match its own mirrored orientation,
        // so the all-match fixture cannot give both orientations equivalent
        // protocol work. No-match is the production-realistic cache A/B.
        ensure!(
            !all_matches,
            "dual-orientation cache comparison requires IRIS_MPC_NETWORK_BENCH_ALL_MATCHES=false"
        );
    }
    let sessions_per_orientation = n_sessions / orientation_mode.orientations();
    let total_comparisons = comparisons
        .checked_mul(repetitions)
        .and_then(|count| count.checked_mul(orientation_mode.orientations()))
        .ok_or_else(|| eyre::eyre!("total comparison count overflow"))?;

    let addresses = addresses()?;
    println!(
        "TCP_PROTOCOL_CONFIG party={party} comparisons={comparisons} \
         warmup_comparisons={warmup_comparisons} addresses={} tls={} \
         connections={connections} sessions={n_sessions} rotations=31 partial_results=true \
         orientation_mode={} orientations={} sessions_per_orientation={sessions_per_orientation} \
         chunk_size={chunk_size} repetitions={repetitions} \
         full_rotation_task_size={full_rotation_task_size} \
         thresholds=anon_prefilter,match_candidates \
         tokio_cores={} dot_cores={} \
         all_matches={all_matches} unique_records={unique_records} \
         numa_realloc={numa_realloc}",
        addresses.join(","),
        tls.is_some(),
        orientation_mode.as_str(),
        orientation_mode.orientations(),
        tokio_cores.unwrap_or_else(iris_mpc_common::helpers::numactl::get_tokio_worker_threads),
        iris_mpc_cpu::execution::hawk_main::iris_worker::select_core_ids(0).len(),
    );
    let shutdown = CancellationToken::new();
    let mut network_handle = build_network_handle(
        NetworkHandleArgs {
            party_index: party,
            outbound_addresses: addresses.clone(),
            addresses,
            connection_parallelism: connections,
            request_parallelism: 1,
            sessions_per_request: n_sessions,
            tls,
        },
        shutdown.clone(),
    )
    .await?;
    let mut control_channel = network_handle.control_channel().await?;
    let (mut sessions, session_error) = network_handle.make_sessions().await?;
    ensure!(sessions.len() == n_sessions, "unexpected MPC session count");

    // Install counters after the replicated PRF setup so only request traffic
    // is included. The TCP multiplexer adds a four-byte session ID per message.
    let counter = Arc::new(TrafficCounter::default());
    for session in &mut sessions {
        let inner = std::mem::replace(
            &mut session.network_session.networking,
            Box::new(UnusedNetworking),
        );
        session.network_session.networking = Box::new(CountingNetworking {
            inner,
            counter: counter.clone(),
        });
    }

    let load_started = Instant::now();
    let (mut normal_stores, query, vector_ids) = build_stores(
        party,
        comparisons,
        all_matches,
        unique_records,
        numa_realloc,
        full_rotation_task_size,
        sessions,
    )
    .await?;
    let mut mirror_stores = if orientation_mode == OrientationMode::Single {
        None
    } else {
        Some(normal_stores.split_off(sessions_per_orientation))
    };
    ensure!(
        normal_stores.len() == sessions_per_orientation
            && mirror_stores
                .as_ref()
                .is_none_or(|stores| stores.len() == sessions_per_orientation),
        "session groups were not split equally"
    );
    let vector_ids: Arc<[VectorId]> = vector_ids.into();
    println!(
        "TCP_PROTOCOL_LOADED party={party} comparisons={comparisons} seconds={:.6}",
        load_started.elapsed().as_secs_f64()
    );

    if warmup_comparisons > 0 {
        let warmup_ids = Arc::<[VectorId]>::from(vector_ids[..warmup_comparisons].to_vec());
        let warmup = run_orientation_protocol(
            orientation_mode,
            normal_stores,
            mirror_stores,
            query,
            warmup_ids,
            all_matches,
            chunk_size,
        )
        .await?;
        normal_stores = warmup.normal_stores;
        mirror_stores = warmup.mirror_stores;
    }

    let before = counter.snapshot();
    let started = Instant::now();
    let mut distance_count = 0;
    let mut rotation_match_count = 0;
    let mut match_count = 0;
    for _ in 0..repetitions {
        let run = run_orientation_protocol(
            orientation_mode,
            normal_stores,
            mirror_stores,
            query,
            vector_ids.clone(),
            all_matches,
            chunk_size,
        )
        .await?;
        normal_stores = run.normal_stores;
        mirror_stores = run.mirror_stores;
        distance_count += run.distance_count;
        rotation_match_count += run.rotation_match_count;
        match_count += run.match_count;
    }
    let elapsed = started.elapsed();
    let traffic = counter.snapshot() - before;
    ensure!(
        !session_error.is_cancelled(),
        "MPC session failed during benchmark"
    );

    println!(
        "TCP_PROTOCOL_RESULT party={party} comparisons={comparisons} repetitions={repetitions} \
         orientation_mode={} orientations={} sessions_per_orientation={sessions_per_orientation} \
         total_comparisons={total_comparisons} elapsed_seconds={:.6} \
         comparisons_per_second={:.3} distance_count={distance_count} \
         rotation_match_count={rotation_match_count} anon_match_count={match_count} \
         sent_payload_bytes={} sent_messages={} sent_framed_bytes={} \
         sent_framed_bytes_per_comparison={:.6} received_payload_bytes={} \
         received_messages={} received_framed_bytes={} \
         received_framed_bytes_per_comparison={:.6}",
        orientation_mode.as_str(),
        orientation_mode.orientations(),
        elapsed.as_secs_f64(),
        total_comparisons as f64 / elapsed.as_secs_f64(),
        traffic.sent_payload_bytes,
        traffic.sent_messages,
        traffic.sent_framed_bytes(),
        traffic.sent_framed_bytes() as f64 / total_comparisons as f64,
        traffic.received_payload_bytes,
        traffic.received_messages,
        traffic.received_framed_bytes(),
        traffic.received_framed_bytes() as f64 / total_comparisons as f64,
    );
    // Keep the fastest party from closing its data stream while another party
    // is still consuming the final opened result.
    control_channel.sync().await?;
    drop(normal_stores);
    drop(mirror_stores);
    shutdown.cancel();
    Ok(())
}
