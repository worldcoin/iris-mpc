//! Measures communication cost (bytes and messages) of the hawk_main workflow
//! on a pre-built HNSW graph.
//!
//! ## Call structure
//!
//! 1. Load plaintext irises and graph from disk.
//! 2. Secret-share iris data across 3 parties; clone the graph for each.
//! 3. Allocate local TCP addresses and spawn a `HawkActor` per party.
//! 4. Swap each actor's `NetworkHandle` with a `CountingNetworkHandle` wrapper
//!    (via `replace_networking`) to intercept all MPC traffic.
//! 5. Promote each actor into a `HawkHandle` (connects networking).
//! 6. For each batch:
//!    a. Generate query shares and call `submit_batch_query` on every party.
//!    b. Collect results and report matches/insertions.
//!    c. Print per-function communication breakdown (hierarchical + flat).
//!    d. Optionally write results to text, JSON, and CSV files.
//!
//! ## Usage
//!
//! This binary requires the `networking_metrics` feature:
//!
//!   1. Generate benchmark data first if `iris-mpc-bins/data/` is not populated:
//!      ```sh
//!      cargo run --release -p iris-mpc-bins --bin construct-graph-ptxt -- \
//!          --job-spec resources/iris-mpc-cpu/construct_graph_ptxt.toml
//!      ```
//!
//!   2. Run this binary:
//!      ```sh
//!      cargo run --release -p iris-mpc-bins --features networking_metrics \
//!          --bin communication-stats -- \
//!          --job-spec resources/iris-mpc-cpu/communication_stats.toml
//!      ```
//!
//!   Make sure that the configuration files have same overlapping parameters and paths.

use std::{
    collections::HashMap, error::Error, fmt::Write as _, path::PathBuf, sync::Arc, time::Duration,
};

use ampc_actor_utils::execution::{player::Identity, session::NetworkSession};
use async_trait::async_trait;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::{job::JobSubmissionHandle, vector_id::VectorId};
use iris_mpc_cpu::{
    execution::{
        hawk_main::{
            test_utils::{batch_of_party, make_batch, make_iris_share_with_seed},
            HawkActor, HawkArgs, HawkHandle, HawkOps,
        },
        local::get_free_local_addresses,
    },
    hawkers::{aby3::aby3_store::Aby3Store, plaintext_store::PlaintextStore},
    hnsw::{
        metrics::network_tree::{NetworkFormatter, SortBy},
        GraphMem,
    },
    network::mpc::{NetworkHandle, NetworkValue, Networking},
    protocol::shared_iris::GaloisRingSharedIris,
    utils::{
        cli::{IrisesConfig, LoadGraphConfig, SearcherConfig, SearcherParams},
        constants::N_PARTIES,
        serialization::load_toml,
    },
};
use itertools::izip;
use serde::Deserialize;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing_forest::{printer::Formatter as _, tag::NoTag, tree::Tree, ForestLayer, PrettyPrinter};
use tracing_subscriber::{
    filter::filter_fn, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

// ---------------------------------------------------------------------------
// CountingNetworking – wraps any Networking impl, tracks bytes and messages
// ---------------------------------------------------------------------------

struct CountingNetworking {
    inner: Box<dyn Networking + Send + Sync>,
}

impl std::fmt::Debug for CountingNetworking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingNetworking").finish()
    }
}

#[async_trait]
impl Networking for CountingNetworking {
    async fn send(&mut self, val: NetworkValue, receiver: &Identity) -> Result<()> {
        let bytes = val.byte_len() as u64;
        tracing::trace!(target: "searcher::network", bytes = bytes, messages = 1, "send");
        self.inner.send(val, receiver).await
    }

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        self.inner.receive(sender).await
    }
}

// ---------------------------------------------------------------------------
// CountingNetworkHandle – wraps a NetworkHandle, injects CountingNetworking
// into every session it creates
// ---------------------------------------------------------------------------

struct CountingNetworkHandle {
    inner: Box<dyn NetworkHandle>,
}

impl CountingNetworkHandle {
    fn new(inner: Box<dyn NetworkHandle>) -> Self {
        Self { inner }
    }

    fn wrap_sessions(&self, sessions: Vec<NetworkSession>) -> Vec<NetworkSession> {
        sessions
            .into_iter()
            .map(|mut s| {
                let old = std::mem::replace(
                    &mut s.networking,
                    Box::new(DummyNet) as Box<dyn Networking + Send + Sync>,
                );
                s.networking = Box::new(CountingNetworking { inner: old });
                s
            })
            .collect()
    }
}

#[async_trait]
impl NetworkHandle for CountingNetworkHandle {
    async fn make_network_sessions(&mut self) -> Result<(Vec<NetworkSession>, CancellationToken)> {
        let (sessions, ct) = self.inner.make_network_sessions().await?;
        Ok((self.wrap_sessions(sessions), ct))
    }

    async fn make_sessions(
        &mut self,
    ) -> Result<(
        Vec<ampc_actor_utils::execution::session::Session>,
        CancellationToken,
    )> {
        self.inner.make_sessions().await
    }

    async fn sync_peers(&mut self) -> Result<()> {
        self.inner.sync_peers().await
    }
}

/// Placeholder networking used during the mem::replace swap.
struct DummyNet;

impl std::fmt::Debug for DummyNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DummyNet")
    }
}

#[async_trait]
impl Networking for DummyNet {
    async fn send(&mut self, _: NetworkValue, _: &Identity) -> Result<()> {
        unreachable!("DummyNet should never be called")
    }
    async fn receive(&mut self, _: &Identity) -> Result<NetworkValue> {
        unreachable!("DummyNet should never be called")
    }
}

/// Placeholder NetworkHandle used during the replace_networking swap.
struct DummyNetHandle;

#[async_trait]
impl NetworkHandle for DummyNetHandle {
    async fn make_network_sessions(&mut self) -> Result<(Vec<NetworkSession>, CancellationToken)> {
        unreachable!("DummyNetHandle should never be called")
    }
    async fn make_sessions(
        &mut self,
    ) -> Result<(
        Vec<ampc_actor_utils::execution::session::Session>,
        CancellationToken,
    )> {
        unreachable!("DummyNetHandle should never be called")
    }
    async fn sync_peers(&mut self) -> Result<()> {
        unreachable!("DummyNetHandle should never be called")
    }
}

// ---------------------------------------------------------------------------
// Data loading: build per-party iris stores and graphs from plaintext files
// ---------------------------------------------------------------------------

type Aby3SharedIrises = iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3SharedIrises;
type BothEyes<T> = [T; 2];

fn build_party_iris_stores(plain_store: &PlaintextStore) -> Vec<BothEyes<Aby3SharedIrises>> {
    use aes_prng::AesRng;
    use rand::SeedableRng;

    let mut rng = AesRng::seed_from_u64(0_u64);
    let mut party_maps: Vec<HashMap<VectorId, Arc<GaloisRingSharedIris>>> =
        vec![HashMap::new(); N_PARTIES];

    let sorted_serial_ids = plain_store.storage.get_sorted_serial_ids();
    for serial_id in sorted_serial_ids {
        let iris = plain_store
            .storage
            .get_vector_by_serial_id(serial_id)
            .expect("Key not found");
        let vector_id = VectorId::from_serial_id(serial_id);
        let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, (**iris).clone());
        for (party_id, share) in shares.into_iter().enumerate() {
            party_maps[party_id].insert(vector_id, Arc::new(share));
        }
    }

    party_maps
        .into_iter()
        .map(|db| {
            let store_left = Aby3Store::<HawkOps>::new_storage(Some(db.clone()));
            let store_right = Aby3Store::<HawkOps>::new_storage(Some(db));
            [store_left, store_right]
        })
        .collect()
}

// ---------------------------------------------------------------------------

#[derive(Parser)]
struct Args {
    /// Path to configuration TOML file
    #[clap(long)]
    job_spec: PathBuf,
}

fn default_batch_size() -> usize {
    5
}
fn default_num_batches() -> usize {
    1
}
fn default_parallelism() -> usize {
    2
}

#[derive(Deserialize)]
struct Config {
    irises: IrisesConfig,
    graph: LoadGraphConfig,
    searcher: SearcherConfig,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_num_batches")]
    num_batches: usize,
    #[serde(default = "default_parallelism")]
    request_parallelism: usize,
    #[serde(default = "default_parallelism")]
    connection_parallelism: usize,
    #[serde(default)]
    sort_by: SortBy,
    #[serde(default)]
    tracing: bool,
    output: Option<String>,
    json_output: Option<String>,
    csv_output: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config: Config = load_toml(&args.job_spec)?;

    // Extract HNSW params from searcher config for HawkArgs
    let (ef_constr, ef_search, m) = match &config.searcher.params {
        SearcherParams::Standard {
            ef_constr,
            ef_search,
            M,
        } => (*ef_constr, *ef_search, *M),
        SearcherParams::Uniform { ef, M } => (*ef, *ef, *M),
        SearcherParams::Custom { .. } => {
            return Err("Custom searcher params not supported in communication-stats".into());
        }
    };

    // Initialize tracing — NetworkFormatter accumulates per-function stats from
    // the tracing-forest tree, which correctly tracks async spans.
    let network_formatter = Arc::new(NetworkFormatter::new().with_tracing_output(config.tracing));

    let formatter_clone = network_formatter.clone();
    let file_processor =
        PrettyPrinter::new().formatter(move |tree: &Tree| formatter_clone.fmt(tree));
    tracing_subscriber::registry()
        .with(
            ForestLayer::new(file_processor, NoTag {}).with_filter(filter_fn(|metadata| {
                metadata.target().starts_with("searcher")
            })),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(EnvFilter::new("communication_stats=info")),
        )
        .init();

    // 1. Load plaintext data
    tracing::info!("Loading iris codes...");
    let irises = iris_mpc_cpu::utils::cli::load_irises(config.irises).await?;
    let database_size = irises.len();
    let plain_store = PlaintextStore::from_irises_iter(irises.into_iter());
    tracing::info!("Loaded {database_size} irises.");

    tracing::info!("Loading graph...");
    let plain_graph = config.graph.read_graph_from_file()?;
    tracing::info!("Graph loaded.");

    // 2. Build per-party secret-shared iris stores and graphs
    tracing::info!("Secret-sharing iris data for {} parties...", N_PARTIES);
    let party_iris_stores = build_party_iris_stores(&plain_store);
    let party_graphs: Vec<BothEyes<GraphMem<_>>> = (0..N_PARTIES)
        .map(|_| [plain_graph.clone(), plain_graph.clone()])
        .collect();
    tracing::info!("Secret sharing done.");

    // 3. Set up TCP addresses for the 3 parties
    let addresses = get_free_local_addresses(N_PARTIES).await?;

    // 4. Create HawkActors with pre-loaded data and counting networking
    tracing::info!("Starting {} HawkActors...", N_PARTIES);
    let mut handles = Vec::new();
    let mut join_handles = Vec::new();

    let request_parallelism = config.request_parallelism;
    let connection_parallelism = config.connection_parallelism;

    for (i, (iris_stores, graphs)) in izip!(party_iris_stores, party_graphs).enumerate() {
        let addresses = addresses.clone();

        let join = tokio::spawn(async move {
            // Stagger startup so TCP listeners bind before connectors
            sleep(Duration::from_millis(100 * i as u64)).await;

            let hawk_args = HawkArgs::parse_from([
                "hawk_main",
                "--addresses",
                &addresses.join(","),
                "--outbound-addrs",
                &addresses.join(","),
                "--party-index",
                &i.to_string(),
                "--hnsw-param-ef-constr",
                &ef_constr.to_string(),
                "--hnsw-param-m",
                &m.to_string(),
                "--hnsw-param-ef-search",
                &ef_search.to_string(),
                "--request-parallelism",
                &request_parallelism.to_string(),
                "--connection-parallelism",
                &connection_parallelism.to_string(),
                "--disable-persistence",
            ]);

            let mut actor = HawkActor::from_cli_with_graph_and_store(
                &hawk_args,
                CancellationToken::new(),
                graphs,
                iris_stores,
            )
            .await?;

            // Swap the real networking handle out, wrap it with counting,
            // and put it back.
            let real_handle = actor.replace_networking(Box::new(DummyNetHandle));
            actor.replace_networking(Box::new(CountingNetworkHandle::new(real_handle)));

            let handle = HawkHandle::new(actor).await?;
            Ok::<_, eyre::Report>(handle)
        });

        join_handles.push(join);
    }

    for jh in join_handles {
        handles.push(jh.await??);
    }
    tracing::info!("All HawkActors started and connected.");

    // 5. Generate query shares
    let query_shares: Vec<_> = (0..N_PARTIES)
        .map(|party_id| make_iris_share_with_seed(config.batch_size, party_id, 42))
        .collect();

    // 6. Submit batches and measure communication
    for batch_idx in 0..config.num_batches {
        network_formatter.reset();

        let batch_template = make_batch(config.batch_size);

        let mut file_output = String::new();

        let batch_header = format!(
            "--- Batch {}/{} ({} queries on {} irises) ---",
            batch_idx + 1,
            config.num_batches,
            config.batch_size,
            database_size
        );
        tracing::info!("{batch_header}");
        writeln!(file_output, "{batch_header}").unwrap();

        // Submit to all parties in parallel
        let mut result_futures = Vec::new();
        for (party_idx, handle) in handles.iter_mut().enumerate() {
            let party_batch = batch_of_party(&batch_template, &query_shares[party_idx]);
            let fut = handle.submit_batch_query(party_batch).await;
            result_futures.push(fut);
        }

        // Await all results
        let results: Vec<_> = futures::future::try_join_all(result_futures).await?;

        // Report results
        let first = &results[0];
        let n_matches = first.matches.iter().filter(|&&m| m).count();
        let n_inserts = first.matches.iter().filter(|&&m| !m).count();
        let results_line = format!("  Results: {} matches, {} insertions", n_matches, n_inserts);
        tracing::info!("{results_line}");
        writeln!(file_output, "{results_line}\n").unwrap();

        // Per-function breakdown (hierarchical)
        let table = network_formatter.format_tree_table(config.sort_by);
        tracing::info!("\n{table}");
        writeln!(file_output, "{table}").unwrap();

        // Per-function breakdown (flat, summed across all call sites)
        let flat_table = network_formatter.format_flat_table(config.sort_by);
        tracing::info!("\n{flat_table}");
        writeln!(file_output, "{flat_table}").unwrap();

        if let Some(ref path) = config.output {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            write!(f, "{file_output}")?;
        }

        if let Some(ref path) = config.json_output {
            let json = network_formatter.to_json(config.sort_by);
            let file = std::fs::File::create(path)?;
            serde_json::to_writer_pretty(file, &json)?;
            tracing::info!("Per-function tree written to {path}");
        }

        if let Some(ref path) = config.csv_output {
            use std::io::Write;
            let csv = network_formatter.to_flat_csv(config.sort_by);
            let mut f = std::fs::File::create(path)?;
            f.write_all(csv.as_bytes())?;
            tracing::info!("Flat function table written to {path}");
        }
    }

    if !config.tracing {
        tracing::info!("For full tracing tree, set tracing = true in config");
    }

    tracing::info!("Done.");
    Ok(())
}
