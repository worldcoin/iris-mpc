use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    config::{ENV_PROD, ENV_STAGE},
    helpers::smpc_request::{
        REAUTH_MESSAGE_TYPE, RECOVERY_UPDATE_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    },
    postgres::{AccessMode, PostgresClient},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, StoreId, LEFT, RIGHT},
    genesis::genesis_checkpoint::{
        download_genesis_checkpoint, get_latest_checkpoint_state, GenesisCheckpointState,
    },
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::{graph_store::GraphPg, layered_graph::GraphMem},
        searcher::HnswParams,
    },
};
use iris_mpc_store::Store;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as FmtWrite,
    fs,
    io::Write as IoWrite,
    path::{Path, PathBuf},
    process,
};

/// Dual-output writer: every line goes to both stdout and an internal buffer.
/// The buffer is saved to `report.txt` alongside the JSON output files.
struct Report(String);

impl Report {
    fn new() -> Self {
        Self(String::new())
    }

    fn log(&mut self, line: &str) {
        println!("{line}");
        self.0.push_str(line);
        self.0.push('\n');
    }

    fn save(&self, dir: &Path) -> eyre::Result<PathBuf> {
        let p = dir.join("report.txt");
        let mut f = fs::File::create(&p)?;
        f.write_all(self.0.as_bytes())?;
        Ok(p)
    }
}

macro_rules! rpt {
    ($report:expr) => {
        $report.log("")
    };
    ($report:expr, $($arg:tt)*) => {
        $report.log(&format!($($arg)*))
    };
}

const STATE_DOMAIN: &str = "genesis";
const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";
const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";

/// Number of random serial IDs to sample for the cross-schema iris comparison.
const SAMPLE_COUNT: usize = 1_000;

/// Iris row from DB: (id, left_code, left_mask, right_code, right_mask).
type IrisRow = (i64, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
/// Iris code data: (left_code, left_mask, right_code, right_mask).
type IrisData = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
/// Number of recent processed modifications whose serial IDs to include in the sample.
const RECENT_MOD_COUNT: i64 = 100;
/// Modification request types that update (overwrite) existing iris code data.
const IRIS_UPDATE_TYPES: &[&str] = &[
    RESET_UPDATE_MESSAGE_TYPE,
    RECOVERY_UPDATE_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE,
];

#[derive(Parser)]
#[command(
    name = "db-sanity-check",
    about = "Validate DB state for a single MPC party"
)]
struct Args {
    /// Postgres connection string for the HNSW (CPU) database
    #[arg(long, env = "HNSW_DATABASE_URL")]
    hnsw_db_url: String,
    /// Postgres connection string for the GPU database
    #[arg(long, env = "GPU_DATABASE_URL")]
    gpu_db_url: String,
    /// HNSW (CPU) schema name
    #[arg(long)]
    hnsw_schema: String,
    /// GPU schema name
    #[arg(long)]
    gpu_schema: String,
    /// HNSW M parameter for degree bound checks
    #[arg(long, default_value_t = 256)]
    m: usize,
    /// Layer probability q for geometric distribution check (default: 1/M).
    /// Each layer should have ~q fraction of the nodes in the layer below.
    #[arg(long)]
    layer_probability: Option<f64>,
    /// S3 URI to JSON exclusions file with {"deleted_serial_ids": [...]}
    /// (e.g. s3://bucket/path/deleted_serial_ids.json)
    #[arg(long)]
    exclusions_s3_uri: Option<String>,
    /// Directory for JSON output files
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,
    /// RNG seed for reproducible cross-schema sampling
    #[arg(long)]
    seed: u64,
    /// S3 URI to upload output files to (e.g. s3://bucket/prefix/)
    #[arg(long)]
    s3_output: Option<String>,
    /// S3 key for a specific graph checkpoint. When set, overrides
    /// auto-discovery from the `genesis_graph_checkpoint` DB table. Only used
    /// when S3 mode is enabled (see `SMPC__GRAPH_CHECKPOINT_BUCKET_NAME` env).
    #[arg(long)]
    checkpoint_s3_key: Option<String>,
}

/// Subset of `iris_mpc_common::config::Config` that the sanity check reads
/// from the environment (variables prefixed `SMPC__`, separator `__`). Field
/// names / defaults mirror the full Config so deployments that already set
/// these values for genesis get the same behavior here.
///
/// - `environment`: controls `force_path_style` (true except in prod/stage).
/// - `graph_checkpoint_bucket_name`: empty string disables S3-checkpoint mode
///   and falls back to loading the graph from the Postgres links table.
/// - `graph_checkpoint_bucket_region`: region for the checkpoint bucket; may
///   differ from the ambient AWS region.
#[derive(Debug, Clone, Deserialize)]
struct SanityCheckConfig {
    #[serde(default)]
    environment: String,
    #[serde(default = "default_graph_checkpoint_bucket_name")]
    graph_checkpoint_bucket_name: String,
    #[serde(default = "default_graph_checkpoint_bucket_region")]
    graph_checkpoint_bucket_region: String,
}

fn default_graph_checkpoint_bucket_name() -> String {
    "wf-smpcv2-dev-hnsw-checkpoint".to_string()
}

fn default_graph_checkpoint_bucket_region() -> String {
    "eu-north-1".to_string()
}

impl SanityCheckConfig {
    fn load() -> Result<Self> {
        let cfg: Self = config::Config::builder()
            .add_source(
                config::Environment::with_prefix("SMPC")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?
            .try_deserialize()?;
        Ok(cfg)
    }

    /// LocalStack / dev need path-style; prod and stage use virtual-hosted.
    fn force_path_style(&self) -> bool {
        self.environment != ENV_PROD && self.environment != ENV_STAGE
    }

    /// Empty bucket name disables S3-checkpoint mode.
    fn s3_checkpoint_enabled(&self) -> bool {
        !self.graph_checkpoint_bucket_name.is_empty()
    }
}

#[derive(Deserialize)]
struct ExclusionsFile {
    deleted_serial_ids: Vec<u32>,
}

#[derive(Serialize)]
struct CheckResult {
    id: String,
    name: String,
    #[serde(rename = "status", serialize_with = "ser_status")]
    passed: bool,
    detail: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

fn ser_status<S: serde::Serializer>(v: &bool, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(if *v { "PASS" } else { "FAIL" })
}

impl CheckResult {
    fn new(id: &str, name: &str, ok: bool, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            passed: ok,
            detail: detail.into(),
            warnings: Vec::new(),
        }
    }

    fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

#[derive(Serialize)]
struct DegreeHistEntry {
    eye: String,
    layer: usize,
    degree: usize,
    node_count: usize,
}

struct Stats(Vec<(String, String)>);
impl Stats {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn add(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.0.push((key.into(), value.into()));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let config = SanityCheckConfig::load()?;

    let mut rpt = Report::new();

    rpt!(rpt, "=== DB Sanity Check ===");
    rpt!(
        rpt,
        "HNSW schema: {}  GPU schema: {}  M: {}",
        args.hnsw_schema,
        args.gpu_schema,
        args.m
    );
    rpt!(
        rpt,
        "environment={:?} checkpoint_bucket={:?} checkpoint_region={:?} force_path_style={}",
        config.environment,
        config.graph_checkpoint_bucket_name,
        config.graph_checkpoint_bucket_region,
        config.force_path_style(),
    );
    rpt!(rpt);

    let hnsw_pg =
        PostgresClient::new(&args.hnsw_db_url, &args.hnsw_schema, AccessMode::ReadOnly).await?;
    let gpu_pg =
        PostgresClient::new(&args.gpu_db_url, &args.gpu_schema, AccessMode::ReadOnly).await?;
    let hnsw_store = Store::new(&hnsw_pg).await?;
    let graph_pg = GraphPg::<Aby3Store>::new(&hnsw_pg).await?;

    let mut checks: Vec<CheckResult> = Vec::new();
    let mut stats = Stats::new();
    let mut degree_hist: Vec<DegreeHistEntry> = Vec::new();

    let raw_exclusions: Option<Vec<u32>> = match &args.exclusions_s3_uri {
        Some(uri) => {
            let parsed = download_exclusions_from_s3(uri, config.force_path_style()).await?;
            Some(parsed.deleted_serial_ids)
        }
        None => None,
    };

    // --- Optional: load graph from S3 checkpoint ---
    let s3_graphs: Option<BothEyes<GraphMem<VectorId>>> = if config.s3_checkpoint_enabled() {
        let bucket = config.graph_checkpoint_bucket_name.as_str();
        rpt!(rpt, "--- Loading graph from S3 checkpoint ---");
        let checkpoint_state = load_checkpoint_state(
            &graph_pg,
            args.checkpoint_s3_key.as_deref(),
            bucket,
            &mut rpt,
        )
        .await?;

        let s3_client = build_checkpoint_s3_client(
            &config.graph_checkpoint_bucket_region,
            config.force_path_style(),
        )
        .await;

        rpt!(
            rpt,
            "  Downloading checkpoint: s3://{}/{}",
            bucket,
            checkpoint_state.s3_key
        );
        let graphs: BothEyes<GraphMem<VectorId>> =
            download_genesis_checkpoint(&s3_client, bucket, &checkpoint_state).await?;
        rpt!(rpt, "  Checkpoint loaded and BLAKE3 verified.");

        // Check 0a: checkpoint metadata vs persistent_state genesis watermarks
        rpt!(rpt, "--- Check 0a: Checkpoint metadata validation ---");
        let ps_iris_id: Option<u32> = graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
            .await?;
        let ps_mod_id: Option<i64> = graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_MODIFICATION_ID)
            .await?;

        let iris_ok = ps_iris_id
            .map(|ps| ps == checkpoint_state.last_indexed_iris_id)
            .unwrap_or(false);
        let mod_ok = ps_mod_id
            .map(|ps| ps == checkpoint_state.last_indexed_modification_id)
            .unwrap_or(false);
        let cp_ok = iris_ok && mod_ok;

        let detail = format!(
            "checkpoint(iris_id={}, mod_id={}) vs persistent_state(iris_id={}, mod_id={})",
            checkpoint_state.last_indexed_iris_id,
            checkpoint_state.last_indexed_modification_id,
            ps_iris_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "not set".into()),
            ps_mod_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "not set".into()),
        );
        rpt!(
            rpt,
            "  [{}] {}",
            if cp_ok { "OK" } else { "MISMATCH" },
            detail
        );
        checks.push(CheckResult::new("0a", "Checkpoint metadata", cp_ok, detail));

        stats.add("checkpoint_s3_key", &checkpoint_state.s3_key);
        stats.add(
            "checkpoint_last_indexed_iris_id",
            checkpoint_state.last_indexed_iris_id.to_string(),
        );
        stats.add(
            "checkpoint_last_indexed_modification_id",
            checkpoint_state.last_indexed_modification_id.to_string(),
        );
        stats.add("checkpoint_blake3_hash", &checkpoint_state.blake3_hash);

        Some(graphs)
    } else {
        None
    };

    rpt!(rpt, "--- Collecting iris IDs ---");
    let iris_ids = collect_iris_ids(&hnsw_store, &mut stats).await?;

    // Filter exclusions to IDs that actually exist in this DB snapshot.
    // Genesis filters deletions to <= max_indexation_id; the S3 file may
    // contain IDs beyond this snapshot's range.
    let iris_max = iris_ids.iter().copied().max().unwrap_or(0) as u32;
    let exclusions: Option<HashSet<u32>> = raw_exclusions.map(|raw| {
        let before = raw.len();
        let filtered: HashSet<u32> = raw.into_iter().filter(|&id| id <= iris_max).collect();
        rpt!(
            rpt,
            "  Exclusions: {} total in file, {} after filtering to id <= {iris_max}",
            before,
            filtered.len()
        );
        filtered
    });

    rpt!(rpt, "--- Check 1: HNSW graph structural checks ---");
    let layer_probability = args.layer_probability.unwrap_or((args.m as f64).recip());
    run_graph_checks(
        &graph_pg,
        s3_graphs.as_ref(),
        &iris_ids,
        &exclusions,
        args.m,
        layer_probability,
        &mut checks,
        &mut degree_hist,
        &mut stats,
        &mut rpt,
    )
    .await?;

    rpt!(rpt, "--- Check 2: Persistent state consistency ---");
    let iris_max = hnsw_store.get_max_serial_id().await?;
    let last_mod_id = run_persistent_state_checks(
        &graph_pg,
        iris_max,
        s3_graphs.as_ref(),
        &mut checks,
        &mut stats,
    )
    .await?;

    rpt!(
        rpt,
        "--- Check 3: HNSW vs GPU iris consistency (sampled, modification-aware) ---"
    );
    run_cross_schema_checks(
        iris_max,
        last_mod_id,
        args.seed,
        &hnsw_store.pool,
        &gpu_pg.pool,
        &mut checks,
        &mut rpt,
    )
    .await?;

    // --- Report ---
    rpt!(rpt, "\n--- Checks ---");
    let pass_count = checks.iter().filter(|c| c.passed).count();
    for c in &checks {
        let tag = if c.passed { "PASS" } else { "FAIL" };
        rpt!(rpt, "[{tag}] {}: {} ({})", c.id, c.name, c.detail);
    }
    rpt!(rpt, "\n--- Stats ---");
    for (k, v) in &stats.0 {
        rpt!(rpt, "{k}: {v}");
    }
    let fail_count = checks.len() - pass_count;
    rpt!(
        rpt,
        "\n=== Summary: {pass_count}/{} checks passed, {fail_count} failed ===",
        checks.len()
    );

    let mut output_files = write_json_reports(&args.output_dir, &checks, &stats, &degree_hist)?;
    let report_path = rpt.save(&args.output_dir)?;
    println!("Wrote {}", report_path.display());
    output_files.push(report_path);

    if let Some(s3_uri) = &args.s3_output {
        upload_to_s3(s3_uri, &output_files, config.force_path_style()).await?;
    }

    if fail_count > 0 {
        process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Collect iris serial IDs (needed for graph orphan / coverage checks)
// ---------------------------------------------------------------------------

async fn collect_iris_ids(store: &Store, stats: &mut Stats) -> Result<HashSet<i64>> {
    let ids: Vec<(i64,)> = sqlx::query_as("SELECT id FROM irises")
        .fetch_all(&store.pool)
        .await?;

    let max_id = ids.iter().map(|(id,)| *id).max().unwrap_or(0);
    stats.add("Total iris count (HNSW)", ids.len().to_string());
    stats.add("Max serial ID (HNSW)", max_id.to_string());

    Ok(ids.into_iter().map(|(id,)| id).collect())
}

// ---------------------------------------------------------------------------
// Check 1: HNSW graph structural checks
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_graph_checks(
    graph_pg: &GraphPg<Aby3Store>,
    s3_graphs: Option<&BothEyes<GraphMem<VectorId>>>,
    iris_ids: &HashSet<i64>,
    exclusions: &Option<HashSet<u32>>,
    m: usize,
    layer_probability: f64,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    stats: &mut Stats,
    rpt: &mut Report,
) -> Result<()> {
    let mut l0_id_sets: Vec<(&str, HashSet<u32>)> = Vec::new();

    if let Some(graphs) = s3_graphs {
        // Graphs already loaded from S3 checkpoint.
        for (eye, idx) in [("left", LEFT), ("right", RIGHT)] {
            rpt!(rpt, "  Checking {eye} graph (from S3 checkpoint)...");
            let l0_ids = check_single_graph(
                eye,
                &graphs[idx],
                iris_ids,
                exclusions,
                m,
                layer_probability,
                checks,
                degree_hist,
                stats,
                rpt,
            );
            l0_id_sets.push((eye, l0_ids));
        }
    } else {
        // Load one graph at a time from Postgres to halve peak memory.
        for (eye, store_id) in [("left", StoreId::Left), ("right", StoreId::Right)] {
            rpt!(rpt, "  Loading {eye} graph...");
            let graph = load_graph(graph_pg, store_id).await?;
            let l0_ids = check_single_graph(
                eye,
                &graph,
                iris_ids,
                exclusions,
                m,
                layer_probability,
                checks,
                degree_hist,
                stats,
                rpt,
            );
            drop(graph);
            l0_id_sets.push((eye, l0_ids));
        }
    }

    // -- 1i: Left/Right graph sync --
    let (left_ids, right_ids) = (&l0_id_sets[0].1, &l0_id_sets[1].1);
    checks.push(CheckResult::new(
        "1i",
        "Left/Right graph sync",
        left_ids == right_ids,
        if left_ids == right_ids {
            format!("{} serial IDs match at layer 0", left_ids.len())
        } else {
            format!(
                "Mismatch: {} only in left, {} only in right",
                left_ids.difference(right_ids).count(),
                right_ids.difference(left_ids).count()
            )
        },
    ));

    Ok(())
}

/// Run per-eye checks (1a–1h, 1j) and return the layer-0 serial ID set.
#[allow(clippy::too_many_arguments)]
fn check_single_graph(
    eye: &str,
    graph: &GraphMem<VectorId>,
    iris_ids: &HashSet<i64>,
    exclusions: &Option<HashSet<u32>>,
    m: usize,
    layer_probability: f64,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    stats: &mut Stats,
    rpt: &mut Report,
) -> HashSet<u32> {
    // ef values are irrelevant
    let params = HnswParams::new(1, 1, m);

    // -- Stats --
    stats.add(
        format!("{eye} graph checksum"),
        graph.checksum().to_string(),
    );
    stats.add(
        format!("{eye} graph num_layers"),
        graph.num_layers().to_string(),
    );
    let ep_desc = graph
        .entry_points
        .iter()
        .map(|ep| format!("{}@L{}", ep.point, ep.layer))
        .collect::<Vec<_>>()
        .join(", ");
    stats.add(format!("{eye} entry points"), ep_desc);

    for (lc, layer) in graph.layers.iter().enumerate() {
        stats.add(
            format!("{eye} layer {lc} node count"),
            layer.links.len().to_string(),
        );
        let mut deg_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for neighbors in layer.links.values() {
            *deg_counts.entry(neighbors.len()).or_insert(0) += 1;
        }
        for (&degree, &count) in &deg_counts {
            degree_hist.push(DegreeHistEntry {
                eye: eye.to_string(),
                layer: lc,
                degree,
                node_count: count,
            });
        }
        if !layer.links.is_empty() {
            let mut degrees: Vec<usize> = layer.links.values().map(|n| n.len()).collect();
            degrees.sort();
            let (min, max) = (degrees[0], degrees[degrees.len() - 1]);
            let avg = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
            let median = degrees[degrees.len() / 2];
            stats.add(
                format!("{eye} layer {lc} degree min/avg/median/max"),
                format!("{min}/{avg:.1}/{median}/{max}"),
            );
        }
    }

    // -- 1a: No orphan graph nodes --
    let orphan_count = graph
        .layers
        .iter()
        .flat_map(|l| l.links.keys())
        .filter(|n| !iris_ids.contains(&(n.serial_id() as i64)))
        .count();
    checks.push(CheckResult::new(
        "1a",
        &format!("No orphan graph nodes ({eye})"),
        orphan_count == 0,
        if orphan_count == 0 {
            "All graph nodes exist in irises table".into()
        } else {
            format!("{orphan_count} graph nodes not found in irises table")
        },
    ));

    // -- 1b: Node coverage --
    let layer0_ids: HashSet<u32> = graph
        .layers
        .first()
        .map(|l| l.links.keys().map(|v| v.serial_id()).collect())
        .unwrap_or_default();
    let uncovered: HashSet<u32> = iris_ids
        .iter()
        .map(|&id| id as u32)
        .filter(|id| !layer0_ids.contains(id))
        .collect();
    checks.push(match exclusions {
        Some(excl) if uncovered == *excl => CheckResult::new(
            "1b",
            &format!("Node coverage ({eye})"),
            true,
            format!(
                "{} uncovered IDs match exclusions list exactly",
                uncovered.len()
            ),
        ),
        Some(excl) => {
            let only_uncov: Vec<_> = uncovered.difference(excl).copied().collect();
            let only_excl: Vec<_> = excl.difference(&uncovered).copied().collect();
            let mut d = format!(
                "Mismatch: {} uncovered, {} excluded",
                uncovered.len(),
                excl.len()
            );
            if !only_uncov.is_empty() {
                let _ = write!(
                    d,
                    "; {} in DB not in exclusions: {:?}",
                    only_uncov.len(),
                    &only_uncov[..only_uncov.len().min(10)]
                );
            }
            if !only_excl.is_empty() {
                let _ = write!(
                    d,
                    "; {} in exclusions not uncovered: {:?}",
                    only_excl.len(),
                    &only_excl[..only_excl.len().min(10)]
                );
            }
            CheckResult::new("1b", &format!("Node coverage ({eye})"), false, d)
        }
        None => CheckResult::new(
            "1b",
            &format!("Node coverage ({eye})"),
            true,
            format!(
                "{} iris IDs not in graph layer 0 (no exclusions file)",
                uncovered.len()
            ),
        ),
    });

    // -- 1c: Layer hierarchy --
    let hierarchy_viol: u64 = graph
        .layers
        .iter()
        .enumerate()
        .skip(1)
        .flat_map(|(lc, layer)| {
            layer.links.keys().flat_map(move |node| {
                (0..lc).filter(move |&lower| !graph.layers[lower].links.contains_key(node))
            })
        })
        .count() as u64;
    checks.push(CheckResult::new(
        "1c",
        &format!("Layer hierarchy ({eye})"),
        hierarchy_viol == 0,
        if hierarchy_viol == 0 {
            "All higher-layer nodes present in lower layers".into()
        } else {
            format!("{hierarchy_viol} hierarchy violations")
        },
    ));

    // -- 1d: Neighbor validity --
    let invalid_nb: u64 = graph
        .layers
        .iter()
        .map(|layer| {
            let nodes: HashSet<&VectorId> = layer.links.keys().collect();
            layer
                .links
                .values()
                .flat_map(|nbs| nbs.iter())
                .filter(|nb| !nodes.contains(nb))
                .count() as u64
        })
        .sum();
    checks.push(CheckResult::new(
        "1d",
        &format!("Neighbor validity ({eye})"),
        invalid_nb == 0,
        if invalid_nb == 0 {
            "All neighbors reference valid nodes at the same layer".into()
        } else {
            format!("{invalid_nb} invalid neighbor references")
        },
    ));

    // -- 1e: No self-loops --
    let self_loops: u64 = graph
        .layers
        .iter()
        .flat_map(|l| l.links.iter())
        .filter(|(node, nbs)| nbs.contains(node))
        .count() as u64;
    checks.push(CheckResult::new(
        "1e",
        &format!("No self-loops ({eye})"),
        self_loops == 0,
        if self_loops == 0 {
            "No node lists itself as neighbor".into()
        } else {
            format!("{self_loops} self-loops found")
        },
    ));

    // -- 1f: No duplicate neighbors --
    let dup_count: u64 = graph
        .layers
        .iter()
        .flat_map(|l| l.links.values())
        .map(|nbs| {
            let unique: HashSet<&VectorId> = nbs.iter().collect();
            (nbs.len() - unique.len()) as u64
        })
        .sum();
    checks.push(CheckResult::new(
        "1f",
        &format!("No duplicate neighbors ({eye})"),
        dup_count == 0,
        if dup_count == 0 {
            "No duplicate neighbors found".into()
        } else {
            format!("{dup_count} duplicate neighbor entries")
        },
    ));

    // -- 1g: Degree bounds --
    let mut degree_viol = 0u64;
    for (lc, layer) in graph.layers.iter().enumerate() {
        let m_limit = params.get_M_limit(lc);
        for (node, nbs) in layer.links.iter() {
            if nbs.len() > m_limit {
                degree_viol += 1;
                if degree_viol <= 5 {
                    rpt!(
                        rpt,
                        "  [1g] {eye} L{lc} node {node} degree {} > M_limit {m_limit}",
                        nbs.len()
                    );
                }
            }
        }
    }
    checks.push(CheckResult::new(
        "1g",
        &format!("Degree bounds ({eye})"),
        degree_viol == 0,
        if degree_viol == 0 {
            format!(
                "L0 M_limit={}, L1+ M_limit={}",
                params.get_M_limit(0),
                params.get_M_limit(1),
            )
        } else {
            format!("{degree_viol} nodes exceed M_limit")
        },
    ));

    // -- 1h: Entry point validity --
    let mut ep_valid = true;
    let mut ep_detail = String::new();
    for ep in &graph.entry_points {
        if ep.layer >= graph.layers.len() {
            ep_valid = false;
            let _ = write!(
                ep_detail,
                "EP {} at layer {} but only {} layers; ",
                ep.point,
                ep.layer,
                graph.layers.len()
            );
        } else if !graph.layers[ep.layer].links.contains_key(&ep.point) {
            ep_valid = false;
            let _ = write!(
                ep_detail,
                "EP {} not found in layer {}; ",
                ep.point, ep.layer
            );
        }
    }
    checks.push(CheckResult::new(
        "1h",
        &format!("Entry point validity ({eye})"),
        ep_valid,
        if ep_valid {
            format!("{} entry points valid", graph.entry_points.len())
        } else {
            ep_detail
        },
    ));

    // -- 1j: Layer density near geometric --
    // Each node independently lands at layer >= L with probability q^L, so
    // count at layer L ~ Binomial(N, q^L).  Flag if actual count is more than
    // 3 standard deviations from the expected value.
    let n = graph.layers.first().map(|l| l.links.len()).unwrap_or(0) as f64;
    if graph.layers.len() < 2 || n == 0.0 {
        checks.push(CheckResult::new(
            "1j",
            &format!("Layer density geometric ({eye})"),
            true,
            "Fewer than 2 layers, skipped",
        ));
    } else {
        let mut violations = Vec::new();
        for lc in 1..graph.layers.len() {
            let p = layer_probability.powi(lc as i32);
            let expected = n * p;
            let std_dev = (n * p * (1.0 - p)).sqrt();
            let actual = graph.layers[lc].links.len() as f64;
            let z = if std_dev > 0.0 {
                (actual - expected).abs() / std_dev
            } else {
                0.0
            };
            if z > 3.0 {
                violations.push(format!(
                    "L{lc}: {actual:.0} nodes, expected {expected:.1} +/- {std_dev:.1} ({z:.1}σ)"
                ));
            }
        }
        checks.push(CheckResult::new(
            "1j",
            &format!("Layer density geometric ({eye})"),
            violations.is_empty(),
            if violations.is_empty() {
                format!(
                    "All layers within 3σ of Binomial(N={n:.0}, q={layer_probability:.4}) ({} layers)",
                    graph.layers.len()
                )
            } else {
                format!("Outliers: {}", violations.join(", "))
            },
        ));
    }

    layer0_ids
}

async fn load_graph(
    graph_pg: &GraphPg<Aby3Store>,
    store_id: StoreId,
) -> Result<GraphMem<VectorId>> {
    let mut tx = graph_pg.tx().await?;
    let mut ops = tx.with_graph(store_id);
    todo!() //ops.load_to_mem(graph_pg.pool(), 4).await
}

/// Resolve checkpoint state: use explicit S3 key (looked up from DB) or
/// auto-discover the latest checkpoint from the genesis_graph_checkpoint table.
async fn load_checkpoint_state(
    graph_pg: &GraphPg<Aby3Store>,
    explicit_key: Option<&str>,
    bucket: &str,
    rpt: &mut Report,
) -> Result<GenesisCheckpointState> {
    if let Some(key) = explicit_key {
        rpt!(
            rpt,
            "  Using explicit checkpoint key: s3://{}/{}",
            bucket,
            key
        );
        // Look up the matching row from DB so we have the blake3 hash for verification.
        let row = graph_pg
            .get_genesis_graph_checkpoint_by_key(key)
            .await?
            .ok_or_else(|| eyre::eyre!("No checkpoint row found in DB for S3 key: {}", key))?;
        Ok(GenesisCheckpointState {
            s3_key: row.s3_key,
            last_indexed_iris_id: row.last_indexed_iris_id.try_into().map_err(|_| {
                eyre::eyre!("Invalid last_indexed_iris_id: {}", row.last_indexed_iris_id)
            })?,
            last_indexed_modification_id: row.last_indexed_modification_id,
            blake3_hash: row.blake3_hash,
            is_archival: row.is_archival,
        })
    } else {
        rpt!(rpt, "  Auto-discovering latest checkpoint from DB...");
        let state = get_latest_checkpoint_state(graph_pg).await?;
        match state {
            Some(s) => {
                rpt!(
                    rpt,
                    "  Found checkpoint: s3://{}/{} (iris_id={}, mod_id={})",
                    bucket,
                    s.s3_key,
                    s.last_indexed_iris_id,
                    s.last_indexed_modification_id
                );
                Ok(s)
            }
            None => Err(eyre::eyre!(
                "No checkpoint found in genesis_graph_checkpoint table"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Check 2: Persistent state consistency
// ---------------------------------------------------------------------------

async fn run_persistent_state_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_max_serial_id: usize,
    s3_graphs: Option<&BothEyes<GraphMem<VectorId>>>,
    checks: &mut Vec<CheckResult>,
    stats: &mut Stats,
) -> Result<Option<i64>> {
    let last_indexed: Option<u32> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
        .await?;
    // When the graph is loaded from an S3 checkpoint, the Postgres links table
    // is not the source of truth for the graph, so read the max serial id from
    // the in-memory graphs instead.
    let (left_max, right_max) = match s3_graphs {
        Some(graphs) => (
            graph_mem_max_serial_id(&graphs[LEFT]),
            graph_mem_max_serial_id(&graphs[RIGHT]),
        ),
        None => (
            get_graph_max_serial_id(graph_pg, StoreId::Left).await?,
            get_graph_max_serial_id(graph_pg, StoreId::Right).await?,
        ),
    };

    // 2a: last_indexed_iris_id matches irises table max serial ID
    match last_indexed {
        Some(last_id) => {
            let ok = last_id as usize == iris_max_serial_id;
            checks.push(CheckResult::new(
                "2a",
                "last_indexed_iris_id",
                ok,
                if ok {
                    format!("Consistent: {last_id}")
                } else {
                    format!("last_indexed={last_id} != irises max={iris_max_serial_id}")
                },
            ));
        }
        None => checks.push(CheckResult::new(
            "2a",
            "last_indexed_iris_id",
            true,
            "Not set (no genesis run yet)",
        )),
    }

    // 2b: left and right graphs have the same max serial ID
    checks.push(CheckResult::new(
        "2b",
        "Graph max serial_id alignment",
        left_max == right_max,
        if left_max == right_max {
            format!("Both graphs max serial_id = {left_max}")
        } else {
            format!("Left max={left_max}, Right max={right_max}")
        },
    ));

    // Stat: last_indexed_modification_id
    let last_mod_id: Option<i64> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_MODIFICATION_ID)
        .await?;
    stats.add(
        "last_indexed_modification_id",
        match last_mod_id {
            Some(id) => id.to_string(),
            None => "not set".into(),
        },
    );

    Ok(last_mod_id)
}

async fn get_graph_max_serial_id(graph_pg: &GraphPg<Aby3Store>, store_id: StoreId) -> Result<i64> {
    let mut tx = graph_pg.tx().await?;
    let mut ops = tx.with_graph(store_id);
    todo!() //ops.get_max_serial_id().await
}

/// Returns the maximum serial_id across all nodes in layer 0 of an in-memory
/// graph, or 0 if the graph is empty. Matches the semantics of
/// `GraphOps::get_max_serial_id`, which returns 0 when the links table is empty.
fn graph_mem_max_serial_id(graph: &GraphMem<VectorId>) -> i64 {
    graph
        .layers
        .first()
        .and_then(|layer| {
            layer
                .get_links_map()
                .keys()
                .map(|vec_id| vec_id.serial_id())
                .max()
        })
        .map(|id| id as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Check 3: Cross-schema consistency (HNSW vs GPU)
// ---------------------------------------------------------------------------

async fn run_cross_schema_checks(
    last_indexed_id: usize,
    last_indexed_mod_id: Option<i64>,
    seed: u64,
    hnsw_pool: &sqlx::PgPool,
    gpu_pool: &sqlx::PgPool,
    checks: &mut Vec<CheckResult>,
    rpt: &mut Report,
) -> Result<()> {
    let lid = last_indexed_id as i64;

    // 3a: Row count comparison (informational — not a hard check)
    let hnsw_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM irises WHERE id <= $1")
        .bind(lid)
        .fetch_one(hnsw_pool)
        .await?;
    let gpu_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM irises WHERE id <= $1")
        .bind(lid)
        .fetch_one(gpu_pool)
        .await?;
    let (hnsw_count, gpu_count) = (hnsw_count.0, gpu_count.0);
    if hnsw_count == gpu_count {
        rpt!(
            rpt,
            "  3a: Row count (id <= {last_indexed_id}): {hnsw_count} (both schemas match)"
        );
    } else {
        rpt!(rpt, "  3a: Row count (id <= {last_indexed_id}): HNSW={hnsw_count}, GPU={gpu_count} (MISMATCH)");
    }

    // 3b: Max serial ID comparison (informational — not a hard check)
    let hnsw_max: (i64,) = sqlx::query_as("SELECT COALESCE(MAX(id), 0) FROM irises WHERE id <= $1")
        .bind(lid)
        .fetch_one(hnsw_pool)
        .await?;
    let gpu_max: (i64,) = sqlx::query_as("SELECT COALESCE(MAX(id), 0) FROM irises WHERE id <= $1")
        .bind(lid)
        .fetch_one(gpu_pool)
        .await?;
    let (hnsw_max, gpu_max) = (hnsw_max.0, gpu_max.0);
    if hnsw_max == gpu_max {
        rpt!(
            rpt,
            "  3b: Max serial ID (id <= {last_indexed_id}): {hnsw_max} (both schemas match)"
        );
    } else {
        rpt!(rpt, "  3b: Max serial ID (id <= {last_indexed_id}): HNSW={hnsw_max}, GPU={gpu_max} (MISMATCH)");
    }

    // 3c: Sampled byte-identical shares, with modification-aware exclusions.
    //
    // Rather than scanning all BYTEA data, we:
    //   1. Sample ~SAMPLE_COUNT random serial IDs from [1, last_indexed_id].
    //   2. Augment with up to RECENT_MOD_COUNT serial IDs from the most-recent
    //      processed modifications (id <= last_indexed_mod_id), so recently
    //      modified irises are always exercised.
    //   3. Fetch iris data from each DB independently, compare in Rust.
    //   4. Load any modifications that arrived *after* last_indexed_mod_id and
    //      that update iris code data; exclude their serial IDs from the check
    //      because GPU may reflect the new code while HNSW still has the old one.
    if last_indexed_id == 0 {
        checks.push(CheckResult::new(
            "3c",
            "Byte-identical shares (sampled)",
            true,
            "No irises indexed yet, skipped",
        ));
        return Ok(());
    }

    // Step 1: Sample random serial IDs.
    let mut sample_set: HashSet<i64> = {
        let mut rng = StdRng::seed_from_u64(seed);
        if last_indexed_id <= SAMPLE_COUNT {
            (1..=lid).collect()
        } else {
            let mut ids: HashSet<i64> = HashSet::with_capacity(SAMPLE_COUNT);
            while ids.len() < SAMPLE_COUNT {
                ids.insert(rng.gen_range(1..=lid));
            }
            ids
        }
    };

    // Step 2: Augment with serial IDs from recently processed modifications.
    // The modifications table lives in the GPU schema.
    let after_mod_id = last_indexed_mod_id.unwrap_or(0);
    if last_indexed_mod_id.is_some() {
        let recent: Vec<(i64,)> = sqlx::query_as(
            "SELECT serial_id FROM modifications \
             WHERE id <= $1 AND serial_id IS NOT NULL \
             ORDER BY id DESC LIMIT $2",
        )
        .bind(after_mod_id)
        .bind(RECENT_MOD_COUNT)
        .fetch_all(gpu_pool)
        .await?;
        for (sid,) in recent {
            if (1..=lid).contains(&sid) {
                sample_set.insert(sid);
            }
        }
    }

    let sample_vec: Vec<i64> = sample_set.into_iter().collect();
    rpt!(
        rpt,
        "  Comparing {} sampled serial IDs between schemas (seed={seed})...",
        sample_vec.len(),
    );

    // Step 3: Fetch iris data from each DB independently and compare in Rust.
    let hnsw_rows: Vec<IrisRow> = sqlx::query_as(
        "SELECT id, left_code, left_mask, right_code, right_mask \
         FROM irises WHERE id = ANY($1)",
    )
    .bind(&sample_vec)
    .fetch_all(hnsw_pool)
    .await?;

    let gpu_rows: Vec<IrisRow> = sqlx::query_as(
        "SELECT id, left_code, left_mask, right_code, right_mask \
         FROM irises WHERE id = ANY($1)",
    )
    .bind(&sample_vec)
    .fetch_all(gpu_pool)
    .await?;

    let hnsw_map: HashMap<i64, IrisData> = hnsw_rows
        .into_iter()
        .map(|(id, lc, lm, rc, rm)| (id, (lc, lm, rc, rm)))
        .collect();
    let gpu_map: HashMap<i64, IrisData> = gpu_rows
        .into_iter()
        .map(|(id, lc, lm, rc, rm)| (id, (lc, lm, rc, rm)))
        .collect();

    let mut mismatched_raw: Vec<i64> = Vec::new();
    let mut only_in_hnsw: Vec<i64> = Vec::new();
    let mut only_in_gpu: Vec<i64> = Vec::new();
    for (&id, hnsw_data) in &hnsw_map {
        match gpu_map.get(&id) {
            Some(gpu_data) if gpu_data != hnsw_data => mismatched_raw.push(id),
            None => only_in_hnsw.push(id),
            _ => {}
        }
    }
    for &id in gpu_map.keys() {
        if !hnsw_map.contains_key(&id) {
            only_in_gpu.push(id);
        }
    }

    // Warn about one-sided IDs.  A handful is expected around identity
    // deletions; a large number suggests a real problem.
    const ONE_SIDED_WARN_THRESHOLD: usize = 10;
    let mut one_sided_warnings: Vec<String> = Vec::new();
    if !only_in_hnsw.is_empty() {
        let level = if only_in_hnsw.len() >= ONE_SIDED_WARN_THRESHOLD {
            "WARNING"
        } else {
            "NOTICE"
        };
        let msg = format!(
            "{} sampled IDs in HNSW but not GPU (deletion after genesis?): {:?}",
            only_in_hnsw.len(),
            &only_in_hnsw[..only_in_hnsw.len().min(10)],
        );
        rpt!(rpt, "  [{level}] {msg}");
        one_sided_warnings.push(msg);
    }
    if !only_in_gpu.is_empty() {
        let level = if only_in_gpu.len() >= ONE_SIDED_WARN_THRESHOLD {
            "WARNING"
        } else {
            "NOTICE"
        };
        let msg = format!(
            "{} sampled IDs in GPU but not HNSW (identity deletion set?): {:?}",
            only_in_gpu.len(),
            &only_in_gpu[..only_in_gpu.len().min(10)],
        );
        rpt!(rpt, "  [{level}] {msg}");
        one_sided_warnings.push(msg);
    }

    // Step 4: Load serial IDs of pending iris-code-updating modifications
    // (id > last_indexed_mod_id).  These may diverge between schemas legitimately.
    // The modifications table lives in the GPU schema.
    let update_types: Vec<String> = IRIS_UPDATE_TYPES.iter().map(|s| s.to_string()).collect();
    let pending: Vec<(i64,)> = sqlx::query_as(
        "SELECT DISTINCT serial_id FROM modifications \
         WHERE id > $1 AND request_type = ANY($2) AND serial_id IS NOT NULL",
    )
    .bind(after_mod_id)
    .bind(&update_types)
    .fetch_all(gpu_pool)
    .await?;
    let pending_set: HashSet<i64> = pending.into_iter().map(|(sid,)| sid).collect();

    // Step 5: Filter out legitimately-diverging IDs.
    // One-sided IDs are not counted as mismatches (logged as warnings above).
    let raw_mismatch_count = mismatched_raw.len();
    let mismatched_ids: Vec<i64> = mismatched_raw
        .into_iter()
        .filter(|id| !pending_set.contains(id))
        .collect();
    let excluded_count = raw_mismatch_count - mismatched_ids.len();

    let one_sided_total = only_in_hnsw.len() + only_in_gpu.len();
    checks.push(
        CheckResult::new(
            "3c",
            "Byte-identical shares (sampled)",
            mismatched_ids.is_empty(),
            if mismatched_ids.is_empty() {
                format!(
                    "{} IDs sampled, {} pending-update IDs identified, \
                     {} excluded, {} one-sided IDs (not counted as mismatches), \
                     0 mismatches",
                    sample_vec.len(),
                    pending_set.len(),
                    excluded_count,
                    one_sided_total,
                )
            } else {
                format!(
                    "{} IDs sampled, {} pending-update IDs identified, \
                     {} excluded, {} one-sided IDs, {} mismatches (first 10): {:?}",
                    sample_vec.len(),
                    pending_set.len(),
                    excluded_count,
                    one_sided_total,
                    mismatched_ids.len(),
                    &mismatched_ids[..mismatched_ids.len().min(10)],
                )
            },
        )
        .with_warnings(one_sided_warnings),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

fn write_json_reports(
    dir: &Path,
    checks: &[CheckResult],
    stats: &Stats,
    hist: &[DegreeHistEntry],
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(dir)?;
    let mut files = Vec::new();

    let p = dir.join("checks.json");
    fs::write(&p, serde_json::to_string_pretty(checks)?)?;
    println!("Wrote {}", p.display());
    files.push(p);

    let p = dir.join("stats.json");
    let map: serde_json::Map<String, serde_json::Value> = stats
        .0
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    fs::write(
        &p,
        serde_json::to_string_pretty(&serde_json::Value::Object(map))?,
    )?;
    println!("Wrote {}", p.display());
    files.push(p);

    let p = dir.join("degree_histogram.csv");
    {
        let mut wtr = csv::Writer::from_path(&p)?;
        for entry in hist {
            wtr.serialize(entry)?;
        }
        wtr.flush()?;
    }
    println!("Wrote {}", p.display());
    files.push(p);

    Ok(files)
}

// ---------------------------------------------------------------------------
// S3 upload
// ---------------------------------------------------------------------------

/// Parse an S3 URI like `s3://bucket/prefix/` into (bucket, key_prefix).
fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let stripped = uri
        .strip_prefix("s3://")
        .ok_or_else(|| eyre::eyre!("S3 URI must start with s3://"))?;
    let (bucket, prefix) = stripped.split_once('/').unwrap_or((stripped, ""));
    eyre::ensure!(!bucket.is_empty(), "S3 URI has empty bucket name");
    Ok((bucket.to_string(), prefix.to_string()))
}

fn build_s3_client(config: &aws_config::SdkConfig, force_path_style: bool) -> aws_sdk_s3::Client {
    let retry_config = aws_config::retry::RetryConfig::standard().with_max_attempts(5);
    let s3_config = aws_sdk_s3::config::Builder::from(config)
        .force_path_style(force_path_style)
        .retry_config(retry_config)
        .build();
    aws_sdk_s3::Client::from_conf(s3_config)
}

/// Build an S3 client for the graph-checkpoint bucket, allowing the region to
/// differ from the ambient default (genesis writes checkpoints into a bucket
/// that may live in a different region from the iris/exclusions buckets).
async fn build_checkpoint_s3_client(region: &str, force_path_style: bool) -> aws_sdk_s3::Client {
    let loader = aws_config::from_env().region(aws_sdk_s3::config::Region::new(region.to_owned()));
    let config = loader.load().await;
    build_s3_client(&config, force_path_style)
}

async fn download_exclusions_from_s3(
    s3_uri: &str,
    force_path_style: bool,
) -> Result<ExclusionsFile> {
    let (bucket, key) = parse_s3_uri(s3_uri)?;
    eyre::ensure!(!key.is_empty(), "S3 URI must include an object key");
    println!("Downloading exclusions from s3://{bucket}/{key}");

    let config = aws_config::from_env().load().await;
    let client = build_s3_client(&config, force_path_style);

    let response = client.get_object().bucket(&bucket).key(&key).send().await?;
    let body = response.body.collect().await?;
    let exclusions: ExclusionsFile = serde_json::from_slice(&body.into_bytes())?;

    println!(
        "  Loaded {} excluded serial IDs",
        exclusions.deleted_serial_ids.len()
    );
    Ok(exclusions)
}

async fn upload_to_s3(s3_uri: &str, files: &[PathBuf], force_path_style: bool) -> Result<()> {
    let (bucket, prefix) = parse_s3_uri(s3_uri)?;
    println!("--- Uploading to S3: s3://{bucket}/{prefix} ---");

    let config = aws_config::from_env().load().await;
    let client = build_s3_client(&config, force_path_style);

    for path in files {
        let file_name = path
            .file_name()
            .ok_or_else(|| eyre::eyre!("no filename for {}", path.display()))?
            .to_string_lossy();
        let key = if prefix.is_empty() {
            file_name.to_string()
        } else {
            let trimmed = prefix.trim_end_matches('/');
            format!("{trimmed}/{file_name}")
        };

        let content_type = if file_name.ends_with(".json") {
            "application/json"
        } else {
            "text/plain"
        };
        let body = aws_sdk_s3::primitives::ByteStream::from_path(path).await?;
        client
            .put_object()
            .bucket(&bucket)
            .key(&key)
            .body(body)
            .content_type(content_type)
            .send()
            .await?;
        println!("  Uploaded s3://{bucket}/{key}");
    }

    Ok(())
}
