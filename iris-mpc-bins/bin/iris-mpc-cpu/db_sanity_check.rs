use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    postgres::{AccessMode, PostgresClient},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::StoreId,
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::{graph_store::GraphPg, layered_graph::GraphMem},
        searcher::{HnswParams, N_PARAM_LAYERS},
    },
};
use iris_mpc_store::Store;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write as FmtWrite,
    fs,
    path::{Path, PathBuf},
    process,
};

const STATE_DOMAIN: &str = "genesis";
const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";
const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";

#[derive(Parser)]
#[command(
    name = "db-sanity-check",
    about = "Validate DB state for a single MPC party"
)]
struct Args {
    /// Postgres connection string
    #[arg(long, env = "DATABASE_URL")]
    db_url: String,
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
    /// Path to JSON exclusions file with {"deleted_serial_ids": [...]}
    #[arg(long)]
    exclusions_file: Option<PathBuf>,
    /// Directory for JSON output files
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,
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
        }
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

    println!("=== DB Sanity Check ===");
    println!(
        "HNSW schema: {}  GPU schema: {}  M: {}",
        args.hnsw_schema, args.gpu_schema, args.m
    );
    println!();

    let hnsw_pg =
        PostgresClient::new(&args.db_url, &args.hnsw_schema, AccessMode::ReadOnly).await?;
    let hnsw_store = Store::new(&hnsw_pg).await?;
    let graph_pg = GraphPg::<Aby3Store>::new(&hnsw_pg).await?;

    let mut checks: Vec<CheckResult> = Vec::new();
    let mut stats = Stats::new();
    let mut degree_hist: Vec<DegreeHistEntry> = Vec::new();

    let exclusions: Option<HashSet<u32>> = match &args.exclusions_file {
        Some(path) => {
            let parsed: ExclusionsFile = serde_json::from_str(&fs::read_to_string(path)?)?;
            Some(parsed.deleted_serial_ids.into_iter().collect())
        }
        None => None,
    };

    println!("--- Collecting iris IDs ---");
    let iris_ids = collect_iris_ids(&hnsw_store, &mut stats).await?;

    println!("--- Check 1: HNSW graph structural checks ---");
    let layer_probability = args.layer_probability.unwrap_or((args.m as f64).recip());
    run_graph_checks(
        &graph_pg,
        &iris_ids,
        &exclusions,
        args.m,
        layer_probability,
        &mut checks,
        &mut degree_hist,
        &mut stats,
    )
    .await?;

    println!("--- Check 2: Persistent state consistency ---");
    let iris_max = hnsw_store.get_max_serial_id().await?;
    run_persistent_state_checks(&graph_pg, iris_max, &mut checks, &mut stats).await?;

    println!("--- Check 3: HNSW vs GPU iris consistency (up to last_indexed_iris_id) ---");
    run_cross_schema_checks(
        &args.hnsw_schema,
        &args.gpu_schema,
        iris_max,
        &hnsw_store.pool,
        &mut checks,
    )
    .await?;

    // --- Report ---
    println!("\n--- Checks ---");
    let pass_count = checks.iter().filter(|c| c.passed).count();
    for c in &checks {
        let tag = if c.passed { "PASS" } else { "FAIL" };
        println!("[{tag}] {}: {} ({})", c.id, c.name, c.detail);
    }
    println!("\n--- Stats ---");
    for (k, v) in &stats.0 {
        println!("{k}: {v}");
    }
    let fail_count = checks.len() - pass_count;
    println!(
        "\n=== Summary: {pass_count}/{} checks passed, {fail_count} failed ===",
        checks.len()
    );

    write_json_reports(&args.output_dir, &checks, &stats, &degree_hist)?;
    if fail_count > 0 {
        process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Collect iris serial IDs (needed for graph orphan / coverage checks)
// ---------------------------------------------------------------------------

async fn collect_iris_ids(store: &Store, stats: &mut Stats) -> Result<HashSet<i64>> {
    let ids: Vec<(i64,)> = sqlx::query_as("SELECT id FROM irises ORDER BY id")
        .fetch_all(&store.pool)
        .await?;

    let max_id = ids.last().map(|(id,)| *id).unwrap_or(0);
    stats.add("Total iris count (HNSW)", ids.len().to_string());
    stats.add("Max serial ID (HNSW)", max_id.to_string());

    Ok(ids.into_iter().map(|(id,)| id).collect())
}

// ---------------------------------------------------------------------------
// Check 1: HNSW graph structural checks
// ---------------------------------------------------------------------------

async fn run_graph_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_ids: &HashSet<i64>,
    exclusions: &Option<HashSet<u32>>,
    m: usize,
    layer_probability: f64,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    stats: &mut Stats,
) -> Result<()> {
    // Load one graph at a time to halve peak memory.
    let mut l0_id_sets: Vec<(&str, HashSet<u32>)> = Vec::new();
    for (eye, store_id) in [("left", StoreId::Left), ("right", StoreId::Right)] {
        println!("  Loading {eye} graph...");
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
        );
        drop(graph);
        l0_id_sets.push((eye, l0_ids));
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
        let m_limit = params.M_limit[lc.min(N_PARAM_LAYERS - 1)];
        for (node, nbs) in layer.links.iter() {
            if nbs.len() > m_limit {
                degree_viol += 1;
                if degree_viol <= 5 {
                    println!(
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
                params.M_limit[0],
                params.M_limit[1.min(N_PARAM_LAYERS - 1)]
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
    Ok(ops.load_to_mem(graph_pg.pool(), 4).await?)
}

// ---------------------------------------------------------------------------
// Check 2: Persistent state consistency
// ---------------------------------------------------------------------------

async fn run_persistent_state_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_max_serial_id: usize,
    checks: &mut Vec<CheckResult>,
    stats: &mut Stats,
) -> Result<()> {
    let last_indexed: Option<u32> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
        .await?;
    let left_max = get_graph_max_serial_id(graph_pg, StoreId::Left).await?;
    let right_max = get_graph_max_serial_id(graph_pg, StoreId::Right).await?;

    // 2a
    match last_indexed {
        Some(last_id) => {
            let mut issues = Vec::new();
            if left_max != right_max {
                issues.push(format!("left max={left_max} != right max={right_max}"));
            }
            if last_id as i64 != left_max {
                issues.push(format!("last_indexed={last_id} != left max={left_max}"));
            }
            if last_id as usize != iris_max_serial_id {
                issues.push(format!(
                    "last_indexed={last_id} != irises max={iris_max_serial_id}"
                ));
            }
            checks.push(CheckResult::new(
                "2a",
                "last_indexed_iris_id",
                issues.is_empty(),
                if issues.is_empty() {
                    format!("Consistent: {last_id}")
                } else {
                    issues.join("; ")
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

    // 2b
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

    Ok(())
}

async fn get_graph_max_serial_id(graph_pg: &GraphPg<Aby3Store>, store_id: StoreId) -> Result<i64> {
    let mut tx = graph_pg.tx().await?;
    let mut ops = tx.with_graph(store_id);
    ops.get_max_serial_id().await
}

// ---------------------------------------------------------------------------
// Check 3: Cross-schema consistency (HNSW vs GPU)
// ---------------------------------------------------------------------------

fn validate_schema_name(name: &str) -> Result<()> {
    eyre::ensure!(
        !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'),
        "invalid schema name: {name}"
    );
    Ok(())
}

async fn run_cross_schema_checks(
    hnsw_schema: &str,
    gpu_schema: &str,
    last_indexed_id: usize,
    pool: &sqlx::PgPool,
    checks: &mut Vec<CheckResult>,
) -> Result<()> {
    validate_schema_name(hnsw_schema)?;
    validate_schema_name(gpu_schema)?;
    let lid = last_indexed_id as i64;

    // 3a: Same row count (up to last_indexed_id)
    let count_query = format!(
        r#"SELECT
             (SELECT COUNT(*) FROM "{}".irises WHERE id <= $1) AS hnsw_count,
             (SELECT COUNT(*) FROM "{}".irises WHERE id <= $1) AS gpu_count"#,
        hnsw_schema, gpu_schema
    );
    let (hnsw_count, gpu_count): (i64, i64) = sqlx::query_as(&count_query)
        .bind(lid)
        .fetch_one(pool)
        .await?;
    checks.push(CheckResult::new(
        "3a",
        "Same row count",
        hnsw_count == gpu_count,
        if hnsw_count == gpu_count {
            format!("Both schemas have {hnsw_count} irises (id <= {last_indexed_id})")
        } else {
            format!("HNSW={hnsw_count}, GPU={gpu_count} (id <= {last_indexed_id})")
        },
    ));

    // 3b: Same max serial ID (up to last_indexed_id)
    let max_id_query = format!(
        r#"SELECT
             (SELECT COALESCE(MAX(id), 0) FROM "{}".irises WHERE id <= $1) AS hnsw_max,
             (SELECT COALESCE(MAX(id), 0) FROM "{}".irises WHERE id <= $1) AS gpu_max"#,
        hnsw_schema, gpu_schema
    );
    let (hnsw_max, gpu_max): (i64, i64) = sqlx::query_as(&max_id_query)
        .bind(lid)
        .fetch_one(pool)
        .await?;
    checks.push(CheckResult::new(
        "3b",
        "Same max serial ID",
        hnsw_max == gpu_max,
        if hnsw_max == gpu_max {
            format!("Both schemas max_id={hnsw_max} (id <= {last_indexed_id})")
        } else {
            format!("HNSW max={hnsw_max}, GPU max={gpu_max} (id <= {last_indexed_id})")
        },
    ));

    // 3c: Byte-identical shares (up to last_indexed_id)
    // TODO: For large databases, consider replacing this full JOIN
    // with random sampling to avoid scanning all BYTEA data.
    println!("  Comparing iris shares between schemas (SQL JOIN, id <= {last_indexed_id})...");
    let mismatch_query = format!(
        r#"SELECT h.id FROM "{}".irises h
           JOIN "{}".irises g ON h.id = g.id
           WHERE h.id <= $1
             AND (h.left_code != g.left_code
              OR h.left_mask != g.left_mask
              OR h.right_code != g.right_code
              OR h.right_mask != g.right_mask)
           LIMIT 10"#,
        hnsw_schema, gpu_schema
    );
    let mismatched_ids: Vec<(i64,)> = sqlx::query_as(&mismatch_query)
        .bind(lid)
        .fetch_all(pool)
        .await?;

    // Informational only — mismatches are expected if modifications landed after
    // genesis processed them.  Always passes; detail reports any divergence.
    checks.push(CheckResult::new(
        "3c",
        "Byte-identical shares",
        true,
        if mismatched_ids.is_empty() {
            format!("{hnsw_count} irises compared, 0 mismatches (id <= {last_indexed_id})")
        } else {
            let ids: Vec<i64> = mismatched_ids.into_iter().map(|(id,)| id).collect();
            format!(
                "{hnsw_count} irises compared, mismatched IDs (first 10): {:?} (id <= {last_indexed_id})",
                ids
            )
        },
    ));

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
) -> Result<()> {
    fs::create_dir_all(dir)?;

    let p = dir.join("checks.json");
    fs::write(&p, serde_json::to_string_pretty(checks)?)?;
    println!("Wrote {}", p.display());

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

    let p = dir.join("degree_histogram.json");
    fs::write(&p, serde_json::to_string_pretty(hist)?)?;
    println!("Wrote {}", p.display());

    Ok(())
}
