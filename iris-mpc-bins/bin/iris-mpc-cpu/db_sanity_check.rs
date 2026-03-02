use clap::Parser;
use eyre::{eyre, Result};
use futures::StreamExt;
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
    let gpu_pg = PostgresClient::new(&args.db_url, &args.gpu_schema, AccessMode::ReadOnly).await?;
    let hnsw_store = Store::new(&hnsw_pg).await?;
    let gpu_store = Store::new(&gpu_pg).await?;
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
    run_graph_checks(
        &graph_pg,
        &iris_ids,
        &exclusions,
        args.m,
        &mut checks,
        &mut degree_hist,
        &mut stats,
    )
    .await?;

    println!("--- Check 2: Persistent state consistency ---");
    let iris_max = hnsw_store.get_max_serial_id().await?;
    run_persistent_state_checks(&graph_pg, iris_max, &mut checks).await?;

    println!("--- Check 3: HNSW vs GPU iris consistency ---");
    run_cross_schema_checks(&hnsw_store, &gpu_store, &mut checks).await?;

    println!("--- Check 4: Modifications table ---");
    run_modification_checks(&hnsw_store, &mut checks, &mut stats).await?;

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
    let mut seen_ids: HashSet<i64> = HashSet::new();
    let mut total = 0u64;

    let mut stream = store.stream_irises().await;
    while let Some(result) = stream.next().await {
        let iris = result.map_err(|e| eyre!("Error streaming iris: {e}"))?;
        total += 1;
        seen_ids.insert(iris.serial_id() as i64);
        if total % 500_000 == 0 {
            println!("  ... streamed {total} irises");
        }
    }

    let iris_count = store.count_irises().await?;
    stats.add("Total iris count (HNSW)", iris_count.to_string());
    stats.add("Max serial ID (HNSW)", store.get_max_serial_id().await?.to_string());

    Ok(seen_ids)
}

// ---------------------------------------------------------------------------
// Check 1: HNSW graph structural checks
// ---------------------------------------------------------------------------

async fn run_graph_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_ids: &HashSet<i64>,
    exclusions: &Option<HashSet<u32>>,
    m: usize,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    stats: &mut Stats,
) -> Result<()> {
    let params = HnswParams::new(1, 1, m);

    println!("  Loading graphs...");
    let graphs = [
        ("left", load_graph(graph_pg, StoreId::Left).await?),
        ("right", load_graph(graph_pg, StoreId::Right).await?),
    ];

    for (eye, graph) in &graphs {
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

        // -- 1f: Degree bounds --
        let mut degree_viol = 0u64;
        for (lc, layer) in graph.layers.iter().enumerate() {
            let m_limit = params.M_limit[lc.min(N_PARAM_LAYERS - 1)];
            for (node, nbs) in layer.links.iter() {
                if nbs.len() > m_limit {
                    degree_viol += 1;
                    if degree_viol <= 5 {
                        println!(
                            "  [1f] {eye} L{lc} node {node} degree {} > M_limit {m_limit}",
                            nbs.len()
                        );
                    }
                }
            }
        }
        checks.push(CheckResult::new(
            "1f",
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

        // -- 1g: Entry point validity --
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
            "1g",
            &format!("Entry point validity ({eye})"),
            ep_valid,
            if ep_valid {
                format!("{} entry points valid", graph.entry_points.len())
            } else {
                ep_detail
            },
        ));
    }

    // -- 1h: Left/Right graph sync --
    let l0_ids = |g: &GraphMem<VectorId>| -> HashSet<u32> {
        g.layers
            .first()
            .map(|l| l.links.keys().map(|v| v.serial_id()).collect())
            .unwrap_or_default()
    };
    let (left_ids, right_ids) = (l0_ids(&graphs[0].1), l0_ids(&graphs[1].1));
    checks.push(CheckResult::new(
        "1h",
        "Left/Right graph sync",
        left_ids == right_ids,
        if left_ids == right_ids {
            format!("{} serial IDs match at layer 0", left_ids.len())
        } else {
            format!(
                "Mismatch: {} only in left, {} only in right",
                left_ids.difference(&right_ids).count(),
                right_ids.difference(&left_ids).count()
            )
        },
    ));

    Ok(())
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
) -> Result<()> {
    let last_indexed: Option<u32> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
        .await?;
    let left_max = get_graph_max_serial_id(graph_pg, StoreId::Left).await?;
    let right_max = get_graph_max_serial_id(graph_pg, StoreId::Right).await?;

    // 3a
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

    // 3b
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

async fn run_cross_schema_checks(
    hnsw_store: &Store,
    gpu_store: &Store,
    checks: &mut Vec<CheckResult>,
) -> Result<()> {
    // 4a
    let hnsw_count = hnsw_store.count_irises().await?;
    let gpu_count = gpu_store.count_irises().await?;
    checks.push(CheckResult::new(
        "3a",
        "Same row count",
        hnsw_count == gpu_count,
        if hnsw_count == gpu_count {
            format!("Both schemas have {hnsw_count} irises")
        } else {
            format!("HNSW={hnsw_count}, GPU={gpu_count}")
        },
    ));

    // 4b
    let hnsw_max = hnsw_store.get_max_serial_id().await?;
    let gpu_max = gpu_store.get_max_serial_id().await?;
    checks.push(CheckResult::new(
        "3b",
        "Same max serial ID",
        hnsw_max == gpu_max,
        if hnsw_max == gpu_max {
            format!("Both schemas max_id={hnsw_max}")
        } else {
            format!("HNSW max={hnsw_max}, GPU max={gpu_max}")
        },
    ));

    // 4c: Byte-identical shares
    println!("  Comparing iris shares between schemas...");
    let mut hnsw_stream = hnsw_store.stream_irises().await;
    let mut gpu_stream = gpu_store.stream_irises().await;
    let (mut mismatches, mut compared) = (0u64, 0u64);
    let mut mismatch_details: Vec<String> = Vec::new();

    loop {
        match (hnsw_stream.next().await, gpu_stream.next().await) {
            (Some(Ok(h)), Some(Ok(g))) => {
                compared += 1;
                let (h_id, g_id) = (h.serial_id() as i64, g.serial_id() as i64);
                if h_id != g_id {
                    mismatches += 1;
                    if mismatch_details.len() < 10 {
                        mismatch_details.push(format!("ID mismatch: HNSW={h_id}, GPU={g_id}"));
                    }
                    break;
                }
                if h.left_code() != g.left_code()
                    || h.left_mask() != g.left_mask()
                    || h.right_code() != g.right_code()
                    || h.right_mask() != g.right_mask()
                {
                    mismatches += 1;
                    if mismatch_details.len() < 10 {
                        mismatch_details.push(format!("Serial ID {h_id}: share data differs"));
                    }
                }
                if compared % 500_000 == 0 {
                    println!("  ... compared {compared} irises");
                }
            }
            (None, None) => break,
            (Some(_), None) => {
                mismatches += 1;
                mismatch_details.push("HNSW has more rows than GPU".into());
                break;
            }
            (None, Some(_)) => {
                mismatches += 1;
                mismatch_details.push("GPU has more rows than HNSW".into());
                break;
            }
            (Some(Err(e)), _) | (_, Some(Err(e))) => {
                return Err(eyre!("Error in cross-schema compare: {e}"));
            }
        }
    }

    checks.push(CheckResult::new(
        "3c",
        "Byte-identical shares",
        mismatches == 0,
        if mismatches == 0 {
            format!("{compared} irises compared")
        } else {
            format!(
                "{mismatches} mismatches (compared {compared}): {}",
                mismatch_details.join("; ")
            )
        },
    ));

    Ok(())
}

// ---------------------------------------------------------------------------
// Check 4: Modifications table
// ---------------------------------------------------------------------------

async fn run_modification_checks(
    store: &Store,
    checks: &mut Vec<CheckResult>,
    stats: &mut Stats,
) -> Result<()> {
    let rows: Vec<(String, bool, i64)> = sqlx::query_as(
        "SELECT status, persisted, COUNT(*) FROM modifications \
         GROUP BY status, persisted ORDER BY status, persisted",
    )
    .fetch_all(&store.pool)
    .await?;

    let (mut total, mut bad) = (0i64, 0i64);
    for (status, persisted, count) in &rows {
        stats.add(
            format!("Modifications status={status} persisted={persisted}"),
            count.to_string(),
        );
        total += count;
        if status != "COMPLETED" || !persisted {
            bad += count;
        }
    }
    stats.add("Total modifications", total.to_string());

    let type_rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT request_type, COUNT(*) FROM modifications \
         GROUP BY request_type ORDER BY request_type",
    )
    .fetch_all(&store.pool)
    .await?;
    for (t, c) in &type_rows {
        stats.add(format!("Modifications type={t}"), c.to_string());
    }

    checks.push(CheckResult::new(
        "4a",
        "Completed & persisted",
        bad == 0,
        if bad == 0 {
            format!("All {total} modifications are COMPLETED and persisted")
        } else {
            format!("{bad}/{total} modifications not COMPLETED+persisted")
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
