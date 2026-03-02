use clap::Parser;
use eyre::{eyre, Result};
use futures::StreamExt;
use iris_mpc_common::{
    postgres::{AccessMode, PostgresClient},
    vector_id::VectorId,
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
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
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as FmtWrite,
    fs,
    path::{Path, PathBuf},
    process,
};

// Persistent state domain/key constants (matching state_accessor.rs)
const STATE_DOMAIN: &str = "genesis";
const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";

#[derive(Parser)]
#[command(name = "db-sanity-check")]
#[command(about = "Validate DB state sanity for a single MPC party after genesis")]
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

    /// Directory for CSV output files
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,
}

#[derive(Deserialize)]
struct ExclusionsFile {
    deleted_serial_ids: Vec<u32>,
}

#[derive(Clone)]
struct CheckResult {
    id: String,
    name: String,
    passed: bool,
    detail: String,
}

impl CheckResult {
    fn pass(id: &str, name: &str, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            passed: true,
            detail: detail.into(),
        }
    }

    fn fail(id: &str, name: &str, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            passed: false,
            detail: detail.into(),
        }
    }
}

struct Stats {
    entries: Vec<(String, String)>,
}

impl Stats {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn add(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.push((key.into(), value.into()));
    }
}

struct DegreeHistEntry {
    eye: String,
    layer: usize,
    degree: usize,
    node_count: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("=== DB Sanity Check ===");
    println!("HNSW schema: {}", args.hnsw_schema);
    println!("GPU schema:  {}", args.gpu_schema);
    println!("M: {}", args.m);
    println!();

    // Connect to both schemas
    let hnsw_pg = PostgresClient::new(&args.db_url, &args.hnsw_schema, AccessMode::ReadOnly).await?;
    let gpu_pg = PostgresClient::new(&args.db_url, &args.gpu_schema, AccessMode::ReadOnly).await?;

    let hnsw_store = Store::new(&hnsw_pg).await?;
    let gpu_store = Store::new(&gpu_pg).await?;
    let graph_pg = GraphPg::<Aby3Store>::new(&hnsw_pg).await?;

    let mut all_checks: Vec<CheckResult> = Vec::new();
    let mut stats = Stats::new();
    let mut degree_hist: Vec<DegreeHistEntry> = Vec::new();

    // Load exclusions if provided
    let exclusions: Option<HashSet<u32>> = if let Some(ref path) = args.exclusions_file {
        let data = fs::read_to_string(path)?;
        let parsed: ExclusionsFile = serde_json::from_str(&data)?;
        Some(parsed.deleted_serial_ids.into_iter().collect())
    } else {
        None
    };

    // --- Check 1: Iris table integrity ---
    println!("--- Check 1: Iris table integrity ---");
    let (iris_checks, iris_ids, version_hist) =
        run_iris_checks(&hnsw_store).await?;
    all_checks.extend(iris_checks);

    // Iris stats
    let iris_count = hnsw_store.count_irises().await?;
    let max_serial_id = hnsw_store.get_max_serial_id().await?;
    stats.add("Total iris count (HNSW)", iris_count.to_string());
    stats.add("Max serial ID (HNSW irises)", max_serial_id.to_string());

    let missing_count = if max_serial_id > 0 {
        max_serial_id - iris_count
    } else {
        0
    };
    stats.add("Missing iris IDs count", missing_count.to_string());

    // Version distribution
    let mut version_entries: Vec<_> = version_hist.iter().collect();
    version_entries.sort_by_key(|(k, _)| **k);
    for (vid, count) in &version_entries {
        stats.add(format!("Version {} count", vid), count.to_string());
    }

    // --- Check 2: HNSW graph structural checks ---
    println!("--- Check 2: HNSW graph structural checks ---");
    let (graph_checks, graph_degree_hist, graph_stats) =
        run_graph_checks(&graph_pg, &iris_ids, &exclusions, args.m).await?;
    all_checks.extend(graph_checks);
    degree_hist.extend(graph_degree_hist);
    for (k, v) in graph_stats {
        stats.add(k, v);
    }

    // --- Check 3: Persistent state consistency ---
    println!("--- Check 3: Persistent state consistency ---");
    let persistent_checks =
        run_persistent_state_checks(&graph_pg, max_serial_id).await?;
    all_checks.extend(persistent_checks);

    // Dump all persistent state rows
    dump_persistent_state(&graph_pg, &mut stats).await?;

    // --- Check 4: HNSW vs GPU iris consistency ---
    println!("--- Check 4: HNSW vs GPU iris consistency ---");
    let cross_checks =
        run_cross_schema_checks(&hnsw_store, &gpu_store).await?;
    all_checks.extend(cross_checks);

    // --- Check 5: Modifications table ---
    println!("--- Check 5: Modifications table ---");
    let mod_checks = run_modification_checks(&hnsw_store, &mut stats).await?;
    all_checks.extend(mod_checks);

    // --- Print report ---
    println!();
    println!("--- Checks ---");
    let mut pass_count = 0;
    let total_count = all_checks.len();
    for check in &all_checks {
        let status = if check.passed {
            pass_count += 1;
            "PASS"
        } else {
            "FAIL"
        };
        println!("[{}] {}: {} ({})", status, check.id, check.name, check.detail);
    }

    println!();
    println!("--- Stats ---");
    for (key, value) in &stats.entries {
        println!("{}: {}", key, value);
    }

    let fail_count = total_count - pass_count;
    println!();
    println!(
        "=== Summary: {}/{} checks passed, {} failed ===",
        pass_count, total_count, fail_count
    );

    // --- Write CSV reports ---
    write_csv_reports(&args.output_dir, &all_checks, &stats, &degree_hist)?;

    if fail_count > 0 {
        process::exit(1);
    }

    Ok(())
}

/// Check 1: Stream HNSW irises and validate integrity
async fn run_iris_checks(
    store: &Store,
) -> Result<(Vec<CheckResult>, HashSet<i64>, HashMap<i16, usize>)> {
    let mut checks = Vec::new();
    let mut seen_ids: HashSet<i64> = HashSet::new();
    let mut version_hist: HashMap<i16, usize> = HashMap::new();
    let mut null_or_empty_count = 0u64;
    let mut bad_size_count = 0u64;
    let mut bad_version_count = 0u64;
    let mut total_count = 0u64;
    let mut max_id: i64 = 0;

    let mut stream = store.stream_irises().await;

    while let Some(result) = stream.next().await {
        let iris = result.map_err(|e| eyre!("Error streaming iris: {e}"))?;
        total_count += 1;
        let id = iris.serial_id() as i64;
        seen_ids.insert(id);
        if id > max_id {
            max_id = id;
        }

        *version_hist.entry(iris.version_id()).or_insert(0) += 1;

        // 1a: No NULL/empty shares
        let lc = iris.left_code();
        let lm = iris.left_mask();
        let rc = iris.right_code();
        let rm = iris.right_mask();
        if lc.is_empty() || lm.is_empty() || rc.is_empty() || rm.is_empty() {
            null_or_empty_count += 1;
        }

        // 1b: Correct byte sizes (the accessors cast u8->u16, so check u16 lengths)
        let lc_ok = lc.len() == IRIS_CODE_LENGTH;
        let lm_ok = lm.len() == MASK_CODE_LENGTH;
        let rc_ok = rc.len() == IRIS_CODE_LENGTH;
        let rm_ok = rm.len() == MASK_CODE_LENGTH;
        if !lc_ok || !lm_ok || !rc_ok || !rm_ok {
            bad_size_count += 1;
        }

        // 1c: Version ID sanity
        if iris.version_id() < 0 {
            bad_version_count += 1;
        }

        if total_count % 500_000 == 0 {
            println!("  ... streamed {} irises", total_count);
        }
    }

    // 1a
    if null_or_empty_count == 0 {
        checks.push(CheckResult::pass(
            "1a",
            "No NULL shares",
            format!("{total_count} irises checked"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "1a",
            "No NULL shares",
            format!("{null_or_empty_count} irises have NULL/empty shares"),
        ));
    }

    // 1b
    if bad_size_count == 0 {
        checks.push(CheckResult::pass(
            "1b",
            "Correct byte sizes",
            format!(
                "All irises have code={} u16s, mask={} u16s",
                IRIS_CODE_LENGTH, MASK_CODE_LENGTH
            ),
        ));
    } else {
        checks.push(CheckResult::fail(
            "1b",
            "Correct byte sizes",
            format!("{bad_size_count} irises have incorrect byte sizes"),
        ));
    }

    // 1c
    if bad_version_count == 0 {
        checks.push(CheckResult::pass(
            "1c",
            "Version ID sanity",
            format!("All version_ids >= 0"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "1c",
            "Version ID sanity",
            format!("{bad_version_count} irises have negative version_id"),
        ));
    }

    // 1d: Contiguous IDs
    if max_id > 0 {
        let mut missing: Vec<i64> = Vec::new();
        for expected in 1..=max_id {
            if !seen_ids.contains(&expected) {
                missing.push(expected);
            }
        }

        if missing.is_empty() {
            checks.push(CheckResult::pass(
                "1d",
                "Contiguous IDs",
                format!("All IDs 1..={max_id} present"),
            ));
        } else {
            let detail = if missing.len() <= 20 {
                format!("{} missing IDs: {:?}", missing.len(), missing)
            } else {
                format!(
                    "{} missing IDs (first 20): {:?}",
                    missing.len(),
                    &missing[..20]
                )
            };
            checks.push(CheckResult::fail("1d", "Contiguous IDs", detail));
        }
    } else {
        checks.push(CheckResult::pass(
            "1d",
            "Contiguous IDs",
            "No irises in table".to_string(),
        ));
    }

    Ok((checks, seen_ids, version_hist))
}

/// Check 2: Load graphs and validate structure
async fn run_graph_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_ids: &HashSet<i64>,
    exclusions: &Option<HashSet<u32>>,
    m: usize,
) -> Result<(Vec<CheckResult>, Vec<DegreeHistEntry>, Vec<(String, String)>)> {
    let mut checks = Vec::new();
    let mut degree_hist = Vec::new();
    let mut graph_stats = Vec::new();

    let params = HnswParams::new(1, 1, m);

    // Load left and right graphs
    println!("  Loading left graph...");
    let left_graph = load_graph(graph_pg, StoreId::Left).await?;
    println!("  Loading right graph...");
    let right_graph = load_graph(graph_pg, StoreId::Right).await?;

    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        // Graph stats
        graph_stats.push((
            format!("{eye_name} graph checksum"),
            graph.checksum().to_string(),
        ));
        graph_stats.push((
            format!("{eye_name} graph num_layers"),
            graph.num_layers().to_string(),
        ));

        for (lc, layer) in graph.layers.iter().enumerate() {
            graph_stats.push((
                format!("{eye_name} layer {lc} node count"),
                layer.links.len().to_string(),
            ));
        }

        // Entry points
        let ep_desc: String = graph
            .entry_points
            .iter()
            .map(|ep| format!("{}@L{}", ep.point, ep.layer))
            .collect::<Vec<_>>()
            .join(", ");
        graph_stats.push((format!("{eye_name} entry points"), ep_desc));

        // Degree stats per layer
        for (lc, layer) in graph.layers.iter().enumerate() {
            let mut deg_counts: BTreeMap<usize, usize> = BTreeMap::new();
            for neighbors in layer.links.values() {
                *deg_counts.entry(neighbors.len()).or_insert(0) += 1;
            }

            for (&degree, &count) in &deg_counts {
                degree_hist.push(DegreeHistEntry {
                    eye: eye_name.to_string(),
                    layer: lc,
                    degree,
                    node_count: count,
                });
            }

            if !layer.links.is_empty() {
                let degrees: Vec<usize> = layer.links.values().map(|n| n.len()).collect();
                let min = *degrees.iter().min().unwrap();
                let max = *degrees.iter().max().unwrap();
                let sum: usize = degrees.iter().sum();
                let avg = sum as f64 / degrees.len() as f64;
                let mut sorted = degrees.clone();
                sorted.sort();
                let median = sorted[sorted.len() / 2];

                graph_stats.push((
                    format!("{eye_name} layer {lc} degree min/avg/median/max"),
                    format!("{min}/{avg:.1}/{median}/{max}"),
                ));
            }
        }
    }

    // 2a: No orphan graph nodes
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut orphan_count = 0u64;
        for layer in &graph.layers {
            for node in layer.links.keys() {
                if !iris_ids.contains(&(node.serial_id() as i64)) {
                    orphan_count += 1;
                }
            }
        }
        if orphan_count == 0 {
            checks.push(CheckResult::pass(
                "2a",
                &format!("No orphan graph nodes ({eye_name})"),
                "All graph nodes exist in irises table",
            ));
        } else {
            checks.push(CheckResult::fail(
                "2a",
                &format!("No orphan graph nodes ({eye_name})"),
                format!("{orphan_count} graph nodes not found in irises table"),
            ));
        }
    }

    // 2b: Node coverage — iris IDs in DB but not in graph layer 0
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let layer0_ids: HashSet<u32> = if !graph.layers.is_empty() {
            graph.layers[0]
                .links
                .keys()
                .map(|v| v.serial_id())
                .collect()
        } else {
            HashSet::new()
        };

        let uncovered: HashSet<u32> = iris_ids
            .iter()
            .map(|&id| id as u32)
            .filter(|id| !layer0_ids.contains(id))
            .collect();

        if let Some(ref excl) = exclusions {
            if uncovered == *excl {
                checks.push(CheckResult::pass(
                    "2b",
                    &format!("Node coverage ({eye_name})"),
                    format!(
                        "{} uncovered IDs match exclusions list exactly",
                        uncovered.len()
                    ),
                ));
            } else {
                let only_in_uncovered: Vec<_> =
                    uncovered.difference(excl).copied().collect();
                let only_in_exclusions: Vec<_> =
                    excl.difference(&uncovered).copied().collect();
                let mut detail = format!(
                    "Mismatch: {} uncovered, {} excluded",
                    uncovered.len(),
                    excl.len()
                );
                if !only_in_uncovered.is_empty() {
                    let _ = write!(
                        detail,
                        "; {} in DB but not in exclusions (first 10: {:?})",
                        only_in_uncovered.len(),
                        &only_in_uncovered[..only_in_uncovered.len().min(10)]
                    );
                }
                if !only_in_exclusions.is_empty() {
                    let _ = write!(
                        detail,
                        "; {} in exclusions but not uncovered (first 10: {:?})",
                        only_in_exclusions.len(),
                        &only_in_exclusions[..only_in_exclusions.len().min(10)]
                    );
                }
                checks.push(CheckResult::fail(
                    "2b",
                    &format!("Node coverage ({eye_name})"),
                    detail,
                ));
            }
        } else {
            // Informational only — no exclusions file to validate against
            let detail = format!(
                "{} iris IDs not in graph layer 0 (no exclusions file to validate against)",
                uncovered.len()
            );
            checks.push(CheckResult::pass(
                "2b",
                &format!("Node coverage ({eye_name})"),
                detail,
            ));
        }
    }

    // 2c: Layer hierarchy
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut violations = 0u64;
        for (lc, layer) in graph.layers.iter().enumerate() {
            if lc == 0 {
                continue;
            }
            for node in layer.links.keys() {
                // Check that this node exists in all layers below
                for lower_lc in 0..lc {
                    if !graph.layers[lower_lc].links.contains_key(node) {
                        violations += 1;
                    }
                }
            }
        }
        if violations == 0 {
            checks.push(CheckResult::pass(
                "2c",
                &format!("Layer hierarchy ({eye_name})"),
                "All higher-layer nodes present in lower layers",
            ));
        } else {
            checks.push(CheckResult::fail(
                "2c",
                &format!("Layer hierarchy ({eye_name})"),
                format!("{violations} hierarchy violations"),
            ));
        }
    }

    // 2d: Neighbor validity
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut invalid_count = 0u64;
        for (_lc, layer) in graph.layers.iter().enumerate() {
            let layer_nodes: HashSet<&VectorId> = layer.links.keys().collect();
            for neighbors in layer.links.values() {
                for nb in neighbors {
                    if !layer_nodes.contains(nb) {
                        invalid_count += 1;
                    }
                }
            }
        }
        if invalid_count == 0 {
            checks.push(CheckResult::pass(
                "2d",
                &format!("Neighbor validity ({eye_name})"),
                "All neighbors reference valid nodes at the same layer",
            ));
        } else {
            checks.push(CheckResult::fail(
                "2d",
                &format!("Neighbor validity ({eye_name})"),
                format!("{invalid_count} invalid neighbor references"),
            ));
        }
    }

    // 2e: No self-loops
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut self_loop_count = 0u64;
        for layer in &graph.layers {
            for (node, neighbors) in layer.links.iter() {
                if neighbors.contains(node) {
                    self_loop_count += 1;
                }
            }
        }
        if self_loop_count == 0 {
            checks.push(CheckResult::pass(
                "2e",
                &format!("No self-loops ({eye_name})"),
                "No node lists itself as neighbor",
            ));
        } else {
            checks.push(CheckResult::fail(
                "2e",
                &format!("No self-loops ({eye_name})"),
                format!("{self_loop_count} self-loops found"),
            ));
        }
    }

    // 2f: Degree bounds
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut violations = 0u64;
        for (lc, layer) in graph.layers.iter().enumerate() {
            let layer_idx = lc.min(N_PARAM_LAYERS - 1);
            let m_limit = params.M_limit[layer_idx];
            for (node, neighbors) in layer.links.iter() {
                if neighbors.len() > m_limit {
                    violations += 1;
                    if violations <= 5 {
                        println!(
                            "  [2f] {eye_name} L{lc} node {} has degree {} > M_limit {}",
                            node,
                            neighbors.len(),
                            m_limit
                        );
                    }
                }
            }
        }
        if violations == 0 {
            checks.push(CheckResult::pass(
                "2f",
                &format!("Degree bounds ({eye_name})"),
                format!(
                    "L0 M_limit={}, L1+ M_limit={}",
                    params.M_limit[0],
                    params.M_limit[1.min(N_PARAM_LAYERS - 1)]
                ),
            ));
        } else {
            checks.push(CheckResult::fail(
                "2f",
                &format!("Degree bounds ({eye_name})"),
                format!("{violations} nodes exceed M_limit"),
            ));
        }
    }

    // 2g: Entry point validity
    for (eye_name, graph) in [("left", &left_graph), ("right", &right_graph)] {
        let mut valid = true;
        let mut detail = String::new();
        for ep in &graph.entry_points {
            if ep.layer >= graph.layers.len() {
                valid = false;
                let _ = write!(
                    detail,
                    "EP {} at layer {} but only {} layers exist; ",
                    ep.point, ep.layer, graph.layers.len()
                );
            } else if !graph.layers[ep.layer].links.contains_key(&ep.point) {
                valid = false;
                let _ = write!(
                    detail,
                    "EP {} not found in layer {}; ",
                    ep.point, ep.layer
                );
            }
        }
        if valid {
            checks.push(CheckResult::pass(
                "2g",
                &format!("Entry point validity ({eye_name})"),
                format!("{} entry points valid", graph.entry_points.len()),
            ));
        } else {
            checks.push(CheckResult::fail(
                "2g",
                &format!("Entry point validity ({eye_name})"),
                detail,
            ));
        }
    }

    // 2h: Left/Right graph sync — same serial IDs at layer 0
    {
        let left_l0_ids: HashSet<u32> = if !left_graph.layers.is_empty() {
            left_graph.layers[0]
                .links
                .keys()
                .map(|v| v.serial_id())
                .collect()
        } else {
            HashSet::new()
        };
        let right_l0_ids: HashSet<u32> = if !right_graph.layers.is_empty() {
            right_graph.layers[0]
                .links
                .keys()
                .map(|v| v.serial_id())
                .collect()
        } else {
            HashSet::new()
        };

        if left_l0_ids == right_l0_ids {
            checks.push(CheckResult::pass(
                "2h",
                "Left/Right graph sync",
                format!("{} serial IDs match at layer 0", left_l0_ids.len()),
            ));
        } else {
            let only_left: Vec<_> = left_l0_ids.difference(&right_l0_ids).copied().collect();
            let only_right: Vec<_> = right_l0_ids.difference(&left_l0_ids).copied().collect();
            checks.push(CheckResult::fail(
                "2h",
                "Left/Right graph sync",
                format!(
                    "Mismatch: {} only in left, {} only in right",
                    only_left.len(),
                    only_right.len()
                ),
            ));
        }
    }

    Ok((checks, degree_hist, graph_stats))
}

/// Load a graph for a specific eye (Left/Right) from the DB
async fn load_graph(
    graph_pg: &GraphPg<Aby3Store>,
    store_id: StoreId,
) -> Result<GraphMem<VectorId>> {
    let mut tx = graph_pg.tx().await?;
    let mut ops = tx.with_graph(store_id);
    let graph = ops.load_to_mem(graph_pg.pool(), 4).await?;
    Ok(graph)
}

/// Check 3: Persistent state consistency
async fn run_persistent_state_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_max_serial_id: usize,
) -> Result<Vec<CheckResult>> {
    let mut checks = Vec::new();

    // 3a: last_indexed_iris_id consistency
    let last_indexed: Option<u32> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
        .await?;

    // Get graph max serial IDs
    let left_max = get_graph_max_serial_id(graph_pg, StoreId::Left).await?;
    let right_max = get_graph_max_serial_id(graph_pg, StoreId::Right).await?;

    match last_indexed {
        Some(last_id) => {
            let last_id_usize = last_id as usize;
            let mut issues = Vec::new();

            if left_max != right_max {
                issues.push(format!(
                    "left graph max={left_max} != right graph max={right_max}"
                ));
            }
            if last_id_usize != left_max as usize {
                issues.push(format!(
                    "last_indexed_iris_id={last_id} != left graph max={left_max}"
                ));
            }
            if last_id_usize > iris_max_serial_id {
                issues.push(format!(
                    "last_indexed_iris_id={last_id} > irises max_id={iris_max_serial_id}"
                ));
            }

            if issues.is_empty() {
                checks.push(CheckResult::pass(
                    "3a",
                    "last_indexed_iris_id",
                    format!("Consistent: {last_id}"),
                ));
            } else {
                checks.push(CheckResult::fail(
                    "3a",
                    "last_indexed_iris_id",
                    issues.join("; "),
                ));
            }
        }
        None => {
            checks.push(CheckResult::pass(
                "3a",
                "last_indexed_iris_id",
                "Not set (no genesis run yet)".to_string(),
            ));
        }
    }

    // 3b: Graph max serial_id alignment
    if left_max == right_max {
        checks.push(CheckResult::pass(
            "3b",
            "Graph max serial_id alignment",
            format!("Both graphs max serial_id = {left_max}"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "3b",
            "Graph max serial_id alignment",
            format!("Left max={left_max}, Right max={right_max}"),
        ));
    }

    Ok(checks)
}

async fn get_graph_max_serial_id(
    graph_pg: &GraphPg<Aby3Store>,
    store_id: StoreId,
) -> Result<i64> {
    let mut tx = graph_pg.tx().await?;
    let mut ops = tx.with_graph(store_id);
    ops.get_max_serial_id().await
}

/// Dump all persistent state rows for stats
async fn dump_persistent_state(
    graph_pg: &GraphPg<Aby3Store>,
    stats: &mut Stats,
) -> Result<()> {
    // Query all rows from persistent_state table
    let rows: Vec<(String, String, serde_json::Value)> = sqlx::query_as(
        "SELECT domain, \"key\", \"value\" FROM persistent_state ORDER BY domain, \"key\"",
    )
    .fetch_all(graph_pg.pool())
    .await?;

    for (domain, key, value) in &rows {
        stats.add(
            format!("persistent_state[{domain}/{key}]"),
            value.to_string(),
        );
    }

    Ok(())
}

/// Check 4: Cross-schema consistency (HNSW vs GPU)
async fn run_cross_schema_checks(
    hnsw_store: &Store,
    gpu_store: &Store,
) -> Result<Vec<CheckResult>> {
    let mut checks = Vec::new();

    // 4a: Same row count
    let hnsw_count = hnsw_store.count_irises().await?;
    let gpu_count = gpu_store.count_irises().await?;
    if hnsw_count == gpu_count {
        checks.push(CheckResult::pass(
            "4a",
            "Same row count",
            format!("Both schemas have {hnsw_count} irises"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "4a",
            "Same row count",
            format!("HNSW={hnsw_count}, GPU={gpu_count}"),
        ));
    }

    // 4b: Same max serial ID
    let hnsw_max = hnsw_store.get_max_serial_id().await?;
    let gpu_max = gpu_store.get_max_serial_id().await?;
    if hnsw_max == gpu_max {
        checks.push(CheckResult::pass(
            "4b",
            "Same max serial ID",
            format!("Both schemas max_id={hnsw_max}"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "4b",
            "Same max serial ID",
            format!("HNSW max={hnsw_max}, GPU max={gpu_max}"),
        ));
    }

    // 4c: Byte-identical shares — stream both ordered by ID and compare
    println!("  Comparing iris shares between schemas...");
    let mut hnsw_stream = hnsw_store.stream_irises().await;
    let mut gpu_stream = gpu_store.stream_irises().await;

    let mut mismatch_count = 0u64;
    let mut compared_count = 0u64;
    let mut mismatch_details: Vec<String> = Vec::new();
    const MAX_MISMATCH_DETAILS: usize = 10;

    loop {
        let hnsw_next = hnsw_stream.next().await;
        let gpu_next = gpu_stream.next().await;

        match (hnsw_next, gpu_next) {
            (Some(Ok(h)), Some(Ok(g))) => {
                compared_count += 1;
                let h_id = h.serial_id() as i64;
                let g_id = g.serial_id() as i64;

                if h_id != g_id {
                    mismatch_count += 1;
                    if mismatch_details.len() < MAX_MISMATCH_DETAILS {
                        mismatch_details
                            .push(format!("ID mismatch: HNSW={h_id}, GPU={g_id}"));
                    }
                    // Can't meaningfully continue comparing once IDs diverge
                    break;
                }

                // Compare all 4 BYTEA columns via the u16 slices
                let codes_match = h.left_code() == g.left_code()
                    && h.left_mask() == g.left_mask()
                    && h.right_code() == g.right_code()
                    && h.right_mask() == g.right_mask();

                if !codes_match {
                    mismatch_count += 1;
                    if mismatch_details.len() < MAX_MISMATCH_DETAILS {
                        mismatch_details
                            .push(format!("Serial ID {h_id}: share data differs"));
                    }
                }

                if compared_count % 500_000 == 0 {
                    println!("  ... compared {} irises", compared_count);
                }
            }
            (None, None) => break,
            (Some(_), None) => {
                mismatch_count += 1;
                mismatch_details.push("HNSW has more rows than GPU".to_string());
                break;
            }
            (None, Some(_)) => {
                mismatch_count += 1;
                mismatch_details.push("GPU has more rows than HNSW".to_string());
                break;
            }
            (Some(Err(e)), _) | (_, Some(Err(e))) => {
                return Err(eyre!("Error streaming for cross-schema compare: {e}"));
            }
        }
    }

    if mismatch_count == 0 {
        checks.push(CheckResult::pass(
            "4c",
            "Byte-identical shares",
            format!("{compared_count} irises compared"),
        ));
    } else {
        let detail = format!(
            "{mismatch_count} mismatches (compared {compared_count}): {}",
            mismatch_details.join("; ")
        );
        checks.push(CheckResult::fail("4c", "Byte-identical shares", detail));
    }

    Ok(checks)
}

/// Check 5: Modifications table
async fn run_modification_checks(store: &Store, stats: &mut Stats) -> Result<Vec<CheckResult>> {
    let mut checks = Vec::new();

    // Query modification counts by status and persisted
    let rows: Vec<(String, bool, i64)> = sqlx::query_as(
        "SELECT status, persisted, COUNT(*) FROM modifications GROUP BY status, persisted ORDER BY status, persisted",
    )
    .fetch_all(&store.pool)
    .await?;

    let mut total_mods = 0i64;
    let mut non_completed_persisted = 0i64;

    for (status, persisted, count) in &rows {
        stats.add(
            format!("Modifications status={status} persisted={persisted}"),
            count.to_string(),
        );
        total_mods += count;
        if status != "COMPLETED" || !persisted {
            non_completed_persisted += count;
        }
    }

    stats.add("Total modifications", total_mods.to_string());

    // Also get counts by request_type
    let type_rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT request_type, COUNT(*) FROM modifications GROUP BY request_type ORDER BY request_type",
    )
    .fetch_all(&store.pool)
    .await?;

    for (req_type, count) in &type_rows {
        stats.add(
            format!("Modifications type={req_type}"),
            count.to_string(),
        );
    }

    // 5a: All completed & persisted
    if non_completed_persisted == 0 {
        checks.push(CheckResult::pass(
            "5a",
            "Completed & persisted",
            format!("All {total_mods} modifications are COMPLETED and persisted"),
        ));
    } else {
        checks.push(CheckResult::fail(
            "5a",
            "Completed & persisted",
            format!(
                "{non_completed_persisted}/{total_mods} modifications are not COMPLETED+persisted"
            ),
        ));
    }

    Ok(checks)
}

/// Write CSV reports
fn write_csv_reports(
    output_dir: &Path,
    checks: &[CheckResult],
    stats: &Stats,
    degree_hist: &[DegreeHistEntry],
) -> Result<()> {
    fs::create_dir_all(output_dir)?;

    // checks.csv
    let checks_path = output_dir.join("checks.csv");
    let mut wtr = csv::Writer::from_path(&checks_path)?;
    wtr.write_record(["id", "name", "status", "detail"])?;
    for c in checks {
        let status = if c.passed { "PASS" } else { "FAIL" };
        wtr.write_record([&c.id, &c.name, status, &c.detail])?;
    }
    wtr.flush()?;
    println!("Wrote {}", checks_path.display());

    // stats.csv
    let stats_path = output_dir.join("stats.csv");
    let mut wtr = csv::Writer::from_path(&stats_path)?;
    wtr.write_record(["key", "value"])?;
    for (k, v) in &stats.entries {
        wtr.write_record([k, v])?;
    }
    wtr.flush()?;
    println!("Wrote {}", stats_path.display());

    // degree_histogram.csv
    let hist_path = output_dir.join("degree_histogram.csv");
    let mut wtr = csv::Writer::from_path(&hist_path)?;
    wtr.write_record(["eye", "layer", "degree", "node_count"])?;
    for entry in degree_hist {
        wtr.write_record([
            &entry.eye,
            &entry.layer.to_string(),
            &entry.degree.to_string(),
            &entry.node_count.to_string(),
        ])?;
    }
    wtr.flush()?;
    println!("Wrote {}", hist_path.display());

    Ok(())
}
