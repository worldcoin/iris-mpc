#![recursion_limit = "256"]

use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    config::{ENV_PROD, ENV_STAGE},
    helpers::smpc_request::{
        REAUTH_MESSAGE_TYPE, RECOVERY_UPDATE_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    },
    postgres::{AccessMode, PostgresClient},
    SerialId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, LEFT, RIGHT},
    graph_checkpoint::{
        download_graph_checkpoint, get_most_recent_checkpoints, GraphCheckpointState,
    },
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::{
            graph_store::{GraphMutationRow, GraphPg},
            layered_graph::{GraphMem, Neighborhood},
        },
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
    /// Single-source layered BFS from the search entry point; emits per-bucket
    /// hop / reachability stats (`hops_by_bucket.csv`). Off by default; O(edges).
    #[arg(long, default_value_t = false)]
    bfs_hops: bool,
    /// Count strongly-connected components per layer (Tarjan). Off by default —
    /// this is an O(nodes + edges) pass over each layer.
    #[arg(long, default_value_t = false)]
    scc: bool,
    /// Number of consecutive serial IDs per bucket for all per-bucket reports
    /// (degree, hops, neighbor-serial matrix). Required.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    bucket_size: u32,
    /// Emit an in-depth per-serial reachability dossier (`probe_report.txt` /
    /// `.json`) for these serial IDs. Comma-separated. Enables the BFS + SCC
    /// passes needed to populate the dossier.
    #[arg(long, value_delimiter = ',')]
    probe_serials: Vec<u32>,
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

/// Per-(eye, layer, serial-id bucket) degree summary, emitted for both in- and
/// out-degree. `bucket` is the 0-based serial-id range index ((serial_id - 1) / bucket_size).
#[derive(Serialize)]
struct DegreeBucketEntry {
    eye: String,
    layer: usize,
    direction: &'static str,
    bucket: u32,
    serial_start: u32,
    serial_end: u32,
    node_count: usize,
    min: usize,
    avg: f64,
    median: usize,
    max: usize,
}

/// One cell of the per-(eye, layer) bucket->bucket directed-edge count matrix.
/// `valid_edges` counts active edges; `raw_edges` counts all edges.
/// Only populated (nonzero) cells are emitted.
#[derive(Serialize)]
struct MatrixEntry {
    eye: String,
    layer: usize,
    src_bucket: u32,
    dst_bucket: u32,
    valid_edges: u64,
    raw_edges: u64,
}

/// min / avg / median / max of a degree list. `degrees` is sorted in place.
fn degree_summary(degrees: &mut [usize]) -> (usize, f64, usize, usize) {
    degrees.sort_unstable();
    let n = degrees.len();
    let sum: usize = degrees.iter().sum();
    (
        degrees[0],
        sum as f64 / n as f64,
        degrees[n / 2],
        degrees[n - 1],
    )
}

/// Per-(eye, serial-id bucket) BFS-hop summary over the layer-0 nodes whose
/// serial ID falls in the bucket, for a single-source search from one entry point.
///
/// `hops_*` aggregate over the nodes in the bucket reachable from the entry point;
/// `reachable_nodes` / `unreachable_nodes` split the bucket by reachability.
#[derive(Serialize)]
struct HopBucketEntry {
    eye: String,
    bucket: u32,
    serial_start: u32,
    serial_end: u32,
    node_count: u64,
    reachable_nodes: u64,
    unreachable_nodes: u64,
    hops_min: u32,
    hops_avg: f64,
    hops_median: u32,
    hops_max: u32,
}

/// min / avg / median / max + total count from a value→count slice, where
/// `counts[v]` is the number of observations equal to `v`. Median is the value
/// at cumulative index `total / 2` (lower median). All-zero when empty.
fn hist_stats(counts: &[u64]) -> (u32, f64, u32, u32, u64) {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return (0, 0.0, 0, 0, 0);
    }
    let min = counts.iter().position(|&c| c > 0).unwrap() as u32;
    let max = counts.iter().rposition(|&c| c > 0).unwrap() as u32;
    let weighted: u128 = counts
        .iter()
        .enumerate()
        .map(|(v, &c)| v as u128 * c as u128)
        .sum();
    let avg = weighted as f64 / total as f64;
    let target = total / 2;
    let mut cum = 0u64;
    let mut median = min;
    for (v, &c) in counts.iter().enumerate() {
        cum += c;
        if cum > target {
            median = v as u32;
            break;
        }
    }
    (min, avg, median, max, total)
}

/// Per-layer adjacency indexed by `serial_id - 1` (`None` = no node at that
/// slot), plus the content-clock seq per slot (`INIT_ABSENT` = serial not a
/// live graph node) and the array length `n`.
type LayerArrays<'a> = (Vec<Vec<Option<&'a Neighborhood>>>, Vec<u64>, usize);

/// Sentinel in `live_versions` for a serial absent from the irises table.
const NO_IRIS: i16 = i16::MIN;

/// Sentinel in the init-seq array for a serial absent from the content clock.
const INIT_ABSENT: u64 = u64::MAX;

/// 0-based array slot of a 1-based serial; `None` for serial 0 (never a real node).
fn slot(serial: SerialId) -> Option<usize> {
    serial.checked_sub(1).map(|i| i as usize)
}

/// Fill a length-`n` adjacency array (indexed by `serial_id - 1`) from one
/// layer's links. An edge is *active* — followable by search — iff its
/// target's content-clock seq does not exceed the referencing neighborhood's
/// seq (the `get_active_links` gate).
fn fill_layer_arrays(
    links: &HashMap<SerialId, Neighborhood>,
    n: usize,
) -> Vec<Option<&Neighborhood>> {
    let mut adj: Vec<Option<&Neighborhood>> = vec![None; n];
    for (serial, nbhd) in links.iter() {
        if let Some(i) = slot(*serial).filter(|&i| i < n) {
            adj[i] = Some(nbhd);
        }
    }
    adj
}

/// Content-clock seq per slot, `INIT_ABSENT` where no live node exists.
fn build_init_seqs(graph: &GraphMem, n: usize) -> Vec<u64> {
    let mut seqs = vec![INIT_ABSENT; n];
    for (serial, init) in graph.node_init.iter() {
        if let Some(i) = slot(*serial).filter(|&i| i < n) {
            seqs[i] = init.seq_no;
        }
    }
    seqs
}

/// Per-layer adjacency plus content-clock seqs, indexed by `serial_id - 1`.
/// Sized to layer 0's max serial (layer 0 holds all nodes). Caller must ensure
/// layer 0 is non-empty. Returns (adj, init_seqs, n).
fn build_layer_arrays(graph: &GraphMem) -> LayerArrays<'_> {
    let n = graph.layers[0]
        .links
        .keys()
        .max()
        .map_or(0, |&m| m as usize);
    let mut adj: Vec<Vec<Option<&Neighborhood>>> = Vec::with_capacity(graph.layers.len());
    for layer in &graph.layers {
        adj.push(fill_layer_arrays(&layer.links, n));
    }
    let init_seqs = build_init_seqs(graph, n);
    (adj, init_seqs, n)
}

/// The search sources: every recorded entry point present in the graph, else the
/// temporary entry point (LinearScan, no recorded entry points). Returns
/// `(serial, layer)` pairs with each layer clamped to existing layers. Seeding
/// BFS from the whole entry-point set mirrors the search, which descends from the
/// entry point nearest the query — so reachability is "reachable from any entry
/// point" and the hop count is the shortest layered path from the nearest one.
fn pick_sources(
    graph: &GraphMem,
    adj: &[Vec<Option<&Neighborhood>>],
    n: usize,
) -> Vec<(SerialId, usize)> {
    let num_layers = graph.layers.len();
    let present = |point: SerialId, layer: usize| {
        slot(point).is_some_and(|i| i < n && adj[layer.min(num_layers - 1)][i].is_some())
    };
    let eps: Vec<(SerialId, usize)> = graph
        .entry_points
        .iter()
        .map(|ep| (ep.point, ep.layer.min(num_layers - 1)))
        .filter(|(p, l)| present(*p, *l))
        .collect();
    if !eps.is_empty() {
        return eps;
    }
    // LinearScan fallback: get_temporary_entry_point returns the min serial.
    let Some((tp, tl)) = graph.get_temporary_entry_point() else {
        return Vec::new();
    };
    let top = tl.min(num_layers - 1);
    if present(tp, top) {
        vec![(tp, top)]
    } else {
        Vec::new()
    }
}

/// Multi-source layered BFS (top layer → 0) over active edges, via a bucket
/// queue (Dial's algorithm). Returns `dist` indexed by `serial_id - 1`
/// (`u32::MAX` = unreachable). Each source enters at hop 0 in its own layer and
/// distances carry down between layers (free descent). Reachability is a topological
/// upper bound (any path from any source) and the hop count an optimistic lower
/// bound on the actual greedy descent, which follows a single path and may stall.
fn layered_bfs(
    adj: &[Vec<Option<&Neighborhood>>],
    init_seqs: &[u64],
    n: usize,
    sources: &[(SerialId, usize)],
) -> Vec<u32> {
    let mut dist = vec![u32::MAX; n];
    let Some(top) = sources.iter().map(|(_, l)| *l).max() else {
        return dist;
    };
    // Seeds entering at each layer (real entry points all sit at the top layer, but
    // a source recorded at a lower layer is introduced when its layer is reached).
    let mut seeds: Vec<Vec<u32>> = vec![Vec::new(); top + 1];
    for &(point, l) in sources {
        if let Some(i) = slot(point).filter(|&i| i < n) {
            if l <= top {
                seeds[l].push(i as u32);
            }
        }
    }
    let mut reached: Vec<u32> = Vec::new();
    let mut queue: Vec<Vec<u32>> = Vec::new();
    for layer in (0..=top).rev() {
        for &s in &seeds[layer] {
            if dist[s as usize] == u32::MAX {
                dist[s as usize] = 0;
                reached.push(s);
            }
        }
        let adj_l = &adj[layer];
        let mut max_d = 0u32;
        for &u in &reached {
            let d = dist[u as usize];
            while queue.len() <= d as usize {
                queue.push(Vec::new());
            }
            queue[d as usize].push(u);
            max_d = max_d.max(d);
        }
        let mut d = 0u32;
        while d <= max_d {
            let mut i = 0;
            while i < queue[d as usize].len() {
                let u = queue[d as usize][i] as usize;
                i += 1;
                if dist[u] != d {
                    continue; // stale bucket-queue entry
                }
                let nd = d + 1;
                let Some(nbhd) = adj_l[u] else {
                    continue;
                };
                for nb in nbhd.neighbors() {
                    let Some(v) = slot(nb).filter(|&v| v < n) else {
                        continue;
                    };
                    if adj_l[v].is_none() || init_seqs[v] > nbhd.seq_no() {
                        continue; // active-strict: only follow edges the search can traverse
                    }
                    if nd < dist[v] {
                        let first_seen = dist[v] == u32::MAX;
                        dist[v] = nd;
                        while queue.len() <= nd as usize {
                            queue.push(Vec::new());
                        }
                        queue[nd as usize].push(v as u32);
                        max_d = max_d.max(nd);
                        if first_seen {
                            reached.push(v as u32);
                        }
                    }
                }
            }
            d += 1;
        }
        for b in queue.iter_mut() {
            b.clear();
        }
    }
    dist
}

/// Iterative Tarjan SCC over one layer's active edges. Returns
/// `(comp_of, sizes)`: `comp_of[i]` = component index of the node at slot `i`
/// (`u32::MAX` if no node there), `sizes[c]` = size of component `c`.
fn scc_layer(adj_l: &[Option<&Neighborhood>], init_seqs: &[u64], n: usize) -> (Vec<u32>, Vec<u64>) {
    const UNVISITED: u32 = u32::MAX;
    // `neighbors()` decodes the neighborhood on every call, and a DFS frame is
    // resumed once per child — decode once at push time and keep the list in
    // the frame (peak extra memory = decoded edges along one DFS path).
    struct Frame {
        node: u32,
        edge_pos: u32,
        neighbors: Vec<SerialId>,
        seq_no: u64,
    }
    fn frame_for(adj_l: &[Option<&Neighborhood>], s: u32) -> Frame {
        let nbhd = adj_l[s as usize].expect("dfs only visits present slots");
        Frame {
            node: s,
            edge_pos: 0,
            neighbors: nbhd.neighbors(),
            seq_no: nbhd.seq_no(),
        }
    }
    let mut idx = vec![UNVISITED; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut comp_of = vec![u32::MAX; n];
    let mut comp_stack: Vec<u32> = Vec::new();
    let mut dfs: Vec<Frame> = Vec::new();
    let mut sizes: Vec<u64> = Vec::new();
    let mut next_index = 0u32;
    for s in 0..n {
        if adj_l[s].is_none() || idx[s] != UNVISITED {
            continue;
        }
        dfs.push(frame_for(adj_l, s as u32));
        while let Some(top) = dfs.len().checked_sub(1) {
            let node = dfs[top].node;
            let u = node as usize;
            if dfs[top].edge_pos == 0 {
                idx[u] = next_index;
                low[u] = next_index;
                next_index += 1;
                comp_stack.push(node);
                on_stack[u] = true;
            }
            let mut p = dfs[top].edge_pos as usize;
            let mut recursed = false;
            while p < dfs[top].neighbors.len() {
                let frame = &dfs[top];
                let nb = frame.neighbors[p];
                let seq_no = frame.seq_no;
                p += 1;
                let Some(v) = slot(nb).filter(|&v| v < n) else {
                    continue;
                };
                if adj_l[v].is_none() || init_seqs[v] > seq_no {
                    continue;
                }
                if idx[v] == UNVISITED {
                    dfs[top].edge_pos = p as u32;
                    dfs.push(frame_for(adj_l, v as u32));
                    recursed = true;
                    break;
                } else if on_stack[v] {
                    low[u] = low[u].min(idx[v]);
                }
            }
            if recursed {
                continue;
            }
            if low[u] == idx[u] {
                let comp = sizes.len() as u32;
                let mut size = 0u64;
                loop {
                    let w = comp_stack.pop().unwrap();
                    on_stack[w as usize] = false;
                    comp_of[w as usize] = comp;
                    size += 1;
                    if w == node {
                        break;
                    }
                }
                sizes.push(size);
            }
            dfs.pop();
            if let Some(parent) = dfs.last() {
                let pu = parent.node as usize;
                low[pu] = low[pu].min(low[u]);
            }
        }
    }
    (comp_of, sizes)
}

/// Layered BFS hop/reachability stats, bucketed by serial ID.
///
/// Seeds from every live entry point (the set the search descends from), so a node
/// counts as reachable if any entry point can reach it. Hop count is the shortest
/// *layered* path: traverse a layer's edges, then descend (free) at any reached
/// node. Since nodes enter a layer carrying the distance from above, each layer's
/// relaxation is a non-uniform-source shortest path, run with a bucket queue
/// (Dial's algorithm). Reachability is active-strict.
fn compute_hop_buckets(
    eye: &str,
    graph: &GraphMem,
    bucket_size: u32,
    out: &mut Vec<HopBucketEntry>,
) {
    let Some(l0) = graph.layers.first() else {
        return;
    };
    if l0.links.is_empty() {
        return;
    }
    let (adj, init_seqs, n) = build_layer_arrays(graph);
    let dist = layered_bfs(&adj, &init_seqs, n, &pick_sources(graph, &adj, n));

    // Aggregate per bucket: hops over reachable nodes, plus reachable/unreachable split.
    let num_buckets = n.div_ceil(bucket_size as usize);
    let mut hop_counts: Vec<Vec<u64>> = vec![Vec::new(); num_buckets]; // [bucket][hop]
    let mut node_count: Vec<u64> = vec![0; num_buckets];
    let mut unreachable: Vec<u64> = vec![0; num_buckets];
    for ui in 0..n {
        if adj[0][ui].is_none() {
            continue;
        }
        let bucket = ui / bucket_size as usize;
        node_count[bucket] += 1;
        let d = dist[ui];
        if d == u32::MAX {
            unreachable[bucket] += 1;
        } else {
            let hc = &mut hop_counts[bucket];
            while hc.len() <= d as usize {
                hc.push(0);
            }
            hc[d as usize] += 1;
        }
    }

    for b in 0..num_buckets {
        if node_count[b] == 0 {
            continue;
        }
        let (hmin, havg, hmed, hmax, reachable) = hist_stats(&hop_counts[b]);
        out.push(HopBucketEntry {
            eye: eye.to_string(),
            bucket: b as u32,
            serial_start: b as u32 * bucket_size + 1,
            serial_end: (b as u32 + 1) * bucket_size,
            node_count: node_count[b],
            reachable_nodes: reachable,
            unreachable_nodes: unreachable[b],
            hops_min: hmin,
            hops_avg: havg,
            hops_median: hmed,
            hops_max: hmax,
        });
    }
}

/// Count strongly-connected components of one layer's directed graph (iterative
/// Tarjan — forward edges only, so no reverse adjacency is needed). Edges are
/// active-strict (a content-stale reference is not an edge), matching the BFS.
/// Returns (number of SCCs, size of the largest SCC). All state is array-indexed
/// by `serial_id - 1`.
fn count_sccs(graph: &GraphMem, layer_idx: usize) -> (u64, u64) {
    let layer = &graph.layers[layer_idx];
    if layer.links.is_empty() {
        return (0, 0);
    }
    let n = layer.links.keys().max().copied().unwrap() as usize;
    let adj = fill_layer_arrays(&layer.links, n);
    let init_seqs = build_init_seqs(graph, n);
    let (_, sizes) = scc_layer(&adj, &init_seqs, n);
    (sizes.len() as u64, sizes.iter().copied().max().unwrap_or(0))
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

#[derive(Serialize, Clone)]
struct ProbeNeighbor {
    serial: u32,
    /// Content-clock version of this neighbor; `None` if not a live graph node.
    version: Option<i16>,
    /// Whether the edge passes the `get_active_links` gate (target's
    /// content-clock seq ≤ the referencing neighborhood's seq).
    active: bool,
    reachable: bool,
    hop: Option<u32>,
}

/// In-depth per-(eye, serial) reachability dossier for a probed node.
#[derive(Serialize)]
struct ProbeReport {
    eye: String,
    serial: u32,
    exists_in_graph: bool,
    in_irises_table: bool,
    /// Content-clock version of this serial's graph node.
    graph_version: Option<i16>,
    /// Whether `graph_version` matches the irises-table version. `false` ⇒
    /// content changed without a re-index (search still traverses; graph
    /// bookkeeping is out of sync with the store). `None` if either is absent.
    version_synced: Option<bool>,
    cpu_version: Option<i16>,
    gpu_version: Option<i16>,
    layers_present: Vec<usize>,
    reachable: bool,
    hop: Option<u32>,
    scc_id: Option<u32>,
    scc_size: Option<u64>,
    same_scc_as_entry: Option<bool>,
    self_loop: bool,
    in_degree_raw: u32,
    in_degree_active: u32,
    stale_in_edges: u32,
    out_degree: u32,
    in_neighbors_active: Vec<ProbeNeighbor>,
    in_neighbors_raw: Vec<ProbeNeighbor>,
    out_neighbors: Vec<ProbeNeighbor>,
    pending_modifications: Vec<String>,
    gpu_byte_match: Option<bool>,
    verdict: String,
}

/// Build a per-(eye, serial) dossier explaining whether/why each probed serial is
/// reachable by the search. Topology (reachability, SCC, neighbor lists) reuses the
/// shared BFS/SCC/array helpers; version/modification/GPU-byte facts come from the DB.
async fn run_probe_reports(
    probes: &[u32],
    graphs: &BothEyes<GraphMem>,
    iris_ids: &HashSet<i64>,
    live_versions: &[i16],
    hnsw_pool: &sqlx::PgPool,
    gpu_pool: &sqlx::PgPool,
) -> Result<Vec<ProbeReport>> {
    // --- DB facts (serial-specific, eye-independent) ---
    let probe_i64: Vec<i64> = probes.iter().map(|&s| s as i64).collect();
    let mods: Vec<(i64, i64, String)> = sqlx::query_as(
        "SELECT serial_id, id, request_type FROM modifications \
         WHERE serial_id = ANY($1) ORDER BY id",
    )
    .bind(&probe_i64)
    .fetch_all(gpu_pool)
    .await
    .unwrap_or_default();
    let mut mods_by_serial: HashMap<u32, Vec<String>> = HashMap::new();
    for (sid, id, rt) in mods {
        mods_by_serial
            .entry(sid as u32)
            .or_default()
            .push(format!("{rt}#{id}"));
    }
    let sql = "SELECT id, left_code, left_mask, right_code, right_mask \
               FROM irises WHERE id = ANY($1)";
    let hnsw_rows: Vec<IrisRow> = sqlx::query_as(sql)
        .bind(&probe_i64)
        .fetch_all(hnsw_pool)
        .await
        .unwrap_or_default();
    let gpu_rows: Vec<IrisRow> = sqlx::query_as(sql)
        .bind(&probe_i64)
        .fetch_all(gpu_pool)
        .await
        .unwrap_or_default();
    let to_map = |rows: Vec<IrisRow>| -> HashMap<i64, IrisData> {
        rows.into_iter()
            .map(|(id, lc, lm, rc, rm)| (id, (lc, lm, rc, rm)))
            .collect()
    };
    let hnsw_map = to_map(hnsw_rows);
    let gpu_map = to_map(gpu_rows);

    let version_sql = "SELECT id, version_id FROM irises WHERE id = ANY($1)";
    let fetch_versions = |pool: &sqlx::PgPool| {
        let pool = pool.clone();
        let ids = probe_i64.clone();
        async move {
            sqlx::query_as::<_, (i64, i16)>(version_sql)
                .bind(&ids)
                .fetch_all(&pool)
                .await
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<i64, i16>>()
        }
    };
    let cpu_versions = fetch_versions(hnsw_pool).await;
    let gpu_versions = fetch_versions(gpu_pool).await;

    // --- Per-eye topology ---
    let probe_set: HashSet<u32> = probes.iter().copied().collect();
    let mut out: Vec<ProbeReport> = Vec::new();
    for (eye, idx) in [("left", LEFT), ("right", RIGHT)] {
        let graph = &graphs[idx];
        if graph.layers.first().is_none_or(|l| l.links.is_empty()) {
            continue;
        }
        let (adj, init_seqs, n) = build_layer_arrays(graph);
        let sources = pick_sources(graph, &adj, n);
        let dist = layered_bfs(&adj, &init_seqs, n, &sources);
        let (comp_of, sizes) = scc_layer(&adj[0], &init_seqs, n);
        let entry_comp = sources
            .first()
            .and_then(|(p, _)| slot(*p))
            .map(|i| comp_of[i]);

        // One edge sweep collects in-neighbors (raw + active) for the probe set.
        let mut in_raw: HashMap<u32, Vec<ProbeNeighbor>> = HashMap::new();
        let mut in_active: HashMap<u32, Vec<ProbeNeighbor>> = HashMap::new();
        for (node, nbhd) in graph.layers[0].links.iter() {
            let si = slot(*node).filter(|&i| i < n);
            let reachable = si.is_some_and(|i| dist[i] != u32::MAX);
            for target in nbhd.neighbors() {
                if !probe_set.contains(&target) {
                    continue;
                }
                let active = graph
                    .node_init
                    .get(&target)
                    .is_some_and(|ni| ni.seq_no <= nbhd.seq_no());
                let pn = ProbeNeighbor {
                    serial: *node,
                    version: graph.node_init.get(node).map(|ni| ni.version),
                    active,
                    reachable,
                    hop: reachable.then(|| dist[si.unwrap()]),
                };
                if active {
                    in_active.entry(target).or_default().push(pn.clone());
                }
                in_raw.entry(target).or_default().push(pn);
            }
        }

        for &serial in probes {
            let pidx = slot(serial).filter(|&i| i < n);
            let live_version = pidx
                .and_then(|i| live_versions.get(i))
                .copied()
                .filter(|&v| v != NO_IRIS);
            let graph_version = graph.node_init.get(&serial).map(|ni| ni.version);
            let version_synced = match (graph_version, live_version) {
                (Some(g), Some(l)) => Some(g == l),
                _ => None,
            };
            let exists_in_graph = pidx.is_some_and(|i| adj[0][i].is_some());
            let probe_nbhd = pidx.and_then(|i| adj[0][i]);
            let probe_seq = probe_nbhd.map(|nbhd| nbhd.seq_no());
            let layers_present: Vec<usize> = (0..graph.layers.len())
                .filter(|&l| pidx.is_some_and(|i| adj[l][i].is_some()))
                .collect();
            let reachable = exists_in_graph && dist[pidx.unwrap()] != u32::MAX;
            let hop = reachable.then(|| dist[pidx.unwrap()]);
            let scc_id = exists_in_graph.then(|| comp_of[pidx.unwrap()]);
            let scc_size = scc_id.map(|c| sizes[c as usize]);
            let same_scc_as_entry = match (scc_id, entry_comp) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            };
            let out_ids: Vec<SerialId> = probe_nbhd.map_or_else(Vec::new, |nbhd| nbhd.neighbors());
            let self_loop = out_ids.contains(&serial);
            let out_neighbors: Vec<ProbeNeighbor> = out_ids
                .iter()
                .map(|&nb| {
                    let i = slot(nb).filter(|&i| i < n);
                    let r = i.is_some_and(|i| dist[i] != u32::MAX);
                    ProbeNeighbor {
                        serial: nb,
                        version: graph.node_init.get(&nb).map(|ni| ni.version),
                        active: probe_seq.is_some_and(|ps| {
                            graph.node_init.get(&nb).is_some_and(|ni| ni.seq_no <= ps)
                        }),
                        reachable: r,
                        hop: r.then(|| dist[i.unwrap()]),
                    }
                })
                .collect();
            let in_neighbors_raw = in_raw.remove(&serial).unwrap_or_default();
            let in_neighbors_active = in_active.remove(&serial).unwrap_or_default();
            let in_degree_raw = in_neighbors_raw.len() as u32;
            let in_degree_active = in_neighbors_active.len() as u32;
            let active_non_self = in_neighbors_active
                .iter()
                .filter(|nbn| nbn.serial != serial)
                .count();

            let verdict = if !exists_in_graph {
                "ABSENT (not a layer-0 node)".to_string()
            } else if reachable {
                format!("REACHABLE@hop {}", hop.unwrap())
            } else if in_degree_raw == 0 {
                "ORPHAN (never linked: zero in-edges)".to_string()
            } else if active_non_self == 0 && self_loop {
                "ORPHAN (self-loop only)".to_string()
            } else if in_degree_active == 0 {
                "ORPHAN (severed: all in-edges content-stale)".to_string()
            } else {
                "CUT-OFF (active in-neighbors all unreachable)".to_string()
            };

            let byte = hnsw_map.get(&(serial as i64));
            let gpu = gpu_map.get(&(serial as i64));
            out.push(ProbeReport {
                eye: eye.to_string(),
                serial,
                exists_in_graph,
                in_irises_table: iris_ids.contains(&(serial as i64)),
                graph_version,
                version_synced,
                cpu_version: cpu_versions.get(&(serial as i64)).copied(),
                gpu_version: gpu_versions.get(&(serial as i64)).copied(),
                layers_present,
                reachable,
                hop,
                scc_id,
                scc_size,
                same_scc_as_entry,
                self_loop,
                in_degree_raw,
                in_degree_active,
                stale_in_edges: in_degree_raw - in_degree_active,
                out_degree: out_neighbors.len() as u32,
                in_neighbors_active,
                in_neighbors_raw,
                out_neighbors,
                pending_modifications: mods_by_serial.get(&serial).cloned().unwrap_or_default(),
                gpu_byte_match: match (byte, gpu) {
                    (Some(h), Some(g)) => Some(h == g),
                    _ => None,
                },
                verdict,
            });
        }
    }
    Ok(out)
}

/// Render a probe neighbor list (capped) as `serial:version[(stale)]@hop` /
/// `…/unreach`, where `(stale)` marks an edge the search cannot follow.
fn fmt_nbrs(nbrs: &[ProbeNeighbor], cap: usize) -> String {
    let shown: Vec<String> = nbrs
        .iter()
        .take(cap)
        .map(|n| {
            let v = n.version.map(|v| v.to_string()).unwrap_or("?".to_string());
            let st = if n.active { "" } else { "(stale)" };
            match n.hop {
                Some(h) => format!("{}:{}{}@{}", n.serial, v, st, h),
                None => format!("{}:{}{}/unreach", n.serial, v, st),
            }
        })
        .collect();
    let extra = if nbrs.len() > cap {
        format!(" …(+{} more)", nbrs.len() - cap)
    } else {
        String::new()
    };
    format!("[{}]{}", shown.join(", "), extra)
}

/// Human-readable multi-line rendering of one probe dossier.
fn format_probe(r: &ProbeReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "[{}] serial {}", r.eye, r.serial);
    let _ = writeln!(s, "  VERDICT: {}", r.verdict);
    let _ = writeln!(
        s,
        "  exists={} in_irises={} version(cpu/gpu)={:?}/{:?} graph_version={:?} synced={:?} layers={:?}",
        r.exists_in_graph,
        r.in_irises_table,
        r.cpu_version,
        r.gpu_version,
        r.graph_version,
        r.version_synced,
        r.layers_present
    );
    let _ = writeln!(
        s,
        "  reachable={} hop={:?} scc=#{:?} scc_size={:?} same_scc_as_entry={:?} self_loop={}",
        r.reachable, r.hop, r.scc_id, r.scc_size, r.same_scc_as_entry, r.self_loop
    );
    let _ = writeln!(
        s,
        "  in_degree raw={} active={} stale={}  out_degree={}",
        r.in_degree_raw, r.in_degree_active, r.stale_in_edges, r.out_degree
    );
    let _ = writeln!(
        s,
        "  gpu_byte_match={:?}  modifications={:?}",
        r.gpu_byte_match, r.pending_modifications
    );
    let _ = writeln!(
        s,
        "  in_neighbors_active: {}",
        fmt_nbrs(&r.in_neighbors_active, 25)
    );
    let _ = writeln!(
        s,
        "  in_neighbors_raw:   {}",
        fmt_nbrs(&r.in_neighbors_raw, 25)
    );
    let _ = writeln!(
        s,
        "  out_neighbors:      {}",
        fmt_nbrs(&r.out_neighbors, 25)
    );
    s
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
    let mut degree_buckets: Vec<DegreeBucketEntry> = Vec::new();
    let mut matrix_entries: Vec<MatrixEntry> = Vec::new();
    let mut hop_buckets: Vec<HopBucketEntry> = Vec::new();

    let raw_exclusions: Option<Vec<u32>> = match &args.exclusions_s3_uri {
        Some(uri) => {
            let parsed = download_exclusions_from_s3(uri, config.force_path_style()).await?;
            Some(parsed.deleted_serial_ids)
        }
        None => None,
    };

    // --- Load graph from S3 checkpoint, then replay any mutations recorded after it ---
    let bucket = config.graph_checkpoint_bucket_name.as_str();
    rpt!(rpt, "--- Loading graph from S3 checkpoint ---");
    let checkpoint_state = load_checkpoint_state(
        &graph_pg,
        args.checkpoint_s3_key.as_deref(),
        bucket,
        &mut rpt,
    )
    .await?;

    // Build the checkpoint S3 client (mirrors AwsClients::checkpoint_s3_client,
    // which may target a different region from the general S3 client).
    let checkpoint_s3_client = build_checkpoint_s3_client(
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
    let mut graphs: BothEyes<GraphMem> =
        download_graph_checkpoint(&checkpoint_s3_client, bucket, &checkpoint_state, None).await?;
    rpt!(rpt, "  Checkpoint loaded and BLAKE3 verified.");

    // Replay any GraphMutations recorded in hawk_graph_mutations after the checkpoint.
    let mutation_rows: Vec<GraphMutationRow> = graph_pg
        .get_hawk_graph_mutations_after(checkpoint_state.graph_mutation_id)
        .await?;
    rpt!(
        rpt,
        "  Applying {} mutation row(s) after checkpoint (graph_mutation_id={:?})...",
        mutation_rows.len(),
        checkpoint_state.graph_mutation_id,
    );
    for row in &mutation_rows {
        let both_eyes = row.deserialize_mutations()?;
        for m in &both_eyes[LEFT] {
            graphs[LEFT].insert_apply(m)?;
        }
        for m in &both_eyes[RIGHT] {
            graphs[RIGHT].insert_apply(m)?;
        }
    }
    rpt!(rpt, "  All mutations applied.");

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

    let s3_graphs: Option<BothEyes<GraphMem>> = Some(graphs);

    rpt!(rpt, "--- Collecting iris IDs ---");
    let (iris_ids, live_versions) = collect_iris_ids(&hnsw_store, &mut stats).await?;

    // Filter exclusions to IDs that actually exist in this DB snapshot.
    // Genesis filters deletions to <= max_indexation_id; the S3 file may
    // contain IDs beyond this snapshot's range.
    let iris_max = iris_ids.iter().copied().max().unwrap_or(0) as u32;
    let exclusions: Option<HashSet<SerialId>> = raw_exclusions.map(|raw| {
        let before = raw.len();
        let filtered: HashSet<SerialId> = raw.into_iter().filter(|&id| id <= iris_max).collect();
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
        s3_graphs.as_ref(),
        &iris_ids,
        &live_versions,
        &exclusions,
        args.m,
        layer_probability,
        &mut checks,
        &mut degree_hist,
        &mut degree_buckets,
        &mut matrix_entries,
        &mut hop_buckets,
        args.bucket_size,
        args.bfs_hops,
        args.scc,
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

    // --- Probe reports (in-depth per-serial reachability dossier) ---
    if !args.probe_serials.is_empty() {
        if let Some(graphs) = s3_graphs.as_ref() {
            rpt!(
                rpt,
                "\n--- Probe reports for serials {:?} ---",
                args.probe_serials
            );
            let reports = run_probe_reports(
                &args.probe_serials,
                graphs,
                &iris_ids,
                &live_versions,
                &hnsw_store.pool,
                &gpu_pg.pool,
            )
            .await?;
            let mut txt = String::new();
            for r in &reports {
                let block = format_probe(r);
                rpt!(rpt, "{}", block.trim_end());
                txt.push_str(&block);
                txt.push('\n');
            }
            let p = args.output_dir.join("probe_report.txt");
            fs::write(&p, &txt)?;
            println!("Wrote {}", p.display());
            let p = args.output_dir.join("probe_report.json");
            fs::write(&p, serde_json::to_string_pretty(&reports)?)?;
            println!("Wrote {}", p.display());
        }
    }

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

    let mut output_files = write_json_reports(
        &args.output_dir,
        &checks,
        &stats,
        &degree_hist,
        &degree_buckets,
        &matrix_entries,
        &hop_buckets,
    )?;
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

/// Returns the set of serial IDs present in the (cpu) irises table and a
/// `live_versions` vector indexed by `serial_id - 1` holding each serial's
/// version (`NO_IRIS` for gaps), for cross-checking graph content-clock
/// versions against the store.
async fn collect_iris_ids(store: &Store, stats: &mut Stats) -> Result<(HashSet<i64>, Vec<i16>)> {
    let rows: Vec<(i64, i16)> = sqlx::query_as("SELECT id, version_id FROM irises")
        .fetch_all(&store.pool)
        .await?;

    let max_id = rows.iter().map(|(id, _)| *id).max().unwrap_or(0);
    stats.add("Total iris count (HNSW)", rows.len().to_string());
    stats.add("Max serial ID (HNSW)", max_id.to_string());

    let mut live_versions = vec![NO_IRIS; max_id.max(0) as usize];
    for (id, v) in &rows {
        if *id >= 1 {
            live_versions[(*id - 1) as usize] = *v;
        }
    }
    let ids = rows.into_iter().map(|(id, _)| id).collect();
    Ok((ids, live_versions))
}

// ---------------------------------------------------------------------------
// Check 1: HNSW graph structural checks
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_graph_checks(
    s3_graphs: Option<&BothEyes<GraphMem>>,
    iris_ids: &HashSet<i64>,
    live_versions: &[i16],
    exclusions: &Option<HashSet<SerialId>>,
    m: usize,
    layer_probability: f64,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    degree_buckets: &mut Vec<DegreeBucketEntry>,
    matrix_entries: &mut Vec<MatrixEntry>,
    hop_buckets: &mut Vec<HopBucketEntry>,
    bucket_size: u32,
    bfs_hops: bool,
    scc: bool,
    stats: &mut Stats,
    rpt: &mut Report,
) -> Result<()> {
    let mut l0_id_sets: Vec<(&str, HashSet<SerialId>)> = Vec::new();

    if let Some(graphs) = s3_graphs {
        for (eye, idx) in [("left", LEFT), ("right", RIGHT)] {
            rpt!(
                rpt,
                "  Checking {eye} graph (from S3 checkpoint + mutations)..."
            );
            let l0_ids = check_single_graph(
                eye,
                &graphs[idx],
                iris_ids,
                live_versions,
                exclusions,
                m,
                layer_probability,
                bucket_size,
                checks,
                degree_hist,
                degree_buckets,
                matrix_entries,
                stats,
                rpt,
            );
            l0_id_sets.push((eye, l0_ids));
            if bfs_hops {
                rpt!(
                    rpt,
                    "  Computing {eye} BFS hop buckets from entry points..."
                );
                compute_hop_buckets(eye, &graphs[idx], bucket_size, hop_buckets);
            }
            if scc {
                for lc in 0..graphs[idx].layers.len() {
                    let (num, largest) = count_sccs(&graphs[idx], lc);
                    rpt!(rpt, "  {eye} layer {lc}: {num} SCC(s), largest {largest}");
                    stats.add(format!("{eye} layer {lc} SCC count"), num.to_string());
                    stats.add(
                        format!("{eye} layer {lc} largest SCC size"),
                        largest.to_string(),
                    );
                }
            }
        }
    } else {
        eyre::bail!(
            "No graph loaded — S3 checkpoint is required (DB graph loading is no longer supported)"
        );
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
    graph: &GraphMem,
    iris_ids: &HashSet<i64>,
    live_versions: &[i16],
    exclusions: &Option<HashSet<SerialId>>,
    m: usize,
    layer_probability: f64,
    bucket_size: u32,
    checks: &mut Vec<CheckResult>,
    degree_hist: &mut Vec<DegreeHistEntry>,
    degree_buckets: &mut Vec<DegreeBucketEntry>,
    matrix_entries: &mut Vec<MatrixEntry>,
    stats: &mut Stats,
    rpt: &mut Report,
) -> HashSet<SerialId> {
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

    // Content changed without a re-index: search still traverses (edges
    // resolve by serial), but graph bookkeeping is out of sync with the store.
    let version_desync = graph
        .node_init
        .iter()
        .filter(|(serial, ni)| {
            slot(**serial)
                .and_then(|i| live_versions.get(i))
                .is_some_and(|&v| v != NO_IRIS && v != ni.version)
        })
        .count();
    stats.add(
        format!("{eye} nodes version-desynced vs irises"),
        version_desync.to_string(),
    );

    for (lc, layer) in graph.layers.iter().enumerate() {
        stats.add(
            format!("{eye} layer {lc} node count"),
            layer.links.len().to_string(),
        );
        let mut deg_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for neighbors in layer.links.values() {
            *deg_counts.entry(neighbors.degree()).or_insert(0) += 1;
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
            let mut degrees: Vec<usize> = layer.links.values().map(|n| n.degree()).collect();
            degrees.sort();
            let (min, max) = (degrees[0], degrees[degrees.len() - 1]);
            let avg = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
            let median = degrees[degrees.len() / 2];
            stats.add(
                format!("{eye} layer {lc} degree min/avg/median/max"),
                format!("{min}/{avg:.1}/{median}/{max}"),
            );
        }

        // Per-bucket out/in-degree and the bucket->bucket edge matrix
        // (valid = active edges with both endpoints in the layer, raw = all).
        // An active edge passes the `get_active_links` gate: the target's
        // content-clock seq does not exceed the referencing neighborhood's seq.
        let in_size = layer.links.keys().max().map_or(0, |&s| s as usize);
        let mut present = vec![false; in_size];
        for node in layer.links.keys() {
            if let Some(i) = slot(*node).filter(|&i| i < in_size) {
                present[i] = true;
            }
        }
        let mut in_deg: Vec<u32> = vec![0; in_size];
        let mut out_by_bucket: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut matrix: HashMap<(u32, u32), (u64, u64)> = HashMap::new();
        for (node, nbhd) in layer.links.iter() {
            let Some(si) = slot(*node) else {
                continue;
            };
            let src = si as u32 / bucket_size;
            out_by_bucket.entry(src).or_default().push(nbhd.degree());
            for nb in nbhd.neighbors() {
                let Some(v) = slot(nb) else {
                    continue;
                };
                let cell = matrix
                    .entry((src, v as u32 / bucket_size))
                    .or_insert((0, 0));
                cell.1 += 1;
                if v < in_size
                    && present[v]
                    && graph
                        .node_init
                        .get(&nb)
                        .is_some_and(|ni| ni.seq_no <= nbhd.seq_no())
                {
                    in_deg[v] += 1;
                    cell.0 += 1;
                }
            }
        }
        for (&(src_bucket, dst_bucket), &(valid_edges, raw_edges)) in &matrix {
            matrix_entries.push(MatrixEntry {
                eye: eye.to_string(),
                layer: lc,
                src_bucket,
                dst_bucket,
                valid_edges,
                raw_edges,
            });
        }
        let mut in_by_bucket: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for node in layer.links.keys() {
            if let Some(i) = slot(*node).filter(|&i| i < in_size) {
                in_by_bucket
                    .entry(i as u32 / bucket_size)
                    .or_default()
                    .push(in_deg[i] as usize);
            }
        }
        for (direction, by_bucket) in [("out", &mut out_by_bucket), ("in", &mut in_by_bucket)] {
            for (&bucket, degrees) in by_bucket.iter_mut() {
                let (min, avg, median, max) = degree_summary(degrees);
                degree_buckets.push(DegreeBucketEntry {
                    eye: eye.to_string(),
                    layer: lc,
                    direction,
                    bucket,
                    serial_start: bucket * bucket_size + 1,
                    serial_end: (bucket + 1) * bucket_size,
                    node_count: degrees.len(),
                    min,
                    avg,
                    median,
                    max,
                });
            }
        }
    }

    // -- 1a: No orphan graph nodes --
    let orphan_count = graph
        .layers
        .iter()
        .flat_map(|l| l.links.keys())
        .filter(|n| !iris_ids.contains(&(**n as i64)))
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
    let layer0_ids: HashSet<SerialId> = graph
        .layers
        .first()
        .map(|l| l.links.keys().copied().collect())
        .unwrap_or_default();
    let uncovered: HashSet<SerialId> = iris_ids
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
            let nodes: HashSet<&SerialId> = layer.links.keys().collect();
            layer
                .links
                .values()
                .flat_map(|nbs| nbs.neighbors())
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
        .filter(|(node, nbs)| nbs.neighbors().contains(node))
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
            let ids = nbs.neighbors();
            let unique: HashSet<&SerialId> = ids.iter().collect();
            (ids.len() - unique.len()) as u64
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
            if nbs.degree() > m_limit {
                degree_viol += 1;
                if degree_viol <= 5 {
                    rpt!(
                        rpt,
                        "  [1g] {eye} L{lc} node {node} degree {} > M_limit {m_limit}",
                        nbs.degree()
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

/// Resolve checkpoint state: use explicit S3 key (looked up from DB) or
/// auto-discover the latest checkpoint from the genesis_graph_checkpoint table.
async fn load_checkpoint_state(
    graph_pg: &GraphPg<Aby3Store>,
    explicit_key: Option<&str>,
    bucket: &str,
    rpt: &mut Report,
) -> Result<GraphCheckpointState> {
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
        Ok(GraphCheckpointState {
            s3_key: row.s3_key,
            last_indexed_iris_id: row.last_indexed_iris_id.try_into().map_err(|_| {
                eyre::eyre!("Invalid last_indexed_iris_id: {}", row.last_indexed_iris_id)
            })?,
            last_indexed_modification_id: row.last_indexed_modification_id,
            graph_mutation_id: row.graph_mutation_id,
            blake3_hash: row.blake3_hash,
            graph_version: row.graph_version,
            is_archival: row.is_archival,
        })
    } else {
        rpt!(rpt, "  Auto-discovering most recent checkpoint from DB...");
        let (checkpoints, _hashes) = get_most_recent_checkpoints(graph_pg).await?;
        if checkpoints.is_empty() {
            return Err(eyre::eyre!(
                "No checkpoints found in genesis_graph_checkpoint table"
            ));
        }
        // Pick the checkpoint with the highest last_indexed_iris_id as the most recent.
        let state = checkpoints
            .into_iter()
            .max_by_key(|c| c.last_indexed_iris_id)
            .expect("non-empty vec");
        rpt!(
            rpt,
            "  Found checkpoint: s3://{}/{} (iris_id={}, mod_id={})",
            bucket,
            state.s3_key,
            state.last_indexed_iris_id,
            state.last_indexed_modification_id
        );
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Check 2: Persistent state consistency
// ---------------------------------------------------------------------------

async fn run_persistent_state_checks(
    graph_pg: &GraphPg<Aby3Store>,
    iris_max_serial_id: usize,
    s3_graphs: Option<&BothEyes<GraphMem>>,
    checks: &mut Vec<CheckResult>,
    stats: &mut Stats,
) -> Result<Option<i64>> {
    let last_indexed: Option<u32> = graph_pg
        .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
        .await?;
    // The graph is always loaded from an S3 checkpoint (+ mutations); the
    // Postgres links table is no longer the source of truth.
    let (left_max, right_max) = match s3_graphs {
        Some(graphs) => (
            graph_mem_max_serial_id(&graphs[LEFT]),
            graph_mem_max_serial_id(&graphs[RIGHT]),
        ),
        None => {
            eyre::bail!("No graph loaded — S3 checkpoint is required (DB graph loading is no longer supported)");
        }
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

/// Returns the maximum serial_id across all nodes in layer 0 of an in-memory
/// graph, or 0 if the graph is empty. Matches the semantics of
/// `GraphOps::get_max_serial_id`, which returns 0 when the links table is empty.
fn graph_mem_max_serial_id(graph: &GraphMem) -> i64 {
    graph
        .layers
        .first()
        .and_then(|layer| layer.get_links_map().keys().copied().max())
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
    degree_buckets: &[DegreeBucketEntry],
    matrix_entries: &[MatrixEntry],
    hop_buckets: &[HopBucketEntry],
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

    let p = dir.join("degree_by_bucket.csv");
    {
        let mut wtr = csv::Writer::from_path(&p)?;
        for entry in degree_buckets {
            wtr.serialize(entry)?;
        }
        wtr.flush()?;
    }
    println!("Wrote {}", p.display());
    files.push(p);

    let p = dir.join("neighbor_serial_matrix.csv");
    {
        let mut wtr = csv::Writer::from_path(&p)?;
        for entry in matrix_entries {
            wtr.serialize(entry)?;
        }
        wtr.flush()?;
    }
    println!("Wrote {}", p.display());
    files.push(p);

    if !hop_buckets.is_empty() {
        let p = dir.join("hops_by_bucket.csv");
        {
            let mut wtr = csv::Writer::from_path(&p)?;
            for entry in hop_buckets {
                wtr.serialize(entry)?;
            }
            wtr.flush()?;
        }
        println!("Wrote {}", p.display());
        files.push(p);
    }

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
