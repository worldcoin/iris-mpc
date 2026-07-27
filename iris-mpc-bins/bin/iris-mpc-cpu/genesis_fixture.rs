//! Dev-cluster fixture tool for the legacy (V4 → V5) migration rehearsal.
//!
//! Builds and resets the rehearsal fixture: an isolated source/destination
//! schema pair cloned from a template schema, a damaged V4 base checkpoint
//! pinned by hash, and the per-party store damage the version-join is supposed
//! to repair. Mirrors the `genesis_111` e2e fixture at dev-cluster scale.
//!
//! Runs against the three party databases over port-forwards, so fixture
//! iteration needs no image rebuild. All S3 work targets the checkpoint bucket
//! the genesis run is configured with.

#![recursion_limit = "256"]

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::PathBuf,
};

use aws_config::Region;
use aws_sdk_s3::{primitives::ByteStream, Client as S3Client};
use clap::{Parser, Subcommand};
use eyre::{ensure, eyre, Result};
use iris_mpc_common::{
    iris_db::get_dummy_shares_for_deletion,
    postgres::{run_migrations, AccessMode, PostgresClient},
};
use iris_mpc_cpu::{
    graph_checkpoint::{
        download_graph_checkpoint, download_graph_checkpoint_pruned, GraphCheckpointState,
    },
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::{GraphCheckpointRow, GraphPg},
    utils::serialization::{
        graph::LegacyPruneContext,
        types::graph_v4::{self, GraphV4},
    },
};
use iris_mpc_store::Store;
use serde::{Deserialize, Serialize};

/// One snapshotted row: `(serial, existed, version, left/right code & mask)`.
type BackupRow = (
    i64,
    bool,
    Option<i16>,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
);

/// Table (in every schema this tool mutates) holding pre-damage row state.
const BACKUP_TABLE: &str = "fixture_row_backup";

/// Rows copied above the base height so the pinned reset has a tail to trim.
const STALE_TAIL_ROWS: u32 = 10;

/// Modification id of the junk WAL row the pinned reset must clear.
const JUNK_WAL_MOD_ID: i64 = 999_999;

#[derive(Parser)]
#[clap(
    about = "Fixture builder for the dev-cluster V4→V5 migration rehearsal",
    long_about = None
)]
struct Args {
    /// Per-party Postgres URLs, in party order, comma separated.
    #[clap(long, value_delimiter = ',')]
    db_urls: Vec<String>,

    /// `SMPC__SCHEMA_NAME`: the shared schema-name prefix.
    #[clap(long, default_value = "SMPC")]
    schema_prefix: String,

    /// `SMPC__ENVIRONMENT`.
    #[clap(long, default_value = "dev")]
    environment: String,

    /// `SMPC__GPU_SCHEMA_NAME_SUFFIX`: the genesis source store.
    #[clap(long, default_value = "_v5reh_src")]
    src_suffix: String,

    /// `SMPC__HNSW_SCHEMA_NAME_SUFFIX`: the genesis destination store.
    #[clap(long, default_value = "_v5reh_dst")]
    dst_suffix: String,

    /// `SMPC__GRAPH_CHECKPOINT_BUCKET_NAME`.
    #[clap(long, default_value = "wf-smpcv2-dev-hnsw-performance-reports")]
    bucket: String,

    /// `SMPC__GRAPH_CHECKPOINT_BUCKET_REGION`.
    #[clap(long, default_value = "eu-central-1")]
    region: String,

    /// Fixture manifest: written by `bake-base`, read by every other command.
    #[clap(long, default_value = "fixture-manifest.json")]
    manifest: PathBuf,

    /// Print the actions instead of performing them (where supported).
    #[clap(long)]
    dry_run: bool,

    #[clap(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create and migrate the schema pair, then clone iris rows into it.
    Provision {
        /// Suffix of the schema to clone rows from (read-only).
        #[clap(long)]
        template_suffix: String,
        /// Highest serial cloned into the source store.
        #[clap(long)]
        src_max: u32,
        /// Highest serial cloned into the destination store (= base height).
        #[clap(long)]
        dst_max: u32,
        /// Rows per INSERT..SELECT chunk.
        #[clap(long, default_value = "5000")]
        chunk: u32,
    },

    /// Bake a damaged V4 base from an existing checkpoint object and pin it.
    BakeBase {
        /// Bucket holding the pristine base object.
        #[clap(long)]
        source_bucket: String,
        /// Key of the pristine base object.
        #[clap(long)]
        source_key: String,
        /// Expected blake3 of the pristine base object (hex).
        #[clap(long)]
        source_hash: String,
        /// Key the baked base is uploaded to.
        #[clap(long)]
        out_key: String,
        /// Base height the checkpoint row records.
        #[clap(long)]
        base_height: u32,
        /// Modification cursor the checkpoint row records.
        #[clap(long, default_value = "0")]
        base_mod_id: i64,
        #[clap(long, default_value = "20")]
        ghosts: usize,
        #[clap(long, default_value = "20")]
        self_loops: usize,
        #[clap(long, default_value = "20")]
        tombstones: usize,
        #[clap(long, default_value = "20")]
        stale: usize,
        #[clap(long, default_value = "20")]
        row_loss: usize,
        /// Party whose destination rows are deleted (cross-party union arm).
        #[clap(long, default_value = "0")]
        row_loss_party: usize,
    },

    /// Load the pinned base through prune-at-read and check the damage classes.
    Precheck {
        /// Parties to decode on (the object is shared, so one is usually enough).
        #[clap(long, value_delimiter = ',', default_value = "0")]
        parties: Vec<usize>,
    },

    /// Apply the per-party store damage, backing up every touched row first.
    DamageRows,

    /// Undo an attempt: restore rows, clear derived state, re-apply damage.
    Reset,

    /// Post-run assertions across all parties.
    Verify {
        /// Indexation target the run was given.
        #[clap(long)]
        target: u32,
        /// Party whose final checkpoint is decoded.
        #[clap(long, default_value = "0")]
        graph_party: usize,
    },

    /// Print schema, row, cursor and checkpoint state per party.
    Status,
}

/// Fixture description: everything a later command needs to know.
#[derive(Serialize, Deserialize, Debug, Default)]
struct Manifest {
    base_height: u32,
    base_mod_id: i64,
    base_hash: String,
    base_key: String,
    src_max: u32,
    dst_max: u32,
    row_loss_party: usize,
    /// Multi-version ghosts: force-included, join-invisible.
    ghost: Vec<u32>,
    /// Self-loop-only nodes: force-included, join-invisible.
    self_loop: Vec<u32>,
    /// Deletion-list serials left in the base graph.
    tombstone: Vec<u32>,
    /// Source rows ahead of the destination (version axis).
    stale: Vec<u32>,
    /// Destination rows missing on `row_loss_party` (row axis).
    row_loss: Vec<u32>,
}

impl Manifest {
    /// Serials only the force-include path can reach.
    fn forced(&self) -> Vec<u32> {
        let mut v = self.ghost.clone();
        v.extend(&self.self_loop);
        v.sort_unstable();
        v
    }

    fn load(path: &PathBuf) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| eyre!("cannot read manifest {}: {e}", path.display()))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn store(&self, path: &PathBuf) -> Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// One party's handles. Both stores live on the same cluster on dev.
struct Party {
    id: usize,
    src_schema: String,
    dst_schema: String,
    src: Store,
    dst: Store,
    graph: GraphPg<PlaintextStore>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    ensure!(
        args.db_urls.len() == 3,
        "--db-urls must list exactly three party URLs"
    );

    match &args.cmd {
        Cmd::Provision {
            template_suffix,
            src_max,
            dst_max,
            chunk,
        } => {
            provision(&args, template_suffix, *src_max, *dst_max, *chunk).await?;
        }
        Cmd::BakeBase { .. } => bake_base(&args).await?,
        Cmd::Precheck { parties } => precheck(&args, parties).await?,
        Cmd::DamageRows => {
            let manifest = Manifest::load(&args.manifest)?;
            for party in 0..3 {
                let p = open_party(&args, party, false).await?;
                damage_rows(&p, &manifest, args.dry_run).await?;
            }
        }
        Cmd::Reset => {
            let manifest = Manifest::load(&args.manifest)?;
            reset(&args, &manifest).await?;
        }
        Cmd::Verify {
            target,
            graph_party,
        } => {
            let manifest = Manifest::load(&args.manifest)?;
            verify(&args, &manifest, *target, *graph_party).await?;
        }
        Cmd::Status => status(&args).await?,
    }
    Ok(())
}

/* ------------------------------ plumbing ------------------------------ */

fn schema_name(args: &Args, suffix: &str, party: usize) -> String {
    format!(
        "{}{}_{}_{}",
        args.schema_prefix, suffix, args.environment, party
    )
}

/// Reject anything that would need quoting beyond `"..."`.
fn check_ident(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "unsafe schema name: {name}"
    );
    Ok(())
}

async fn s3_client(args: &Args) -> S3Client {
    let sdk = aws_config::from_env()
        .region(Region::new(args.region.clone()))
        .load()
        .await;
    S3Client::new(&sdk)
}

/// Open one party's stores.
///
/// # Errors
/// Bails unless both schemas already exist, so a mistyped suffix cannot
/// silently create an empty schema (`AccessMode::ReadWrite` would).
async fn open_party(args: &Args, party: usize, allow_create: bool) -> Result<Party> {
    let src_schema = schema_name(args, &args.src_suffix, party);
    let dst_schema = schema_name(args, &args.dst_suffix, party);
    check_ident(&src_schema)?;
    check_ident(&dst_schema)?;

    if !allow_create {
        // A throwaway read-only client on `public` to probe the catalog.
        let probe =
            PostgresClient::new(&args.db_urls[party], "public", AccessMode::ReadOnly).await?;
        for schema in [&src_schema, &dst_schema] {
            let exists: (bool,) =
                sqlx::query_as("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)")
                    .bind(schema)
                    .fetch_one(&probe.pool)
                    .await?;
            ensure!(exists.0, "schema {schema} does not exist — run `provision`");
        }
    }

    let src_pg =
        PostgresClient::new(&args.db_urls[party], &src_schema, AccessMode::ReadWrite).await?;
    let dst_pg =
        PostgresClient::new(&args.db_urls[party], &dst_schema, AccessMode::ReadWrite).await?;
    let src = Store::new(&src_pg).await?;
    let dst = Store::new(&dst_pg).await?;
    let graph = GraphPg::<PlaintextStore>::new(&dst_pg).await?;

    Ok(Party {
        id: party,
        src_schema,
        dst_schema,
        src,
        dst,
        graph,
    })
}

/* ------------------------------ provision ----------------------------- */

async fn provision(
    args: &Args,
    template_suffix: &str,
    src_max: u32,
    dst_max: u32,
    chunk: u32,
) -> Result<()> {
    ensure!(dst_max <= src_max, "--dst-max must not exceed --src-max");

    // One task per party: the three clusters are independent, and a single
    // party's copy is bounded by that cluster's write throughput.
    let mut set = tokio::task::JoinSet::new();
    for party in 0..3 {
        let template = schema_name(args, template_suffix, party);
        let src_schema = schema_name(args, &args.src_suffix, party);
        let dst_schema = schema_name(args, &args.dst_suffix, party);
        check_ident(&template)?;
        check_ident(&src_schema)?;
        check_ident(&dst_schema)?;
        tracing::info!("party {party}: template {template} -> {src_schema} / {dst_schema}");
        if args.dry_run {
            continue;
        }
        let url = args.db_urls[party].clone();
        set.spawn(async move {
            provision_party(party, url, template, src_schema, dst_schema, src_max, dst_max, chunk)
                .await
        });
    }
    while let Some(joined) = set.join_next().await {
        joined??;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn provision_party(
    party: usize,
    url: String,
    template: String,
    src_schema: String,
    dst_schema: String,
    src_max: u32,
    dst_max: u32,
    chunk: u32,
) -> Result<()> {
    // ReadWrite creates the schema; run_migrations brings it to head
    // (including the explicit-version trigger the repair path needs).
    for schema in [&src_schema, &dst_schema] {
        let pg = PostgresClient::new(&url, schema, AccessMode::ReadWrite).await?;
        run_migrations(&pg.pool, false).await?;
        tracing::info!("party {party}: {schema} migrated");
    }

    let pg = PostgresClient::new(&url, "public", AccessMode::ReadOnly).await?;
    let copied_src = copy_irises(&pg.pool, &template, &src_schema, src_max, chunk).await?;
    let copied_dst = copy_irises(&pg.pool, &template, &dst_schema, dst_max, chunk).await?;
    tracing::info!("party {party}: cloned {copied_src} source rows, {copied_dst} destination rows");

    // Cursors describe the base the pinned run resets to.
    let dst_pg = PostgresClient::new(&url, &dst_schema, AccessMode::ReadWrite).await?;
    let graph = GraphPg::<PlaintextStore>::new(&dst_pg).await?;
    let mut tx = graph.tx().await?.tx;
    GraphPg::<PlaintextStore>::set_persistent_state(
        &mut tx,
        "genesis",
        "last_indexed_iris_id",
        &dst_max,
    )
    .await?;
    GraphPg::<PlaintextStore>::set_persistent_state(
        &mut tx,
        "genesis",
        "last_indexed_modification_id",
        &0i64,
    )
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Columns present in both schemas' `irises` table, in the destination's order.
async fn common_columns(
    pool: &sqlx::PgPool,
    from_schema: &str,
    to_schema: &str,
) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT a.column_name FROM information_schema.columns a \
         JOIN information_schema.columns b \
           ON b.column_name = a.column_name AND b.table_schema = $2 AND b.table_name = 'irises' \
         WHERE a.table_schema = $1 AND a.table_name = 'irises' \
         ORDER BY a.ordinal_position",
    )
    .bind(to_schema)
    .bind(from_schema)
    .fetch_all(pool)
    .await?;
    ensure!(
        !rows.is_empty(),
        "no shared irises columns between {from_schema} and {to_schema}"
    );
    Ok(rows.into_iter().map(|(c,)| c).collect())
}

/// Server-side chunked copy of `irises` rows `1..=max_id`. Idempotent.
async fn copy_irises(
    pool: &sqlx::PgPool,
    from_schema: &str,
    to_schema: &str,
    max_id: u32,
    chunk: u32,
) -> Result<u64> {
    let cols = common_columns(pool, from_schema, to_schema).await?;
    let col_list = cols
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let mut total = 0u64;
    let mut lo = 1u32;
    while lo <= max_id {
        let hi = (lo + chunk - 1).min(max_id);
        let sql = format!(
            "INSERT INTO \"{to_schema}\".irises ({col_list}) \
             SELECT {col_list} FROM \"{from_schema}\".irises \
             WHERE id >= $1 AND id <= $2 \
             ON CONFLICT (id) DO NOTHING"
        );
        let done = sqlx::query(&sql)
            .bind(lo as i64)
            .bind(hi as i64)
            .execute(pool)
            .await?
            .rows_affected();
        total += done;
        tracing::info!("{to_schema}: copied through serial {hi} ({total} rows)");
        lo = hi + 1;
    }
    Ok(total)
}

/* ------------------------------ bake-base ----------------------------- */

async fn bake_base(args: &Args) -> Result<()> {
    let Cmd::BakeBase {
        source_bucket,
        source_key,
        source_hash,
        out_key,
        base_height,
        base_mod_id,
        ghosts,
        self_loops,
        tombstones,
        stale,
        row_loss,
        row_loss_party,
    } = &args.cmd
    else {
        unreachable!("bake_base called for another subcommand")
    };
    ensure!(*row_loss_party < 3, "--row-loss-party must be 0, 1 or 2");

    let s3 = s3_client(args).await;
    tracing::info!("downloading pristine base s3://{source_bucket}/{source_key}");
    let body = s3
        .get_object()
        .bucket(source_bucket)
        .key(source_key)
        .send()
        .await
        .map_err(|e| eyre!("cannot read pristine base: {e}"))?
        .body
        .collect()
        .await?
        .into_bytes();
    let got = blake3::hash(&body).to_hex().to_string();
    ensure!(
        got == *source_hash,
        "pristine base hash mismatch: expected {source_hash}, got {got}"
    );
    tracing::info!("pristine base verified ({} bytes)", body.len());

    let mut pair: [GraphV4; 2] = bincode::deserialize(&body)?;
    drop(body);

    // The row's height defines the fixture's serial range; anything above it is
    // not backed by cloned iris rows, so it must not stay in the graph.
    for (eye, g) in pair.iter_mut().enumerate() {
        let (nodes, edges) = restrict_to_prefix(g, *base_height);
        tracing::info!(
            "eye {eye}: restricted to serials <= {base_height} (dropped {nodes} nodes, {edges} edges); \
             bottom layer now {} nodes, {} layers",
            g.layers[0].links.len(),
            g.layers.len()
        );
    }

    let plan = pick_serials(
        &pair[0],
        &[*ghosts, *self_loops, *tombstones, *stale, *row_loss],
    )?;
    let mut manifest = Manifest {
        base_height: *base_height,
        base_mod_id: *base_mod_id,
        base_hash: String::new(),
        base_key: out_key.clone(),
        src_max: 0,
        dst_max: *base_height,
        row_loss_party: *row_loss_party,
        ghost: plan[0].clone(),
        self_loop: plan[1].clone(),
        tombstone: plan[2].clone(),
        stale: plan[3].clone(),
        row_loss: plan[4].clone(),
    };

    for (eye, g) in pair.iter_mut().enumerate() {
        bake_damage(g, &manifest)?;
        tracing::info!("eye {eye}: damage baked");
    }

    let bytes = bincode::serialize(&pair)?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    manifest.base_hash = hash.clone();
    tracing::info!(
        "baked V4 base: {} bytes, blake3 {hash}, key {out_key}",
        bytes.len()
    );

    if args.dry_run {
        tracing::info!("dry run: not uploading, not inserting rows");
        manifest.store(&args.manifest)?;
        return Ok(());
    }

    s3.put_object()
        .bucket(&args.bucket)
        .key(out_key)
        .body(ByteStream::from(bytes))
        .send()
        .await
        .map_err(|e| eyre!("cannot upload baked base: {e}"))?;

    for party in 0..3 {
        let p = open_party(args, party, false).await?;
        if p.graph
            .get_genesis_graph_checkpoint_by_hash(&hash)
            .await?
            .is_some()
        {
            tracing::info!("party {party}: base row already present");
            continue;
        }
        let mut tx = p.graph.tx().await?.tx;
        GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
            &mut tx,
            out_key,
            *base_height as i64,
            *base_mod_id,
            None,
            &hash,
            true,
            4,
        )
        .await?;
        tx.commit().await?;
        tracing::info!("party {party}: pinned base row inserted");
    }

    manifest.store(&args.manifest)?;
    tracing::info!("manifest written to {}", args.manifest.display());
    Ok(())
}

/// Restrict a V4 graph to serials `1..=max_serial`, dropping out-of-range nodes
/// and every edge that pointed at one.
///
/// Lets the rehearsal run a smaller, self-consistent slice of a large base: the
/// prune then has nothing to report beyond the fixture's own damage. `set_hash`
/// is left stale deliberately — legacy bases are only ever read through the
/// prune, which rebuilds the hashes from the kept keys.
fn restrict_to_prefix(g: &mut GraphV4, max_serial: u32) -> (usize, usize) {
    let mut dropped_nodes = 0usize;
    let mut dropped_edges = 0usize;
    for layer in g.layers.iter_mut() {
        let before = layer.links.len();
        layer.links.retain(|k, _| k.id <= max_serial);
        dropped_nodes += before - layer.links.len();
        for edges in layer.links.values_mut() {
            let before = edges.0.len();
            edges.0.retain(|t| t.id <= max_serial);
            dropped_edges += before - edges.0.len();
        }
    }
    while g.layers.len() > 1 && g.layers.last().is_some_and(|l| l.links.is_empty()) {
        g.layers.pop();
    }
    g.entry_points.retain(|e| e.point.id <= max_serial && e.layer < g.layers.len());
    if g.entry_points.is_empty() {
        // Every declared entry point was out of range: promote a surviving node
        // from the highest non-empty layer.
        let layer = g.layers.len() - 1;
        if let Some(point) = g.layers[layer].links.keys().min().copied() {
            g.entry_points.push(graph_v4::EntryPoint { point, layer });
        }
    }
    (dropped_nodes, dropped_edges)
}

/// Pick disjoint serial sets, evenly spread over the bottom layer.
///
/// Candidates are bottom-layer-only, non-entry-point nodes with a single graph
/// key, so the fixture's own damage is the only thing the prune can report.
fn pick_serials(g: &GraphV4, counts: &[usize]) -> Result<Vec<Vec<u32>>> {
    let entry: HashSet<u32> = g.entry_points.iter().map(|e| e.point.id).collect();
    let mut upper: HashSet<u32> = HashSet::new();
    for layer in g.layers.iter().skip(1) {
        upper.extend(layer.links.keys().map(|k| k.id));
    }
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for key in g.layers[0].links.keys() {
        *seen.entry(key.id).or_default() += 1;
    }
    let mut candidates: Vec<u32> = seen
        .iter()
        .filter(|(serial, keys)| **keys == 1 && !entry.contains(serial) && !upper.contains(serial))
        .map(|(serial, _)| *serial)
        .collect();
    candidates.sort_unstable();

    let total: usize = counts.iter().sum();
    ensure!(
        candidates.len() > total * 4,
        "too few clean bottom-layer candidates ({}) for {total} damage serials",
        candidates.len()
    );

    // Even spread, deterministic: no RNG, so every party and every rerun of
    // bake-base produces the identical fixture.
    let stride = candidates.len() / (total + 1);
    let picks: Vec<u32> = (1..=total).map(|i| candidates[i * stride]).collect();

    let mut out = Vec::new();
    let mut it = picks.into_iter();
    for n in counts {
        let mut set: Vec<u32> = it.by_ref().take(*n).collect();
        set.sort_unstable();
        out.push(set);
    }
    Ok(out)
}

/// Rewrite one eye's V4 graph with the ghost and self-loop damage.
///
/// Tombstone, stale and row-loss classes are store-side only: their graph
/// entries must stay pristine or their axes would not be what the run reports.
fn bake_damage(g: &mut GraphV4, manifest: &Manifest) -> Result<()> {
    let l0 = &mut g.layers[0].links;
    let live: HashMap<u32, graph_v4::VectorId> = l0.keys().map(|k| (k.id, *k)).collect();

    // Any surviving key: the ghost's dangling edge target must exist so the
    // only reported anomaly is the multi-version key itself.
    let anchor = *live
        .get(manifest.tombstone.first().unwrap_or(&0))
        .or_else(|| l0.keys().next())
        .ok_or_else(|| eyre!("empty bottom layer"))?;

    for serial in &manifest.ghost {
        let key = *live
            .get(serial)
            .ok_or_else(|| eyre!("ghost serial {serial} absent from the bottom layer"))?;
        l0.get_mut(&key).expect("key just resolved").0.clear();
        l0.insert(
            graph_v4::VectorId {
                id: key.id,
                version: key.version + 1,
            },
            graph_v4::EdgeIds(vec![anchor]),
        );
    }

    for serial in &manifest.self_loop {
        let key = *live
            .get(serial)
            .ok_or_else(|| eyre!("self-loop serial {serial} absent from the bottom layer"))?;
        l0.get_mut(&key).expect("key just resolved").0 = vec![key];
    }
    Ok(())
}

/* ------------------------------- precheck ----------------------------- */

async fn precheck(args: &Args, parties: &[usize]) -> Result<()> {
    let manifest = Manifest::load(&args.manifest)?;
    let s3 = s3_client(args).await;

    // The pin must resolve identically on every party before decoding anything.
    for party in 0..3 {
        let p = open_party(args, party, false).await?;
        let row = p
            .graph
            .get_genesis_graph_checkpoint_by_hash(&manifest.base_hash)
            .await?
            .ok_or_else(|| eyre!("party {party}: pinned base row missing"))?;
        ensure!(row.graph_version == 4, "party {party}: base row is not V4");
        ensure!(row.is_archival, "party {party}: base row is not archival");
        ensure!(
            row.last_indexed_iris_id == manifest.base_height as i64,
            "party {party}: base row height {} != {}",
            row.last_indexed_iris_id,
            manifest.base_height
        );
        tracing::info!("party {party}: pinned base row ok (row id {})", row.id);
    }

    let ghost: BTreeSet<u32> = manifest.ghost.iter().copied().collect();
    let self_loop: BTreeSet<u32> = manifest.self_loop.iter().copied().collect();
    let deleted: HashSet<u32> = manifest.tombstone.iter().copied().collect();

    for &party in parties {
        let p = open_party(args, party, false).await?;
        let state = base_state(&p, &manifest).await?;
        let version_map = version_map(&p.src, manifest.base_height).await?;
        ensure!(
            version_map.len() == manifest.base_height as usize,
            "party {party}: source covers {} of {} serials",
            version_map.len(),
            manifest.base_height
        );

        let (graphs, reports) = download_graph_checkpoint_pruned(
            &s3,
            &args.bucket,
            &state,
            Some(LegacyPruneContext {
                version_map,
                deleted: deleted.clone(),
            }),
        )
        .await?;
        let reports = reports.ok_or_else(|| eyre!("legacy load emitted no prune reports"))?;

        for (eye, (graph, report)) in graphs.iter().zip(reports.iter()).enumerate() {
            tracing::info!(
                "party {party} eye {eye}: multi_version={} self_loop={} \
                 dropped(deleted={}, stale={}) edges_self_loop={} zero_out={} zero_in={}",
                report.multi_version_serials.len(),
                report.self_loop_serials.len(),
                report.nodes_dropped_deleted,
                report.nodes_dropped_stale,
                report.edges_dropped_self_loop,
                report.zero_out_degree.len(),
                report.zero_in_degree.len(),
            );
            ensure!(
                report.multi_version_serials == ghost,
                "party {party} eye {eye}: multi-version class mismatch"
            );
            ensure!(
                report.self_loop_serials == self_loop,
                "party {party} eye {eye}: self-loop class mismatch"
            );
            ensure!(
                report.nodes_dropped_deleted as usize >= manifest.tombstone.len(),
                "party {party} eye {eye}: tombstones not dropped"
            );
            for serial in &manifest.tombstone {
                ensure!(
                    graph.vector_id_of(*serial).is_none(),
                    "party {party} eye {eye}: tombstone {serial} survived the prune"
                );
            }
            for serial in manifest.forced() {
                ensure!(
                    graph.vector_id_of(serial).is_some(),
                    "party {party} eye {eye}: forced serial {serial} lost by the prune"
                );
            }
        }
    }
    tracing::info!("precheck passed");
    Ok(())
}

/// The pinned base as a checkpoint state.
async fn base_state(p: &Party, manifest: &Manifest) -> Result<GraphCheckpointState> {
    let row = p
        .graph
        .get_genesis_graph_checkpoint_by_hash(&manifest.base_hash)
        .await?
        .ok_or_else(|| eyre!("party {}: pinned base row missing", p.id))?;
    row.try_into()
}

/// `serial → version_id` for `1..=max_id` from a store.
async fn version_map(store: &Store, max_id: u32) -> Result<HashMap<u32, i16>> {
    let rows: Vec<(i64, i16)> =
        sqlx::query_as("SELECT id, version_id FROM irises WHERE id <= $1 ORDER BY id")
            .bind(max_id as i64)
            .fetch_all(&store.pool)
            .await?;
    Ok(rows.into_iter().map(|(id, v)| (id as u32, v)).collect())
}

/* ----------------------------- damage rows ---------------------------- */

async fn ensure_backup_table(pool: &sqlx::PgPool, schema: &str) -> Result<()> {
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS \"{schema}\".{BACKUP_TABLE} (
             serial      bigint PRIMARY KEY,
             kind        text NOT NULL,
             existed     boolean NOT NULL,
             version_id  smallint,
             left_code   bytea,
             left_mask   bytea,
             right_code  bytea,
             right_mask  bytea
         )"
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

/// Snapshot a row before mutating it. Never overwrites an existing snapshot.
async fn backup_row(pool: &sqlx::PgPool, schema: &str, serial: u32, kind: &str) -> Result<()> {
    let sql = format!(
        "INSERT INTO \"{schema}\".{BACKUP_TABLE} \
             (serial, kind, existed, version_id, left_code, left_mask, right_code, right_mask) \
         SELECT $1, $2, true, version_id, left_code, left_mask, right_code, right_mask \
         FROM \"{schema}\".irises WHERE id = $1 \
         ON CONFLICT (serial) DO NOTHING"
    );
    let affected = sqlx::query(&sql)
        .bind(serial as i64)
        .bind(kind)
        .execute(pool)
        .await?
        .rows_affected();
    if affected == 0 {
        // Either already snapshotted, or the row is absent: record absence so
        // restore knows to delete rather than resurrect.
        let sql = format!(
            "INSERT INTO \"{schema}\".{BACKUP_TABLE} (serial, kind, existed) \
             VALUES ($1, $2, false) ON CONFLICT (serial) DO NOTHING"
        );
        sqlx::query(&sql)
            .bind(serial as i64)
            .bind(kind)
            .execute(pool)
            .await?;
    }
    Ok(())
}

async fn damage_rows(p: &Party, manifest: &Manifest, dry_run: bool) -> Result<()> {
    if dry_run {
        tracing::info!(
            "party {}: would damage {} source rows and {} destination rows",
            p.id,
            manifest.tombstone.len() + manifest.stale.len(),
            if p.id == manifest.row_loss_party {
                manifest.row_loss.len()
            } else {
                0
            }
        );
        return Ok(());
    }

    ensure_backup_table(&p.src.pool, &p.src_schema).await?;
    ensure_backup_table(&p.dst.pool, &p.dst_schema).await?;

    // Tombstones: the source content becomes this party's dummy shares (the
    // trigger bumps the version); the deletion list carries the serial.
    let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(p.id);
    for serial in &manifest.tombstone {
        backup_row(&p.src.pool, &p.src_schema, *serial, "tombstone").await?;
        p.src
            .update_iris(
                None,
                *serial as i64,
                &dummy_code,
                &dummy_mask,
                &dummy_code,
                &dummy_mask,
            )
            .await?;
    }

    // Stale: nudge the source content so the trigger moves the source version
    // ahead of the destination row and the base graph key.
    for serial in &manifest.stale {
        backup_row(&p.src.pool, &p.src_schema, *serial, "stale").await?;
        sqlx::query(
            "UPDATE irises SET left_code = set_byte(left_code, 0, (get_byte(left_code, 0) + 1) % 256) \
             WHERE id = $1",
        )
        .bind(*serial as i64)
        .execute(&p.src.pool)
        .await?;
    }

    // Row loss on a single party: the cross-party union must still bring every
    // party to the same surgery list.
    if p.id == manifest.row_loss_party {
        for serial in &manifest.row_loss {
            backup_row(&p.dst.pool, &p.dst_schema, *serial, "row_loss").await?;
            sqlx::query("DELETE FROM irises WHERE id = $1")
                .bind(*serial as i64)
                .execute(&p.dst.pool)
                .await?;
        }
    }

    // A destination tail above the base height, plus a junk WAL row: both must
    // be gone after the pinned reset.
    let tail_from = manifest.base_height + 1;
    let tail_to = manifest.base_height + STALE_TAIL_ROWS;
    let sql = format!(
        "INSERT INTO \"{}\".irises (id, left_code, left_mask, right_code, right_mask, version_id) \
         SELECT id, left_code, left_mask, right_code, right_mask, version_id \
         FROM \"{}\".irises WHERE id >= $1 AND id <= $2 ON CONFLICT (id) DO NOTHING",
        p.dst_schema, p.src_schema
    );
    sqlx::query(&sql)
        .bind(tail_from as i64)
        .bind(tail_to as i64)
        .execute(&p.dst.pool)
        .await?;

    sqlx::query(
        "INSERT INTO hawk_graph_mutations (modification_id, serialized_mutations, mutation_format_version) \
         VALUES ($1, $2, $3) ON CONFLICT (modification_id) DO NOTHING",
    )
    .bind(JUNK_WAL_MOD_ID)
    .bind(vec![0u8, 1, 2])
    .bind(1i16)
    .execute(p.graph.pool())
    .await?;

    tracing::info!(
        "party {}: damage applied (tombstone {}, stale {}, row_loss {})",
        p.id,
        manifest.tombstone.len(),
        manifest.stale.len(),
        if p.id == manifest.row_loss_party {
            manifest.row_loss.len()
        } else {
            0
        }
    );
    Ok(())
}

/* -------------------------------- reset ------------------------------- */

/// Restore every snapshotted row in `schema`, then drop the snapshots.
async fn restore_rows(pool: &sqlx::PgPool, schema: &str) -> Result<u64> {
    let exists: (bool,) = sqlx::query_as(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables \
         WHERE table_schema = $1 AND table_name = $2)",
    )
    .bind(schema)
    .bind(BACKUP_TABLE)
    .fetch_one(pool)
    .await?;
    if !exists.0 {
        return Ok(0);
    }

    let rows: Vec<BackupRow> = sqlx::query_as(&format!(
        "SELECT serial, existed, version_id, left_code, left_mask, right_code, right_mask \
             FROM \"{schema}\".{BACKUP_TABLE} ORDER BY serial"
    ))
    .fetch_all(pool)
    .await?;

    let mut restored = 0u64;
    for (serial, existed, version_id, lc, lm, rc, rm) in &rows {
        let mut tx = pool.begin().await?;
        // The trigger rejects a hand-set version_id without this flag.
        sqlx::query("SET LOCAL \"app.explicit_version_id\" = 'on'")
            .execute(&mut *tx)
            .await?;
        if *existed {
            let sql = format!(
                "INSERT INTO \"{schema}\".irises \
                     (id, left_code, left_mask, right_code, right_mask, version_id) \
                 VALUES ($1, $2, $3, $4, $5, $6) \
                 ON CONFLICT (id) DO UPDATE SET \
                     left_code = EXCLUDED.left_code, left_mask = EXCLUDED.left_mask, \
                     right_code = EXCLUDED.right_code, right_mask = EXCLUDED.right_mask, \
                     version_id = EXCLUDED.version_id"
            );
            sqlx::query(&sql)
                .bind(serial)
                .bind(lc)
                .bind(lm)
                .bind(rc)
                .bind(rm)
                .bind(version_id)
                .execute(&mut *tx)
                .await?;
        } else {
            sqlx::query(&format!("DELETE FROM \"{schema}\".irises WHERE id = $1"))
                .bind(serial)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        restored += 1;
    }

    sqlx::query(&format!("TRUNCATE \"{schema}\".{BACKUP_TABLE}"))
        .execute(pool)
        .await?;
    Ok(restored)
}

async fn reset(args: &Args, manifest: &Manifest) -> Result<()> {
    let s3 = s3_client(args).await;

    for party in 0..3 {
        let p = open_party(args, party, false).await?;
        if args.dry_run {
            tracing::info!("party {party}: would restore rows and clear derived state");
            continue;
        }

        let src_restored = restore_rows(&p.src.pool, &p.src_schema).await?;
        let dst_restored = restore_rows(&p.dst.pool, &p.dst_schema).await?;

        // Everything the run derives: rows above the base, WAL, modifications,
        // cursors, and checkpoints newer than the pin (with their objects).
        sqlx::query("DELETE FROM irises WHERE id > $1")
            .bind(manifest.base_height as i64)
            .execute(&p.dst.pool)
            .await?;
        sqlx::query("TRUNCATE hawk_graph_mutations")
            .execute(p.graph.pool())
            .await?;
        sqlx::query("TRUNCATE modifications")
            .execute(&p.dst.pool)
            .await?;

        let base_row = p
            .graph
            .get_genesis_graph_checkpoint_by_hash(&manifest.base_hash)
            .await?
            .ok_or_else(|| eyre!("party {party}: pinned base row missing"))?;
        let newer: Vec<GraphCheckpointRow> = p
            .graph
            .get_genesis_graph_checkpoints()
            .await?
            .into_iter()
            .filter(|r| r.id > base_row.id)
            .collect();
        for row in &newer {
            // Only ever delete objects this fixture's runs produced.
            if row.s3_key == manifest.base_key {
                continue;
            }
            match s3
                .delete_object()
                .bucket(&args.bucket)
                .key(&row.s3_key)
                .send()
                .await
            {
                Ok(_) => {
                    tracing::info!("party {party}: deleted s3://{}/{}", args.bucket, row.s3_key)
                }
                Err(e) => tracing::warn!("party {party}: could not delete {}: {e}", row.s3_key),
            }
        }
        let mut tx = p.graph.tx().await?;
        tx.delete_checkpoints_after_id(base_row.id).await?;
        let mut tx = tx.tx;
        GraphPg::<PlaintextStore>::set_persistent_state(
            &mut tx,
            "genesis",
            "last_indexed_iris_id",
            &manifest.base_height,
        )
        .await?;
        GraphPg::<PlaintextStore>::set_persistent_state(
            &mut tx,
            "genesis",
            "last_indexed_modification_id",
            &manifest.base_mod_id,
        )
        .await?;
        tx.commit().await?;

        tracing::info!(
            "party {party}: reset (restored {src_restored} source rows, {dst_restored} \
             destination rows, dropped {} newer checkpoints)",
            newer.len()
        );

        damage_rows(&p, manifest, false).await?;
    }
    Ok(())
}

/* -------------------------------- verify ------------------------------ */

async fn verify(args: &Args, manifest: &Manifest, target: u32, graph_party: usize) -> Result<()> {
    let s3 = s3_client(args).await;
    let mut final_hashes = Vec::new();

    for party in 0..3 {
        let p = open_party(args, party, false).await?;
        let rows = p.graph.get_genesis_graph_checkpoints().await?;
        ensure!(
            rows.len() >= 2,
            "party {party}: expected a post-delta and a final checkpoint"
        );
        ensure!(
            rows[0].last_indexed_iris_id == target as i64,
            "party {party}: newest checkpoint at {}, expected {target}",
            rows[0].last_indexed_iris_id
        );
        ensure!(
            rows[0].graph_version == 5,
            "party {party}: final checkpoint is not V5"
        );
        let post_delta = rows
            .iter()
            .find(|r| r.last_indexed_iris_id == manifest.base_height as i64 && r.graph_version == 5)
            .ok_or_else(|| {
                eyre!("party {party}: no V5 post-delta checkpoint at the base height")
            })?;
        ensure!(
            post_delta.is_archival,
            "party {party}: post-delta checkpoint is not archival"
        );
        ensure!(
            post_delta.blake3_hash != manifest.base_hash,
            "party {party}: post-delta checkpoint equals the V4 base"
        );
        final_hashes.push(rows[0].blake3_hash.clone());

        // Rows: destination must match the source on content and version.
        let src = row_digests(&p.src.pool, target).await?;
        let dst = row_digests(&p.dst.pool, target).await?;
        ensure!(
            src == dst,
            "party {party}: destination rows differ from source (first mismatch at {:?})",
            src.iter()
                .zip(dst.iter())
                .find(|(a, b)| a != b)
                .map(|(a, _)| a.0)
        );
        ensure!(
            dst.len() == target as usize,
            "party {party}: {} destination rows, expected {target}",
            dst.len()
        );

        let wal: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM hawk_graph_mutations")
            .fetch_one(p.graph.pool())
            .await?;
        ensure!(wal.0 == 0, "party {party}: WAL is not empty");
        tracing::info!("party {party}: rows, cursors and checkpoint rows ok");
    }
    ensure!(
        final_hashes.iter().all(|h| h == &final_hashes[0]),
        "final checkpoint hashes disagree: {final_hashes:?}"
    );
    tracing::info!(
        "final checkpoint hash agrees on all parties: {}",
        final_hashes[0]
    );

    // Graph-level assertions on one party (the checkpoints are byte-identical).
    let p = open_party(args, graph_party, false).await?;
    let rows = p.graph.get_genesis_graph_checkpoints().await?;
    let state: GraphCheckpointState = rows[0].clone().try_into()?;
    let graphs = download_graph_checkpoint(&s3, &args.bucket, &state, None).await?;
    let versions = version_map(&p.src, target).await?;

    for (eye, graph) in graphs.iter().enumerate() {
        let clock: BTreeSet<u32> = graph.node_init.keys().copied().collect();
        let layer0: BTreeSet<u32> = graph.layers[0].get_links_map().keys().copied().collect();
        ensure!(
            clock == layer0,
            "eye {eye}: content clock and bottom layer disagree"
        );
        for serial in manifest.forced() {
            let version = *versions
                .get(&serial)
                .ok_or_else(|| eyre!("serial {serial} absent from the source"))?;
            let vid = graph
                .vector_id_of(serial)
                .ok_or_else(|| eyre!("eye {eye}: forced serial {serial} missing"))?;
            ensure!(
                vid.version_id() == version,
                "eye {eye}: serial {serial} at version {}, source is {version}",
                vid.version_id()
            );
            let init = graph
                .node_init
                .get(&serial)
                .ok_or_else(|| eyre!("eye {eye}: no clock entry for {serial}"))?;
            ensure!(
                init.seq_no > 0,
                "eye {eye}: serial {serial} still carries the prune-seeded clock — not surged"
            );
            let links = graph.layers[0]
                .get_links(&serial)
                .ok_or_else(|| eyre!("eye {eye}: no bottom-layer node for {serial}"))?;
            let neighbors = links.neighbors();
            ensure!(
                !neighbors.is_empty(),
                "eye {eye}: serial {serial} was not re-linked"
            );
            ensure!(
                !neighbors.contains(&serial),
                "eye {eye}: serial {serial} kept a self-edge"
            );
        }
        for serial in &manifest.tombstone {
            ensure!(
                graph.vector_id_of(*serial).is_none(),
                "eye {eye}: tombstone {serial} is back in the graph"
            );
        }
        tracing::info!("eye {eye}: forced serials healed, tombstones absent");
    }
    tracing::info!("verify passed");
    Ok(())
}

/// `(id, version_id, md5 of the four share columns)` for `1..=max_id`.
async fn row_digests(pool: &sqlx::PgPool, max_id: u32) -> Result<Vec<(i64, i16, String)>> {
    Ok(sqlx::query_as(
        "SELECT id, version_id, \
         md5(left_code) || md5(left_mask) || md5(right_code) || md5(right_mask) \
         FROM irises WHERE id <= $1 ORDER BY id",
    )
    .bind(max_id as i64)
    .fetch_all(pool)
    .await?)
}

/* -------------------------------- status ------------------------------ */

async fn status(args: &Args) -> Result<()> {
    for party in 0..3 {
        let p = open_party(args, party, false).await?;
        let src: (Option<i64>, i64) = sqlx::query_as("SELECT MAX(id), COUNT(*) FROM irises")
            .fetch_one(&p.src.pool)
            .await?;
        let dst: (Option<i64>, i64) = sqlx::query_as("SELECT MAX(id), COUNT(*) FROM irises")
            .fetch_one(&p.dst.pool)
            .await?;
        let last_iris: Option<u32> = p
            .graph
            .get_persistent_state("genesis", "last_indexed_iris_id")
            .await?;
        let wal: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM hawk_graph_mutations")
            .fetch_one(p.graph.pool())
            .await?;
        let rows = p.graph.get_genesis_graph_checkpoints().await?;

        println!("party {party}");
        println!(
            "  source      {} max={:?} rows={}",
            p.src_schema, src.0, src.1
        );
        println!(
            "  destination {} max={:?} rows={}",
            p.dst_schema, dst.0, dst.1
        );
        println!(
            "  cursor last_indexed_iris_id={last_iris:?}  wal_rows={}",
            wal.0
        );
        for row in rows.iter().take(5) {
            println!(
                "  checkpoint id={} v{} height={} archival={} hash={}",
                row.id,
                row.graph_version,
                row.last_indexed_iris_id,
                row.is_archival,
                &row.blake3_hash[..12.min(row.blake3_hash.len())]
            );
        }
    }
    Ok(())
}
