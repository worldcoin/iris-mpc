//! Legacy V4 base migration: load through prune-at-read, force-include the
//! prune-detected damage classes into the surgery set, repair, index, and end
//! on native V5 checkpoints.
//!
//! Phase A builds a clean base. Phase B rewrites it as a damaged V4
//! checkpoint — a multi-version ghost key, a self-loop, and a tombstoned
//! serial still present in the graph — plus single-party row damage. The
//! ghost and self-loop serials are version- and row-clean after the prune, so
//! no join axis can reach them: only the force-include path heals them, which
//! is what the emptied neighborhoods assert. Phase C runs genesis pinned to
//! the V4 base and asserts routing, healing, and the V5 format transition.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use crate::join_runners;
use crate::run_genesis;
use crate::utils::{
    genesis_runner::{self, DEFAULT_GENESIS_ARGS},
    mpc_node::{db_ops, MpcNode},
    s3_deletions::{get_aws_clients, upload_iris_deletions},
    HawkConfigs, TestRun, TestRunContextInfo,
};
use aws_sdk_s3::primitives::ByteStream;
use eyre::{ensure, eyre, Result};
use iris_mpc_common::{config::Config, iris_db::get_dummy_shares_for_deletion, VectorId};
use iris_mpc_cpu::{
    graph_checkpoint::{
        download_graph_checkpoint, download_graph_checkpoint_pruned, get_latest_checkpoint_state,
    },
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::GraphPg,
    utils::serialization::{
        graph::LegacyPruneContext,
        types::graph_v4::{self, GraphV4},
    },
};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;
use tokio::task::JoinSet;

// Base graph height, final indexation height, and the size of the stale HNSW tail.
const BASE_HEIGHT: u32 = 80;
const MAX_HEIGHT: u32 = 100;
const TAIL_ROWS: usize = 10;

// Damage serials (all ≤ BASE_HEIGHT so they exist in the base graph).
//
// GH1 and SL1 are join-invisible on purpose: after the prune their surviving
// key sits at the source-current version with clean rows, so only the
// force-included prune-report classes can route them to surgery.
const GH1: u32 = 20; // ghost: straggler key at a stale version; live key's edges emptied
const SL1: u32 = 40; // self-loop at the live version, as the node's only edge
const TB1: u32 = 55; // tombstone: S3-listed deletion, dummy source content, key kept in base
const KP1: u32 = 30; // HNSW row deleted on party 0 only (cross-party union)

pub struct Test {
    configs: HawkConfigs,
}

impl Test {
    pub fn new() -> Self {
        Self {
            configs: genesis_runner::get_node_configs(),
        }
    }

    async fn get_nodes(&self) -> impl Iterator<Item = Arc<MpcNode>> {
        crate::utils::mpc_node::MpcNodes::new(&self.configs)
            .await
            .into_iter()
    }
}

impl TestRun for Test {
    async fn exec(&mut self) -> Result<()> {
        // Phase A — build a clean base graph (native V5 checkpoint).
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = BASE_HEIGHT;
        run_genesis!(self, args);

        // Phase B — rewrite the base as a damaged V4 checkpoint on every
        // party, then damage the stores.
        let mut hashes = Vec::new();
        for config in self.configs.iter() {
            hashes.push(bake_v4_base(config).await?);
        }
        ensure!(
            hashes.iter().all(|h| h == &hashes[0]),
            "baked V4 checkpoint hashes must agree across parties"
        );
        let base_hash = hashes[0].clone();

        // TB1 joins the deletion list only now: the base indexed it, then it
        // was deleted (list entry + dummy source content).
        let aws = get_aws_clients(&self.configs[0]).await?;
        upload_iris_deletions(&vec![TB1], &aws.s3_client, &self.configs[0].environment).await?;

        let mut join_set = JoinSet::new();
        for (party_id, node) in self.get_nodes().await.enumerate() {
            join_set.spawn(async move { inject_store_damage(&node, party_id).await });
        }
        join_runners!(join_set);

        // Fixture self-check: the baked base must classify exactly as intended
        // under prune-at-read.
        assert_v4_fixture(&self.configs, &base_hash).await?;

        // Phase C — genesis pinned to the V4 base, indexing to MAX_HEIGHT.
        run_migration(&self.configs, MAX_HEIGHT, base_hash.clone()).await?;

        assert_reconciled(&self.configs).await?;
        assert_graph_state(&self.configs, &[GH1, SL1, KP1], &[TB1]).await?;
        assert_v5_checkpoints(&self.configs, &base_hash).await?;

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        genesis_runner::base_genesis_e2e_init(&self.configs, vec![]).await
    }

    async fn setup_assert(&mut self) -> Result<()> {
        genesis_runner::base_genesis_e2e_init_assertions(&self.configs, 0).await
    }

    async fn teardown(&mut self) -> Result<()> {
        crate::utils::mpc_node::MpcNodes::new(&self.configs)
            .await
            .cleanup_s3_checkpoints(&self.configs)
            .await
    }
}

/// Bake the damage into one eye's V4 wire graph.
///
/// All phase-A serials sit at version 0, matching the version-0 keys the
/// `GraphMem → GraphV4` conversion emits, so every undamaged entry survives
/// the prune verbatim.
fn bake_damage(g: &mut GraphV4) {
    let vid = |id: u32, version: i16| graph_v4::VectorId { id, version };
    let l0 = &mut g.layers[0].links;

    // GH1: a straggler key at a stale version (flags multi_version) and an
    // emptied live neighborhood (healed only if GH1 is surged).
    l0.get_mut(&vid(GH1, 0))
        .expect("GH1 must be in the base")
        .0
        .clear();
    l0.insert(vid(GH1, 1), graph_v4::EdgeIds(vec![vid(1, 0)]));

    // SL1: a self-edge at the live version as the only edge (flags self_loop;
    // the prune drops it, leaving out-degree 0 unless SL1 is surged).
    l0.get_mut(&vid(SL1, 0)).expect("SL1 must be in the base").0 = vec![vid(SL1, 0)];

    // TB1: untouched — its key stays in the base while the deletion list and
    // the source tombstone appear afterwards.
}

/// Download one party's phase-A checkpoint, rewrite it as a damaged V4
/// checkpoint, upload it, and record its row. Returns the blake3 hash.
async fn bake_v4_base(config: &Config) -> Result<String> {
    let node = MpcNode::new(config.clone()).await;
    let aws = get_aws_clients(config).await?;
    let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
        .await?
        .ok_or_else(|| eyre!("no checkpoint recorded after phase A"))?;
    let graphs = download_graph_checkpoint(
        &aws.checkpoint_s3_client,
        &config.graph_checkpoint_bucket_name,
        &state,
        None,
    )
    .await?;

    let [left, right] = graphs;
    let mut pair: [GraphV4; 2] = [left.into(), right.into()];
    for g in pair.iter_mut() {
        bake_damage(g);
    }

    // V4 layer serialization is canonical (sorted links), so the bytes — and
    // the hash — agree across parties.
    let bytes = bincode::serialize(&pair)?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let s3_key = format!("checkpoints/genesis_111_v4_base_{}.dat", config.party_id);
    aws.checkpoint_s3_client
        .put_object()
        .bucket(&config.graph_checkpoint_bucket_name)
        .key(&s3_key)
        .body(ByteStream::from(bytes))
        .send()
        .await
        .map_err(|e| eyre!("failed to upload baked V4 checkpoint: {e}"))?;

    let graph_tx = node.cpu_stores.graph.tx().await?;
    let mut tx = graph_tx.tx;
    GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
        &mut tx,
        &s3_key,
        BASE_HEIGHT as i64,
        state.last_indexed_modification_id,
        None,
        &hash,
        true,
        4,
    )
    .await?;
    tx.commit().await?;

    Ok(hash)
}

/// Apply the store-level damage to one party.
async fn inject_store_damage(node: &MpcNode, party_id: usize) -> Result<()> {
    // TB1: deletion — source content becomes the party's dummy shares (the
    // trigger bumps the version); no modification row.
    let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(party_id);
    node.gpu_stores
        .iris
        .update_iris(
            None,
            TB1 as i64,
            &dummy_code,
            &dummy_mask,
            &dummy_code,
            &dummy_mask,
        )
        .await?;

    // KP1: HNSW row deleted on party 0 only; the cross-party union must bring
    // every party to the same surgery list.
    if party_id == 0 {
        sqlx::query("DELETE FROM irises WHERE id = $1")
            .bind(KP1 as i64)
            .execute(&node.cpu_stores.iris.pool)
            .await?;
    }

    // HNSW stale tail rows > BASE_HEIGHT; the pinned reset must trim them.
    node.insert_extra_irises_into_cpu_store(BASE_HEIGHT as usize, TAIL_ROWS)
        .await?;

    // Junk WAL row that reset must clear.
    sqlx::query(
        "INSERT INTO hawk_graph_mutations (modification_id, serialized_mutations, mutation_format_version) VALUES ($1, $2, $3)",
    )
    .bind(999_i64)
    .bind(vec![0u8, 1, 2])
    .bind(1_i16)
    .execute(node.cpu_stores.graph.pool())
    .await?;

    Ok(())
}

/// Assert the baked V4 base classifies exactly as intended under
/// prune-at-read, on every party.
async fn assert_v4_fixture(configs: &HawkConfigs, base_hash: &str) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let aws = get_aws_clients(config).await?;
        let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
            .await?
            .ok_or_else(|| eyre!("no baked base checkpoint"))?;
        assert_eq!(state.blake3_hash, base_hash, "baked row must be the latest");
        assert_eq!(state.graph_version, 4, "baked row must be V4");

        let src = db_ops::get_iris_vector_ids(&node.gpu_stores.iris).await?;
        let version_map: HashMap<u32, i16> = src
            .iter()
            .map(|v| (v.serial_id(), v.version_id()))
            .collect();
        let (graphs, reports) = download_graph_checkpoint_pruned(
            &aws.checkpoint_s3_client,
            &config.graph_checkpoint_bucket_name,
            &state,
            Some(LegacyPruneContext {
                version_map,
                deleted: HashSet::from([TB1]),
            }),
        )
        .await?;
        let reports = reports.ok_or_else(|| eyre!("legacy load must emit prune reports"))?;

        for (eye, (graph, report)) in graphs.iter().zip(reports.iter()).enumerate() {
            assert_eq!(
                report.multi_version_serials,
                BTreeSet::from([GH1]),
                "multi-version class (eye {eye})"
            );
            assert_eq!(
                report.self_loop_serials,
                BTreeSet::from([SL1]),
                "self-loop class (eye {eye})"
            );
            assert!(
                report.zero_out_degree.contains(&GH1) && report.zero_out_degree.contains(&SL1),
                "GH1 and SL1 must survive with no out-edges (eye {eye})"
            );
            assert!(
                report.edges_dropped_self_loop >= 1,
                "the SL1 self-edge must be dropped (eye {eye})"
            );
            assert!(
                report.nodes_dropped_deleted >= 1,
                "the TB1 key must be dropped via the deletion list (eye {eye})"
            );
            assert_eq!(
                graph.vector_id_of(TB1),
                None,
                "TB1 must not survive the prune (eye {eye})"
            );
            assert_eq!(
                graph.vector_id_of(GH1),
                Some(VectorId::new(GH1, 0)),
                "GH1 must collapse onto the live version-0 entry (eye {eye})"
            );
        }
    }
    Ok(())
}

/// Run genesis across all parties, pinned to the baked V4 base.
async fn run_migration(configs: &HawkConfigs, max_indexation_id: u32, base: String) -> Result<()> {
    use tracing::Instrument as _;
    let mut join_set = JoinSet::new();
    for (idx, config) in configs.iter().cloned().enumerate() {
        let span = tracing::info_span!("genesis", idx = idx);
        let base = base.clone();
        join_set.spawn(async move {
            let mut ga = DEFAULT_GENESIS_ARGS;
            ga.max_indexation_id = max_indexation_id;
            let mut ea = ExecutionArgs::from_plaintext_args(ga, false);
            ea.base_checkpoint_hash = Some(base);
            let r = iris_mpc_upgrade_hawk::genesis::exec(ea, config)
                .instrument(span)
                .await;
            tracing::info!(genesis_id = idx, "migration exec returned {:?}", r);
            r
        });
    }
    let res: Result<Vec<_>, eyre::Report> = join_set.join_all().await.into_iter().collect();
    let _ = res?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    Ok(())
}

/// Per-row content digests: `(id, version_id, md5s of the four share columns)`.
async fn iris_row_digests(pool: &sqlx::PgPool) -> Result<Vec<(i64, i16, String)>> {
    Ok(sqlx::query_as(
        "SELECT id, version_id, \
         md5(left_code) || md5(left_mask) || md5(right_code) || md5(right_mask) \
         FROM irises ORDER BY id ASC",
    )
    .fetch_all(pool)
    .await?)
}

/// Assert every party's HNSW iris store matches the source byte-wise, the
/// stale tail is gone, the WAL and modifications are empty, and TB1's row
/// holds the party's dummy shares.
async fn assert_reconciled(configs: &HawkConfigs) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;

        let src = iris_row_digests(&node.gpu_stores.iris.pool).await?;
        let dst = iris_row_digests(&node.cpu_stores.iris.pool).await?;
        assert_eq!(
            src, dst,
            "HNSW irises must match source on (id, version, content) for all serials"
        );
        assert_eq!(
            src.len(),
            MAX_HEIGHT as usize,
            "expected exactly {MAX_HEIGHT} irises (tail rows trimmed)"
        );

        let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(config.party_id);
        let tb1 = node.cpu_stores.iris.get_iris_data_by_id(TB1 as i64).await?;
        assert_eq!(tb1.left_code(), &dummy_code.coefs[..], "TB1 left code");
        assert_eq!(tb1.left_mask(), &dummy_mask.coefs[..], "TB1 left mask");
        assert_eq!(tb1.right_code(), &dummy_code.coefs[..], "TB1 right code");
        assert_eq!(tb1.right_mask(), &dummy_mask.coefs[..], "TB1 right mask");

        let hnsw_mods = node.cpu_stores.iris.last_modifications(1000).await?;
        assert!(
            hnsw_mods.is_empty(),
            "HNSW modifications table must be empty"
        );

        let wal_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM hawk_graph_mutations")
            .fetch_one(node.cpu_stores.graph.pool())
            .await?;
        assert_eq!(wal_count.0, 0, "hawk_graph_mutations must be empty");

        let last_iris: Option<u32> = node
            .cpu_stores
            .graph
            .get_persistent_state("genesis", "last_indexed_iris_id")
            .await?;
        assert_eq!(last_iris, Some(MAX_HEIGHT), "last_indexed_iris_id");
    }
    Ok(())
}

/// Assert every party's latest checkpoint holds each `healed` serial at its
/// source-current version with a rebuilt (non-empty, self-free) bottom-layer
/// neighborhood, and no trace of `removed` serials.
async fn assert_graph_state(configs: &HawkConfigs, healed: &[u32], removed: &[u32]) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let src = db_ops::get_iris_vector_ids(&node.gpu_stores.iris).await?;
        let version_by_serial: HashMap<u32, i16> = src
            .iter()
            .map(|v| (v.serial_id(), v.version_id()))
            .collect();

        let aws = get_aws_clients(config).await?;
        let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
            .await?
            .ok_or_else(|| eyre!("no checkpoint after migration"))?;
        let graphs = download_graph_checkpoint(
            &aws.checkpoint_s3_client,
            &config.graph_checkpoint_bucket_name,
            &state,
            None,
        )
        .await?;

        for (eye, graph) in graphs.iter().enumerate() {
            // The prune seeds the content clock from kept keys only; surgery
            // and indexation must keep it aligned with the bottom layer.
            let clock_keys: BTreeSet<u32> = graph.node_init.keys().copied().collect();
            let layer0_keys: BTreeSet<u32> =
                graph.layers[0].get_links_map().keys().copied().collect();
            assert_eq!(
                clock_keys, layer0_keys,
                "content clock and bottom layer disagree (eye {eye})"
            );
            for &serial in healed {
                let version = *version_by_serial
                    .get(&serial)
                    .ok_or_else(|| eyre!("serial {serial} absent from source"))?;
                let vid = VectorId::new(serial, version);
                assert_eq!(
                    graph.vector_id_of(serial),
                    Some(vid),
                    "content clock must hold exactly {vid:?} (eye {eye})"
                );
                let links = graph.layers[0]
                    .get_links(&serial)
                    .ok_or_else(|| eyre!("layer 0 must hold serial {serial} (eye {eye})"))?;
                let neighbors = links.neighbors();
                assert!(
                    !neighbors.is_empty(),
                    "serial {serial} must be re-linked by surgery (eye {eye})"
                );
                assert!(
                    !neighbors.contains(&serial),
                    "serial {serial} must not keep a self-edge (eye {eye})"
                );
            }
            for &serial in removed {
                assert_eq!(
                    graph.vector_id_of(serial),
                    None,
                    "content clock must hold no entry for serial {serial} (eye {eye})"
                );
                assert!(
                    graph.layers[0].get_links(&serial).is_none(),
                    "layer 0 must hold no node for serial {serial} (eye {eye})"
                );
            }
        }
    }
    Ok(())
}

/// Assert the format transition: the post-delta and final checkpoints are V5,
/// the post-delta checkpoint is archival at `BASE_HEIGHT` with a hash
/// differing from the V4 base, and final hashes agree across parties.
async fn assert_v5_checkpoints(configs: &HawkConfigs, base_hash: &str) -> Result<()> {
    let mut final_hashes = Vec::new();
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let rows = node
            .cpu_stores
            .graph
            .get_genesis_graph_checkpoints()
            .await?;
        assert!(rows.len() >= 2, "expected final + post-delta checkpoints");
        assert_eq!(
            rows[0].last_indexed_iris_id, MAX_HEIGHT as i64,
            "newest checkpoint must be the final indexation checkpoint"
        );
        assert_eq!(rows[0].graph_version, 5, "final checkpoint must be V5");
        assert_eq!(
            rows[1].last_indexed_iris_id, BASE_HEIGHT as i64,
            "second-newest checkpoint must be the post-delta checkpoint"
        );
        assert_eq!(rows[1].graph_version, 5, "post-delta checkpoint must be V5");
        assert!(
            rows[1].is_archival,
            "post-delta checkpoint must be archival"
        );
        assert_ne!(
            rows[1].blake3_hash, base_hash,
            "post-delta checkpoint must differ from the V4 base"
        );
        final_hashes.push(rows[0].blake3_hash.clone());
    }
    assert!(
        final_hashes.iter().all(|h| h == &final_hashes[0]),
        "final checkpoint hashes must agree across parties"
    );
    Ok(())
}
