use std::{collections::HashMap, sync::Arc};

use crate::join_runners;
use crate::run_genesis;
use crate::utils::{
    genesis_runner::{self, DEFAULT_GENESIS_ARGS},
    modifications::{ModificationInput, ModificationType::Reauth},
    mpc_node::{db_ops, MpcNode},
    s3_deletions::get_aws_clients,
    HawkConfigs, TestRun, TestRunContextInfo,
};
use eyre::{eyre, Result};
use iris_mpc_common::{
    helpers::{smpc_request::REAUTH_MESSAGE_TYPE, sync::Modification},
    iris_db::get_dummy_shares_for_deletion,
    VectorId,
};
use iris_mpc_cpu::{
    genesis::DeltaMode,
    graph_checkpoint::{download_graph_checkpoint, get_latest_checkpoint_state},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::GraphPg,
};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;
use tokio::task::JoinSet;

// Base graph height, final indexation height, and the size of the stale HNSW tail.
const BASE_HEIGHT: u32 = 80;
const MAX_HEIGHT: u32 = 100;
const TAIL_ROWS: usize = 10;

// Divergence serials (all ≤ BASE_HEIGHT so they exist in the base graph).
const A1: u32 = 5; // version bump + mod row  (agreement: join ∩ mod)
const A2: u32 = 15; // version bump + mod row  (agreement: join ∩ mod)
const B1: u32 = 25; // version bump, no mod row (join-only)
const C1: u32 = 35; // mod row, no version bump (mod-only, invariant violation)
const D1: u32 = 45; // HNSW row version drift   (store_repair)
const C2: u32 = 55; // rejected reauth (persisted=false) — must appear nowhere
const E1: u32 = 65; // ghost baked into the base graph (multi-version surgery)
const F1: u32 = 70; // deleted: source content = dummy shares (remove-only)
const G1: u32 = 75; // deleted then reinserted with a live iris (replay)
const G1_DONOR: u32 = 2; // untouched serial whose shares G1's reinsert copies

// Max persisted+completed source modification id after divergence: id 1 (E1,
// phase A) and ids 2,3,4 (A1, A2, C1) are persisted; C2 (id 5) is not.
const EXPECTED_MOD_CURSOR: i64 = 4;

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
        // Phase A — build the base graph to BASE_HEIGHT via the default
        // (modifications) mode, then bake a ghost: bump E1 through a reauth
        // modification and rerun modifications mode, whose replay inserts the
        // new version and leaves the old node in the graph. The resulting
        // checkpoint is the version-join base.
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = BASE_HEIGHT;
        run_genesis!(self, args.clone());

        for node in self.get_nodes().await {
            node.apply_modifications(
                &[],
                &[ModificationInput::new(1, E1 as i64, Reauth, true, true)],
            )
            .await?;
        }
        run_genesis!(self, args);

        let base_hash = {
            let nodes: Vec<_> = self.get_nodes().await.collect();
            let state = get_latest_checkpoint_state(&nodes[0].cpu_stores.graph)
                .await?
                .ok_or_else(|| eyre!("no base checkpoint recorded after phase A"))?;
            state.blake3_hash
        };

        // Phase B — inject divergence on every party.
        let mut join_set = JoinSet::new();
        for (party_id, node) in self.get_nodes().await.enumerate() {
            join_set.spawn(async move { inject_divergence(&node, party_id).await });
        }
        join_runners!(join_set);

        // Phase C — version-join pinned to the base, indexing up to MAX_HEIGHT.
        run_version_join(&self.configs, MAX_HEIGHT, Some(base_hash.clone())).await?;
        assert_reconciled(&self.configs, EXPECTED_MOD_CURSOR).await?;
        assert_graph_state(&self.configs, &[A1, A2, B1, E1, G1], &[F1]).await?;

        // Phase D — rerun with the latest common checkpoint (no pin). The join
        // plan is empty; the run converges to the same state (idempotency).
        run_version_join(&self.configs, MAX_HEIGHT, None).await?;
        assert_reconciled(&self.configs, EXPECTED_MOD_CURSOR).await?;

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

/// Apply all Phase B divergence to one party's databases.
async fn inject_divergence(node: &MpcNode, party_id: usize) -> Result<()> {
    // A1, A2: version bump + persisted+completed modification rows (agreement).
    node.apply_modifications(
        &[],
        &[
            ModificationInput::new(2, A1 as i64, Reauth, true, true),
            ModificationInput::new(3, A2 as i64, Reauth, true, true),
        ],
    )
    .await?;

    // B1: version bump WITHOUT a modification row (unlogged write → join-only).
    node.increment_iris_version(B1 as i64).await?;

    // C1: persisted+completed modification row WITHOUT a version bump
    //     (invariant violation → mod-only, must not be replayed).
    // C2: rejected reauth (persisted=false) — must appear in neither set.
    {
        let mut tx = node.gpu_stores.iris.tx().await?;
        db_ops::write_modification(
            &mut tx,
            &Modification {
                id: 4,
                serial_id: Some(C1 as i64),
                request_type: REAUTH_MESSAGE_TYPE.to_string(),
                s3_url: None,
                status: "COMPLETED".to_string(),
                persisted: true,
                result_message_body: None,
            },
        )
        .await?;
        db_ops::write_modification(
            &mut tx,
            &Modification {
                id: 5,
                serial_id: Some(C2 as i64),
                request_type: REAUTH_MESSAGE_TYPE.to_string(),
                s3_url: None,
                status: "COMPLETED".to_string(),
                persisted: false,
                result_message_body: None,
            },
        )
        .await?;
        tx.commit().await?;
    }

    // F1: deletion — source content becomes the party's dummy shares (the
    // content-change trigger bumps the version). No modification row: the
    // tombstone is detected from content alone.
    // G1: deletion followed by reinsertion — the final content is live (copied
    // from an untouched donor serial, a valid cross-party sharing).
    let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(party_id);
    for serial in [F1, G1] {
        node.gpu_stores
            .iris
            .update_iris(
                None,
                serial as i64,
                &dummy_code,
                &dummy_mask,
                &dummy_code,
                &dummy_mask,
            )
            .await?;
    }
    sqlx::query(
        "UPDATE irises SET (left_code, left_mask, right_code, right_mask) = \
         (SELECT left_code, left_mask, right_code, right_mask FROM irises WHERE id = $2) \
         WHERE id = $1",
    )
    .bind(G1 as i64)
    .bind(G1_DONOR as i64)
    .execute(&node.gpu_stores.iris.pool)
    .await?;

    // D1: drift the HNSW iris row version (local-only damage → store_repair).
    sqlx::query("UPDATE irises SET version_id = version_id + 5 WHERE id = $1")
        .bind(D1 as i64)
        .execute(&node.cpu_stores.iris.pool)
        .await?;

    // HNSW stale tail rows > BASE_HEIGHT + clobber last_indexed_iris_id.
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

    // Clobber the HNSW modification cursor with a garbage value; version-join
    // must source its cursor from the checkpoint, not from here.
    {
        let graph_tx = node.cpu_stores.graph.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::set_persistent_state(
            &mut tx,
            "genesis",
            "last_indexed_modification_id",
            &9_999_999_i64,
        )
        .await?;
        tx.commit().await?;
    }

    Ok(())
}

/// Run genesis in version-join mode across all parties.
async fn run_version_join(
    configs: &HawkConfigs,
    max_indexation_id: u32,
    base: Option<String>,
) -> Result<()> {
    use tracing::Instrument as _;
    let mut join_set = JoinSet::new();
    for (idx, config) in configs.iter().cloned().enumerate() {
        let span = tracing::info_span!("genesis", idx = idx);
        let base = base.clone();
        join_set.spawn(async move {
            let mut ga = DEFAULT_GENESIS_ARGS;
            ga.max_indexation_id = max_indexation_id;
            let mut ea = ExecutionArgs::from_plaintext_args(ga, false);
            ea.delta_mode = DeltaMode::VersionJoin;
            ea.base_checkpoint_hash = base;
            let r = iris_mpc_upgrade_hawk::genesis::exec(ea, config)
                .instrument(span)
                .await;
            tracing::info!(genesis_id = idx, "version-join exec returned {:?}", r);
            r
        });
    }
    let res: Result<Vec<_>, eyre::Report> = join_set.join_all().await.into_iter().collect();
    let _ = res?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    Ok(())
}

/// Assert that every party's HNSW iris store matches the source on
/// `(id, version)`, the stale tail is gone, the WAL and HNSW modifications are
/// empty, and the cursors are correct.
async fn assert_reconciled(configs: &HawkConfigs, expected_mod_cursor: i64) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;

        let src = db_ops::get_iris_vector_ids(&node.gpu_stores.iris).await?;
        let dst = db_ops::get_iris_vector_ids(&node.cpu_stores.iris).await?;
        assert_eq!(
            src, dst,
            "HNSW irises must match source on (id, version) for all serials"
        );
        assert_eq!(
            src.len(),
            MAX_HEIGHT as usize,
            "expected exactly {MAX_HEIGHT} irises (tail rows trimmed)"
        );

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

        let last_mod: Option<i64> = node
            .cpu_stores
            .graph
            .get_persistent_state("genesis", "last_indexed_modification_id")
            .await?;
        assert_eq!(
            last_mod,
            Some(expected_mod_cursor),
            "last_indexed_modification_id"
        );
    }
    Ok(())
}

/// Assert every party's latest checkpoint graph holds, in layer 0, exactly one
/// key per serial in `current` — the current source version (stale nodes
/// replayed, ghosts removed) — and no key at all for serials in `removed`.
async fn assert_graph_state(configs: &HawkConfigs, current: &[u32], removed: &[u32]) -> Result<()> {
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
            .ok_or_else(|| eyre!("no checkpoint after version-join"))?;
        let graphs = download_graph_checkpoint(
            &aws.checkpoint_s3_client,
            &config.graph_checkpoint_bucket_name,
            &state,
        )
        .await?;

        for (eye, graph) in graphs.iter().enumerate() {
            assert!(!graph.layers.is_empty(), "graph is empty (eye {eye})");
            let mut keys_per_serial: HashMap<u32, Vec<VectorId>> = HashMap::new();
            for vid in graph.layers[0].links.keys() {
                keys_per_serial
                    .entry(vid.serial_id())
                    .or_default()
                    .push(*vid);
            }

            for &serial in current {
                let version = *version_by_serial
                    .get(&serial)
                    .ok_or_else(|| eyre!("serial {serial} absent from source"))?;
                let vid = VectorId::new(serial, version);
                assert_eq!(
                    keys_per_serial.get(&serial),
                    Some(&vec![vid]),
                    "graph layer 0 must hold exactly {vid:?} (eye {eye})"
                );
            }
            for &serial in removed {
                assert_eq!(
                    keys_per_serial.get(&serial),
                    None,
                    "graph layer 0 must hold no key for serial {serial} (eye {eye})"
                );
            }
        }
    }
    Ok(())
}
