use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use crate::join_runners;
use crate::run_genesis;
use crate::utils::{
    genesis_runner::{self, DEFAULT_GENESIS_ARGS},
    modifications::{ModificationInput, ModificationType::Reauth},
    mpc_node::{db_ops, MpcNode},
    s3_deletions::get_aws_clients,
    HawkConfigs, TestRun, TestRunContextInfo,
};
use eyre::{ensure, eyre, Result};
use iris_mpc_common::{
    helpers::{smpc_request::REAUTH_MESSAGE_TYPE, sync::Modification},
    iris_db::get_dummy_shares_for_deletion,
    VectorId,
};
use iris_mpc_cpu::{
    graph_checkpoint::{download_graph_checkpoint, get_latest_checkpoint_state},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::GraphPg,
};
use iris_mpc_store::ExplicitVersionToken;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;
use std::ops::DerefMut;
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
const D1: u32 = 45; // HNSW row version drift   (row mismatch → surgery)
const C2: u32 = 55; // rejected reauth (persisted=false) — must appear nowhere
const H1: u32 = 60; // HNSW row deleted, source live (row missing → insert + surgery)
const E1: u32 = 65; // base graph clock stale, row aligned (graph-axis-only surgery)
const F1: u32 = 70; // deleted: source content = dummy shares (remove-only)
const G1: u32 = 75; // deleted then reinserted with a live iris (replay)
const G1_DONOR: u32 = 2; // untouched serial whose shares G1's reinsert copies

// Asymmetric damage on a single party; the cross-party surgery union must
// bring every party to the same replay list or the MPC jobs desync.
const K1: u32 = 50; // HNSW row deleted on party 0 only
const L1: u32 = 30; // HNSW row version drift on party 1 only

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
        // Phase A — build the base graph, then bake the E1 fixture: bump E1 in
        // the source (reauth mod + row version) and align the HNSW row, so the
        // base checkpoint's content clock is stale while the rows agree. The
        // phase-A checkpoint is the version-join base.
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = BASE_HEIGHT;
        run_genesis!(self, args);

        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move { bake_e1_stale_clock(&node).await });
        }
        let hashes: Vec<String> = join_set
            .join_all()
            .await
            .into_iter()
            .collect::<Result<_>>()?;
        ensure!(
            hashes.iter().all(|h| h == &hashes[0]),
            "baked checkpoint hashes must agree across parties"
        );
        let base_hash = hashes[0].clone();

        // Fixture self-check: the base checkpoint's content clock still holds
        // E1 at version 0 (graph-axis divergence exists).
        assert_e1_stale_in_base(&self.configs, &base_hash).await?;

        // Phase B — inject divergence on every party.
        let mut join_set = JoinSet::new();
        for (party_id, node) in self.get_nodes().await.enumerate() {
            join_set.spawn(async move { inject_divergence(&node, party_id).await });
        }
        join_runners!(join_set);

        // Phase C — version-join pinned to the base, indexing up to MAX_HEIGHT.
        run_version_join(&self.configs, MAX_HEIGHT, Some(base_hash.clone())).await?;
        assert_reconciled(&self.configs, EXPECTED_MOD_CURSOR).await?;
        assert_graph_state(&self.configs, &[A1, A2, B1, D1, E1, G1, H1, K1, L1], &[F1]).await?;
        assert_post_delta_checkpoint(&self.configs, &base_hash).await?;
        let phase_c_hashes = latest_checkpoint_hashes(&self.configs).await?;

        // Phase D — rerun with the latest common checkpoint (no pin). The join
        // plan is empty; the run converges to the same state (idempotency) and
        // is quiescent: no new checkpoint is recorded.
        run_version_join(&self.configs, MAX_HEIGHT, None).await?;
        assert_reconciled(&self.configs, EXPECTED_MOD_CURSOR).await?;
        let phase_d_hashes = latest_checkpoint_hashes(&self.configs).await?;
        assert_eq!(
            phase_c_hashes, phase_d_hashes,
            "phase D must not record a new checkpoint (graph unchanged)"
        );

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

/// Bake the E1 stale-clock fixture on one party and return the base
/// checkpoint's blake3 hash (equal across parties: graph structure is public
/// and its serialization canonical).
///
/// Steps: reauth mod row + source version bump; HNSW row version aligned to
/// the source. Rows agree at version 1 while the base graph's content clock
/// still holds version 0 — a pure graph-axis divergence.
async fn bake_e1_stale_clock(node: &MpcNode) -> Result<String> {
    node.apply_modifications(
        &[],
        &[ModificationInput::new(1, E1 as i64, Reauth, true, true)],
    )
    .await?;

    // Align the HNSW row version with the source (mimics a persisted replay).
    {
        let mut tx = node.cpu_stores.iris.tx().await?;
        {
            let mut ev = ExplicitVersionToken::enable(&mut tx).await?;
            sqlx::query("UPDATE irises SET version_id = version_id + 1 WHERE id = $1")
                .bind(E1 as i64)
                .execute(ev.tx().deref_mut())
                .await?;
        }
        tx.commit().await?;
    }

    let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
        .await?
        .ok_or_else(|| eyre!("no checkpoint recorded after phase A"))?;
    Ok(state.blake3_hash)
}

/// Assert the base checkpoint's content clock holds E1 at version 0 in both
/// eyes, on every party (the source row is already at 1).
async fn assert_e1_stale_in_base(configs: &HawkConfigs, base_hash: &str) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let aws = get_aws_clients(config).await?;
        let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
            .await?
            .ok_or_else(|| eyre!("no base checkpoint"))?;
        assert_eq!(state.blake3_hash, base_hash, "base row must be the latest");
        let graphs = download_graph_checkpoint(
            &aws.checkpoint_s3_client,
            &config.graph_checkpoint_bucket_name,
            &state,
            None,
        )
        .await?;
        for (eye, graph) in graphs.iter().enumerate() {
            assert_eq!(
                graph.vector_id_of(E1),
                Some(VectorId::new(E1, 0)),
                "E1 must sit at version 0 in the base content clock (eye {eye})"
            );
        }
    }
    Ok(())
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

    // F1: deletion — source content becomes the party's dummy shares (trigger
    // bumps the version); no modification row.
    // G1: deletion then reinsertion — final content copied from a donor serial
    // (a valid cross-party sharing).
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

    // D1: drift the HNSW iris row version (local-only damage → surgery).
    // Hand-set version: needs the explicit-version flag past the trigger.
    {
        let mut tx = node.cpu_stores.iris.tx().await?;
        {
            let mut ev = ExplicitVersionToken::enable(&mut tx).await?;
            sqlx::query("UPDATE irises SET version_id = version_id + 5 WHERE id = $1")
                .bind(D1 as i64)
                .execute(ev.tx().deref_mut())
                .await?;
        }
        tx.commit().await?;
    }

    // H1: delete the HNSW iris row while the source stays live (missing row →
    // surgery; the row is INSERTed at the post-checkpoint flush).
    sqlx::query("DELETE FROM irises WHERE id = $1")
        .bind(H1 as i64)
        .execute(&node.cpu_stores.iris.pool)
        .await?;

    // K1 / L1: same damage classes on a single party only.
    if party_id == 0 {
        sqlx::query("DELETE FROM irises WHERE id = $1")
            .bind(K1 as i64)
            .execute(&node.cpu_stores.iris.pool)
            .await?;
    }
    if party_id == 1 {
        let mut tx = node.cpu_stores.iris.tx().await?;
        {
            let mut ev = ExplicitVersionToken::enable(&mut tx).await?;
            sqlx::query("UPDATE irises SET version_id = version_id + 3 WHERE id = $1")
                .bind(L1 as i64)
                .execute(ev.tx().deref_mut())
                .await?;
        }
        tx.commit().await?;
    }

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

    // Clobber the HNSW modification cursor with a DIFFERENT garbage value per
    // party; the cursor is party-local state, so this must neither wedge the
    // sync handshake nor leak into the join (which sources its cursor from the
    // checkpoint).
    {
        let graph_tx = node.cpu_stores.graph.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::set_persistent_state(
            &mut tx,
            "genesis",
            "last_indexed_modification_id",
            &(9_999_999_i64 + party_id as i64),
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

/// Assert every party recorded a post-delta checkpoint: the two newest rows
/// are the final indexation checkpoint at `MAX_HEIGHT` and, below it, an
/// archival delta checkpoint at `BASE_HEIGHT` whose hash differs from the
/// base (the graph changed). Row writes are deferred until after this
/// checkpoint, so its existence is a precondition of the reconciled rows.
async fn assert_post_delta_checkpoint(configs: &HawkConfigs, base_hash: &str) -> Result<()> {
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let rows = node
            .cpu_stores
            .graph
            .get_genesis_graph_checkpoints()
            .await?;
        assert!(rows.len() >= 2, "expected final + delta checkpoints");
        assert_eq!(
            rows[0].last_indexed_iris_id, MAX_HEIGHT as i64,
            "newest checkpoint must be the final indexation checkpoint"
        );
        assert_eq!(
            rows[1].last_indexed_iris_id, BASE_HEIGHT as i64,
            "second-newest checkpoint must be the post-delta checkpoint"
        );
        assert!(
            rows[1].is_archival,
            "post-delta checkpoint must be archival"
        );
        assert_ne!(
            rows[1].blake3_hash, base_hash,
            "post-delta checkpoint must differ from the base (graph changed)"
        );
    }
    Ok(())
}

/// The latest checkpoint blake3 hash per party.
async fn latest_checkpoint_hashes(configs: &HawkConfigs) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for config in configs.iter() {
        let node = MpcNode::new(config.clone()).await;
        let state = get_latest_checkpoint_state(&node.cpu_stores.graph)
            .await?
            .ok_or_else(|| eyre!("no checkpoint recorded"))?;
        out.push(state.blake3_hash);
    }
    Ok(out)
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

/// Assert that every party's HNSW iris store matches the source byte-wise on
/// `(id, version, content)`, the stale tail is gone, the WAL and HNSW
/// modifications are empty, and the cursors are correct. Also checks the
/// fixture content: F1 holds the party's dummy shares, G1 the donor's shares.
async fn assert_reconciled(configs: &HawkConfigs, expected_mod_cursor: i64) -> Result<()> {
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

        // F1 holds the party's deletion dummy shares.
        let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(config.party_id);
        let f1 = node.cpu_stores.iris.get_iris_data_by_id(F1 as i64).await?;
        assert_eq!(f1.left_code(), &dummy_code.coefs[..], "F1 left code");
        assert_eq!(f1.left_mask(), &dummy_mask.coefs[..], "F1 left mask");
        assert_eq!(f1.right_code(), &dummy_code.coefs[..], "F1 right code");
        assert_eq!(f1.right_mask(), &dummy_mask.coefs[..], "F1 right mask");

        // G1 holds the donor's shares (content digest only; versions differ).
        let digest_of = |rows: &[(i64, i16, String)], serial: u32| {
            rows.iter()
                .find(|(id, _, _)| *id == serial as i64)
                .map(|(_, _, digest)| digest.clone())
                .ok_or_else(|| eyre!("serial {serial} missing"))
        };
        assert_eq!(
            digest_of(&dst, G1)?,
            digest_of(&dst, G1_DONOR)?,
            "G1 must hold the donor's content"
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

/// Assert every party's latest checkpoint graph holds each serial in
/// `current` at exactly the current source version in its content clock
/// (stale nodes replayed), and no trace at all of serials in `removed`.
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
            None,
        )
        .await?;

        for (eye, graph) in graphs.iter().enumerate() {
            assert!(!graph.layers.is_empty(), "graph is empty (eye {eye})");
            // The content clock and the bottom layer must hold exactly the
            // same serials: surgery, removals and replays all maintain the
            // clock through the same apply path as the edges.
            let clock_keys: BTreeSet<u32> = graph.node_init.keys().copied().collect();
            let layer0_keys: BTreeSet<u32> =
                graph.layers[0].get_links_map().keys().copied().collect();
            assert_eq!(
                clock_keys, layer0_keys,
                "content clock and bottom layer disagree (eye {eye})"
            );
            for &serial in current {
                let version = *version_by_serial
                    .get(&serial)
                    .ok_or_else(|| eyre!("serial {serial} absent from source"))?;
                let vid = VectorId::new(serial, version);
                assert_eq!(
                    graph.vector_id_of(serial),
                    Some(vid),
                    "content clock must hold exactly {vid:?} (eye {eye})"
                );
                assert!(
                    graph.layers[0].get_links(&serial).is_some(),
                    "graph layer 0 must hold serial {serial} (eye {eye})"
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
                    "graph layer 0 must hold no node for serial {serial} (eye {eye})"
                );
            }
        }
    }
    Ok(())
}
