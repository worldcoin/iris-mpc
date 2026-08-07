/// startup_122 — one party restarts *after* the fleet is serving, with unchanged
/// data, and must not be allowed back in.
///
/// Setup: three empty parties, no holds — the fleet boots all the way to ready.
/// Exec: party 2 is stopped and restarted without touching its database, so it
/// re-derives exactly the sync state it had before. Every party must still exit and
/// none may report ready.
///
/// This is the mirror image of [`crate::workflows::startup_120`], and the pair is
/// the point: an identical sync state makes a restart absorbable *during* the
/// handshake and must not make it absorbable once sessions exist. The digest-keyed
/// rejoin is scoped to startup — `DesyncWatch::stop` is called on reaching
/// [`Phase::Serving`], after which the heartbeat's UUID check owns peer supervision
/// — because the MPC session state the parties share by then is tied to a peer's
/// incarnation, not to its data, and cannot be reconstructed from a matching
/// digest. A returning party carries none of it. So the whole fleet has to come
/// down and re-run startup together.
///
/// What breaks this test is precisely that scoping going wrong: a rejoin path still
/// live at `Serving` would let the survivors read party 2's unchanged digest,
/// re-verify it and carry on serving over stale sessions — the fleet would return
/// to all-ready with the survivors in their original processes, and
/// `wait_all_exited` would fail.
///
/// The exit route is not asserted, only the outcome. In practice the survivors go
/// first and on the *absence*: three consecutive 2s heartbeat failures fire at
/// ~6s, well before party 2 finishes rebuilding its sync state, so they trigger
/// graceful shutdown without ever seeing the new UUID. Party 2 then finds no peer
/// to sync with and dies on its own barrier. Absence and return are two halves of
/// one restart and there is no interleaving in which only one of them is visible,
/// so pinning the workflow to a particular error string would only make it brittle;
/// the invariant under test is that nobody keeps serving.
use crate::utils::{
    cpu_node::CpuNodes,
    hawk_fleet::{FleetOptions, HawkFleet, FLEET_TIMEOUT, RESTARTED_PARTY},
    runner::{CpuTestContext, TestRun},
};
use iris_mpc::server::startup_phase::Phase;

/// Startup-barrier budget for every party in this workflow.
///
/// Bounds the returning party: its peers are gone, so its first barrier round just
/// retries a refused connection until this budget expires, and only then does it
/// exit. The 300s default would put that well past [`FLEET_TIMEOUT`] and `exec`
/// would fail on the harness deadline instead of on the behaviour it asserts. Ample
/// for the initial boot, where all three parties enter the barriers within
/// milliseconds of each other and clear them in a couple of one-second retry
/// rounds.
const BARRIER_TIMEOUT_SECS: u64 = 30;

#[derive(Default)]
pub struct Startup122 {
    nodes: Option<CpuNodes>,
    fleet: Option<HawkFleet>,
}

impl Startup122 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Startup122 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        self.nodes = Some(CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?);
        Ok(())
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Hand the fleet to `self` before anything can fail, so `teardown` always
        // gets a chance to stop it (`HawkFleet::drop` is the backstop).
        let options = FleetOptions {
            startup_sync_timeout_secs: Some(BARRIER_TIMEOUT_SECS),
            ..Default::default()
        };
        self.fleet = Some(HawkFleet::start_all_with(&ctx.configs, ctx, options).await?);
        let fleet = self.fleet.as_mut().unwrap();

        // No holds: the restart under test is post-handshake, so the fleet has to be
        // fully serving first. Ready is what arms the heartbeat and retires the
        // startup watch, and it is the state whose session data a returning party
        // cannot reconstruct.
        fleet.wait_all_ready(FLEET_TIMEOUT).await?;
        tracing::info!(
            "fleet serving on fleet sync-state digest {}",
            fleet.agreed_fleet_sync_state_digest().await?
        );

        // Party 2's own sync state, as it stands while serving. Nothing below
        // touches its database, so the restart has to reproduce this exactly — that
        // is the "even though the data is unchanged" in the workflow name, and
        // without it this would be startup_121 (a mismatch) wearing a different hat.
        let digest_before = fleet
            .wait_for_party_sync_state_digest(RESTARTED_PARTY, FLEET_TIMEOUT)
            .await?;

        fleet.stop_party(RESTARTED_PARTY).await?;
        fleet.start_party(RESTARTED_PARTY, ctx, None);

        let digest_after = fleet
            .wait_for_party_sync_state_digest(RESTARTED_PARTY, FLEET_TIMEOUT)
            .await?;
        if digest_after != digest_before {
            eyre::bail!(
                "party {RESTARTED_PARTY} came back on sync state {digest_after}, not the \
                 {digest_before} it left on — something mutated its state across the restart, so \
                 this run exercises a sync-state mismatch (startup_121) rather than the \
                 unchanged-state rejoin this workflow is about"
            );
        }
        tracing::info!(
            "party {RESTARTED_PARTY} restarted on its original sync state {digest_after}"
        );

        // Party 2 must not talk its way back into a serving fleet on the strength of
        // that digest, and the survivors must not keep serving without it. Both are
        // the same assertion here: every party exits. `wait_all_exited` fails if any
        // is still running at the deadline, which is what a silently-absorbed restart
        // would look like.
        let outcomes = fleet.wait_all_exited(FLEET_TIMEOUT).await?;
        tracing::info!("fleet tore itself down after the post-ready restart: {outcomes:?}");
        Ok(())
    }

    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // No party may be left serving. A survivor still ready here would be one
        // holding MPC session state shared with an incarnation that no longer exists;
        // a ready party 2 would be one that rejoined a fleet it has no session state
        // for.
        for config in ctx.configs.iter() {
            let url = format!("http://127.0.0.1:{}/ready", config.healthcheck_port);
            if let Ok(response) = reqwest::get(&url).await {
                if response.status().is_success() {
                    eyre::bail!(
                        "party {} reports ready after a peer restarted post-{}",
                        config.party_id,
                        Phase::Serving
                    );
                }
            }
        }
        Ok(())
    }

    async fn teardown(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(fleet) = self.fleet.as_mut() {
            fleet.stop_all().await?;
        }
        // The fleet reached the load, so it left checkpoint state behind; clear it so
        // the next workflow starts from a clean fleet.
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }
}
