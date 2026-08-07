/// startup_122 — one party restarts *after* the fleet is serving, with unchanged
/// data, and must not be allowed back in.
///
/// Setup: three empty parties, no holds — the fleet boots all the way to ready.
/// Exec: party 2 is stopped and restarted without touching its database, so it
/// re-derives exactly the sync state it had before. The survivors must decide to
/// tear down, must never re-admit party 2's new incarnation, and party 2 must never
/// reach [`Phase::Serving`].
///
/// This is the mirror image of [`crate::workflows::startup_120`], and the pair is
/// the point: an identical sync state makes a restart absorbable *during* the
/// handshake and must not make it absorbable once sessions exist. The digest-keyed
/// rejoin is scoped to startup — `DesyncWatch::stop` is called on entering
/// `Phase::Serving`, after which the heartbeat's UUID check owns peer supervision —
/// because the MPC session state the parties share by then is tied to a peer's
/// incarnation, not to its data, and a returning party carries none of it. So the
/// whole fleet has to come down and re-run startup together.
///
/// # What this asserts, and what it deliberately does not
///
/// The survivors *decide* to shut down within about six seconds and never finish:
/// `trigger_manual_shutdown` flips `shutting_down` on `/health` immediately, but
/// `server_main` does not return, so the processes are still heartbeating minutes
/// later (observed 2026-08-07). Requiring them to exit therefore proves nothing
/// about the behaviour under test and hangs the workflow instead, so this asserts
/// on the *decision* and on the peer set, both of which are readable over HTTP
/// while the parties are still up. `wait_for_teardown` accepts an exit too, so
/// fixing the propagation later does not turn this red. Teardown's `stop_all`
/// cancels the tasks directly, which is the path that does work.
///
/// The decision itself is not attributable to the return, only to the restart as a
/// whole: the survivors' three consecutive 2s heartbeat failures fire while party 2
/// is still rebuilding its sync state, so in practice they act on its *absence*
/// before ever seeing the new UUID. That is why [`Startup122`] also asserts the
/// peer set — `verified_peers` is frozen at startup and nothing writes to it once a
/// node is serving, so party 2's new UUID appearing there is exactly the signature
/// of a rejoin path that leaked past `Serving`, and it is a claim about the return
/// rather than about the gap.
use crate::utils::{
    cpu_node::CpuNodes,
    hawk_fleet::{FleetOptions, HawkFleet, FLEET_TIMEOUT, RESTARTED_PARTY},
    runner::{CpuTestContext, TestRun},
    COUNT_OF_PARTIES,
};
use iris_mpc::server::startup_phase::Phase;
use std::time::Duration;

/// Startup-barrier budget for every party in this workflow.
///
/// Bounds the returning party's in-process retry: it finds its peers ready but no
/// longer holding its UUID, and `wait_for_others_unready` deliberately retries that
/// case without exiting, so with the 300s default it would still be looping long
/// after the workflow is done with it. Ample for the initial boot, where all three
/// parties enter the barriers within milliseconds of each other.
const BARRIER_TIMEOUT_SECS: u64 = 30;

/// How long the survivors get to notice the restart.
///
/// Three consecutive failures at `heartbeat_interval_secs = 2` is ~6s; the rest is
/// slack for a loaded CI machine.
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(60);

/// How long party 2 is watched for an illegitimate rejoin after the survivors have
/// committed to tearing down.
///
/// Nothing is expected to happen here — this is the window in which a leaked rejoin
/// path would have its chance, and the workflow needs to have looked.
const REJOIN_WATCH: Duration = Duration::from_secs(15);

#[derive(Default)]
pub struct Startup122 {
    nodes: Option<CpuNodes>,
    fleet: Option<HawkFleet>,
    /// Coordination UUID of party 2's *new* incarnation, for `exec_assert`.
    restarted_uuid: Option<String>,
}

impl Startup122 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fail if any surviving party has admitted `uuid` to its startup-verified peer
    /// set, or if party 2 has talked its way to ready.
    async fn assert_not_rejoined(&self, uuid: &str) -> eyre::Result<()> {
        let fleet = self.fleet.as_ref().unwrap();

        for party in (0..COUNT_OF_PARTIES).filter(|p| *p != RESTARTED_PARTY) {
            // A survivor that has stopped answering has certainly not re-admitted
            // anyone; only a live answer can carry the failure.
            if let Ok(probe) = fleet.health(party).await {
                if probe.verified_peers.contains(uuid) {
                    eyre::bail!(
                        "party {party} admitted party {RESTARTED_PARTY}'s new incarnation {uuid} \
                         to its verified peers after it restarted post-serving; the startup \
                         rejoin must not apply once sessions exist"
                    );
                }
            }
        }

        // Likewise for party 2's own view: unreachable means it is gone, which is a
        // pass. Reaching Serving, or reporting ready, would mean it rejoined.
        if let Ok(phase) = fleet.phase(RESTARTED_PARTY).await {
            if phase >= Phase::Serving {
                eyre::bail!(
                    "party {RESTARTED_PARTY} reached phase {phase} after restarting post-serving"
                );
            }
        }
        if let Ok(probe) = fleet.health(RESTARTED_PARTY).await {
            if probe.is_ready {
                eyre::bail!("party {RESTARTED_PARTY} reports ready after restarting post-serving");
            }
        }
        Ok(())
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

        // Party 2's own sync state, as it stands while serving. Nothing below touches
        // its database, so the restart has to reproduce this exactly — that is the
        // "even though the data is unchanged" in the workflow name, and without it
        // this would be startup_121 (a mismatch) wearing a different hat.
        let before = fleet
            .wait_for_startup_state(RESTARTED_PARTY, FLEET_TIMEOUT)
            .await?;

        fleet.stop_party(RESTARTED_PARTY).await?;
        fleet.start_party(RESTARTED_PARTY, ctx, None);

        let after = fleet
            .wait_for_startup_state(RESTARTED_PARTY, FLEET_TIMEOUT)
            .await?;
        if after.party_sync_state_digest != before.party_sync_state_digest {
            eyre::bail!(
                "party {RESTARTED_PARTY} came back on sync state {}, not the {} it left on — \
                 something mutated its state across the restart, so this run exercises a \
                 sync-state mismatch (startup_121) rather than the unchanged-state rejoin this \
                 workflow is about",
                after.party_sync_state_digest,
                before.party_sync_state_digest
            );
        }

        let restarted_uuid = after
            .uuid
            .clone()
            .ok_or_else(|| eyre::eyre!("party {RESTARTED_PARTY} restarted without a UUID"))?;
        tracing::info!(
            "party {RESTARTED_PARTY} restarted on its original sync state {} as {restarted_uuid} \
             (was {:?})",
            after.party_sync_state_digest,
            before.uuid
        );
        self.restarted_uuid = Some(restarted_uuid.clone());

        // Every survivor must commit to tearing down. Not "must have exited": the
        // decision is what the fleet owes us here, and it is the only part that
        // happens on a bounded clock. See the module docs.
        for party in (0..COUNT_OF_PARTIES).filter(|p| *p != RESTARTED_PARTY) {
            fleet.wait_for_teardown(party, TEARDOWN_TIMEOUT).await?;
            tracing::info!("party {party} committed to tearing down");
        }

        // Then watch for the thing that must not happen. A leaked rejoin would land
        // in this window: party 2 is up, publishing a digest its peers recognise, and
        // the peers are still alive to act on it.
        let watch_until = tokio::time::Instant::now() + REJOIN_WATCH;
        while tokio::time::Instant::now() < watch_until {
            self.assert_not_rejoined(&restarted_uuid).await?;
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        // Re-check once more at the end of the workflow, after the survivors have had
        // the whole of `exec` to change their minds.
        let uuid = self
            .restarted_uuid
            .clone()
            .ok_or_else(|| eyre::eyre!("exec did not record the restarted party's UUID"))?;
        self.assert_not_rejoined(&uuid).await
    }

    async fn teardown(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        // Cancels each party's task directly. The parties do not exit on their own
        // graceful shutdown, so this is what actually frees the ports for the next
        // workflow.
        if let Some(fleet) = self.fleet.as_mut() {
            fleet.stop_all().await?;
        }
        // The fleet reached the load, so it left checkpoint state behind.
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }
}
