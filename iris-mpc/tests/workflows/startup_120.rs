/// startup_120 — one party restarts during the startup handshake with unchanged
/// data and rejoins the fleet's startup epoch.
///
/// Setup: three empty parties, party 2 configured to hold in [`Phase::Propose`].
/// Exec: party 2 is stopped there, the survivors are checked to be parked below
/// [`Phase::Converge`], then party 2 is restarted without the hold and the fleet
/// must come ready.
///
/// Asserts the point of the epoch design — the restart is absorbed. Before the
/// epoch layer this could not work at all: the returning party minted a fresh UUID,
/// absent from the survivors' startup-verified set, so `wait_for_others_ready` and
/// the heartbeat's first-contact check killed two healthy nodes.
///
/// Also covers a potential deadlock: party 2's visibility barrier needs the
/// survivors to publish its *new* UUID, and they can only do that from inside
/// `wait_for_epoch_commit`, which records peer UUIDs *before* it evaluates the
/// epoch. Reverse that ordering and this workflow hangs.
///
/// See [`crate::utils::hawk_fleet`] for why the kill point is the handshake rather
/// than the DB load.
use crate::utils::{
    cpu_node::CpuNodes,
    hawk_fleet::{FleetOptions, HawkFleet, FLEET_TIMEOUT, RESTARTED_PARTY},
    runner::{CpuTestContext, TestRun},
    COUNT_OF_PARTIES,
};
use iris_mpc::server::startup_phase::Phase;
use std::array;

#[derive(Default)]
pub struct Startup120 {
    nodes: Option<CpuNodes>,
    fleet: Option<HawkFleet>,
}

impl Startup120 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Startup120 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        self.nodes = Some(CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?);
        Ok(())
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Hand the fleet to `self` before anything can fail, so `teardown` always
        // gets a chance to stop it (`HawkFleet::drop` is the backstop). The barrier
        // budget stays at its default: the survivors have to outlast a peer restart
        // here, which is the whole point of the workflow.
        let options = FleetOptions {
            holds: array::from_fn(|party| (party == RESTARTED_PARTY).then_some(Phase::Propose)),
            startup_sync_timeout_secs: None,
        };
        self.fleet = Some(HawkFleet::start_all_with(&ctx.configs, ctx, options).await?);
        let fleet = self.fleet.as_mut().unwrap();

        // Kill at Propose — after the mutual-visibility barriers, before the party
        // has published an epoch. That ordering makes this deterministic: with no
        // epoch from this party the survivors cannot satisfy their commit barrier, so
        // they cannot have moved on whatever interleaving they were in.
        fleet
            .wait_for_phase(RESTARTED_PARTY, Phase::Propose, FLEET_TIMEOUT)
            .await?;
        fleet.stop_party(RESTARTED_PARTY).await?;

        // The survivors must be parked, not progressing. At or past Converge would
        // mean a party started mutating local storage under a plan its peers never
        // agreed to.
        fleet.assert_others_running(RESTARTED_PARTY)?;
        for party in (0..COUNT_OF_PARTIES).filter(|p| *p != RESTARTED_PARTY) {
            let phase = fleet.phase(party).await?;
            if phase >= Phase::Converge {
                eyre::bail!(
                    "party {party} reached phase {phase} while party {RESTARTED_PARTY} was down; \
                     the commit barrier should have held it"
                );
            }
        }

        // Restarted without the hold, so it runs the handshake through to serving.
        fleet.start_party(RESTARTED_PARTY, ctx, None);
        fleet.wait_all_ready(FLEET_TIMEOUT).await?;
        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let fleet = self.fleet.as_ref().unwrap();

        // The two untouched parties are still in their original processes — they
        // absorbed a peer restart instead of reloading.
        fleet.assert_others_running(RESTARTED_PARTY)?;

        // All three converged on one epoch, so the restarted party rejoined the
        // fleet's startup rather than forming a separate one.
        tracing::info!(
            "all parties agree on startup epoch {}",
            fleet.agreed_epoch().await?
        );
        Ok(())
    }

    async fn teardown(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(fleet) = self.fleet.as_mut() {
            fleet.stop_all().await?;
        }
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }
}
