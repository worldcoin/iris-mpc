/// startup_121 — one party restarts during the startup handshake with *changed*
/// data, and the whole fleet must refuse to come up.
///
/// Setup: three empty parties. Exec: party 2 is stopped once it reaches
/// [`Phase::Commit`], an extra iris row is inserted into its database, and it is
/// restarted. Every party must exit and none may report ready.
///
/// The surviving parties hold an epoch derived from party 2's previous facts; it
/// now derives a different one. Nobody may proceed, because the agreed plan
/// describes a fleet state that no longer exists.
///
/// Deliberately does not assert the error text. The mismatch is caught by whichever
/// check the survivors reach first — the commit barrier if they are still parked,
/// the startup watch if they had already moved into converge/load — and in the
/// latter case the cross-party checkpoint round inside the load can surface its own
/// peer timeout first. What is invariant, and all this workflow needs, is that
/// nobody serves.
///
/// See [`crate::utils::hawk_fleet`] for why the kill point is the handshake rather
/// than the DB load.
use crate::utils::{
    cpu_node::CpuNodes,
    hawk_fleet::{HawkFleet, FLEET_TIMEOUT, RESTARTED_PARTY},
    runner::{CpuTestContext, TestRun},
};
use iris_mpc::server::startup_phase::Phase;
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

#[derive(Default)]
pub struct Startup121 {
    nodes: Option<CpuNodes>,
    fleet: Option<HawkFleet>,
}

impl Startup121 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Startup121 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        self.nodes = Some(CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?);
        Ok(())
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // Hand the fleet to `self` before anything can fail, so `teardown` always
        // gets a chance to stop it. `HawkFleet::drop` is the backstop, but a
        // graceful stop leaves the ports free for the next workflow sooner.
        self.fleet = Some(HawkFleet::start_all(&ctx.configs, ctx).await?);
        let fleet = self.fleet.as_mut().unwrap();

        // Kill at Commit, not Propose: the survivors must already have derived an
        // epoch from this party's *old* facts. Otherwise they would simply read the
        // new ones on restart and legitimately agree — correct behaviour, but not
        // the case under test.
        fleet
            .wait_for_phase(RESTARTED_PARTY, Phase::Commit, FLEET_TIMEOUT)
            .await?;
        fleet.stop_party(RESTARTED_PARTY).await?;

        // Change the party's data while it is down. One extra iris row moves
        // `db_len`, which is part of the fact digest; appending at `count + 1` keeps
        // the `COUNT(*) == MAX(id)` store invariant intact so the party fails on the
        // epoch, not on `check_store_consistency`.
        let node = &nodes.0[RESTARTED_PARTY];
        let next_id = node.store.iris_store.count_irises().await? as i64 + 1;
        let code = vec![0u16; IRIS_CODE_LENGTH];
        let mask = vec![0u16; MASK_CODE_LENGTH];
        node.insert_iris_share(next_id, &code, &mask, &code, &mask)
            .await?;
        tracing::info!(
            "inserted iris {next_id} into party {RESTARTED_PARTY} to change its startup facts"
        );

        fleet.start_party(RESTARTED_PARTY, ctx);

        // Every party must exit. `wait_all_exited` fails if any is still running at
        // the deadline, which is what a silently-accepted mismatch would look like.
        let outcomes = fleet.wait_all_exited(FLEET_TIMEOUT).await?;
        tracing::info!("fleet refused to start: {outcomes:?}");

        // At least one party must have *failed*, not merely stopped. All-clean
        // exits would mean the fleet wound down for some other reason — the test
        // process aborting, say — and this workflow would pass without ever having
        // exercised a mismatch.
        if outcomes.iter().all(|o| o.contains("without an error")) {
            eyre::bail!("no party reported an error; the mismatch was never detected: {outcomes:?}");
        }
        Ok(())
    }

    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // No party may be serving. Ready here would mean a party accepted a fleet
        // state its peers had not agreed to.
        for config in ctx.configs.iter() {
            let url = format!("http://127.0.0.1:{}/ready", config.healthcheck_port);
            if let Ok(response) = reqwest::get(&url).await {
                if response.status().is_success() {
                    eyre::bail!(
                        "party {} reports ready after an epoch mismatch",
                        config.party_id
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
        // Drop the extra iris row so the next workflow starts from a clean fleet.
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }
}
