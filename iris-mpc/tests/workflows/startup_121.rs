/// startup_121 — one party restarts during the startup handshake with *changed*
/// data, and the whole fleet must refuse to come up.
///
/// Setup: three empty parties, party 2 configured to hold in [`Phase::Propose`].
/// Exec: once the survivors have committed an epoch derived from party 2's facts,
/// party 2 is stopped, an extra iris row is inserted into its database, and it is
/// restarted without the hold. Every party must exit and none may report ready.
///
/// Only *some* party has to name the mismatch. The one that comes back always
/// detects it: its first barrier round sees both peers on the old epoch. Its peers
/// usually do not — it publishes the new epoch and bails within milliseconds, far
/// inside their poll interval, after which their polls only see a closed port, which
/// the barrier correctly reads as "a peer is restarting" rather than as
/// disagreement, so they exit on their own timeout. Both routes satisfy what this
/// workflow needs: nobody serves, and the fleet tears itself down. Hence
/// [`BARRIER_TIMEOUT_SECS`] — with the 300s default the survivors would still be
/// waiting long after `FLEET_TIMEOUT`.
///
/// The hold is at `Propose`, i.e. *before* party 2 publishes an epoch, which keeps
/// the survivors parked in the commit barrier. Holding at `Commit` instead would
/// release them into the DB load while party 2 has no MPC listener, and they would
/// die of a checkpoint peer timeout before the data was even changed — a pass for
/// entirely the wrong reason.
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
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use std::array;

/// Commit-barrier budget for every party in this workflow.
///
/// Long enough that the returning party gets its chance to publish a mismatched
/// epoch — a restart costs about eleven seconds here, ten of them the empty-queue
/// long poll in `build_sync_state` — and short enough that a fleet which never
/// observes it still gives up well inside `FLEET_TIMEOUT`. Whichever way it goes,
/// `exec` finishes in under a minute instead of waiting out the 300s default.
const BARRIER_TIMEOUT_SECS: u64 = 30;

/// Substrings identifying a party that failed *on the epoch comparison* rather
/// than on a barrier timeout. The two checks that can reach an
/// `EpochVerdict::Void` word it differently, and either one proves the
/// comparison under test actually ran.
const EPOCH_MISMATCH_ERRORS: [&str; 2] = [
    // wait_for_epoch_commit
    "startup epoch mismatch",
    // spawn_startup_watch
    "startup watch found a peer on a different epoch",
];

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
        // gets a chance to stop it (`HawkFleet::drop` is the backstop).
        let options = FleetOptions {
            holds: array::from_fn(|party| (party == RESTARTED_PARTY).then_some(Phase::Propose)),
            startup_sync_timeout_secs: Some(BARRIER_TIMEOUT_SECS),
        };
        self.fleet = Some(HawkFleet::start_all_with(&ctx.configs, ctx, options).await?);
        let fleet = self.fleet.as_mut().unwrap();

        fleet
            .wait_for_phase(RESTARTED_PARTY, Phase::Propose, FLEET_TIMEOUT)
            .await?;

        // The survivors must already hold an epoch derived from this party's *old*
        // facts before it goes away — otherwise they would read the new ones on
        // restart and legitimately agree, which is correct behaviour but not the case
        // under test. Reaching Commit is exactly that, and leaves them parked in the
        // commit barrier waiting for party 2, which never publishes an epoch because
        // it is held in Propose.
        for party in (0..COUNT_OF_PARTIES).filter(|p| *p != RESTARTED_PARTY) {
            fleet
                .wait_for_phase(party, Phase::Commit, FLEET_TIMEOUT)
                .await?;
        }

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

        // Restarted without the hold: it re-derives an epoch, now a different one.
        fleet.start_party(RESTARTED_PARTY, ctx, None);

        // Every party must exit. `wait_all_exited` fails if any is still running at
        // the deadline, which is what a silently-accepted mismatch would look like.
        let outcomes = fleet.wait_all_exited(FLEET_TIMEOUT).await?;
        tracing::info!("fleet refused to start: {outcomes:?}");

        // Some party must have named the mismatch. This ties the workflow to the
        // check under test: the survivors' own error is a barrier *timeout*, which a
        // peer that never came back at all would produce just as well, so "the fleet
        // exited" alone would pass even with the epoch comparison broken.
        let detected = outcomes.iter().any(|outcome| {
            EPOCH_MISMATCH_ERRORS
                .iter()
                .any(|reason| outcome.contains(reason))
        });
        if !detected {
            eyre::bail!(
                "no party detected the epoch mismatch; the fleet stopped for some other \
                 reason: {outcomes:?}"
            );
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
