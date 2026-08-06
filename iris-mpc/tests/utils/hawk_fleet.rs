//! Per-party control over the three `server_main` instances.
//!
//! [`super::wait_conditions`] treats the fleet as a unit: one
//! `CancellationToken` for all three parties and a `JoinSet` that only reports
//! *that* something exited. The startup-epoch tests need finer control — stop
//! exactly one party, prove the other two are still running, restart the first
//! — so this module keeps a `JoinHandle` and a token per party, and exposes the
//! `/startup-state` document so a test can act on a party's real phase instead
//! of guessing with sleeps.
//!
//! # Why the kill point is a configured hold, not a poll
//!
//! The startup workflows have to stop a party *in* a named phase. Polling
//! `/startup-state` for that cannot work: on an empty local fleet `propose -> load`
//! takes ~180 ms end to end, so a 100 ms poll grid routinely first observes the
//! party already past the intended kill point, having published its epoch and
//! released its peers' commit barrier. So the party is *held* instead:
//! [`HawkFleet::start_party`] sets `Config::startup_hold_at_phase` and the party
//! parks there until stopped; the restart is spawned without the hold. The hold is
//! excluded from `CommonConfig`, so it does not perturb the derived epoch.
//!
//! # Why the commit-barrier budget is overridable
//!
//! `wait_for_epoch_commit` treats an unreachable peer as "it is restarting, keep
//! waiting" — which is what makes rejoin work, and why a party that *dies* is
//! indistinguishable from one coming back. The survivors then wait out the whole
//! 300s `startup_sync_timeout_secs`, longer than [`FLEET_TIMEOUT`], so a workflow
//! whose expected outcome is "the fleet gives up" would hit the harness deadline
//! instead of the behaviour it asserts. [`FleetOptions::startup_sync_timeout_secs`]
//! shortens it for those. Rejoin workflows leave the default, since there the
//! survivors *must* outlast a peer restart.
//!
//! # Why the startup workflows kill during the handshake
//!
//! `init_hawk_actor` runs `restart_from_checkpoint` — a cross-party consensus round
//! with a 10s `PEER_ROUND_TIMEOUT` — concurrently with the iris load via
//! `try_join!`. A party that disappears during the load therefore fails its peers
//! *inside* the load, whatever the epoch layer would have decided. Killing during
//! the handshake parks the survivors in the commit barrier, where the outcome is
//! determined entirely by the epoch logic under test. A load-window rejoin test
//! becomes meaningful once the checkpoint round tolerates a peer restart, or once
//! `Phase::Load` is split into local iris and cross-party graph phases.

use std::time::Duration;

use ampc_server_utils::ReadyProbeResponse;
use eyre::{bail, eyre, WrapErr};
use iris_mpc::server::startup_phase::{Epoch, Phase, StartupState};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info_span, Instrument};

use super::runner::CpuTestContext;
use super::{CpuConfigs, COUNT_OF_PARTIES, TEST_THREAD_STACK_SIZE};

/// Party the startup-epoch workflows restart. Arbitrary — nothing in the design
/// distinguishes the parties and the commit barrier is symmetric.
pub const RESTARTED_PARTY: usize = 2;

/// Wall-clock allowance for a fleet to reach ready, or to give up. Generous:
/// covers three parties' migrations, coordination barriers and DB load on a cold
/// CI machine.
pub const FLEET_TIMEOUT: Duration = Duration::from_secs(120);

/// Startup-config overrides for a whole fleet.
#[derive(Debug, Clone, Copy, Default)]
pub struct FleetOptions {
    /// Phase each party parks in, if any. Applies to the parties' first boot; a
    /// restart is spawned by the workflow and passes its own hold (normally none).
    pub holds: [Option<Phase>; COUNT_OF_PARTIES],
    /// Replaces `ServerCoordinationConfig::startup_sync_timeout_secs`, which bounds
    /// the startup barriers — including the commit barrier. Inherited by restarts.
    /// See the module docs for when a workflow needs to shorten it.
    pub startup_sync_timeout_secs: Option<u64>,
}

/// Spawn `server_main` for one party.
///
/// `server_main` holds `!Send` state, so the party runs on its own OS thread via
/// `spawn_blocking` with a dedicated multi-thread runtime.
///
/// Mirrors the per-party body of [`crate::workflows::run_hawk`], which spawns all
/// three into a `JoinSet` under one shared token. That function is left alone
/// because every `wal_*` test depends on its cancellation semantics; the two must
/// be kept in sync.
fn spawn_hawk_party(
    party: usize,
    configs: &CpuConfigs,
    shutdown: CancellationToken,
    ctx: &CpuTestContext,
    hold_at: Option<Phase>,
    startup_sync_timeout_secs: Option<u64>,
) -> JoinHandle<eyre::Result<()>> {
    let mut config = crate::utils::configs::make_hawk_config(&configs[party], configs, &ctx.env);
    config.startup_hold_at_phase = hold_at.map(|phase| phase.as_str().to_string());
    if let Some(secs) = startup_sync_timeout_secs {
        config
            .server_coordination
            .as_mut()
            .expect("test configs always set server_coordination")
            .startup_sync_timeout_secs = secs;
    }
    let abort = ctx.abort.clone();

    tokio::spawn(async move {
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(TEST_THREAD_STACK_SIZE)
                .build()
                .expect("failed to build server runtime");
            let span = info_span!("mpc_node", idx = party);
            rt.block_on(async move {
                tokio::select! {
                    res = iris_mpc::server::server_main(config).instrument(span) => res,
                    _ = shutdown.cancelled() => Ok(()),
                    _ = abort.cancelled() => Ok(()),
                }
            })
        })
        .await
        .map_err(|e| eyre!("server task panicked: {e}"))
        .and_then(|r| r)
    })
}

/// Read one party's `/startup-state` document.
async fn fetch_startup_state(port: u16) -> eyre::Result<StartupState> {
    let url = format!("http://127.0.0.1:{port}/startup-state");
    let body = reqwest::get(&url)
        .await
        .wrap_err_with(|| format!("GET {url} failed"))?
        .bytes()
        .await
        .wrap_err_with(|| format!("reading body of {url} failed"))?;
    serde_json::from_slice(&body).wrap_err_with(|| format!("deserializing {url} failed"))
}

/// The three parties, each independently stoppable.
pub struct HawkFleet {
    tasks: [Option<JoinHandle<eyre::Result<()>>>; COUNT_OF_PARTIES],
    tokens: [Option<CancellationToken>; COUNT_OF_PARTIES],
    configs: CpuConfigs,
    /// Kept so restarts get the same barrier budget as the first boot; the
    /// holds deliberately are not kept.
    startup_sync_timeout_secs: Option<u64>,
}

/// Cancel and abort every party when the fleet goes out of scope.
///
/// Essential, not tidiness. Dropping a `JoinHandle` *detaches* its task, so
/// without this a workflow that fails partway through `exec` — before it hands
/// the fleet to `self` — leaves three `server_main` instances running with their
/// coordination ports held, and the next workflow in the same test process dies
/// with `Address already in use`.
///
/// Cancelling the token is the part that matters: the party runs inside
/// `spawn_blocking`, which cannot be aborted once started, so it only stops when
/// `server_main`'s `select!` observes the cancellation. The port is therefore
/// freed shortly *after* the drop, not during it — which is why
/// [`HawkFleet::start_all`] waits for ports rather than assuming they are free.
impl Drop for HawkFleet {
    fn drop(&mut self) {
        for party in 0..COUNT_OF_PARTIES {
            if let Some(token) = self.tokens[party].take() {
                token.cancel();
            }
            if let Some(task) = self.tasks[party].take() {
                task.abort();
            }
        }
    }
}

/// Poll until nothing accepts connections on `port`.
async fn wait_for_port_free(port: u16, dur: Duration) -> eyre::Result<()> {
    let deadline = tokio::time::Instant::now() + dur;
    loop {
        match TcpStream::connect(("127.0.0.1", port)).await {
            Err(_) => return Ok(()),
            Ok(stream) => drop(stream),
        }
        if tokio::time::Instant::now() >= deadline {
            bail!("something is still listening on port {port} after {dur:?}");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

impl HawkFleet {
    /// Start all three parties, once their coordination ports are free.
    ///
    /// The wait covers a previous fleet still shutting down: `Drop` cancels but
    /// cannot join, so ports may be released a moment later. Failing here with
    /// "still listening" is also a far better diagnostic than letting one party
    /// die of `Address already in use` several seconds into its startup.
    pub async fn start_all(configs: &CpuConfigs, ctx: &CpuTestContext) -> eyre::Result<Self> {
        Self::start_all_with(configs, ctx, FleetOptions::default()).await
    }

    /// Start all three parties under `options`.
    ///
    /// A hold has to be in place from a party's *first* boot: it is the only way to
    /// be sure it has not already run past the phase the workflow means to stop it
    /// in. See [`HawkFleet::start_party`].
    pub async fn start_all_with(
        configs: &CpuConfigs,
        ctx: &CpuTestContext,
        options: FleetOptions,
    ) -> eyre::Result<Self> {
        for config in configs.iter() {
            wait_for_port_free(config.healthcheck_port, Duration::from_secs(30))
                .await
                .wrap_err("a previous fleet has not released its coordination port")?;
        }

        let mut fleet = HawkFleet {
            tasks: [None, None, None],
            tokens: [None, None, None],
            configs: configs.clone(),
            startup_sync_timeout_secs: options.startup_sync_timeout_secs,
        };
        for party in 0..COUNT_OF_PARTIES {
            fleet.start_party(party, ctx, options.holds[party]);
        }
        Ok(fleet)
    }

    /// Start (or restart) one party, which must not currently be running.
    ///
    /// With `hold_at` set the party parks on entering that phase and stays there
    /// until stopped, so a test can kill it at an exact point in the handshake (see
    /// the module docs for why this is a hold rather than a well-timed poll). A held
    /// party never becomes ready, so the workflow must stop it — via
    /// [`HawkFleet::stop_party`] or the `Drop` backstop.
    pub fn start_party(&mut self, party: usize, ctx: &CpuTestContext, hold_at: Option<Phase>) {
        assert!(
            self.tasks[party].is_none(),
            "party {party} is already running"
        );
        let token = CancellationToken::new();
        let task = spawn_hawk_party(
            party,
            &self.configs,
            token.clone(),
            ctx,
            hold_at,
            self.startup_sync_timeout_secs,
        );
        self.tokens[party] = Some(token);
        self.tasks[party] = Some(task);
        tracing::info!("fleet: started party {party} (hold_at {hold_at:?})");
    }

    /// Stop one party and wait until its coordination port is free again, so a
    /// restart cannot race the old listener's teardown.
    pub async fn stop_party(&mut self, party: usize) -> eyre::Result<()> {
        let Some(token) = self.tokens[party].take() else {
            bail!("party {party} is not running");
        };
        token.cancel();

        let mut task = self.tasks[party]
            .take()
            .ok_or_else(|| eyre!("party {party} has a token but no task"))?;

        // `&mut task` so the handle survives a timeout and can be aborted. Letting
        // it drop would detach the task instead of stopping it.
        match tokio::time::timeout(Duration::from_secs(60), &mut task).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(err))) => {
                tracing::warn!("party {party} returned an error while stopping: {err:#}")
            }
            Ok(Err(join_err)) if join_err.is_cancelled() => {}
            Ok(Err(join_err)) => bail!("party {party} panicked while stopping: {join_err}"),
            Err(_) => {
                task.abort();
                bail!("party {party} did not stop within 60s");
            }
        }

        // `server_main`'s `TaskMonitor` aborts the coordination listener when the
        // future is dropped, but the socket close is not synchronous with the task
        // returning; binding too early would fail the restarted party's listener.
        wait_for_port_free(
            self.configs[party].healthcheck_port,
            Duration::from_secs(30),
        )
        .await
        .wrap_err_with(|| format!("party {party} did not release its coordination port"))?;
        tracing::info!("fleet: stopped party {party}");
        Ok(())
    }

    /// Fail if a party is not running, or if its task has already exited.
    ///
    /// This is the assertion the rejoin test rests on: the parties that were never
    /// touched must still be in their original process.
    pub fn assert_still_running(&self, party: usize) -> eyre::Result<()> {
        match &self.tasks[party] {
            None => bail!("party {party} is not running"),
            Some(task) if task.is_finished() => {
                bail!("party {party} exited when it should have kept running")
            }
            Some(_) => Ok(()),
        }
    }

    /// [`HawkFleet::assert_still_running`] for every party but `except`.
    pub fn assert_others_running(&self, except: usize) -> eyre::Result<()> {
        (0..COUNT_OF_PARTIES)
            .filter(|party| *party != except)
            .try_for_each(|party| self.assert_still_running(party))
    }

    /// Wait until a party is sitting in exactly `phase`.
    ///
    /// Exact, not `>=`: the callers act on the party at this point, and one already
    /// past it has published state its peers have acted on, so the rest of the
    /// workflow would assert against a different scenario than the one it describes.
    /// Overshoot therefore fails here, naming the phase actually seen, rather than
    /// surfacing later as a confusing assertion about a peer that behaved correctly.
    /// Pair with a `hold_at`, which makes overshoot impossible.
    ///
    /// Tolerates connection failures while the party's coordination server is still
    /// coming up.
    pub async fn wait_for_phase(
        &self,
        party: usize,
        phase: Phase,
        dur: Duration,
    ) -> eyre::Result<StartupState> {
        let port = self.configs[party].healthcheck_port;
        let deadline = tokio::time::Instant::now() + dur;
        let mut last: Option<Phase> = None;

        loop {
            match fetch_startup_state(port).await {
                Ok(state) if state.phase == phase => return Ok(state),
                Ok(state) if state.phase > phase => bail!(
                    "party {party} ran past phase {phase} to {} before it could be observed — \
                     it should have been started with a hold at {phase}",
                    state.phase
                ),
                Ok(state) => last = Some(state.phase),
                Err(err) => tracing::debug!("party {party} startup state unavailable: {err:#}"),
            }

            self.assert_still_running(party)?;

            if tokio::time::Instant::now() >= deadline {
                bail!(
                    "party {party} did not reach phase {phase} within {dur:?} (last seen {:?})",
                    last
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Current phase of a running party.
    pub async fn phase(&self, party: usize) -> eyre::Result<Phase> {
        Ok(fetch_startup_state(self.configs[party].healthcheck_port)
            .await?
            .phase)
    }

    /// Every party's published epoch. Fails if any party has not derived one, or if
    /// they do not all agree.
    pub async fn agreed_epoch(&self) -> eyre::Result<Epoch> {
        let mut epochs = Vec::with_capacity(COUNT_OF_PARTIES);
        for party in 0..COUNT_OF_PARTIES {
            let state = fetch_startup_state(self.configs[party].healthcheck_port).await?;
            epochs.push(state.epoch.ok_or_else(|| {
                eyre!(
                    "party {party} has not derived an epoch (phase {})",
                    state.phase
                )
            })?);
        }
        if epochs.iter().any(|epoch| *epoch != epochs[0]) {
            bail!("parties disagree on the startup epoch: {epochs:?}");
        }
        Ok(epochs[0])
    }

    /// Wait until all three parties report `is_ready` on `/health`.
    ///
    /// Fails fast if any party's task exits first — a party that dies is never
    /// going to become ready, and the exit is the interesting diagnostic.
    pub async fn wait_all_ready(&self, dur: Duration) -> eyre::Result<()> {
        let deadline = tokio::time::Instant::now() + dur;

        loop {
            let mut all_ready = true;
            for party in 0..COUNT_OF_PARTIES {
                self.assert_still_running(party)?;

                let port = self.configs[party].healthcheck_port;
                let url = format!("http://127.0.0.1:{port}/health");
                match reqwest::get(&url).await {
                    Ok(response) => match response.json::<ReadyProbeResponse>().await {
                        Ok(probe) if probe.is_ready => {}
                        Ok(_) => all_ready = false,
                        Err(err) => {
                            tracing::debug!("party {party} health body unreadable: {err:#}");
                            all_ready = false;
                        }
                    },
                    Err(err) => {
                        tracing::debug!("party {party} health unreachable: {err:#}");
                        all_ready = false;
                    }
                }
            }

            if all_ready {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                bail!("parties did not all become ready within {dur:?}");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Wait until every running party's task has exited, returning one outcome
    /// description per party.
    ///
    /// Fails if any party died of a port conflict: that is a harness failure (a
    /// previous fleet leaked) and is indistinguishable from success to a caller that
    /// only asserts "nobody came up", so it must not silently pass a workflow.
    pub async fn wait_all_exited(&mut self, dur: Duration) -> eyre::Result<Vec<String>> {
        let deadline = tokio::time::Instant::now() + dur;
        let mut errors = Vec::new();

        for party in 0..COUNT_OF_PARTIES {
            let Some(mut task) = self.tasks[party].take() else {
                continue;
            };
            let token = self.tokens[party].take();

            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, &mut task).await {
                Ok(Ok(Ok(()))) => errors.push(format!("party {party} exited without an error")),
                Ok(Ok(Err(err))) => errors.push(format!("party {party}: {err:#}")),
                Ok(Err(join_err)) => {
                    task.abort();
                    bail!("party {party} panicked: {join_err}")
                }
                Err(_) => {
                    // A party still running is the failure this reports, but it must
                    // not be left detached holding its port for the next workflow.
                    if let Some(token) = token {
                        token.cancel();
                    }
                    task.abort();
                    bail!("party {party} was still running after {dur:?}");
                }
            }
        }

        for outcome in &errors {
            if outcome.contains("Address already in use") {
                bail!(
                    "a party died of a port conflict, not of the condition under test — a \
                     previous fleet leaked its ports: {errors:?}"
                );
            }
        }

        Ok(errors)
    }

    /// Stop every still-running party. Safe to call in teardown regardless of
    /// what the test already stopped.
    pub async fn stop_all(&mut self) -> eyre::Result<()> {
        let mut first_err: Option<eyre::Report> = None;
        for party in 0..COUNT_OF_PARTIES {
            if self.tasks[party].is_none() {
                continue;
            }
            if let Err(err) = self.stop_party(party).await {
                tracing::warn!("error stopping party {party}: {err:#}");
                first_err.get_or_insert(err);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}
