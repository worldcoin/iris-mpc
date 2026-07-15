//! Cycle drivers: [`sidecar_main`] (daemon loop) and
//! [`restart_from_checkpoint`] (one-shot Hawk startup).
//!
//! Each cycle opens a fresh `ControlChannel`; no transport state crosses
//! cycle boundaries, so per-cycle failures stay isolated.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ampc_actor_utils::network::mpc::NetworkHandle;
use ampc_actor_utils::network::tcp::deserialize_yaml_json_string;
use eyre::{eyre, Result};
use iris_mpc_common::object_store::ObjectStoreClient;
use serde::Deserialize;
use serde_with::{serde_as, DurationSeconds};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::checkpoint_protocol::{
    run_cycle, Blake3GraphHasher, CheckpointDownload, CycleConfig, CycleError, GraphMutationId,
    InstallAsServing, MostRecentCommon, Outcome, RebuildFromCheckpoint, RingConsensusTransport,
    SkipReason, UploadAndRecord,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::PruningMode;
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};

// ── In-process Sidecar  ───────────────────────────────────────────────────────

// the sidecar daemon gets these passed in via the CLI but for in-process these need to be
// environment variables.
//
// The sidecar process will not be run if the config field is None.
#[derive(Debug, Clone, Deserialize)]
pub struct SidecarConfigWrapper {
    /// listen addresses per party for the networking handle.
    /// corresponds to both inbound addrs and outbound addrs
    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub addresses: Vec<String>,
    #[serde(default = "default_sidecar_parallelism")]
    pub request_parallelism: usize,
    #[serde(default = "default_sidecar_parallelism")]
    pub connection_parallelism: usize,
    #[serde(default)]
    pub config: Option<SidecarConfig>,
}

fn default_sidecar_parallelism() -> usize {
    1
}

fn default_make_connections_timeout() -> Duration {
    Duration::from_secs(300)
}

impl SidecarConfigWrapper {
    pub fn load_config(prefix: &str) -> Result<Self> {
        let settings = config::Config::builder();
        let settings = settings
            .add_source(
                config::Environment::with_prefix(prefix)
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;
        let config: Self = settings.try_deserialize::<Self>()?;
        Ok(config)
    }
}

// ── Sidecar daemon ───────────────────────────────────────────────────────

/// Sidecar daemon configuration. Construct from the binary's CLI args /
/// environment.
#[serde_as]
#[derive(Debug, Clone, Deserialize)]
pub struct SidecarConfig {
    /// Bucket containing the checkpoint S3 objects.
    pub bucket: String,
    /// Party index (0, 1, 2). Drives S3 key prefix and metric labels.
    pub party_id: usize,
    /// Sleep between successful (or skipped) cycles.
    #[serde_as(as = "DurationSeconds<u64>")]
    pub cycle_interval: Duration,
    /// Sleep after a transient cycle error before retrying.
    #[serde_as(as = "DurationSeconds<u64>")]
    pub retry_interval: Duration,
    /// Per-peer-round timeout passed into the protocol's `CycleConfig`.
    #[serde_as(as = "DurationSeconds<u64>")]
    pub peer_round_timeout: Duration,
    /// Wall-clock bound on opening the control channel (establishing the peer
    /// mesh). The underlying dial retries with no deadline of its own, so an
    /// unreachable peer would otherwise wedge the cycle indefinitely; this
    /// converts that into a transient error (the next fire retries).
    #[serde(default = "default_make_connections_timeout")]
    #[serde_as(as = "DurationSeconds<u64>")]
    pub make_connections_timeout: Duration,
    /// Lower-bound for cycle work; below this the cycle is skipped (no
    /// upload / no DB write) to avoid hammering S3 with near-empty deltas.
    pub min_mutations_per_cycle: u64,
    /// Newest-first window of recent checkpoints advertised in Phase 1.
    pub checkpoint_window: usize,
    /// Marks the produced row as archival (skipped by pruning).
    pub is_archival: bool,
    /// Optionally remove old checkpoints from S3
    pub pruning_mode: PruningMode,
    /// Run exactly one cycle and exit instead of looping. Used when deployed
    /// as a CronJob: a transient/fatal cycle error returns `Err` (the next
    /// scheduled fire is the retry), so no party retries independently and
    /// desyncs the ring.
    pub one_shot: bool,
}

/// Daemon loop. Returns `Ok(())` only on graceful shutdown via
/// `shutdown_ct`; any `CycleError::Fatal` is propagated as an `Err`.
pub async fn sidecar_main<V: VectorStore + Send + Sync>(
    cfg: SidecarConfig,
    graph_store: &GraphPg<V>,
    s3_client: &ObjectStoreClient,
    networking: &mut Box<dyn NetworkHandle>,
    shutdown_ct: CancellationToken,
) -> Result<()> {
    tracing::info!(
        bucket = %cfg.bucket,
        party_id = cfg.party_id,
        cycle_interval = ?cfg.cycle_interval,
        "sidecar daemon starting"
    );

    loop {
        if shutdown_ct.is_cancelled() {
            tracing::info!("sidecar daemon shutting down (cancelled)");
            return Ok(());
        }

        let cycle_start = Instant::now();
        let outcome = sidecar_cycle(&cfg, graph_store, s3_client, networking).await;

        if cfg.one_shot {
            // Single completion line per outcome; a fire that never logs this
            // (e.g. wedged dialling an absent peer until activeDeadlineSeconds)
            // is itself the signal. Derive a log-based metric off `outcome`.
            let duration_secs = cycle_start.elapsed().as_secs_f64();
            return match outcome {
                Ok(Outcome::Finalized { height, .. }) => {
                    tracing::info!(
                        outcome = "finalized",
                        duration_secs,
                        height,
                        "sidecar one-shot complete"
                    );
                    Ok(())
                }
                Ok(Outcome::Skipped(reason)) => {
                    tracing::info!(
                        outcome = "skipped",
                        duration_secs,
                        ?reason,
                        "sidecar one-shot complete"
                    );
                    Ok(())
                }
                Err(CycleError::Transient(msg)) => {
                    tracing::warn!(outcome = "transient", duration_secs, error = %msg, "sidecar one-shot complete");
                    Err(eyre!("sidecar one-shot transient error: {msg}"))
                }
                Err(CycleError::Fatal(msg)) => {
                    tracing::error!(outcome = "fatal", duration_secs, error = %msg, "sidecar one-shot complete");
                    Err(eyre!("sidecar fatal: {msg}"))
                }
            };
        }

        let sleep_after = match outcome {
            Ok(Outcome::Finalized { height, .. }) => {
                tracing::info!(height, "sidecar cycle finalized");
                cfg.cycle_interval
            }
            Ok(Outcome::Skipped(reason)) => {
                tracing::debug!(?reason, "sidecar cycle skipped");
                cfg.cycle_interval
            }
            Err(CycleError::Transient(msg)) => {
                tracing::warn!(error = %msg, "sidecar cycle transient error");
                cfg.retry_interval
            }
            Err(CycleError::Fatal(msg)) => {
                tracing::error!(error = %msg, "sidecar cycle fatal error; exiting");
                return Err(eyre!("sidecar fatal: {msg}"));
            }
        };

        tokio::select! {
            _ = shutdown_ct.cancelled() => {
                tracing::info!("sidecar daemon shutting down (cancelled during sleep)");
                return Ok(());
            }
            _ = tokio::time::sleep(sleep_after) => {}
        }
    }
}

async fn sidecar_cycle<V: VectorStore + Send + Sync>(
    cfg: &SidecarConfig,
    graph_store: &GraphPg<V>,
    s3_client: &ObjectStoreClient,
    networking: &mut Box<dyn NetworkHandle>,
) -> Result<Outcome, CycleError> {
    let channel = tokio::time::timeout(cfg.make_connections_timeout, networking.control_channel())
        .await
        .map_err(|_| {
            CycleError::Transient(format!(
                "control_channel: peer mesh not established within {:?}",
                cfg.make_connections_timeout
            ))
        })?
        .map_err(|e| CycleError::Transient(format!("open control_channel: {e}")))?;
    let transport = RingConsensusTransport::new(channel);

    let mut materializer = RebuildFromCheckpoint::new(
        graph_store,
        s3_client,
        cfg.bucket.clone(),
        CheckpointDownload::Streaming,
    );
    let mut finalizer = UploadAndRecord::new(
        graph_store,
        s3_client,
        cfg.bucket.clone(),
        cfg.party_id,
        cfg.is_archival,
        cfg.pruning_mode,
    );
    let hasher = Blake3GraphHasher::new();

    let cycle_cfg = CycleConfig {
        min_mutations_to_apply: cfg.min_mutations_per_cycle,
        peer_round_timeout: cfg.peer_round_timeout,
        checkpoint_window: cfg.checkpoint_window,
    };

    run_cycle(
        &mut materializer,
        &mut finalizer,
        &transport,
        graph_store,
        &hasher,
        &MostRecentCommon,
        &cycle_cfg,
    )
    .await
}

// ── Restart ──────────────────────────────────────────────────────────────

/// Result of a restart attempt. The caller decides how to act on each.
#[derive(Debug)]
pub enum RestartOutcome {
    /// A graph was installed at the reported height.
    Installed { height: GraphMutationId },
    /// No party has any `genesis_graph_checkpoint` row (agreed via the ring,
    /// not inferred locally); caller falls back to its own bootstrap path
    /// (empty graph, plaintext checkpoint migration).
    NoCheckpoint,
    /// A peer's `current_max_mutation_id` came in below the agreed base.
    /// Restart can be retried once peers catch up; not a fatal error.
    PeerBehindBase {
        freeze: GraphMutationId,
        base: GraphMutationId,
    },
    /// The selector found no checkpoint shared across all parties (even
    /// within the recent window). Operator intervention is likely needed.
    NoCommonBase,
}

/// Max attempts for the restart consensus before giving up (and exiting).
const RESTART_MAX_ATTEMPTS: usize = 30;
/// Delay between restart consensus attempts.
const RESTART_RETRY_DELAY: Duration = Duration::from_secs(5);

/// Rebuild the live graph from the latest checkpoint + WAL replay, then
/// install it into `target`. `min_mutations_to_apply=0` so the cycle
/// proceeds even when there are no new mutations beyond the base.
pub async fn restart_from_checkpoint<V: VectorStore + Send + Sync>(
    graph_store: &GraphPg<V>,
    s3_client: &ObjectStoreClient,
    bucket: String,
    networking: &mut Box<dyn NetworkHandle>,
    target: BothEyes<Arc<RwLock<GraphMem>>>,
    peer_round_timeout: Duration,
    checkpoint_window: usize,
) -> Result<RestartOutcome> {
    tracing::info!(
        bucket = %bucket,
        checkpoint_window,
        peer_round_timeout = ?peer_round_timeout,
        "restart: restoring graph from checkpoint via ring consensus"
    );
    // No local-table fast path here: a party with an empty table must still
    // join the ring, or its peers block in the Phase 1 exchange until
    // timeout and crash-loop while this party silently boots empty.
    // "Nobody has a checkpoint" is a consensus outcome, not a local one.
    let hasher = Blake3GraphHasher::new();
    let cfg = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout,
        checkpoint_window,
    };

    // Retry transient failures instead of exiting. Re-open the channel each
    // attempt: a timed-out round can leave buffered frames that desync a reuse.
    let outcome = {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let channel = networking
                .control_channel()
                .await
                .map_err(|e| eyre!("open control_channel: {e}"))?;
            let transport = RingConsensusTransport::new(channel);
            let mut materializer = RebuildFromCheckpoint::new(
                graph_store,
                s3_client,
                bucket.clone(),
                CheckpointDownload::Buffered,
            );
            let mut finalizer = InstallAsServing::new(target.clone());

            match run_cycle(
                &mut materializer,
                &mut finalizer,
                &transport,
                graph_store,
                &hasher,
                &MostRecentCommon,
                &cfg,
            )
            .await
            {
                Ok(outcome) => break outcome,
                Err(CycleError::Transient(msg)) if attempt < RESTART_MAX_ATTEMPTS => {
                    tracing::warn!(
                        attempt,
                        max = RESTART_MAX_ATTEMPTS,
                        error = %msg,
                        "restart run_cycle transient; retrying"
                    );
                    tokio::time::sleep(RESTART_RETRY_DELAY).await;
                }
                Err(CycleError::Transient(msg)) => {
                    return Err(eyre!(
                        "restart run_cycle: transient after {attempt} attempts: {msg}"
                    ));
                }
                Err(CycleError::Fatal(msg)) => {
                    return Err(eyre!("restart run_cycle: {msg}"));
                }
            }
        }
    };

    match outcome {
        Outcome::Finalized { height, .. } => {
            tracing::info!(height, "restart installed graph from checkpoint");
            Ok(RestartOutcome::Installed { height })
        }
        Outcome::Skipped(SkipReason::PeerBehindBase { freeze, base }) => {
            tracing::warn!(freeze, base, "restart skipped: peer behind base");
            Ok(RestartOutcome::PeerBehindBase { freeze, base })
        }
        Outcome::Skipped(SkipReason::NoCheckpoints) => {
            tracing::info!("restart: no party has any checkpoint");
            Ok(RestartOutcome::NoCheckpoint)
        }
        Outcome::Skipped(SkipReason::NoCommonBase) => {
            tracing::warn!("restart skipped: no common base across parties");
            Ok(RestartOutcome::NoCommonBase)
        }
        Outcome::Skipped(SkipReason::NotEnoughMutations { .. }) => Err(eyre!(
            "restart unexpectedly skipped on NotEnoughMutations despite min=0"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_checkpoint::PruningMode;

    // Unique prefix per test: env is process-global and tests run in parallel.
    #[test]
    fn load_config_round_trips_env() {
        let vars = [
            (
                "WRAPPERTEST__ADDRESSES",
                r#"["10.0.0.1:7000","10.0.0.2:7000","10.0.0.3:7000"]"#,
            ),
            ("WRAPPERTEST__REQUEST_PARALLELISM", "4"),
            ("WRAPPERTEST__CONNECTION_PARALLELISM", "2"),
            ("WRAPPERTEST__CONFIG__BUCKET", "my-bucket"),
            ("WRAPPERTEST__CONFIG__PARTY_ID", "1"),
            ("WRAPPERTEST__CONFIG__CYCLE_INTERVAL", "30"),
            ("WRAPPERTEST__CONFIG__RETRY_INTERVAL", "5"),
            ("WRAPPERTEST__CONFIG__PEER_ROUND_TIMEOUT", "10"),
            ("WRAPPERTEST__CONFIG__MAKE_CONNECTIONS_TIMEOUT", "120"),
            ("WRAPPERTEST__CONFIG__MIN_MUTATIONS_PER_CYCLE", "100"),
            ("WRAPPERTEST__CONFIG__CHECKPOINT_WINDOW", "8"),
            ("WRAPPERTEST__CONFIG__IS_ARCHIVAL", "false"),
            ("WRAPPERTEST__CONFIG__PRUNING_MODE", "older-non-archival"),
            ("WRAPPERTEST__CONFIG__ONE_SHOT", "false"),
        ];
        for (k, v) in vars {
            std::env::set_var(k, v);
        }

        let wrapper =
            SidecarConfigWrapper::load_config("WRAPPERTEST").expect("env should deserialize");

        assert_eq!(
            wrapper.addresses,
            vec!["10.0.0.1:7000", "10.0.0.2:7000", "10.0.0.3:7000"]
        );
        assert_eq!(wrapper.request_parallelism, 4);
        assert_eq!(wrapper.connection_parallelism, 2);

        let cfg = wrapper.config.expect("config section should be present");
        assert_eq!(cfg.bucket, "my-bucket");
        assert_eq!(cfg.party_id, 1);
        assert_eq!(cfg.cycle_interval, Duration::from_secs(30));
        assert_eq!(cfg.retry_interval, Duration::from_secs(5));
        assert_eq!(cfg.peer_round_timeout, Duration::from_secs(10));
        assert_eq!(cfg.make_connections_timeout, Duration::from_secs(120));
        assert_eq!(cfg.min_mutations_per_cycle, 100);
        assert_eq!(cfg.checkpoint_window, 8);
        assert!(!cfg.is_archival);
        assert_eq!(cfg.pruning_mode, PruningMode::OlderNonArchival);

        for (k, _) in vars {
            std::env::remove_var(k);
        }
    }

    #[test]
    fn missing_config_section_disables_sidecar() {
        // No env set: wrapper parses via defaults, config=None disables the sidecar.
        let wrapper = SidecarConfigWrapper::load_config("WRAPPERTESTNONE")
            .expect("empty env should still deserialize");
        assert!(wrapper.config.is_none());
        assert!(wrapper.addresses.is_empty());
        assert_eq!(wrapper.request_parallelism, 1);
    }
}
