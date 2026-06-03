//! WAL-sidecar daemon entry point.
//!
//! Wires together a GraphPg connection, an S3 client, an MPC NetworkHandle,
//! and the checkpoint protocol's `sidecar_main` daemon loop.

#![recursion_limit = "256"]

use clap::Parser;
use eyre::Result;
use std::process::exit;
use std::time::Duration;

use ampc_actor_utils::network::tcp::TlsConfig;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_cpu::{
    checkpoint_protocol::{sidecar_main, SidecarConfig},
    execution::hawk_main::{build_hawk_network_handle, HawkArgs},
    graph_checkpoint::PruningMode,
    hnsw::graph::graph_store::GraphPg,
};
use tokio::signal::unix::{signal, SignalKind};
use tokio_util::sync::CancellationToken;

/// Sidecar CLI args. Mirrors the networking shape of [`HawkArgs`] and adds
/// the sidecar-specific knobs (bucket, intervals, mutation gate).
#[derive(Debug, Parser, Clone)]
pub struct SidecarArgs {
    /// 0, 1, or 2.
    #[clap(long)]
    pub party_index: usize,

    /// Listen addresses for each party (in party-index order). Same shape
    /// as `HawkArgs::addresses`.
    #[clap(long, value_delimiter = ',')]
    pub addresses: Vec<String>,

    /// Outbound dial addresses for each party (used when a proxy sits
    /// between the listener and the dial target, e.g. in tests).
    #[clap(long, value_delimiter = ',')]
    pub outbound_addrs: Vec<String>,

    /// Postgres connection URL.
    #[clap(long, env = "SMPC__CPU_DATABASE__URL")]
    pub db_url: String,

    /// Postgres schema name.
    #[clap(long, env = "SMPC__CPU_DATABASE__SCHEMA")]
    pub db_schema: String,

    /// S3 bucket holding graph checkpoint objects.
    #[clap(long, env = "SMPC__GRAPH_CHECKPOINT_BUCKET_NAME")]
    pub bucket: String,

    /// Sleep between successful (or skipped) cycles, in seconds.
    #[clap(long, default_value = "300")]
    pub cycle_interval_secs: u64,

    /// Sleep after a transient cycle error before retrying, in seconds.
    #[clap(long, default_value = "10")]
    pub retry_interval_secs: u64,

    /// Per-peer-round timeout passed to the protocol, in seconds.
    #[clap(long, default_value = "10")]
    pub peer_round_timeout_secs: u64,

    /// Minimum new mutations beyond the base to run a cycle. Smaller deltas
    /// are skipped to avoid hammering S3.
    #[clap(long, default_value = "10000")]
    pub min_mutations_per_cycle: u64,

    /// Window of recent checkpoints advertised in Phase 1's base-list
    /// exchange. Matches genesis's 10-deep horizon.
    #[clap(long, default_value = "10")]
    pub checkpoint_window: usize,

    /// Mark produced rows as archival (kept by pruning).
    #[clap(long, default_value_t = false)]
    pub is_archival: bool,

    /// Run a single cycle and exit, rather than looping at `cycle_interval`.
    /// Required when deployed as a CronJob (one fire = one cycle).
    #[clap(long, default_value_t = false)]
    pub one_shot: bool,

    #[clap(long, default_value = "1")]
    pub connection_parallelism: usize,

    #[clap(long, default_value = "1")]
    pub request_parallelism: usize,

    /// Pruning mode for old checkpoints after uploading a new checkpoint
    ///
    /// Accepted values:
    ///   - none                 — do not prune any checkpoints
    ///   - older-non-archival   — prune older non-archival checkpoints (default)
    ///   - all-older            — prune all older checkpoints
    #[clap(long("pruning-mode"))]
    pruning_mode: Option<String>,
}

impl SidecarArgs {
    fn cycle_interval(&self) -> Duration {
        Duration::from_secs(self.cycle_interval_secs)
    }
    fn retry_interval(&self) -> Duration {
        Duration::from_secs(self.retry_interval_secs)
    }
    fn peer_round_timeout(&self) -> Duration {
        Duration::from_secs(self.peer_round_timeout_secs)
    }
}

fn hawk_args_from(args: &SidecarArgs) -> HawkArgs {
    HawkArgs {
        party_index: args.party_index,
        addresses: args.addresses.clone(),
        outbound_addrs: args.outbound_addrs.clone(),
        request_parallelism: args.request_parallelism,
        connection_parallelism: args.connection_parallelism,
        hnsw_param_ef_constr: 0,
        hnsw_param_m: 0,
        hnsw_param_ef_search: 0,
        hnsw_param_ef_supermatch: 0,
        hnsw_param_ef_saturation_margin: 0,
        hnsw_layer_density: None,
        hnsw_fixed_layer_search_batch_size: None,
        hnsw_prf_key: None,
        disable_persistence: true,
        hnsw_disable_memory_persistence: true,
        tls: None::<TlsConfig>,
        numa: false,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Logs go to stdout (captured by the k8s log pipeline); no StatsD recorder.
    // Holds the handle for the process lifetime so the subscriber stays live.
    let _tracing = initialize_tracing(None)?;

    let args = SidecarArgs::parse();
    let pruning_mode = if let Some(mode_str) = args.pruning_mode.as_ref() {
        mode_str.parse::<PruningMode>().map_err(|e| {
            eprintln!("Error: --pruning-mode argument invalid: {}", e);
            e
        })?
    } else {
        PruningMode::OlderNonArchival
    };

    println!(
        "Starting WAL sidecar daemon: party={}, bucket={}",
        args.party_index, args.bucket
    );

    let shutdown_ct = CancellationToken::new();
    {
        let ct = shutdown_ct.clone();
        tokio::spawn(async move {
            let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
            let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("received SIGTERM; signalling shutdown"),
                _ = sigint.recv() => tracing::info!("received SIGINT; signalling shutdown"),
            }
            ct.cancel();
        });
    }

    let postgres =
        PostgresClient::new(&args.db_url, &args.db_schema, AccessMode::ReadWrite).await?;
    let graph_store: GraphPg<iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store> =
        GraphPg::new(&postgres).await?;

    let aws_cfg = aws_config::from_env().load().await;
    let s3_client = aws_sdk_s3::Client::new(&aws_cfg);

    let hawk_args = hawk_args_from(&args);
    let mut networking = build_hawk_network_handle(&hawk_args, shutdown_ct.clone()).await?;

    let cfg = SidecarConfig {
        bucket: args.bucket.clone(),
        party_id: args.party_index,
        cycle_interval: args.cycle_interval(),
        retry_interval: args.retry_interval(),
        peer_round_timeout: args.peer_round_timeout(),
        min_mutations_per_cycle: args.min_mutations_per_cycle,
        checkpoint_window: args.checkpoint_window,
        is_archival: args.is_archival,
        pruning_mode,
        one_shot: args.one_shot,
    };

    match sidecar_main(cfg, &graph_store, &s3_client, &mut networking, shutdown_ct).await {
        Ok(()) => {
            tracing::info!("sidecar exited cleanly");
            Ok(())
        }
        Err(e) => {
            tracing::error!("sidecar exited with fatal error: {e}");
            exit(1);
        }
    }
}
