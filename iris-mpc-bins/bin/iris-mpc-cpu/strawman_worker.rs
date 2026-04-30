#![recursion_limit = "256"]

use clap::{Parser, ValueEnum};
use eyre::Result;
use iris_mpc_cpu::{
    hawkers::aby3::aby3_store::DistanceMode,
    strawman_worker::{run_local, StrawmanWorkerArgs},
};
use std::process::exit;
use tokio::signal;
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
#[command(
    about = "Strawman worker for the remote IrisWorkerPool. Throwaway dev/test \
             companion until the production worker binary lands; expected to \
             match its wire behavior via iris-mpc-worker-protocol."
)]
struct Cli {
    /// Identity of this worker (free-form string passed to ampc-actor-utils).
    #[arg(long)]
    worker_id: String,
    /// `host:port` this worker listens on.
    #[arg(long)]
    worker_address: String,
    /// Identity of the leader (Hawk Main) connecting in.
    #[arg(long)]
    leader_id: String,
    /// `host:port` of the leader, used for handshake validation.
    #[arg(long)]
    leader_address: String,
    /// 0-based party id; drives the dummy iris used for `delete_irises`.
    #[arg(long)]
    party_id: usize,
    /// Distance mode for `compute_dot_products` and pairwise distances.
    /// Must match what Hawk Main is configured with — silent divergence
    /// here corrupts results.
    #[arg(long, value_enum, default_value_t = DistanceModeArg::MinRotation)]
    distance_mode: DistanceModeArg,
    /// Pin worker threads to NUMA-local cores. Off by default for dev.
    #[arg(long, default_value_t = false)]
    numa: bool,
    /// Reserved for future shard routing. Ignored by v1.
    #[arg(long, default_value_t = 0)]
    shard_index: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DistanceModeArg {
    Simple,
    MinRotation,
}

impl From<DistanceModeArg> for DistanceMode {
    fn from(v: DistanceModeArg) -> Self {
        match v {
            DistanceModeArg::Simple => DistanceMode::Simple,
            DistanceModeArg::MinRotation => DistanceMode::MinRotation,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let args = StrawmanWorkerArgs {
        worker_id: cli.worker_id,
        worker_address: cli.worker_address,
        leader_id: cli.leader_id,
        leader_address: cli.leader_address,
        party_id: cli.party_id,
        distance_mode: cli.distance_mode.into(),
        numa: cli.numa,
        shard_index: cli.shard_index,
    };

    let shutdown_ct = CancellationToken::new();
    let ct_for_signal = shutdown_ct.clone();
    let signal_task = tokio::spawn(async move {
        if signal::ctrl_c().await.is_ok() {
            tracing::info!("SIGINT received, shutting down strawman worker");
            ct_for_signal.cancel();
        }
    });

    match run_local(args, shutdown_ct).await {
        Ok(()) => {
            tracing::info!("strawman worker exited cleanly");
        }
        Err(e) => {
            tracing::error!("strawman worker error: {e}");
            signal_task.abort();
            exit(1);
        }
    }
    signal_task.abort();
    Ok(())
}
