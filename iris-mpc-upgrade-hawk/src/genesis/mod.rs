mod delta;
mod graph_checkpoint;
mod indexation;
mod retry;
mod setup;
mod snapshot;

use eyre::{eyre, Result};
use iris_mpc_common::{config::Config, helpers::sync::Modification, SerialId};
pub use iris_mpc_cpu::genesis::BatchSizeConfig;
use iris_mpc_cpu::{
    genesis::state_accessor::set_last_indexed_modification_id, graph_checkpoint::PruningMode,
};

pub use graph_checkpoint::{reset_to_checkpoint, upload_and_sync_genesis_checkpoint};
pub use iris_mpc_cpu::graph_checkpoint::{
    get_common_checkpoint, get_most_recent_checkpoints, get_others_graph_hashes,
};

use delta::exec_delta;
use indexation::exec_indexation;
use setup::{exec_setup, SetupOutput};
use snapshot::exec_snapshot;

pub const PERSIST_DELAY: usize = 16;

/// Process input arguments typically passed from command line.
#[derive(Debug, Clone)]
pub struct ExecutionArgs {
    // Serial identifier of maximum indexed Iris.
    pub max_indexation_id: SerialId,

    // Batch size configuration (static or dynamic with cap).
    pub batch_size_config: BatchSizeConfig,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    pub perform_snapshot: bool,

    // Number of irises to index between checkpoints.
    pub checkpoint_frequency: usize,

    // Controls which older checkpoints are pruned after loading a common checkpoint.
    pub pruning_mode: PruningMode,

    // Pinned base checkpoint blake3 hash; None selects the latest common checkpoint.
    pub base_checkpoint_hash: Option<String>,
}

impl ExecutionArgs {
    // this is for integration tests
    pub fn from_plaintext_args(
        args: iris_mpc_cpu::genesis::plaintext::GenesisArgs,
        perform_snapshot: bool,
    ) -> Self {
        Self {
            max_indexation_id: args.max_indexation_id,
            batch_size_config: args.batch_size_config,
            perform_snapshot,
            checkpoint_frequency: args.checkpoint_frequency,
            pruning_mode: args.pruning_mode,
            base_checkpoint_hash: None,
        }
    }
}

/// Information associated with inner execution context.
struct ExecutionContextInfo {
    /// Process input args.
    args: ExecutionArgs,

    /// Process configuration.
    config: Config,

    // Serial idenitifer of last indexed Iris.
    last_indexed_id: SerialId,

    // Set identifiers of Iris's to be excluded from indexation.
    excluded_serial_ids: Vec<SerialId>,

    // Modifications since the base checkpoint's cursor; comparison-log input only.
    modifications: Vec<Modification>,

    // The largest modification id completed and persisted in the source store.
    // Used to track up to which modification the next run of Genesis can start from
    max_modification_persist_id: i64,

    // Whether a common base checkpoint was found (false = fresh start).
    has_base_checkpoint: bool,
}

/// Constructor.
impl ExecutionContextInfo {
    fn new(
        args: &ExecutionArgs,
        config: &Config,
        last_indexed_id: SerialId,
        excluded_serial_ids: Vec<SerialId>,
        modifications: Vec<Modification>,
        max_modification_persist_id: i64,
        has_base_checkpoint: bool,
    ) -> Self {
        Self {
            args: args.clone(),
            config: config.clone(),
            excluded_serial_ids,
            last_indexed_id,
            modifications,
            max_modification_persist_id,
            has_base_checkpoint,
        }
    }
}

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares
///
/// # Arguments
///
/// * `args` - Process arguments.
/// * `config` - Process configuration instance.
///
pub async fn exec(args: ExecutionArgs, config: Config) -> Result<()> {
    tracing::info!("running genesis with \n {:?} \n {:?}", args, config);

    // Phase 0: setup.
    let SetupOutput {
        ctx,
        shutdown_handler,
        mut task_monitor_bg,
        checkpoint_s3_client,
        aws_rds_client,
        registries,
        worker_pools,
        imem_graph_stores,
        mut hawk_handle,
        tx_results,
        graph_store,
        hnsw_iris_store,
        delta_exchange,
        prune_reports,
    } = exec_setup(&args, &config).await?;

    tracing::info!("Setup complete.");
    tracing::info!(
        "Starting Genesis indexing process with the following parameters:\n  Max indexation ID: {}\n  Batch size config: {}\n  Perform snapshot: {}",
        args.max_indexation_id,
        args.batch_size_config,
        args.perform_snapshot,
    );

    // Phase 1: apply delta. A fresh start (no base checkpoint, empty state)
    // has nothing to reconcile; only the modification-cursor stamp is written.
    if ctx.has_base_checkpoint {
        hawk_handle = exec_delta(
            &config,
            &ctx,
            graph_store.clone(),
            &checkpoint_s3_client,
            &registries,
            &worker_pools,
            &hnsw_iris_store,
            &imem_graph_stores,
            &delta_exchange,
            prune_reports.as_ref(),
            hawk_handle,
            &tx_results,
            &mut task_monitor_bg,
            &shutdown_handler,
        )
        .await?;
        tracing::info!("Delta complete.");
    } else {
        tracing::info!(
            "No base checkpoint; skipping delta. Setting last indexed modification id to {}",
            ctx.max_modification_persist_id
        );
        let mut graph_tx = graph_store.tx().await?;
        set_last_indexed_modification_id(&mut graph_tx.tx, ctx.max_modification_persist_id).await?;
        graph_tx.tx.commit().await?;
    }

    // Phase 2: indexation.
    exec_indexation(
        &ctx,
        &checkpoint_s3_client,
        &registries,
        &worker_pools,
        &imem_graph_stores,
        hawk_handle,
        &tx_results,
        task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    tracing::info!("Indexation complete.");

    // Phase 3: snapshot.
    if !args.perform_snapshot {
        tracing::info!("Snapshot skipped.");
    } else {
        exec_snapshot(&ctx, &aws_rds_client).await?;
        tracing::info!("Snapshot complete.");
    };

    // Clear modifications from the HNSW iris store
    // This is because after a genesis run - there should be no modifications left in the HNSW iris store
    let mut tx = hnsw_iris_store.tx().await?;
    hnsw_iris_store
        .clear_modifications_table(&mut tx)
        .await
        .map_err(|err| {
            let msg = format!("Failed to clear modifications: {:?}", err);
            tracing::error!("{}", msg);
            eyre!(msg)
        })?;
    tx.commit().await?;

    tracing::info!("Cleared modifications from the HNSW iris store");

    // trigger manual shutdown to ensure the health check services terminate
    shutdown_handler.trigger_manual_shutdown();

    Ok(())
}
