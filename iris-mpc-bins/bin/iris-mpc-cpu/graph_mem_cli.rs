#![recursion_limit = "256"]

use clap::{Parser, Subcommand};
use eyre::Result;
use iris_mpc_common::object_store::ObjectStoreClient;
use iris_mpc_cpu::hnsw::graph::test_utils::{DbContext, DiffMethod};
use iris_mpc_utils::misc::write_bin;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "graph_mem_cli")]
#[command(about = "Convert from db -> memory -> file or file -> memory -> db", long_about = None)]
struct Cli {
    /// Database URL
    #[arg(long)]
    db_url: String,
    /// Database schema
    #[arg(long)]
    schema: String,
    /// File path (input or output depending on mode)
    #[arg(long)]
    file: PathBuf,
    /// Display the graph in the terminal
    #[arg(long)]
    dbg: bool,
    /// S3 bucket for graph checkpoints
    #[arg(long)]
    s3_bucket: String,
    /// Party ID (0, 1, or 2) — used as part of S3 key namespacing
    #[arg(long)]
    party_id: usize,
    /// AWS region (defaults to env/instance metadata)
    #[arg(long)]
    aws_region: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Obtain the most recent checkpoint, apply all relevant graph mutations, and save it to a file
    BackupGraph,
    /// (testing only) creates random data and stores it as a checkpoint
    RandomCheckpoint,
    /// Load a graph from a file to memory, then upload it to s3 and add it to the checkpoint table.
    LoadCheckpoint,
    /// (testing only) run BackupGraph and compare it against a file.
    VerifyBackup,
    /// Load a graph from a file and compare it to the graph stored in the database.
    CompareToDb {
        /// The diffing method to use for the comparison.
        #[arg(long, value_enum, default_value_t = DiffMethod::DetailedJaccard)]
        diff_method: DiffMethod,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cli {
        db_url,
        schema,
        file,
        dbg,
        s3_bucket,
        party_id,
        aws_region,
        command,
    } = Cli::parse();

    let s3_client = ObjectStoreClient::new(aws_region, false);

    let db_context = DbContext::new(&db_url, &schema, s3_client, s3_bucket, party_id).await;

    match command {
        Command::BackupGraph => {
            let graph = db_context.get_both_eyes().await?;
            write_bin(&graph, &file)?;
        }
        Command::LoadCheckpoint => {
            db_context.make_new_checkpoint(&file, dbg).await?;
        }
        Command::VerifyBackup => {
            db_context.verify_backup(&file, dbg).await?;
        }
        Command::RandomCheckpoint => {
            let graph = db_context.store_random_graph().await?;
            write_bin(&graph, &file)?;
        }
        Command::CompareToDb { diff_method } => {
            db_context.compare_to_db(&file, diff_method, dbg).await?;
        }
    }

    // N.B. this command has it own output
    if !matches!(command, Command::CompareToDb { diff_method: _ }) {
        println!("Command succeeded");
    }

    Ok(())
}
