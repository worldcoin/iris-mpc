use clap::{Parser, Subcommand};
use eyre::Result;
use iris_mpc_cpu::hnsw::graph::test_utils::DbContext;
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
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load a graph from a database to memory, then write it to a file.
    BackupDb,
    /// (testing only) creates random data, stores it in the database, and then runs backup-db.
    StoreRandom,
    /// Load a graph from a file to memory, then write it to a database
    RestoreDb,
    /// (testing only) verify that Load/Store works as expected
    VerifyBackup,
    /// Load a graph from a file and compare it to the graph stored in the database.
    CompareToDb,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let Cli {
        db_url,
        schema,
        file,
        dbg,
        command,
    } = cli;

    let db_context = DbContext::new(&db_url, &schema).await;

    match command {
        Command::BackupDb => {
            db_context.write_graph_to_file(&file, dbg).await?;
        }
        Command::RestoreDb => {
            db_context.load_graph_from_file(&file, dbg).await?;
        }
        Command::VerifyBackup => {
            db_context.verify_backup(&file, dbg).await?;
        }
        Command::StoreRandom => {
            db_context.store_random_graph().await?;
            db_context.write_graph_to_file(&file, dbg).await?;
        }
        Command::CompareToDb => {
            db_context.compare_to_db(&file, dbg).await?;
        }
    }
    // this command has it own output
    if !matches!(command, Command::CompareToDb) {
        println!("Command succeeded");
    }

    Ok(())
}
