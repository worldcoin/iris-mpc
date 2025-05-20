use clap::{Parser, Subcommand};
use iris_mpc_cpu::hnsw::graph::test_utils::DbContext;
use std::path::PathBuf;
use eyre::Result;

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
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load a graph from a database to memory, then write it to a file.
    DumpDb,
    /// Load a graph from a file to memory, then write it to a database
    LoadDb,
    /// verify that Dump/Load works as expected
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let Cli {
        db_url,
        schema,
        file,
        command,
    } = cli;

    let db_context = DbContext::new(&db_url, &schema).await;

    match command {
        Command::DumpDb => {
            db_context.write_graph_to_file(&file).await?;
        }
        Command::LoadDb => {
            db_context.load_graph_from_file(&file).await?;
        }
        Command::Test => {
            db_context.test_load_store(&file).await?;
        }
    }
    println!("Command succeeded");
    Ok(())
}
