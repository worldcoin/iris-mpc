use clap::{Parser, Subcommand, ValueEnum};
use eyre::Result;
use iris_mpc_cpu::{
    execution::hawk_main::{LEFT, RIGHT},
    hnsw::graph::test_utils::{DbContext, DiffMethod},
};
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
    /// serializes both the left and right eye together
    BackupDb,
    /// (testing only) creates random data, stores it in the database, and then runs backup-db.
    StoreRandom,
    /// Load a graph from a file to memory, then write it to a database
    RestoreDb,
    /// restore the left or right eye
    RestoreSide {
        /// the left or right eye
        side: Side,
    },
    /// (testing only) verify that Load/Store works as expected
    VerifyBackup,
    /// Load a graph from a file and compare it to the graph stored in the database.
    CompareToDb {
        /// The diffing method to use for the comparison.
        #[arg(long, value_enum, default_value_t = DiffMethod::DetailedJaccard)]
        diff_method: DiffMethod,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Side {
    Left,
    Right,
}

impl From<Side> for usize {
    fn from(side: Side) -> usize {
        match side {
            Side::Left => LEFT,
            Side::Right => RIGHT,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cli {
        db_url,
        schema,
        file,
        dbg,
        command,
    } = Cli::parse();
    let db_context = DbContext::new(&db_url, &schema).await;

    match command {
        Command::BackupDb => {
            db_context.write_graph_to_file(&file, dbg).await?;
        }
        Command::RestoreDb => {
            db_context.load_graph_from_file(&file, dbg).await?;
        }
        Command::RestoreSide { side } => {
            db_context
                .load_side_from_file(&file, side.into(), dbg)
                .await?;
        }
        Command::VerifyBackup => {
            db_context.verify_backup(&file, dbg).await?;
        }
        Command::StoreRandom => {
            db_context.store_random_graph().await?;
            db_context.write_graph_to_file(&file, dbg).await?;
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
