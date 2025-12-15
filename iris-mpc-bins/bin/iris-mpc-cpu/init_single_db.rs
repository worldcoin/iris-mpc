use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, vector_id::SerialId};
use iris_mpc_cpu::{
    hnsw::graph::test_utils::DbContext,
    protocol::shared_iris::{GaloisRingSharedIris, GaloisRingSharedIrisPair},
    utils::serialization::types::iris_base64::Base64IrisCode,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use serde_json::Deserializer;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    /// the peer in the MPC protocol
    #[clap(long)]
    party_id: usize,

    /// Postgres db schema
    #[clap(long)]
    db_schema: String,

    /// Postgres db server url: party 1.
    /// Example URL format: `postgres://postgres:postgres@localhost:5432`
    #[clap(long)]
    db_url: String,

    /// The source file for plaintext iris codes, in NDJSON file format.
    #[clap(long = "source")]
    path_to_iris_codes: PathBuf,

    /// The target number of left/right iris pairs to build the databases from.
    /// If existing entries are already in the database, then additional entries
    /// are added only to increase the total database size to this value.
    ///
    /// If the persisted databases already have more entries than this number,
    /// then the binary will do nothing, rather than shrinking the databases.
    ///
    /// If this parameter is omitted, then all entries in the source iris code
    /// file will be used.
    #[clap(short('s'), long)]
    target_db_size: Option<usize>,

    /// Shares seed for iris shares insertion, used to generate secret
    /// shares of iris codes.
    /// The default is 42 to match the default in `shares_encoding.rs`.
    #[clap(long, default_value = "42")]
    iris_shares_rnd: u64,

    /// Skip insertion of specific serial IDs ... used primarily in genesis testing.
    #[clap(long, num_args = 1..)]
    skip_insert_serial_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IrisCodeWithSerialId {
    pub iris_code: IrisCode,
    pub serial_id: SerialId,
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    tracing::info!("Initialized tracing subscriber");

    tracing::info!("Parsing CLI arguments");
    let args = Args::parse();
    let party = args.party_id;

    tracing::info!("Setting database connections");
    let db = init_db(&args).await;

    tracing::info!("Setting count of previously indexed irises");
    let n_existing_irises = db.store.get_max_serial_id().await?;
    tracing::info!("Found {} existing database irises", n_existing_irises);
    tracing::warn!(
        "TODO: Escape if count of persisted irises is not equivalent across all parties"
    );

    // number of iris pairs that need to be inserted to increase the DB size to at least the target
    let n_irises = args
        .target_db_size
        .map(|target| target.saturating_sub(n_existing_irises));

    tracing::info!("⚓ ANCHOR: Converting plaintext iris codes locally into secret shares");

    let mut batch: Vec<GaloisRingSharedIrisPair> = Vec::with_capacity(SECRET_SHARING_BATCH_SIZE);
    let mut n_read: usize = 0;

    let file = File::open(args.path_to_iris_codes.as_path()).unwrap();
    let reader = BufReader::new(file);
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(2 * n_existing_irises)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(n_irises.unwrap_or(usize::MAX))
        .chunks(SECRET_SHARING_BATCH_SIZE);

    for (batch_idx, vectors_batch) in stream.into_iter().enumerate() {
        let vectors_batch: Vec<(_, _)> = vectors_batch.collect();
        n_read += vectors_batch.len();

        for (left, right) in vectors_batch {
            // Reset RNG for each pair to match shares_encoding.rs behavior
            let mut shares_seed = StdRng::seed_from_u64(args.iris_shares_rnd);

            let left_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, left.clone());
            let right_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, right.clone());
            batch.push((left_shares[party].clone(), right_shares[party].clone()));
        }

        let cur_batch_len = batch.len();
        let last_idx = batch_idx * SECRET_SHARING_BATCH_SIZE + cur_batch_len + n_existing_irises;

        #[allow(clippy::drain_collect)]
        let (_, end_serial_id) = db.persist_vector_shares(batch.drain(..).collect()).await?;
        assert_eq!(end_serial_id, last_idx);

        tracing::info!(
            "Persisted {} locally generated shares",
            last_idx - n_existing_irises
        );
    }

    tracing::info!("Finished persisting {} locally generated shares", n_read);

    Ok(())
}

async fn init_db(args: &Args) -> DbContext {
    DbContext::new(&args.db_url, &args.db_schema).await
}
