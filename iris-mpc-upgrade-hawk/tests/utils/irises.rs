use std::{fs::File, io::BufReader, ops::DerefMut, path::PathBuf};

use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hnsw::graph::test_utils::DbContext, protocol::shared_iris::GaloisRingSharedIris,
    py_bindings::plaintext_store::Base64IrisCode,
};
use itertools::{izip, Itertools};
use rand::{rngs::StdRng, SeedableRng};

use crate::utils::logger::log_info;

/// Component name for logging purposes.
const COMPONENT: &str = "SystemState-PgresIrises";

/// Number of MPC parties.
const N_PARTIES: usize = 3;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

pub async fn read_irises_from_ndjson(
    ndjson_path: PathBuf,
    num_pairs: usize,
) -> Result<Vec<(IrisCode, IrisCode)>> {
    log_info(
        COMPONENT,
        &format!(
            "Reading {num_pairs} iris code pairs from file {}",
            ndjson_path.display()
        ),
    );

    let file = File::open(ndjson_path.as_path())?;
    let reader = BufReader::new(file);

    let iris_pairs = serde_json::Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples::<(_, _)>()
        .take(num_pairs)
        .collect_vec();

    Ok(iris_pairs)
}

pub fn encode_plaintext_iris_for_party(
    pairs: &[(IrisCode, IrisCode)],
    rng_state: u64,
    party_idx: usize,
) -> Vec<(GaloisRingSharedIris, GaloisRingSharedIris)> {
    pairs
        .iter()
        .map(|code_pair| {
            // Set RNG for each pair to match shares_encoding.rs behavior
            let mut shares_seed = StdRng::seed_from_u64(rng_state);

            // Set MPC participant specific Iris shares from Iris code + entropy.
            let (code_l, code_r) = code_pair;
            let shares_l =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_l.to_owned());
            let shares_r =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_r.to_owned());

            (shares_l[party_idx].clone(), shares_r[party_idx].clone())
        })
        .collect()
}

pub fn share_irises_locally(
    irises: &Vec<(IrisCode, IrisCode)>,
    shares_rng_seed: u64,
) -> Result<Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>>> {
    let mut shared_irises: Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>> = (0..N_PARTIES)
        .map(|_| Vec::with_capacity(SECRET_SHARING_BATCH_SIZE))
        .collect();

    for (left_iris, right_iris) in irises {
        // Reset RNG for each pair to match shares_encoding.rs behavior
        let mut shares_seed = StdRng::seed_from_u64(shares_rng_seed);

        let left_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, left_iris.clone());
        let right_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, right_iris.clone());

        for (party, (shares_l, shares_r)) in izip!(left_shares, right_shares).enumerate() {
            shared_irises[party].push((shares_l, shares_r));
        }
    }

    Ok(shared_irises)
}

pub async fn persist_iris_shares(
    iris_shares: &Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>>,
    dbs: &Vec<DbContext>,
) -> Result<()> {
    for (db, shares) in izip!(dbs, iris_shares) {
        #[allow(clippy::drain_collect)]
        db.persist_vector_shares(shares.clone()).await?;
    }

    Ok(())
}

pub async fn init_dbs(db_urls: Vec<String>, db_schemas: Vec<String>) -> Vec<DbContext> {
    let mut dbs = Vec::new();
    for (url, schema) in izip!(db_urls.iter(), db_schemas.iter()).take(N_PARTIES) {
        dbs.push(DbContext::new(url, schema).await);
    }
    dbs
}

pub async fn clear_iris_shares(db: &DbContext) -> Result<()> {
    let mut query = sqlx::QueryBuilder::new("TRUNCATE irises");

    let mut tx = db.store.tx().await?;
    query.build().execute(tx.deref_mut()).await?;
    tx.commit().await?;

    Ok(())
}
