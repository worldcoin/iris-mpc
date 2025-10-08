use crate::misc::log_info;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    protocol::shared_iris::GaloisRingSharedIris, py_bindings::plaintext_store::Base64IrisCode,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use std::{fs::File, io::BufReader, path::PathBuf};

/// Component name for logging purposes.
const COMPONENT: &str = "SystemState-PgresIrises";

/// Number of MPC parties.
const N_PARTIES: usize = 3;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

pub fn read_irises_from_ndjson(
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

        let left_shares = Box::new(GaloisRingSharedIris::generate_shares_locally(
            &mut shares_seed,
            left_iris.clone(),
        ));
        let right_shares = Box::new(GaloisRingSharedIris::generate_shares_locally(
            &mut shares_seed,
            right_iris.clone(),
        ));

        for (idx, arr) in shared_irises.iter_mut().enumerate() {
            arr.push((left_shares[idx].clone(), right_shares[idx].clone()))
        }
    }

    Ok(shared_irises)
}
