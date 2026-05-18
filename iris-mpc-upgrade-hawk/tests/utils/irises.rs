use std::path::PathBuf;

use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    protocol::shared_iris::GaloisRingSharedIris,
    utils::serialization::iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
};
use iris_mpc_utils::irises::{
    generate_iris_shares_for_upload_both_eyes, GaloisRingSharedIrisForUpload,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};

/// Number of MPC parties.
const N_PARTIES: usize = 3;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

pub fn read_irises_from_ndjson(
    ndjson_path: PathBuf,
    num_pairs: usize,
) -> Result<Vec<(IrisCode, IrisCode)>> {
    tracing::info!(
        "Reading {num_pairs} iris code pairs from file {}",
        ndjson_path.display()
    );

    let iris_pairs = irises_from_ndjson_iter(
        ndjson_path.as_path(),
        Some(2 * num_pairs),
        IrisSelection::All,
    )?
    .tuples::<(_, _)>()
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

/// Share irises locally for upload (full-size mask shares).
/// Returns one `BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>` per iris pair.
pub fn share_irises_for_upload_locally(
    irises: &[(IrisCode, IrisCode)],
    rng_seed: u64,
) -> Result<Vec<BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>>> {
    let mut result = Vec::with_capacity(irises.len());

    for (left_iris, right_iris) in irises {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let both_eyes = generate_iris_shares_for_upload_both_eyes(
            &mut rng,
            Some(left_iris.clone()),
            Some(right_iris.clone()),
        );
        result.push(both_eyes);
    }

    Ok(result)
}
