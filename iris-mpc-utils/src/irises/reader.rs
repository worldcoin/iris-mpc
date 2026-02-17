use std::{
    fs::{self},
    io::Error,
    path::Path,
};

use rand::{CryptoRng, Rng};

use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;

use super::{generate_iris_shares_for_upload, GaloisRingSharedIrisForUpload};
use crate::{constants::N_PARTIES, fsys};

/// Returns iterator over Iris codes deserialized from an ndjson file.
pub fn read_iris_codes(path_to_ndjson: &Path) -> Result<impl Iterator<Item = IrisCode>, Error> {
    println!("path to ndjson: {:?}", path_to_ndjson);
    let stream = fsys::reader::read_json_iter::<Base64IrisCode>(path_to_ndjson)
        .unwrap()
        .map(|res| res.unwrap())
        .map(|res| IrisCode::from(&res));

    Ok(stream)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn read_iris_shares<'a, R: Rng + CryptoRng + 'a>(
    path_to_ndjson: &Path,
    rng: &'a mut R,
) -> Result<impl Iterator<Item = [GaloisRingSharedIris; N_PARTIES]> + 'a, Error> {
    Ok(read_iris_codes(path_to_ndjson)
        .unwrap()
        .map(move |iris_code| GaloisRingSharedIris::generate_shares_locally(rng, iris_code)))
}

/// Returns iterator over Iris shares for upload, deserialized from a stream of Iris Code pairs.
/// These shares contain the full-size mask (not trimmed) for compatibility with the MPC server.
pub fn read_iris_shares_for_upload<'a, R: Rng + CryptoRng + 'a>(
    path_to_ndjson: &Path,
    rng: &'a mut R,
) -> Result<impl Iterator<Item = [GaloisRingSharedIrisForUpload; N_PARTIES]> + 'a, Error> {
    Ok(read_iris_codes(path_to_ndjson)
        .unwrap()
        .map(move |iris_code| generate_iris_shares_for_upload(rng, Some(iris_code))))
}

/// Returns node configuration deserialized from a toml file.
#[allow(dead_code)]
pub fn read_node_config(path_to_config: &Path) -> Result<NodeConfig, Error> {
    assert!(path_to_config.exists());

    Ok(toml::from_str(&fs::read_to_string(path_to_config)?).unwrap())
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{read_iris_codes, read_iris_shares};
    use crate::fsys::local::get_path_to_ndjson;

    const N_TO_SKIP: usize = 900;
    const N_TO_TAKE: usize = 100;
    const RNG_SEED: u64 = 42;

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(RNG_SEED)
    }

    #[tokio::test]
    async fn test_ndjson_file_exists() {
        assert!(get_path_to_ndjson().exists());
    }

    #[tokio::test]
    async fn test_read_iris_codes() {
        let iris_codes: Vec<_> = read_iris_codes(&get_path_to_ndjson())
            .unwrap()
            .skip(N_TO_SKIP)
            .take(N_TO_TAKE)
            .collect();
        assert_eq!(iris_codes.len(), N_TO_TAKE);
    }

    #[tokio::test]
    async fn test_read_iris_shares() {
        let mut rng = get_rng();
        let shares: Vec<_> = read_iris_shares(&get_path_to_ndjson(), &mut rng)
            .unwrap()
            .skip(N_TO_SKIP)
            .take(N_TO_TAKE)
            .collect();
        assert_eq!(shares.len(), N_TO_TAKE);
    }
}
