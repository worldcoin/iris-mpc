use std::{
    fs::{self},
    io::Error,
    path::Path,
};

use itertools::{IntoChunks, Itertools};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};

use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;

use crate::{constants::N_PARTIES, fsys};}

/// Returns iterator over base64 encoded Iris codes deserialized from an ndjson file.
pub fn read_b64_iris_codes(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
) -> Result<impl Iterator<Item = Base64IrisCode>, Error> {
    Ok(
        fsys::reader::read_json_iter::<Base64IrisCode>(path_to_ndjson, n_to_skip, n_to_take)
            .unwrap()
            .map(|res| res.unwrap()),
    )
}

/// Returns chunked iterator over base64 encoded Iris codes deserialized from an ndjson file.
pub fn read_b64_iris_codes_chunks(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    chunk_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = Base64IrisCode>>, Error> {
    Ok(read_b64_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .chunks(chunk_size))
}

/// Returns iterator over Iris codes deserialized from an ndjson file.
pub fn read_iris_codes(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
) -> Result<impl Iterator<Item = IrisCode>, Error> {
    Ok(read_b64_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .map(|res| IrisCode::from(&res)))
}

/// Returns chunked iterator over Iris codes deserialized from an ndjson file.
pub fn read_iris_codes_chunks(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    chunk_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCode>>, Error> {
    Ok(read_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .chunks(chunk_size))
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn read_iris_shares<'a, R: Rng + CryptoRng + 'a>(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    rng: &'a mut R,
) -> Result<impl Iterator<Item = [GaloisRingSharedIris; N_PARTIES]> + 'a, Error> {
    Ok(read_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .map(move |iris_code| GaloisRingSharedIris::generate_shares_locally(rng, iris_code)))
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn read_iris_shares_chunks<'a, R: Rng + CryptoRng + 'a>(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    chunk_size: usize,
    rng: &'a mut R,
) -> Result<IntoChunks<impl Iterator<Item = [GaloisRingSharedIris; N_PARTIES]> + 'a>, Error> {
    Ok(read_iris_shares(path_to_ndjson, n_to_skip, n_to_take, rng)
        .unwrap()
        .chunks(chunk_size))
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

    use super::{
        read_b64_iris_codes, read_b64_iris_codes_chunks, read_iris_codes, read_iris_codes_chunks,
        read_iris_shares, read_iris_shares_chunks,
    };
    use crate::fsys::local::get_path_to_ndjson;

    const N_TO_SKIP: usize = 900;
    const N_TO_TAKE: usize = 100;
    const CHUNK_SIZE: usize = 10;
    const RNG_SEED: u64 = 42;

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(RNG_SEED)
    }

    #[tokio::test]
    async fn test_ndjson_file_exists() {
        assert!(get_path_to_ndjson().exists());
    }

    #[tokio::test]
    async fn test_read_b64_iris_codes() {
        let iris_codes: Vec<_> = read_b64_iris_codes(&get_path_to_ndjson(), N_TO_SKIP, N_TO_TAKE)
            .unwrap()
            .collect();
        assert_eq!(iris_codes.len(), N_TO_TAKE);
    }

    #[tokio::test]
    async fn test_read_b64_iris_codes_chunks() {
        let chunks: Vec<_> =
            read_b64_iris_codes_chunks(&get_path_to_ndjson(), N_TO_SKIP, N_TO_TAKE, CHUNK_SIZE)
                .unwrap()
                .into_iter()
                .map(|chunk| chunk.collect::<Vec<_>>())
                .collect();

        assert_eq!(chunks.len(), N_TO_TAKE / CHUNK_SIZE);
        for chunk in chunks {
            assert_eq!(chunk.len(), CHUNK_SIZE);
        }
    }

    #[tokio::test]
    async fn test_read_iris_codes() {
        let iris_codes: Vec<_> = read_iris_codes(&get_path_to_ndjson(), N_TO_SKIP, N_TO_TAKE)
            .unwrap()
            .collect();
        assert_eq!(iris_codes.len(), N_TO_TAKE);
    }

    #[tokio::test]
    async fn test_read_iris_codes_chunks() {
        let chunks: Vec<_> =
            read_iris_codes_chunks(&get_path_to_ndjson(), N_TO_SKIP, N_TO_TAKE, CHUNK_SIZE)
                .unwrap()
                .into_iter()
                .map(|chunk| chunk.collect::<Vec<_>>())
                .collect();
        assert_eq!(chunks.len(), N_TO_TAKE / CHUNK_SIZE);

        for chunk in chunks {
            assert_eq!(chunk.len(), CHUNK_SIZE);
        }
    }

    #[tokio::test]
    async fn test_read_iris_shares() {
        let mut rng = get_rng();
        let shares: Vec<_> =
            read_iris_shares(&get_path_to_ndjson(), N_TO_SKIP, N_TO_TAKE, &mut rng)
                .unwrap()
                .collect();
        assert_eq!(shares.len(), N_TO_TAKE);
    }

    #[tokio::test]
    async fn test_read_iris_shares_chunks() {
        let mut rng = get_rng();
        let chunks: Vec<_> = read_iris_shares_chunks(
            &get_path_to_ndjson(),
            N_TO_SKIP,
            N_TO_TAKE,
            CHUNK_SIZE,
            &mut rng,
        )
        .unwrap()
        .into_iter()
        .map(|chunk| chunk.collect::<Vec<_>>())
        .collect();
        assert_eq!(chunks.len(), N_TO_TAKE / CHUNK_SIZE);

        for chunk in chunks {
            assert_eq!(chunk.len(), CHUNK_SIZE);
        }
    }
}
