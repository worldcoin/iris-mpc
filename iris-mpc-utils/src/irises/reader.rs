use std::{
    fs::{self, File},
    io::{BufReader, Error},
    path::Path,
};

use itertools::{IntoChunks, Itertools};
use serde::{Deserialize, Serialize};
use serde_json;

use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;

use crate::{constants::N_PARTIES, irises::convertor};

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq)]
enum IrisSelectionStrategy {
    // All Iris codes are selected.
    All,
    // Every other Iris code is selected beginning at an even offset.
    Even,
    // Every other Iris code is selected beginning at an odd offset.
    Odd,
}

/// Returns iterator over base64 encoded Iris codes deserialized from an ndjson file.
#[allow(dead_code)]
pub fn read_b64_iris_codes(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
) -> Result<impl Iterator<Item = Base64IrisCode>, Error> {
    let handle = File::open(path_to_ndjson).unwrap();
    let reader = BufReader::new(handle);
    let iterable = serde_json::Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(n_to_skip)
        .map(|res| res.unwrap())
        .take(n_to_take);

    Ok(iterable)
}

/// Returns chunked iterator over base64 encoded Iris codes deserialized from an ndjson file.
#[allow(dead_code)]
pub fn read_b64_iris_codes_chunks(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    chunk_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = Base64IrisCode>>, Error> {
    let iterable = read_b64_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .chunks(chunk_size);

    Ok(iterable)
}

/// Returns iterator over Iris codes deserialized from an ndjson file.
#[allow(dead_code)]
pub fn read_iris_codes(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
) -> Result<impl Iterator<Item = IrisCode>, Error> {
    let iterable = read_b64_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .map(|res| IrisCode::from(&res));

    Ok(iterable)
}

/// Returns chunked iterator over Iris codes deserialized from an ndjson file.
#[allow(dead_code)]
pub fn read_iris_codes_chunks(
    path_to_ndjson: &Path,
    n_to_skip: usize,
    n_to_take: usize,
    chunk_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCode>>, Error> {
    let iterable = read_iris_codes(path_to_ndjson, n_to_skip, n_to_take)
        .unwrap()
        .chunks(chunk_size);

    Ok(iterable)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
#[allow(dead_code)]
pub fn read_iris_shares(
    path_to_ndjson: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    rng_state: u64,
) -> Result<impl Iterator<Item = Box<[GaloisRingSharedIris; N_PARTIES]>>, Error> {
    let stream = read_iris_codes(path_to_ndjson, n_to_read, n_to_skip)
        .unwrap()
        .map(move |code_pair| convertor::to_galois_ring_shares(rng_state, &code_pair));

    Ok(stream)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
pub fn read_iris_shares_chunks(
    path_to_ndjson: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    chunk_size: usize,
    rng_state: u64,
) -> Result<IntoChunks<impl Iterator<Item = Box<[GaloisRingSharedIris; N_PARTIES]>>>, Error> {
    let stream = read_iris_shares(path_to_ndjson, n_to_read, n_to_skip, rng_state)
        .unwrap()
        .chunks(chunk_size);

    Ok(stream)
}

/// Returns node configuration deserialized from a toml file.
#[allow(dead_code)]
pub fn read_node_config(path_to_config: &Path) -> Result<NodeConfig, Error> {
    assert!(path_to_config.exists());

    Ok(toml::from_str(&fs::read_to_string(path_to_config)?).unwrap())
}
