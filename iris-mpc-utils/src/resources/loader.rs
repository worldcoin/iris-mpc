use crate::{
    convertor,
    types::{GaloisRingSharedIrisPairSet, IrisCodePair},
    utils::misc::limited_iterator,
};
use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode, IrisVectorId};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, py_bindings::plaintext_store::Base64IrisCode,
};
use itertools::{IntoChunks, Itertools};
use serde_json::{self, Deserializer};
use std::{
    fs::{self, File},
    io::{self, BufReader, Error},
    path::Path,
    sync::Arc,
};

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
#[allow(dead_code)]
pub fn load_iris_codes(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    let reader = BufReader::new(File::open(path_to_codes).unwrap());
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(n_to_skip)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(n_to_read);

    Ok(stream)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
#[allow(dead_code)]
pub fn load_iris_codes_batch(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let stream = load_iris_codes(path_to_codes, n_to_read, n_to_skip)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
#[allow(dead_code)]
pub fn load_iris_shares(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    rng_state: u64,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    let stream = load_iris_codes(path_to_codes, n_to_read, n_to_skip)
        .unwrap()
        .map(move |code_pair| convertor::to_galois_ring_share_pair_set(rng_state, &code_pair));

    Ok(stream)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
#[allow(dead_code)]
pub fn load_iris_shares_batch(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
    rng_state: u64,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    let stream = load_iris_shares(path_to_codes, n_to_read, n_to_skip, rng_state)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns node configuration deserialized from a toml file.
#[allow(dead_code)]
pub fn load_node_config(path_to_config: &Path) -> Result<NodeConfig, Error> {
    assert!(path_to_config.exists());

    Ok(toml::from_str(&fs::read_to_string(path_to_config)?).unwrap())
}

/// Returns plaintext store deserialized from a json file.
pub fn load_plaintext_store(fpath: &Path, len: Option<usize>) -> io::Result<PlaintextStore> {
    let file = File::open(fpath)?;
    let reader = BufReader::new(file);

    // Create an iterator over deserialized objects
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let stream = limited_iterator(stream, len);

    // Iterate over each deserialized object
    let mut vector = PlaintextStore::new();
    for (idx, json_pt) in stream.into_iter().enumerate() {
        let json_pt = json_pt?;
        let iris = (&json_pt).into();
        let id = IrisVectorId::from_0_index(idx as u32);
        vector.insert_with_id(id, Arc::new(iris));
    }

    if let Some(num) = len {
        if vector.len() != num {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File {:?} contains too few entries; number read: {}",
                    fpath,
                    vector.len()
                ),
            ));
        }
    }

    Ok(vector)
}
