use crate::utils::fsys::{get_assets_root, get_data_root};
use aes_prng::AesRng;
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodePair};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{HnswParams, HnswSearcher},
    py_bindings::{
        io::write_bin,
        plaintext_store::{to_ndjson_file, Base64IrisCode},
    },
};
use itertools::{IntoChunks, Itertools};
use rand::SeedableRng;
use serde_json::Deserializer;
use std::{
    fs::File,
    io::{BufReader, Error},
};

/// Name of ndjson file containing a set of Iris codes.
const FNAME_1K: &str = "iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson";

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub fn read_iris_codes(
    n_to_read: usize,
    skip_offset: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    let path_to_resources = format!("{}/{}", get_assets_root(), FNAME_1K,);

    Ok(
        Deserializer::from_reader(BufReader::new(File::open(path_to_resources).unwrap()))
            .into_iter::<Base64IrisCode>()
            .skip(skip_offset)
            .map(|x| IrisCode::from(&x.unwrap()))
            .tuples()
            .take(n_to_read),
    )
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris code pairs.
///
pub fn read_iris_codes_batch(
    batch_size: usize,
    n_to_read: usize,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    Ok(read_iris_codes(n_to_read, skip_offset)
        .unwrap()
        .chunks(batch_size))
}

/// Writes to data directory an ndjson file plus associated data files.
///
/// # Arguments
///
/// * `rng_seed` - RNG seed used when generating Iris codes.
/// * `n_to_generate` - Number of Iris codes to generate.
/// * `graph_size_range` - Range of graph sizes to generate.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub async fn write_plaintext_iris_codes(
    rng_seed: u64,
    n_to_generate: usize,
    graph_size_range: Vec<usize>,
) {
    // Set RNG from seed.
    let mut rng = AesRng::seed_from_u64(rng_seed);

    // Write plaintext store.
    let resource_path = format!("{}/iris-shares-plaintext/store.ndjson", get_data_root());
    println!("HNSW :: Writing plaintext store: {}", resource_path);
    let mut store = PlaintextStore::new_random(&mut rng, n_to_generate);
    to_ndjson_file(&store, resource_path.as_str()).unwrap();

    // Write graphs.
    let searcher = HnswSearcher {
        params: HnswParams::new(320, 256, 256),
    };
    for graph_size in graph_size_range {
        let resource_path = format!(
            "{}/iris-shares-plaintext/graph_{graph_size}.dat",
            get_data_root()
        );
        println!(
            "HNSW :: Generating graph: vertices={} :: output={}",
            graph_size, resource_path
        );

        write_bin(
            &store
                .generate_graph(&mut rng, graph_size, &searcher)
                .await
                .unwrap(),
            resource_path.as_str(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::{read_iris_codes, read_iris_codes_batch};

    #[test]
    fn test_read_iris_code_pairs() {
        for (n_to_read, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in read_iris_codes(n_to_read, skip_offset).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_read_iris_code_pairs_batch() {
        for (batch_size, n_to_read, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in read_iris_codes_batch(batch_size, n_to_read, skip_offset)
                .unwrap()
                .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for _ in chunk.into_iter() {
                    n_items += 1;
                }
                assert_eq!(batch_size, n_items);
            }
            assert_eq!(expected_batches, n_chunks);
        }
    }
}
