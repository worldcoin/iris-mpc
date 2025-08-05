use crate::utils::fsys::get_assets_root;
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodePair};
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::{IntoChunks, Itertools};
use serde_json::Deserializer;
use std::{fs::File, io::BufReader, io::Error};

/// Name of ndjson file containing a set of Iris codes.
const FNAME_1K: &str = "iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson";

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub fn read_iris_codes(
    read_maximum: usize,
    skip_offset: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    // Set path.
    let path_to_resources = format!("{}/{}", get_assets_root(), FNAME_1K,);

    // Set file stream.
    let file = File::open(path_to_resources).unwrap();
    let reader = BufReader::new(file);
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(skip_offset)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(read_maximum);

    Ok(stream)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris code pairs.
///
pub fn read_iris_codes_batch(
    batch_size: usize,
    read_maximum: usize,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let stream = read_iris_codes(read_maximum, skip_offset)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::{read_iris_codes, read_iris_codes_batch};

    #[test]
    fn test_read_iris_code_pairs() {
        for (read_maximum, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in read_iris_codes(read_maximum, skip_offset).unwrap() {
                n_read += 1;
            }
            assert_eq!(read_maximum, n_read);
        }
    }

    #[test]
    fn test_read_iris_code_pairs_batch() {
        for (batch_size, read_maximum, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in read_iris_codes_batch(batch_size, read_maximum, skip_offset)
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
