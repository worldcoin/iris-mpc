use super::iris_codes::read_iris_codes;
use crate::utils::convertor::to_galois_ring_share_pair_set;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIrisPairSet;
use itertools::{IntoChunks, Itertools};
use std::io::Error;

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris shares.
///
pub fn read_iris_shares(
    n_to_read: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    Ok(read_iris_codes(n_to_read, skip_offset)
        .unwrap()
        .map(move |code_pair| to_galois_ring_share_pair_set(rng_state, &code_pair)))
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris shares.
///
pub fn read_iris_shares_batch(
    batch_size: usize,
    n_to_read: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    Ok(read_iris_shares(n_to_read, rng_state, skip_offset)
        .unwrap()
        .chunks(batch_size))
}

#[cfg(test)]
mod tests {
    use super::{read_iris_shares, read_iris_shares_batch};
    use crate::utils::constants::PARTY_COUNT;

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_read_iris_shares() {
        for (n_to_read, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for shares in read_iris_shares(n_to_read, DEFAULT_RNG_STATE, skip_offset).unwrap() {
                n_read += 1;
                assert_eq!(shares.len(), PARTY_COUNT);
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_read_iris_shares_batch() {
        for (batch_size, n_to_read, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in
                read_iris_shares_batch(batch_size, n_to_read, DEFAULT_RNG_STATE, skip_offset)
                    .unwrap()
                    .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for item in chunk.into_iter() {
                    assert_eq!(item.len(), PARTY_COUNT);
                    n_items += 1;
                }
                assert_eq!(batch_size, n_items);
            }
            assert_eq!(expected_batches, n_chunks);
        }
    }
}
