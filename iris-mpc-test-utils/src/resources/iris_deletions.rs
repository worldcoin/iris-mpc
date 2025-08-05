use crate::utils::fsys::get_assets_root;
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::genesis::utils::aws::IrisDeletionsForS3;
use serde_json;
use std::io::Error;

/// Name of ndjson file containing a set of Iris codes.
const FNAME_1K: &str = "iris-deletions/20250805-iris-deletions-1k.json";

/// Returns a set of Iris serial identifiers to be associated with deleted Iris shares.
///
/// # Arguments
///
/// * `n_deletions` - The number of deletions to generate.
///
/// # Returns
///
/// A set of Iris serial identifiers to be associated with deleted Iris shares.
///
pub fn generate_iris_deletions(n_deletions: usize) -> Vec<IrisSerialId> {
    // Every 50 Iris serial identifiers will be marked for deletion.
    (0..n_deletions)
        .map(|i| ((i + 1) * 50) as IrisSerialId)
        .collect()
}

/// Returns serial identifiers associated with deleted Iris's.
///
/// # Arguments
///
/// * `n_to_read` - Number of deletions to read into memory.
/// * `skip_offset` - Offset from which to start reading deletions.
///
/// # Returns
///
/// Vec of serial identifiers associated with deleted Iris's.
///
pub fn read_iris_deletions(
    n_to_read: usize,
    skip_offset: usize,
) -> Result<Vec<IrisSerialId>, Error> {
    let path_to_resource = format!("{}/{}", get_assets_root(), FNAME_1K);
    let IrisDeletionsForS3 { deleted_serial_ids } =
        serde_json::from_str(&std::fs::read_to_string(path_to_resource)?)?;

    Ok(deleted_serial_ids
        .into_iter()
        .skip(skip_offset)
        .take(n_to_read)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::read_iris_deletions;

    #[test]
    fn test_read_iris_deletions() {
        for (n_take, skip_offset) in [(2, 0), (10, 2)] {
            let n_read = read_iris_deletions(n_take, skip_offset).unwrap().len();
            assert_eq!(n_read, n_take);
        }
    }
}
