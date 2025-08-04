use super::utils::get_path_to_assets;
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::genesis::utils::aws::IrisDeletionsForS3;
use std::io::Error;

/// Returns serial identifiers associated with deleted Iris's.
///
/// # Arguments
///
/// * `n_take` - Number of deletions to read into memory.
/// * `skip_offset` - Offset from which to start reading deletions.
///
/// # Returns
///
/// Vec of serial identifiers associated with deleted Iris's.
///
pub fn read_iris_deletions(n_take: usize, skip_offset: usize) -> Result<Vec<IrisSerialId>, Error> {
    use serde_json;

    let path_to_resource = format!("{}/iris-deletions/data.json", get_path_to_assets(),);
    let IrisDeletionsForS3 { deleted_serial_ids } =
        serde_json::from_str(&std::fs::read_to_string(path_to_resource)?)?;

    Ok(deleted_serial_ids
        .into_iter()
        .skip(skip_offset)
        .take(n_take)
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
