use iris_mpc_common::IrisSerialId;

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

#[cfg(test)]
mod tests {
    use super::generate_iris_deletions;

    #[test]
    fn test_generate_iris_deletions() {
        let data = generate_iris_deletions(100);
        assert!(data.len() == 100);
        for idx in data {
            assert!(idx % 50 == 0);
        }
    }
}
