use std::io::Error;

/// Returns a set of Iris modifications to be applied.
///
/// # Arguments
///
/// * `n_take` - Maximum number of modifications to read.
/// * `skip_offset` - Offset from which to start reading.
///
/// # Returns
///
/// Vec of Iris modifications.
///
pub fn read_iris_modifications(_n_take: usize, _skip_offset: usize) -> Result<Vec<i64>, Error> {
    // TODO: placeholder for some form of implementation.
    Ok(vec![])
}

#[cfg(test)]
mod tests {
    use super::read_iris_modifications;

    #[test]
    fn test_read_iris_modifications() {
        for (n_take, skip_offset) in [(2, 0), (10, 2)] {
            let n_read = read_iris_modifications(n_take, skip_offset).unwrap().len();
            assert_eq!(n_read, n_take);
        }
    }
}
