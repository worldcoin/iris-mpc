use std::io::Error;

/// Returns identifiers associated with Iris modifications.
///
/// # Arguments
///
/// * `read_maximum` - Maximum number of deletions to read.
/// * `skip_offset` - Offset from which to start reading deletions.
///
/// # Returns
///
/// Vec of identifiers associated with Iris modifications.
///
pub fn read_iris_modifications(
    _read_maximum: usize,
    _skip_offset: usize,
) -> Result<Vec<i64>, Error> {
    // TODO: placeholder for some form of implementation.
    Ok(vec![])
}

#[cfg(test)]
mod tests {
    // TODO
}
