use iris_mpc_common::IrisSerialId;
use std::io::Error;

/// Returns serial identifiers associated with deleted Iris's.
///
/// # Arguments
///
/// * `read_maximum` - Maximum number of deletions to read.
/// * `skip_offset` - Offset from which to start reading deletions.
///
/// # Returns
///
/// Vec of serial identifiers associated with deleted Iris's.
///
pub fn read_iris_deletions(
    _read_maximum: usize,
    _skip_offset: usize,
) -> Result<Vec<IrisSerialId>, Error> {
    // TODO.
    Ok(vec![])
}

#[cfg(test)]
mod tests {
    // TODO
}
