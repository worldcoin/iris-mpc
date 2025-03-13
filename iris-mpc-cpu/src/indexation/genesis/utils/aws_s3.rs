use super::super::errors::IndexationError;
use iris_mpc_common::config::Config;
use rand::prelude::*;

/// Fetches V1 serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - System configuration information.
///
/// # Returns
///
/// A set of V1 serial identifiers marked as deleted.
///
pub(crate) async fn fetch_iris_v1_deletions(_: &Config) -> Result<Vec<i64>, IndexationError> {
    let mut rng = rand::thread_rng();
    let mut identifiers = (1_i64..1000_i64).choose_multiple(&mut rng, 50);
    identifiers.sort();

    Ok(identifiers)
}
