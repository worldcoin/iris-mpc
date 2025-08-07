use iris_mpc_common::config::Config as NodeConfig;
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

/// Returns a node's config deserialized from SMPC environment variables.
///
/// # Returns
///
/// A node's config deserialized from environment variables.
///
pub fn generate_node_config_from_env_vars() -> NodeConfig {
    // Activates environment variables.
    dotenvy::dotenv().ok();

    NodeConfig::load_config("SMPC").unwrap()
}
