use iris_mpc_common::config::Config as NodeConfig;

/// Returns a node's config deserialized from environment variables.
///
/// # Returns
///
/// A node's config deserialized from environment variables.
///
pub fn generate_node_config_from_env_vars() -> NodeConfig {
    // Activate environment variables.
    dotenvy::dotenv().ok();

    NodeConfig::load_config("SMPC").unwrap()
}
