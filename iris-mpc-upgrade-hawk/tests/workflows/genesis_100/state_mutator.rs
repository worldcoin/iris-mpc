use crate::system_state;
use crate::{
    resources, utils::pgres::NetDbProvider, workflows::genesis_shared::params::TestParams,
};
use eyre::Result;
use iris_mpc_common::config::NetConfig;

/// Inserts Iris deletions into AWS S3 bucket.
///
/// # Arguments
///
/// * `net_config` - Network wide configuration.
/// * `params` - Test parameters.
///
pub async fn upload_iris_deletions(net_config: &NetConfig, params: &TestParams) -> Result<()> {
    // Set deletions.
    let data = match params.max_deletions() {
        Some(max) => {
            let skip_offset = 0;
            resources::read_iris_deletions(max, skip_offset).unwrap()
        }
        None => vec![],
    };

    // Upload to AWS S3.
    system_state::upload_iris_deletions(net_config, &data)
        .await
        .unwrap();

    Ok(())
}

/// Inserts Iris shares into GPU store.
///
/// # Arguments
///
/// * `net_config` - Network wide configuration.
/// * `params` - Test parameters.
///
pub async fn insert_iris_shares_into_gpu_stores(
    net_config: &NetConfig,
    params: &TestParams,
) -> Result<()> {
    // Set shares batch generator.
    let batch_size = params.shares_generator_batch_size();
    let read_maximum = params.max_indexation_id() as usize;
    let rng_state = params.shares_generator_rng_state();
    let skip_offset = 0;
    let batch_generator =
        resources::read_iris_shares_batch(batch_size, read_maximum, rng_state, skip_offset)
            .unwrap();

    // Iterate over batches and insert into GPU store.
    // TODO: process serial id ranges.
    let db_provider = NetDbProvider::new_from_config(net_config).await;
    let tx_batch_size = params.shares_pgres_tx_batch_size();
    let _ = system_state::insert_iris_shares(&batch_generator, &db_provider, tx_batch_size)
        .await
        .unwrap();

    // TODO: process serial id ranges.

    Ok(())
}
