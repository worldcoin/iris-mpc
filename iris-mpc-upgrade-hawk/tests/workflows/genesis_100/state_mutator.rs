use super::{inputs::NetArgs, params::Params};
use crate::utils::{pgres::NetDbProvider, resources::read_iris_shares_batch, sys_state};
use iris_mpc_common::{config::NetConfig, PARTY_COUNT};
use iris_mpc_cpu::genesis::state_mutator::insert_iris_deletions;
use itertools::Itertools;

/// Inserts Iris deletions into AWS S3 bucket.
///
/// # Arguments
///
/// * `params` - Test parameters.
/// * `args` - Net args.
/// * `config` - Net configuration.
///
pub async fn upload_iris_deletions_into_s3(_params: &Params, _args: &NetArgs, _config: &NetConfig) {
    println!("TODO: insert_iris_deletions");
}

/// Inserts Iris shares into GPU store.
///
/// # Arguments
///
/// * `params` - Test parameters.
/// * `config` - Net configuration.
///
pub async fn insert_iris_shares_into_gpu_stores(config: &NetConfig, params: &Params) {
    // Set shares batch generator.
    // TODO: move these vars to test params.
    let max_items = 100;
    let rng_state = 93;
    let skip_offset = 0;
    let shares_batch_generator =
        read_iris_shares_batch(params.batch_size(), rng_state, skip_offset, max_items).unwrap();

    // Set db provider.
    let db_provider = NetDbProvider::new_from_config(config).await;

    // Iterate over batches by party and insert into GPU store.
    for chunk in shares_batch_generator.into_iter() {
        let shares = chunk.into_iter().map(|x| x.to_vec()).collect_vec();
        for party_idx in 0..PARTY_COUNT {
            let (_start_serial_id, _end_serial_id) = sys_state::insert_iris_shares(
                db_provider.of_node(party_idx).gpu_store().iris_store(),
                params.pgres_tx_batch_size(),
                shares.iter().map(|i| i[party_idx].to_owned()).collect_vec(),
            )
            .await
            .unwrap();
        }
    }
}
