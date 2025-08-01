use super::{inputs::NetArgs, params::TestParams};
use crate::system_state;
use crate::utils::{pgres::NetDbProvider, resources::read_iris_shares_batch};
use iris_mpc_common::{config::NetConfig, PARTY_IDX_SET};
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
pub async fn upload_iris_deletions_into_s3(
    _params: &TestParams,
    _args: &NetArgs,
    _config: &NetConfig,
) {
    println!("TODO: insert_iris_deletions");
}

/// Inserts Iris shares into GPU store.
///
/// # Arguments
///
/// * `params` - Test parameters.
/// * `config` - Net configuration.
///
pub async fn insert_iris_shares_into_gpu_stores(config: &NetConfig, params: &TestParams) {
    // Set shares batch generator.
    // TODO: move these vars to test params.
    let skip_offset = None;
    let shares_batch_generator = read_iris_shares_batch(
        params.shares_generator_batch_size(),
        params.arg_max_indexation_id() as usize,
        params.shares_generator_rng_state(),
        skip_offset,
    )
    .unwrap();

    // Set db provider.
    let db_provider = NetDbProvider::new_from_config(config).await;

    // Iterate over batches by party and insert into GPU store.
    for chunk in shares_batch_generator.into_iter() {
        let shares = chunk.into_iter().map(|x| x.to_vec()).collect_vec();
        for party_idx in PARTY_IDX_SET {
            let (_start_serial_id, _end_serial_id) = system_state::insert_iris_shares(
                db_provider.of_node(party_idx).gpu_store().iris_store(),
                params.shares_pgres_tx_batch_size(),
                shares.iter().map(|i| i[party_idx].to_owned()).collect_vec(),
            )
            .await
            .unwrap();
        }
    }
}
