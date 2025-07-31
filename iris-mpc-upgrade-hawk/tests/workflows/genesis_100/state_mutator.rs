use super::{inputs::NetArgs, params::Params};
use crate::utils::{
    constants::COUNT_OF_PARTIES, resources::read_iris_shares_batch, store, NetConfig, NetDbProvider,
};
use itertools::Itertools;

/// Inserts Iris deletions into AWS S3 bucket.
///
/// # Arguments
///
/// * `params` - Test parameters.
/// * `args` - Net args.
/// * `config` - Net configuration.
///
pub async fn insert_iris_deletions(_params: &Params, _args: &NetArgs, _config: &NetConfig) {
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
        for party_idx in 0..COUNT_OF_PARTIES {
            let (_start_serial_id, _end_serial_id) = store::insert_iris_shares(
                db_provider.of_node(party_idx).gpu_store().iris_store(),
                params.pgres_tx_batch_size(),
                shares.iter().map(|i| i[party_idx].to_owned()).collect_vec(),
            )
            .await
            .unwrap();
        }
    }
}
