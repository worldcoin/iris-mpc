use super::{inputs::NetArgs, params::Params};
use crate::utils::{constants::COUNT_OF_PARTIES, NetConfig, NetDbProvider, NodeDbProvider};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

pub async fn insert_iris_deletions(_params: &Params, _args: &NetArgs, _config: &NetConfig) {
    println!("TODO: insert_iris_deletions");
}

pub async fn insert_iris_shares(params: &Params, args: &NetArgs, config: &NetConfig) {
    let db_provider = NetDbProvider::new_from_config(config).await;
    for party_idx in 0..COUNT_OF_PARTIES {
        insert_iris_shares_node(
            params,
            args[party_idx].to_owned(),
            db_provider.of_node(party_idx),
        )
        .await;
    }
}

async fn insert_iris_shares_node(_params: &Params, _args: NodeArgs, _db_provider: &NodeDbProvider) {
    println!("TODO: insert_iris_shares_node");
}
