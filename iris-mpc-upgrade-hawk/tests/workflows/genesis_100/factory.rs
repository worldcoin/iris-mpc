use super::{
    inputs::{Inputs, NetInputs, NodeInputs, SystemStateInputs},
    params::Params,
};
use crate::utils::{
    constants, resources, DbConnectionInfo, NetDbProvider, NodeDbProvider, NodeDbStore,
    TestRunContextInfo,
};
use eyre::eyre;
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns a network dB provider.
async fn get_db_provider_net(inputs: &NetInputs) -> NetDbProvider {
    NetDbProvider::new(
        get_db_provider_node(inputs.get_node_inputs(constants::PARTY_IDX_0).config()).await,
        get_db_provider_node(inputs.get_node_inputs(constants::PARTY_IDX_1).config()).await,
        get_db_provider_node(inputs.get_node_inputs(constants::PARTY_IDX_2).config()).await,
    )
}

/// Returns a node dB provider.
async fn get_db_provider_node(config: &NodeConfig) -> NodeDbProvider {
    NodeDbProvider::new(
        config.party_id,
        NodeDbStore::new(DbConnectionInfo::new_read_write(
            get_db_schema(config, config.hnsw_schema_name_suffix()),
            get_db_url(config),
        ))
        .await,
        NodeDbStore::new(DbConnectionInfo::new_read_only(
            get_db_schema(config, config.gpu_schema_name_suffix()),
            get_db_url(config),
        ))
        .await,
    )
}

/// Returns name of a dB schema for connecting to a node's dB.
fn get_db_schema(config: &NodeConfig, schema_suffix: &String) -> String {
    let NodeConfig {
        schema_name,
        environment,
        party_id,
        ..
    } = config;

    format!(
        "{}{}_{}_{}",
        schema_name, schema_suffix, environment, party_id
    )
}

/// Returns name of a dB url for connecting to a node's dB.
fn get_db_url(config: &NodeConfig) -> String {
    config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))
        .unwrap()
        .url
        .clone()
}

/// Returns inputs for running a test.
pub(super) fn get_inputs(ctx: &TestRunContextInfo, params: Params) -> Inputs {
    Inputs::new(
        get_inputs_of_net(ctx, params),
        get_inputs_of_system_state(params),
    )
}

/// Returns inputs for launching a network.
fn get_inputs_of_net(ctx: &TestRunContextInfo, params: Params) -> NetInputs {
    NetInputs::new([
        get_inputs_of_node(ctx, params, constants::PARTY_IDX_0),
        get_inputs_of_node(ctx, params, constants::PARTY_IDX_1),
        get_inputs_of_node(ctx, params, constants::PARTY_IDX_2),
    ])
}

/// Returns inputs for launching a node.
fn get_inputs_of_node(ctx: &TestRunContextInfo, params: Params, party_idx: usize) -> NodeInputs {
    let args = NodeArgs::new(
        params.batch_size(),
        params.batch_size_error_rate(),
        params.max_indexation_id(),
        params.perform_db_snapshot(),
        params.use_db_backup_as_source(),
    );
    let config =
        resources::read_node_config(ctx.exec_env(), format!("node-{}-genesis-0", party_idx))
            .unwrap();

    NodeInputs::new(args, config)
}

/// Returns inputs for initializing system state.
fn get_inputs_of_system_state(params: Params) -> SystemStateInputs {
    SystemStateInputs::new(params, vec![])
}
