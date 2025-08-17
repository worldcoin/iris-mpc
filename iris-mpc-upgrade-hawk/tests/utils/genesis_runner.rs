use iris_mpc_common::config::Config;

use crate::utils::{constants::COUNT_OF_PARTIES, resources, TestRunContextInfo};

pub fn get_node_configs() -> [Config; 3] {
    let exec_env = TestRunContextInfo::new(0, 0);

    (0..COUNT_OF_PARTIES)
        .map(|idx| resources::read_node_config(&exec_env, format!("node-{idx}-genesis-0")).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
