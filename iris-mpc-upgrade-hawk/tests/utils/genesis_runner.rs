use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::config::Config;
use iris_mpc_cpu::genesis::{get_iris_deletions, plaintext::GenesisArgs, BatchSizeConfig};
use itertools::izip;
use rand::{thread_rng, Rng};
use tokio::task::JoinSet;

use crate::utils::{
    constants::COUNT_OF_PARTIES,
    irises,
    mpc_node::{DbAssertions, MpcNode, MpcNodes},
    resources,
    s3_deletions::{get_aws_clients, upload_iris_deletions},
    HawkConfigs, IrisCodePair, TestRunContextInfo,
};

pub const NUM_GPU_IRISES_INIT: usize = 100;
pub const MAX_INDEXATION_ID: usize = 100;

pub const DEFAULT_GENESIS_ARGS: GenesisArgs = GenesisArgs {
    max_indexation_id: MAX_INDEXATION_ID as u32,
    batch_size_config: BatchSizeConfig::Static { size: 1 },
};

pub fn get_node_configs() -> [Config; 3] {
    let exec_env = TestRunContextInfo::new(0, 0);

    (0..COUNT_OF_PARTIES)
        .map(|idx| resources::read_node_config(&exec_env, format!("node-{idx}-genesis-0")).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn get_irises() -> Vec<IrisCodePair> {
    let irises_path =
        resources::get_resource_path("iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson");
    irises::read_irises_from_ndjson(irises_path, NUM_GPU_IRISES_INIT).unwrap()
}

async fn get_nodes(configs: &HawkConfigs) -> impl Iterator<Item = Arc<MpcNode>> {
    MpcNodes::new(configs).await.into_iter()
}

pub async fn base_genesis_e2e_init(configs: &HawkConfigs, s3_deletion_ids: Vec<u32>) -> Result<()> {
    let hawk_prf0 = configs[0].hawk_prf_key;
    assert!(
        configs.iter().all(|c| c.hawk_prf_key == hawk_prf0),
        "All hawk_prf_key values in configs must be equal"
    );

    let plaintext_irises = get_irises();

    let shares_rng_seed: u64 = thread_rng().gen();
    let secret_shared_irises =
        irises::share_irises_locally(&plaintext_irises, shares_rng_seed).unwrap();

    let mut join_set = JoinSet::new();
    for (node, shares) in izip!(get_nodes(configs).await, secret_shared_irises.into_iter()) {
        join_set.spawn(async move {
            node.init_tables(&shares).await.unwrap();
        });
    }
    join_set.join_all().await;

    // any config file is sufficient to connect to S3
    let config = &configs[0];

    let aws_clients = get_aws_clients(config).await.unwrap();
    upload_iris_deletions(
        &s3_deletion_ids,
        &aws_clients.s3_client,
        &config.environment,
    )
    .await
    .unwrap();

    Ok(())
}

pub async fn base_genesis_e2e_init_assertions(
    configs: &HawkConfigs,
    n_deletions: usize,
) -> Result<()> {
    // Assert databases
    let gpu_asserts = DbAssertions::new()
        .assert_num_irises(NUM_GPU_IRISES_INIT)
        .assert_num_modifications(0);

    let cpu_asserts = DbAssertions::new()
        .assert_num_irises(0)
        .assert_num_modifications(0)
        .assert_last_indexed_iris_id(0)
        .assert_last_indexed_modification_id(0);

    let nodes = MpcNodes::new(configs).await;
    nodes.apply_assertions(gpu_asserts, cpu_asserts).await;

    // Assert localstack
    let config = &configs[0];
    let aws_clients = get_aws_clients(config).await.unwrap();
    let deletions = get_iris_deletions(config, &aws_clients.s3_client, 100)
        .await
        .unwrap();
    assert_eq!(deletions.len(), n_deletions);

    Ok(())
}
