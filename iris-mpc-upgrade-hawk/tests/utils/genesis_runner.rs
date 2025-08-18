use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::config::Config;
use itertools::izip;
use tokio::task::JoinSet;

use crate::utils::{
    constants::COUNT_OF_PARTIES,
    irises,
    mpc_node::{MpcNode, MpcNodes},
    resources,
    s3_deletions::{get_aws_clients, upload_iris_deletions},
    HawkConfigs, IrisCodePair, TestRunContextInfo,
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
    irises::read_irises_from_ndjson(irises_path, 100).unwrap()
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
    let secret_shared_irises =
        irises::share_irises_locally(&plaintext_irises, hawk_prf0.unwrap_or_default()).unwrap();

    let mut join_set = JoinSet::new();
    for (node, shares) in izip!(get_nodes(configs).await, secret_shared_irises.into_iter()) {
        join_set.spawn(async move {
            node.init_tables(&shares).await.unwrap();
        });
    }

    while let Some(r) = join_set.join_next().await {
        r.unwrap();
    }

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
