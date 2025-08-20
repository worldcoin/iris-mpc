use std::sync::LazyLock;

use crate::utils::{
    constants::COUNT_OF_PARTIES,
    irises,
    modifications::{ModificationInput, ModificationType},
    mpc_node::{MpcNode, MpcNodes},
    plaintext_genesis,
    resources::{self},
    s3_deletions::{get_aws_clients, upload_iris_deletions},
    HawkConfigs, IrisCodePair, TestError, TestRun, TestRunContextInfo,
};
use eyre::Result;
use iris_mpc_cpu::genesis::{
    get_iris_deletions,
    plaintext::{run_plaintext_genesis, GenesisArgs, GenesisState},
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use itertools::izip;
use tokio::task::JoinSet;

pub struct Test {
    configs: HawkConfigs,
}

const DEFAULT_GENESIS_ARGS: GenesisArgs = GenesisArgs {
    max_indexation_id: 100,
    batch_size: 1,
    batch_size_error_rate: 250,
};

static MODIFICATIONS: LazyLock<Vec<ModificationInput>> = LazyLock::new(|| {
    ModificationInput::from_slice(&[
        (1, ModificationType::Uniqueness, true, true),
        (2, ModificationType::ResetUpdate, true, true),
        (3, ModificationType::Reauth, true, true),
    ])
});

fn get_irises() -> Vec<IrisCodePair> {
    let irises_path =
        resources::get_resource_path("iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson");
    irises::read_irises_from_ndjson(irises_path, 100).unwrap()
}

impl Test {
    pub fn new() -> Self {
        let exec_env = TestRunContextInfo::new(0, 0);

        let configs = (0..COUNT_OF_PARTIES)
            .map(|idx| {
                resources::read_node_config(&exec_env, format!("node-{idx}-genesis-0")).unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self { configs }
    }

    async fn get_nodes(&self) -> impl Iterator<Item = MpcNode> {
        MpcNodes::new(&self.configs).await.into_iter()
    }
}

impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        DEFAULT_GENESIS_ARGS.batch_size,
                        DEFAULT_GENESIS_ARGS.batch_size_error_rate,
                        DEFAULT_GENESIS_ARGS.max_indexation_id,
                        false,
                        false,
                    ),
                    config,
                )
                .await
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap()?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        // Simulate genesis execution in plaintext
        let mut state_0 = GenesisState::default();
        state_0.src_db.irises = plaintext_genesis::init_plaintext_irises_db(&get_irises());
        state_0.config = plaintext_genesis::init_plaintext_config(&self.configs[0]);
        plaintext_genesis::apply_src_modifications(&mut state_0.src_db, &MODIFICATIONS)?;
        state_0.args = DEFAULT_GENESIS_ARGS;

        let expected = run_plaintext_genesis(state_0)
            .await
            .expect("Plaintext genesis execution failed");

        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            let expected = expected.clone();
            let max_indexation_id = DEFAULT_GENESIS_ARGS.max_indexation_id as usize;
            join_set.spawn(async move {
                // inspect the postgres tables - iris counts, modifications, and persisted_state
                let num_irises = node.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_modifications = node.gpu_iris_store.last_modifications(100).await.unwrap();
                assert_eq!(num_modifications.len(), MODIFICATIONS.len());

                let num_modifications = node.cpu_iris_store.last_modifications(100).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                assert_eq!(100, node.get_last_indexed_iris_id().await);
                assert_eq!(3, node.get_last_indexed_modification_id().await);

                node.assert_graphs_match(&expected).await;

                assert_eq!(
                    expected.dst_db.graphs[0].layers[0].links.len(),
                    DEFAULT_GENESIS_ARGS.max_indexation_id as usize
                );
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<(), TestError> {
        let hawk_prf0 = self.configs[0].hawk_prf_key;
        assert!(
            self.configs.iter().all(|c| c.hawk_prf_key == hawk_prf0),
            "All hawk_prf_key values in configs must be equal"
        );

        let plaintext_irises = get_irises();
        let secret_shared_irises =
            irises::share_irises_locally(&plaintext_irises, hawk_prf0.unwrap_or_default()).unwrap();

        let mut join_set = JoinSet::new();
        for (node, shares) in izip!(self.get_nodes().await, secret_shared_irises.into_iter()) {
            join_set.spawn(async move {
                node.init_tables(&shares).await.unwrap();
                node.insert_modifications(&MODIFICATIONS).await.unwrap();
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        // any config file is sufficient to connect to S3
        let config = &self.configs[0];

        let deleted_serial_ids = vec![];
        let aws_clients = get_aws_clients(config).await.unwrap();
        upload_iris_deletions(
            &deleted_serial_ids,
            &aws_clients.s3_client,
            &config.environment,
        )
        .await
        .unwrap();

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move {
                let num_irises = node.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, DEFAULT_GENESIS_ARGS.max_indexation_id as usize);

                let num_irises = node.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, 0);

                let num_modifications = node.gpu_iris_store.last_modifications(100).await.unwrap();
                assert_eq!(num_modifications.len(), MODIFICATIONS.len());

                let num_modifications = node.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                assert_eq!(0, node.get_last_indexed_iris_id().await);
                assert_eq!(0, node.get_last_indexed_modification_id().await);
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        // Assert localstack.

        let config = &self.configs[0];
        let aws_clients = get_aws_clients(config).await.unwrap();
        let deletions = get_iris_deletions(config, &aws_clients.s3_client, 100)
            .await
            .unwrap();
        assert_eq!(deletions.len(), 0);

        Ok(())
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        Ok(())
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        Ok(())
    }
}
