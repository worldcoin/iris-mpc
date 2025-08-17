use crate::utils::{
    constants::COUNT_OF_PARTIES,
    irises,
    modifications::{
        self, ModificationInput,
        ModificationType::{Reauth, ResetUpdate},
    },
    mpc_node::{MpcNode, MpcNodes},
    plaintext_genesis::{self, get_vector_ids},
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

const MODIFICATIONS: [ModificationInput; 2] = [
    ModificationInput::new(1, 3, Reauth, true, true),
    ModificationInput::new(2, 5, ResetUpdate, true, true),
];

pub struct Test {
    configs: HawkConfigs,
}

const DEFAULT_GENESIS_ARGS: GenesisArgs = GenesisArgs {
    max_indexation_id: 100,
    batch_size: 1,
    batch_size_error_rate: 250,
};

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
    // index 50 irises
    // then insert modifications
    // then run genesis again
    async fn exec(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        DEFAULT_GENESIS_ARGS.batch_size,
                        DEFAULT_GENESIS_ARGS.batch_size_error_rate,
                        50,
                        false,
                        false,
                    ),
                    config,
                )
                .await
                .unwrap()
            });
        }
        join_set.join_all().await;

        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move {
                node.apply_modifications(&[], &MODIFICATIONS).await.unwrap();
            });
        }
        join_set.join_all().await;

        // hack to get around an "address already in use" error, emitted by HawkHandle
        let service_ports = vec!["4003".into(), "4004".into(), "4005".into()];

        let mut join_set = JoinSet::new();
        for mut config in self.configs.iter().cloned() {
            let genesis_args = DEFAULT_GENESIS_ARGS;
            config.service_ports = service_ports.clone();
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        genesis_args.batch_size,
                        genesis_args.batch_size_error_rate,
                        100,
                        false,
                        false,
                    ),
                    config,
                )
                .await
                .unwrap()
            });
        }
        join_set.join_all().await;

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        // Simulate genesis execution in plaintext
        let mut state_0 = GenesisState::default();
        state_0.src_db.irises = plaintext_genesis::init_plaintext_irises_db(&get_irises());
        state_0.config = plaintext_genesis::init_plaintext_config(&self.configs[0]);
        state_0.args = DEFAULT_GENESIS_ARGS;
        state_0.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(state_0)
            .await
            .expect("Stage 1 of plaintext genesis execution failed");

        plaintext_genesis::apply_modifications(&mut state_1.src_db, &[], &MODIFICATIONS)?;
        state_1.args.max_indexation_id = 100;

        let expected = run_plaintext_genesis(state_1)
            .await
            .expect("Stage 2 of plaintext genesis execution failed");

        // Number of modifications on already-indexed irises which are processed by genesis delta
        let num_updating_modifications =
            modifications::modifications_extension_updates(&[], &MODIFICATIONS)
                .into_iter()
                .filter(|serial_id| *serial_id <= 50)
                .count();

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

                let cpu_vector_ids = node.get_cpu_iris_vector_ids(1000).await.unwrap();
                let expected_vector_ids = get_vector_ids(&expected.dst_db.irises);
                assert_eq!(cpu_vector_ids, expected_vector_ids);

                let num_modifications = node.gpu_iris_store.last_modifications(100).await.unwrap();
                assert_eq!(num_modifications.len(), MODIFICATIONS.len());

                let num_modifications = node.cpu_iris_store.last_modifications(100).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                assert_eq!(100, node.get_last_indexed_iris_id().await);
                assert_eq!(2, node.get_last_indexed_modification_id().await);

                node.assert_graphs_match(&expected).await;

                // New graph entries are created for modified iris codes
                assert_eq!(
                    expected.dst_db.graphs[0].layers[0].links.len(),
                    max_indexation_id + num_updating_modifications
                );
            });
        }
        join_set.join_all().await;

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
            });
        }
        join_set.join_all().await;

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
            let genesis_args = DEFAULT_GENESIS_ARGS;
            let max_indexation_id = genesis_args.max_indexation_id as usize;
            join_set.spawn(async move {
                let num_irises = node.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, 0);

                let num_modifications = node.gpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = node.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                assert_eq!(0, node.get_last_indexed_iris_id().await);
                assert_eq!(0, node.get_last_indexed_modification_id().await);
            });
        }
        join_set.join_all().await;

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
