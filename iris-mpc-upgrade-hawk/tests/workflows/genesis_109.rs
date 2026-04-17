use crate::run_genesis;
use crate::utils::{
    genesis_runner::{self, DEFAULT_GENESIS_ARGS},
    mpc_node::MpcNodes,
    HawkConfigs, TestRun, TestRunContextInfo,
};
use eyre::Result;
use iris_mpc_cpu::genesis::PruningMode;

macro_rules! assert_checkpoints {
    ($expected:expr, $configs:expr) => {{
        let checkpoints = MpcNodes::new(&$configs).await.get_num_checkpoints().await?;
        for num_cp in checkpoints {
            assert_eq!(num_cp, $expected);
        }
    }};
}

pub struct Test {
    configs: HawkConfigs,
}

impl Test {
    pub fn new() -> Self {
        Self {
            configs: genesis_runner::get_node_configs(),
        }
    }
}

impl TestRun for Test {
    async fn exec(&mut self) -> Result<()> {
        // Run 1: index irises 1-25.
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = 25;
        args.checkpoint_frequency = 10;
        args.pruning_mode = PruningMode::OlderNonArchival;

        run_genesis!(self, args);
        // should have 3 checkpoints: 10, 20, and 25
        assert_checkpoints!(3, self.configs.clone());

        // Run 2: index irises 26-50.
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = 50;
        args.checkpoint_frequency = 50;
        args.pruning_mode = PruningMode::None;

        run_genesis!(self, args);
        // should have 4 checkpoints: 10, 20, 25, and 50
        assert_checkpoints!(4, self.configs.clone());

        // Run 3: index irises 51-75.
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = 75;
        args.checkpoint_frequency = 50;
        args.pruning_mode = PruningMode::OlderNonArchival;

        run_genesis!(self, args);
        // should have 3 checkpoints: 25, 50, and 75 (pruned 10 and 20)
        assert_checkpoints!(3, self.configs.clone());

        // Run 4: index irises 76-100
        let mut args = DEFAULT_GENESIS_ARGS;
        args.max_indexation_id = 100;
        args.checkpoint_frequency = 50;
        args.pruning_mode = PruningMode::AllOlder;

        run_genesis!(self, args);
        // pruned all but the most recent when the test began, then added a checkpoint
        assert_checkpoints!(2, self.configs.clone());

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        let test_deletions = vec![];
        genesis_runner::base_genesis_e2e_init(&self.configs, test_deletions).await
    }

    async fn setup_assert(&mut self) -> Result<()> {
        genesis_runner::base_genesis_e2e_init_assertions(&self.configs, 0).await
    }

    async fn teardown(&mut self) -> Result<()> {
        let nodes = MpcNodes::new(&self.configs).await;
        nodes.cleanup_s3_checkpoints(&self.configs).await
    }
}
