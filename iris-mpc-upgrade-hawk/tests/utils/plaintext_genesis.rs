use crate::utils::IrisCodePair;
use eyre::Result;
use iris_mpc_common::{config::Config, iris_db::iris::IrisCode, IrisSerialId, IrisVersionId};
use iris_mpc_cpu::{
    genesis::plaintext::{
        run_plaintext_genesis, GenesisArgs, GenesisConfig, GenesisDstDbState, GenesisSrcDbState,
        GenesisState, PersistentState,
    },
    hnsw::GraphMem,
};
use std::collections::HashMap;

/// use the builder pattern to run plaintext genesis, to make it easy to add arguments later without having to go back and change old tests
pub struct PlaintextGenesis<'a> {
    genesis_args: GenesisArgs,
    config: &'a Config,
    pairs: &'a [IrisCodePair],
    deletions: Vec<u32>,
}

impl<'a> PlaintextGenesis<'a> {
    pub fn new(genesis_args: GenesisArgs, config: &'a Config, pairs: &'a [IrisCodePair]) -> Self {
        Self {
            genesis_args,
            config,
            pairs,
            deletions: vec![],
        }
    }

    pub fn with_deletions(mut self, deletions: Vec<u32>) -> Self {
        self.deletions = deletions;
        self
    }

    pub async fn run(self) -> Result<GenesisState> {
        simulate_genesis(self.genesis_args, self.config, self.pairs, self.deletions).await
    }
}

async fn simulate_genesis(
    genesis_args: GenesisArgs,
    config: &Config,
    pairs: &[IrisCodePair],
    deletions: Vec<u32>,
) -> Result<GenesisState> {
    let genesis_input = get_genesis_input(pairs);

    let genesis_config = GenesisConfig {
        hnsw_M: config.hnsw_param_M,
        hnsw_ef_constr: config.hnsw_param_ef_constr,
        hnsw_ef_search: config.hnsw_param_ef_search,
        hawk_prf_key: config.hawk_prf_key,
    };

    let genesis_state =
        construct_initial_genesis_state(genesis_config, genesis_args, genesis_input, deletions);

    let expected_genesis_state = run_plaintext_genesis(genesis_state)
        .await
        .expect("plaintext genesis failed");
    Ok(expected_genesis_state)
}

fn construct_initial_genesis_state(
    genesis_config: GenesisConfig,
    genesis_args: GenesisArgs,
    input: HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)>,
    s3_deletions: Vec<u32>,
) -> GenesisState {
    GenesisState {
        src_db: GenesisSrcDbState {
            irises: input,
            modifications: (),
        },
        dst_db: GenesisDstDbState {
            irises: HashMap::new(),
            graphs: [GraphMem::new(), GraphMem::new()],
            persistent_state: PersistentState {
                last_indexed_iris_id: None,
                last_indexed_modification_id: None,
            },
        },
        config: genesis_config,
        args: genesis_args,
        s3_deletions,
    }
}

// construct_initial_genesis_state() needs a special HashMap. Build it from the provided list of plaintext iris shares.
fn get_genesis_input(
    pairs: &[IrisCodePair],
) -> HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)> {
    let mut r = HashMap::new();
    for (idx, (left, right)) in pairs.iter().enumerate() {
        // warning: iris id can't be zero
        r.insert(idx as u32 + 1, (0, left.clone(), right.clone()));
    }
    r
}
