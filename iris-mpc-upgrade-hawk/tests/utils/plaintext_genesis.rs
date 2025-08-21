use crate::utils::{
    modifications::{ModificationInput, ModificationType},
    IrisCodePair,
};
use eyre::{OptionExt, Result};
use iris_mpc_common::{config::Config, iris_db::iris::IrisCode, IrisSerialId, IrisVersionId};
use iris_mpc_cpu::{
    genesis::plaintext::{
        run_plaintext_genesis, GenesisArgs, GenesisConfig, GenesisDstDbState, GenesisSrcDbState,
        GenesisState, IrisesTable, PersistentState,
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
    let genesis_input = init_plaintext_irises_db(pairs);

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
            modifications: HashMap::new(),
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

/// Create a `HashMap`` representing the irises db table for plaintext genesis
/// execution.
pub fn init_plaintext_irises_db(pairs: &[IrisCodePair]) -> IrisesTable {
    pairs
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, (left, right))| (idx as u32 + 1, (0i16, left, right)))
        .collect()
}

/// Update plaintext genesis source database state to reflect application of the
/// specified modification inputs.  Appends modifications to the `modifications`
/// field, and increments the version number of the associated serial ids for
/// modifications updating an iris in the database.  Makes no change to the
/// iris database for a uniqueness modification entry.
pub fn apply_src_modifications(
    src_db: &mut GenesisSrcDbState,
    modifications: &[ModificationInput],
) -> Result<()> {
    let max_modification_id = src_db.modifications.keys().cloned().max().unwrap_or(0);

    for (idx, m) in modifications.iter().enumerate() {
        if matches!(
            m.request_type,
            ModificationType::ResetUpdate | ModificationType::Reauth
        ) {
            let entry = src_db
                .irises
                .get_mut(&(m.serial_id as u32))
                .ok_or_eyre("Modified iris serial id missing from plaintext database")?;
            entry.0 += 1;
        }

        src_db.modifications.insert(
            max_modification_id + (idx as i64) + 1,
            (
                m.serial_id as u32,
                m.request_type.to_string(),
                m.completed,
                m.persisted,
            ),
        );
    }

    Ok(())
}

/// Initialize relevant configuration parameters for plaintext genesis execution
/// from a full server `Config` struct.
pub fn init_plaintext_config(config: &Config) -> GenesisConfig {
    GenesisConfig {
        hnsw_M: config.hnsw_param_M,
        hnsw_ef_constr: config.hnsw_param_ef_constr,
        hnsw_ef_search: config.hnsw_param_ef_search,
        hawk_prf_key: config.hawk_prf_key,
    }
}
