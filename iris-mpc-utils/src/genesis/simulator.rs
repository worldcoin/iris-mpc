use crate::{
    irises::modifications::{self, ModificationInput},
    types::IrisCodePair,
};
use eyre::{bail, OptionExt, Result};
use iris_mpc_common::{
    config::Config, iris_db::iris::IrisCode, IrisSerialId, IrisVectorId, IrisVersionId,
};
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
/// specified modifications `cur_mods`, relative to prior state `last_mods`.
/// Appends modifications to the `modifications` field, and increments the
/// version number of the associated serial ids for modifications updating an
/// iris in the database which took place after `last_mods`.  Makes no change to
/// the iris database for a uniqueness modification entry.
pub fn apply_modifications(
    src_db: &mut GenesisSrcDbState,
    last_mods: &[ModificationInput],
    cur_mods: &[ModificationInput],
) -> Result<()> {
    if !modifications::modifications_extension_is_valid(last_mods, cur_mods) {
        bail!("Specified modifications are not a valid extension of the last modifications state.")
    }

    for m in cur_mods.iter() {
        src_db.modifications.insert(
            m.mod_id,
            (
                m.serial_id as u32,
                m.request_type.to_string(),
                m.completed,
                m.persisted,
            ),
        );
    }

    let update_serial_ids = modifications::modifications_extension_updates(last_mods, cur_mods);
    for serial_id in update_serial_ids {
        src_db
            .irises
            .get_mut(&(serial_id as u32))
            .ok_or_eyre("Modification specifies invalid iris serial id.")?
            .0 += 1;
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

pub fn get_vector_ids(irises: &IrisesTable) -> Vec<IrisVectorId> {
    let mut ids: Vec<_> = irises
        .iter()
        .map(|(serial_id, data)| IrisVectorId::new(*serial_id, data.0))
        .collect();
    ids.sort_by_key(|id| id.serial_id());
    ids
}
