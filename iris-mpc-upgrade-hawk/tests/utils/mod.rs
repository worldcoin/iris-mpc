pub mod constants;
mod errors;
mod logger;
pub mod resources;
pub mod runner;
pub mod s3_client;
mod store;
mod types;

use std::collections::HashMap;

pub use errors::TestError;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId, IrisVersionId};
use iris_mpc_cpu::{
    genesis::plaintext::{
        GenesisArgs, GenesisConfig, GenesisDstDbState, GenesisSrcDbState, GenesisState,
        PersistentState,
    },
    hnsw::GraphMem,
};
pub use runner::{TestExecutionEnvironment, TestRun, TestRunContextInfo};
pub use store::NetDbProvider;
pub use types::{GaloisRingSharedIrisPair, HawkConfigs, IrisCodePair};

pub fn construct_initial_genesis_state(
    genesis_config: GenesisConfig,
    genesis_args: GenesisArgs,
    input: HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)>,
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
        s3_deletions: Vec::new(),
    }
}
