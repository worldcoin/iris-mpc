use std::collections::HashMap;

use super::types::GaloisRingSharedIrisPair;
use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId, IrisVersionId};
use iris_mpc_cpu::{
    genesis::plaintext::{
        GenesisArgs, GenesisConfig, GenesisDstDbState, GenesisSrcDbState, GenesisState,
        PersistentState,
    },
    hnsw::GraphMem,
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use itertools::Itertools;

/// Persists Iris shares to remote databases.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `iris_shares` - set of Iris shares to be inserted into store.
/// * `pgres_tx_size` - Constraint over number of Iris shares to persist in a single pgres transaction.
///
pub async fn insert_iris_shares(
    iris_store: &IrisStore,
    pgres_tx_batch_size: usize,
    iris_shares: Vec<GaloisRingSharedIrisPair>,
) -> Result<(IrisSerialId, IrisSerialId)> {
    let start_serial_id = iris_store.get_max_serial_id().await.unwrap_or(0) + 1;
    let end_serial_id = start_serial_id + iris_shares.len() - 1;

    let mut tx = iris_store.tx().await?;
    for batch in &iris_shares.iter().enumerate().chunks(pgres_tx_batch_size) {
        let iris_refs: Vec<_> = batch
            .map(|(idx, (iris_l, iris_r))| StoredIrisRef {
                id: (start_serial_id + idx) as i64,
                left_code: &iris_l.code.coefs,
                left_mask: &iris_l.mask.coefs,
                right_code: &iris_r.code.coefs,
                right_mask: &iris_r.mask.coefs,
            })
            .collect();

        iris_store.insert_irises(&mut tx, &iris_refs).await?;
        tx.commit().await?;
        tx = iris_store.tx().await?;
    }

    Ok((
        start_serial_id as IrisSerialId,
        end_serial_id as IrisSerialId,
    ))
}

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
