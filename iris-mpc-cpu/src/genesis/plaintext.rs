use std::{collections::HashMap, sync::Arc};

use eyre::{OptionExt, Result};
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId, IrisVectorId, IrisVersionId};
use itertools::izip;
use rand::{thread_rng, Rng};

use crate::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        BothEyes, STORE_IDS,
    },
    genesis::BatchSize,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{vector_store::VectorStoreMut, GraphMem, HnswParams, HnswSearcher},
};

/// Represents irises db table, mapping serial ids to version, and left and right iris codes.
pub type IrisesTable = HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)>;

/// Represents a left/right pair of plaintext in-memory HNSW graphs.
pub type PlaintextGraphs = BothEyes<GraphMem<PlaintextStore>>;

/// List of serial ids to treat as deleted enrollments in the source iris database.
pub type GenesisDeletions = Vec<IrisSerialId>;

/// Plaintext representation of global state of genesis indexer.
pub struct GenesisState {
    pub src_db: GenesisSrcDbState,

    pub dst_db: GenesisDstDbState,

    pub config: GenesisConfig,

    pub args: GenesisArgs,

    pub s3_deletions: GenesisDeletions,
}

/// State of the source database from the GPU server.
pub struct GenesisSrcDbState {
    pub irises: IrisesTable,

    pub modifications: (),
}

/// State of the destination database for the CPU server.
pub struct GenesisDstDbState {
    pub irises: IrisesTable,

    pub graphs: PlaintextGraphs,

    pub persistent_state: PersistentState,
}

/// Database entries for the PersistentState table.
pub struct PersistentState {
    pub last_indexed_iris_id: Option<IrisSerialId>,

    pub last_indexed_modification_id: Option<i64>,
}

/// Logical configuration parameters of genesis `Config` struct.
#[allow(non_snake_case)]
pub struct GenesisConfig {
    pub hnsw_M: usize,

    pub hnsw_ef_constr: usize,

    pub hnsw_ef_search: usize,

    pub hawk_prf_key: Option<u64>,
}

/// Logical CLI arguments for genesis process.
pub struct GenesisArgs {
    pub max_indexation_id: IrisSerialId,

    pub batch_size: usize,

    pub batch_size_error_rate: usize,
}

/// Execute the genesis indexer over a plaintext state representation, and
/// return the resulting state.
pub async fn run_plaintext_genesis(mut state: GenesisState) -> Result<GenesisState> {
    // Id currently being inspected
    let mut id = state
        .dst_db
        .persistent_state
        .last_indexed_iris_id
        .unwrap_or(0);
    let target_id = state.args.max_indexation_id;

    let batch_size = match state.args.batch_size {
        0 => BatchSize::get_dynamic_size(id, state.args.batch_size_error_rate, state.config.hnsw_M),
        batch_size => batch_size,
    };

    // Initialize existing vector stores
    let mut left_store = PlaintextStore::new();
    let mut right_store = PlaintextStore::new();
    for (serial_id, (version, left_iris, right_iris)) in state.dst_db.irises.iter() {
        let vector_id = IrisVectorId::new(*serial_id, *version);
        left_store
            .insert_at(&vector_id, &Arc::new(left_iris.clone()))
            .await?;
        right_store
            .insert_at(&vector_id, &Arc::new(right_iris.clone()))
            .await?;
    }

    let search_params = HnswParams::new(
        state.config.hnsw_ef_constr,
        state.config.hnsw_ef_search,
        state.config.hnsw_M,
    );
    let searcher = HnswSearcher {
        params: search_params,
    };

    let prf_key: [u8; 16] = state
        .config
        .hawk_prf_key
        .map(|key_u64| (key_u64 as u128).to_le_bytes())
        .unwrap_or_else(|| thread_rng().gen());

    // Generate and process batches until we've reached the target indexation id
    while id < target_id {
        // 1. Generate new batch
        let mut batch: Vec<(IrisSerialId, bool)> = Vec::new(); // Iris serial id, whether it should be indexed
        let mut n_to_index = 0;
        while n_to_index < batch_size && id < target_id {
            id += 1;

            let do_index = !state.s3_deletions.contains(&id);
            batch.push((id, do_index));

            if do_index {
                n_to_index += 1;
            }
        }

        // 2. Insert non-deleted entries into HNSW graphs
        let mut left_insert_plans = Vec::new();
        let mut right_insert_plans = Vec::new();

        let mut ids = Vec::new();

        for (cur_id, _) in batch.iter().filter(|(_, do_index)| *do_index) {
            let (version, left_iris, right_iris) = state
                .src_db
                .irises
                .get(cur_id)
                .ok_or_eyre("Expected iris id missing")?
                .clone();
            let vector_id = IrisVectorId::new(*cur_id, version);
            ids.push(Some(vector_id));

            // Initial search and construct insert plans
            for (side, iris, store, graph, results) in izip!(
                STORE_IDS,
                [left_iris, right_iris],
                [&mut left_store, &mut right_store],
                &state.dst_db.graphs,
                [&mut left_insert_plans, &mut right_insert_plans]
            ) {
                let query = Arc::new(iris);
                let identifier = (vector_id, side);
                let insertion_layer = searcher.select_layer_prf(&prf_key, &identifier)?;

                let (links, set_ep) = searcher
                    .search_to_insert(store, graph, &query, insertion_layer)
                    .await?;

                let insert_plan: InsertPlanV<PlaintextStore> = InsertPlanV {
                    query,
                    links,
                    set_ep,
                };

                results.push(Some(insert_plan));
            }
        }

        // Insert batch of insert plans using HawkActor insertion logic
        for (store, graph, plans) in izip!(
            [&mut left_store, &mut right_store],
            &mut state.dst_db.graphs,
            [left_insert_plans, right_insert_plans]
        ) {
            insert::insert(store, graph, &searcher, plans, &ids).await?;
        }

        // 3. Copy all irises to destination db
        for (cur_id, _) in batch {
            let irises = state.src_db.irises.get(&cur_id).unwrap().clone();
            state.dst_db.irises.insert(cur_id, irises);
        }

        // 4. Update last_indexed_iris_id in destination db
        state.dst_db.persistent_state.last_indexed_iris_id = Some(id);
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;

    use super::*;

    fn gen_base_state(n_src_enrollments: usize) -> GenesisState {
        let mut rng = AesRng::seed_from_u64(0);
        let irises_left = IrisDB::new_random_rng(n_src_enrollments, &mut rng).db;
        let irises_right = IrisDB::new_random_rng(n_src_enrollments, &mut rng).db;
        let src_db_irises: HashMap<_, _> = izip!(irises_left, irises_right)
            .enumerate()
            .map(|(id, (left, right))| (id as IrisSerialId, (0, left, right)))
            .collect();

        GenesisState {
            src_db: GenesisSrcDbState {
                irises: src_db_irises,
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
            config: GenesisConfig {
                hnsw_M: 256,
                hnsw_ef_constr: 320,
                hnsw_ef_search: 320,
                hawk_prf_key: Some(0),
            },
            args: GenesisArgs {
                max_indexation_id: 100,
                batch_size: 0,
                batch_size_error_rate: 128,
            },
            s3_deletions: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_plaintext_genesis() -> Result<()> {
        let init_state = gen_base_state(200);

        let new_state = run_plaintext_genesis(init_state).await?;

        assert_eq!(new_state.dst_db.irises.len(), 100);
        assert_eq!(new_state.dst_db.graphs[0].layers[0].links.len(), 100);
        assert_eq!(new_state.dst_db.graphs[1].layers[0].links.len(), 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_plaintext_genesis_with_deletions() -> Result<()> {
        let mut init_state = gen_base_state(200);
        init_state.s3_deletions = vec![25, 40, 50, 60, 90];

        let new_state = run_plaintext_genesis(init_state).await?;

        assert_eq!(new_state.dst_db.irises.len(), 100);
        assert_eq!(new_state.dst_db.graphs[0].layers[0].links.len(), 95);
        assert_eq!(new_state.dst_db.graphs[1].layers[0].links.len(), 95);

        Ok(())
    }

    #[tokio::test]
    async fn test_plaintext_genesis_repeated() -> Result<()> {
        let mut init_state = gen_base_state(200);
        init_state.s3_deletions = vec![25, 40, 50, 60, 90];
        init_state.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(init_state).await?;

        assert_eq!(state_1.dst_db.irises.len(), 50);
        assert_eq!(state_1.dst_db.graphs[0].layers[0].links.len(), 47);
        assert_eq!(state_1.dst_db.graphs[1].layers[0].links.len(), 47);

        state_1.args.max_indexation_id = 100;
        let state_2 = run_plaintext_genesis(state_1).await?;

        assert_eq!(state_2.dst_db.irises.len(), 100);
        assert_eq!(state_2.dst_db.graphs[0].layers[0].links.len(), 95);
        assert_eq!(state_2.dst_db.graphs[1].layers[0].links.len(), 95);

        Ok(())
    }

    #[tokio::test]
    async fn test_plaintext_genesis_batched() -> Result<()> {
        let mut init_state = gen_base_state(200);
        init_state.s3_deletions = vec![25, 40, 50, 60, 90];
        init_state.args.batch_size = 10;

        let new_state = run_plaintext_genesis(init_state).await?;

        assert_eq!(new_state.dst_db.irises.len(), 100);
        assert_eq!(new_state.dst_db.graphs[0].layers[0].links.len(), 95);
        assert_eq!(new_state.dst_db.graphs[1].layers[0].links.len(), 95);

        Ok(())
    }
}
