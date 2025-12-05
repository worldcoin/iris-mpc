//! # Genesis Protocol Plaintext Reference Implementation
//!
//! This module provides a local, in-memory reference implementation of the
//! genesis indexer protocol, which extracts the core application logic from the
//! surrounding network services, MPC functionality, and parallel execution
//! infrastructure.  The goal is to represent the intended logic of the genesis
//! server application in as simple and understandable a package as possible, to
//! make it clear what the complex server software is supposed to be doing.
//!
//! As an invariant, the outcome of successfully running the genesis server
//! binary on MPC servers in the context of properly initialized network
//! resources should logically match that produced by this plaintext
//! implementation.  This equivalence is partially validated by e2e test cases.

use std::{collections::HashMap, sync::Arc};

use eyre::{bail, OptionExt, Result};
use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    },
    iris_db::iris::IrisCode,
    IrisSerialId, IrisVectorId, IrisVersionId,
};
use itertools::izip;
use rand::{thread_rng, Rng};

use crate::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        BothEyes, STORE_IDS,
    },
    genesis::BatchSize,
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef},
    hnsw::{
        graph::neighborhood::Neighborhood, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
        SortedNeighborhood,
    },
};

/// Represents irises db table, mapping serial ids to version, and left and right iris codes.
pub type IrisesTable = HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)>;

/// Represents modifications db table, mapping modification ids to tuples of serial id,
/// request type, completion flag, and persisted flag.
pub type ModificationsTable = HashMap<i64, (IrisSerialId, String, bool, bool)>;

/// Represents a left/right pair of plaintext in-memory HNSW graphs.
pub type PlaintextGraphs = BothEyes<GraphMem<PlaintextVectorRef>>;

/// List of serial ids to treat as deleted enrollments in the source iris database.
pub type GenesisDeletions = Vec<IrisSerialId>;

/// Plaintext representation of global state of genesis indexer.
#[derive(Default, Clone)]
pub struct GenesisState {
    pub src_db: GenesisSrcDbState,

    pub dst_db: GenesisDstDbState,

    pub config: GenesisConfig,

    pub args: GenesisArgs,

    pub s3_deletions: GenesisDeletions,
}

/// State of the source database from the GPU server.
#[derive(Default, Clone)]
pub struct GenesisSrcDbState {
    pub irises: IrisesTable,

    pub modifications: ModificationsTable,
}

/// State of the destination database for the CPU server.
#[derive(Default, Clone)]
pub struct GenesisDstDbState {
    pub irises: IrisesTable,

    pub graphs: PlaintextGraphs,

    pub persistent_state: PersistentState,
}

/// Database entries for the PersistentState table.
#[derive(Debug, Default, Clone, Copy)]
pub struct PersistentState {
    pub last_indexed_iris_id: Option<IrisSerialId>,

    pub last_indexed_modification_id: Option<i64>,
}

/// Logical configuration parameters of genesis `Config` struct.
#[derive(Debug, Default, Clone, Copy)]
#[allow(non_snake_case)]
pub struct GenesisConfig {
    pub hnsw_M: usize,

    pub hnsw_ef_constr: usize,

    pub hnsw_ef_search: usize,

    pub hawk_prf_key: Option<u64>,
}

/// Logical CLI arguments for genesis process.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenesisArgs {
    pub max_indexation_id: IrisSerialId,

    pub batch_size: usize,

    pub batch_size_error_rate: usize,
}

/// Execute the genesis indexer over a plaintext state representation, and
/// return the resulting state.
pub async fn run_plaintext_genesis(mut state: GenesisState) -> Result<GenesisState> {
    // Unpack persistent state fields
    let last_indexed_iris_id = state
        .dst_db
        .persistent_state
        .last_indexed_iris_id
        .unwrap_or(0);
    let last_indexed_modification_id = state
        .dst_db
        .persistent_state
        .last_indexed_modification_id
        .unwrap_or(0);

    // Id currently being inspected
    let mut id = last_indexed_iris_id;
    let target_id = state.args.max_indexation_id;

    let batch_size = match state.args.batch_size {
        0 => BatchSize::get_dynamic_size(id, state.args.batch_size_error_rate, state.config.hnsw_M),
        batch_size => batch_size,
    };

    // Initialize existing vector stores
    let mut left_store = PlaintextStore::new();
    let mut right_store = PlaintextStore::new();
    for (serial_id, (version, left_iris, right_iris)) in state.src_db.irises.iter() {
        let vector_id = IrisVectorId::new(*serial_id, *version);
        left_store
            .insert_at(&vector_id, &Arc::new(left_iris.clone()))
            .await?;
        right_store
            .insert_at(&vector_id, &Arc::new(right_iris.clone()))
            .await?;
    }

    let searcher = HnswSearcher::new_standard(
        state.config.hnsw_ef_constr,
        state.config.hnsw_ef_search,
        state.config.hnsw_M,
    );

    let prf_key: [u8; 16] = state
        .config
        .hawk_prf_key
        .map(|key_u64| (key_u64 as u128).to_le_bytes())
        .unwrap_or_else(|| thread_rng().gen());

    // ⚓ Start: Delta protocol

    // Filter modifications for those which apply to current genesis index
    let mut applicable_modifications: Vec<_> = state
        .src_db
        .modifications
        .iter()
        .filter(
            |(mod_id, (serial_id, request_type, completed, persisted))| {
                **mod_id > last_indexed_modification_id
                    && *serial_id <= last_indexed_iris_id
                    && *completed
                    && *persisted
                    && (request_type == IDENTITY_DELETION_MESSAGE_TYPE
                        || request_type == REAUTH_MESSAGE_TYPE
                        || request_type == RESET_UPDATE_MESSAGE_TYPE)
            },
        )
        .map(|(mod_id, (serial_id, request_type, _status, _persisted))| {
            (*mod_id, *serial_id, request_type.clone())
        })
        .collect();
    applicable_modifications.sort_by_key(|(mod_id, _, _)| *mod_id);

    // Process applicable modifications entries
    for (mod_id, serial_id, request_type) in applicable_modifications {
        match request_type.as_str() {
            RESET_UPDATE_MESSAGE_TYPE | REAUTH_MESSAGE_TYPE => {
                let (vector_id, left_iris, right_iris) = state
                    .src_db
                    .irises
                    .get(&serial_id)
                    .map(|(version, left_iris, right_iris)| {
                        (
                            IrisVectorId::new(serial_id, *version),
                            left_iris.clone(),
                            right_iris.clone(),
                        )
                    })
                    .ok_or_eyre(format!(
                        "Modified iris serial id {serial_id} not found in src_db"
                    ))?;

                for (side, store, graph, iris) in izip!(
                    STORE_IDS,
                    [&mut left_store, &mut right_store],
                    &mut state.dst_db.graphs,
                    vec![left_iris, right_iris]
                ) {
                    let query = Arc::new(iris);

                    let identifier = (vector_id, side);
                    let insertion_layer = searcher.gen_layer_prf(&prf_key, &identifier)?;

                    let (links, update_ep) = searcher
                        .search_to_insert::<_, SortedNeighborhood<_>>(
                            store,
                            graph,
                            &query,
                            insertion_layer,
                        )
                        .await?;

                    // Trim and extract unstructured vector lists
                    let mut links_unstructured = Vec::new();
                    for (lc, mut l) in links.into_iter().enumerate() {
                        let m = searcher.params.get_M(lc);
                        l.trim(store, m).await?;
                        links_unstructured.push(l.edge_ids())
                    }

                    let insert_plan = InsertPlanV {
                        query,
                        links: links_unstructured,
                        update_ep,
                    };

                    insert::insert(
                        store,
                        graph,
                        &searcher,
                        vec![Some(insert_plan)],
                        &vec![Some(vector_id)],
                    )
                    .await?;
                }

                // Insert modified iris into destination db
                let irises = state.src_db.irises.get(&serial_id).unwrap().clone();
                state.dst_db.irises.insert(serial_id, irises);
            }
            _ => {
                bail!("Genesis does not support modifications of type {request_type}")
            }
        }

        // Update last_indexed_modification_id in destination db
        state.dst_db.persistent_state.last_indexed_modification_id = Some(mod_id);
    }

    // Update last_indexed_modification_id in destination db to largest persisted id
    let max_persisted_modification_id = state
        .src_db
        .modifications
        .iter()
        .filter(|(_, (_, _, complete, persisted))| *complete && *persisted)
        .map(|(id, _)| *id)
        .max()
        .unwrap_or(0);
    state.dst_db.persistent_state.last_indexed_modification_id =
        Some(max_persisted_modification_id);

    // ⚓ Start: Genesis indexing

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
                let insertion_layer = searcher.gen_layer_prf(&prf_key, &identifier)?;

                let (links, update_ep) = searcher
                    .search_to_insert::<_, SortedNeighborhood<_>>(
                        store,
                        graph,
                        &query,
                        insertion_layer,
                    )
                    .await?;

                // Trim and extract unstructured vector lists
                let mut links_unstructured = Vec::new();
                for (lc, mut l) in links.into_iter().enumerate() {
                    let m = searcher.params.get_M(lc);
                    l.trim(store, m).await?;
                    links_unstructured.push(l.edge_ids())
                }

                let insert_plan: InsertPlanV<PlaintextStore> = InsertPlanV {
                    query,
                    links: links_unstructured,
                    update_ep,
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
    use iris_mpc_common::{helpers::smpc_request, iris_db::db::IrisDB};
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

    fn apply_modification(
        state: &mut GenesisState,
        id: i64,
        serial_id: IrisSerialId,
        request_type: &str,
    ) {
        let mut rng = thread_rng();
        let updated_left_iris = IrisCode::random_rng(&mut rng);
        let updated_right_iris = IrisCode::random_rng(&mut rng);

        let old_version = state.src_db.irises.get(&serial_id).unwrap().0;
        state.src_db.irises.insert(
            serial_id,
            (old_version + 1, updated_left_iris, updated_right_iris),
        );

        add_modification(state, id, serial_id, request_type, true, true);
    }

    fn add_modification(
        state: &mut GenesisState,
        id: i64,
        serial_id: IrisSerialId,
        request_type: &str,
        completed: bool,
        persisted: bool,
    ) {
        state.src_db.modifications.insert(
            id,
            (serial_id, request_type.to_string(), completed, persisted),
        );
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

    #[tokio::test]
    async fn test_plaintext_genesis_delta() -> Result<()> {
        let mut init_state = gen_base_state(200);
        init_state.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(init_state).await?;

        assert_eq!(state_1.dst_db.irises.len(), 50);

        apply_modification(&mut state_1, 2, 20, smpc_request::REAUTH_MESSAGE_TYPE);
        assert_ne!(
            state_1.dst_db.irises.get(&20),
            state_1.src_db.irises.get(&20)
        );

        state_1.args.max_indexation_id = 100;

        let state_2 = run_plaintext_genesis(state_1).await?;

        assert_eq!(state_2.dst_db.irises.len(), 100);
        assert_eq!(
            state_2.dst_db.irises.get(&20),
            state_2.src_db.irises.get(&20)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_plaintext_genesis_delta_with_skipped() -> Result<()> {
        let mut init_state = gen_base_state(200);

        // Add modification that doesn't apply
        apply_modification(
            &mut init_state,
            1,
            10,
            smpc_request::RESET_UPDATE_MESSAGE_TYPE,
        );
        assert_ne!(
            init_state.dst_db.irises.get(&10),
            init_state.src_db.irises.get(&10)
        );
        init_state.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(init_state).await?;

        assert_eq!(state_1.dst_db.irises.len(), 50);
        assert_eq!(
            state_1.dst_db.irises.get(&10),
            state_1.src_db.irises.get(&10)
        );

        apply_modification(&mut state_1, 2, 20, smpc_request::REAUTH_MESSAGE_TYPE);
        assert_ne!(
            state_1.dst_db.irises.get(&20),
            state_1.src_db.irises.get(&20)
        );
        apply_modification(&mut state_1, 3, 70, smpc_request::RESET_UPDATE_MESSAGE_TYPE);

        state_1.args.max_indexation_id = 100;

        let state_2 = run_plaintext_genesis(state_1).await?;

        assert_eq!(state_2.dst_db.irises.len(), 100);
        assert_eq!(
            state_2.dst_db.irises.get(&20),
            state_2.src_db.irises.get(&20)
        );
        assert_eq!(
            state_2.dst_db.irises.get(&70),
            state_2.src_db.irises.get(&70)
        );

        Ok(())
    }
}
