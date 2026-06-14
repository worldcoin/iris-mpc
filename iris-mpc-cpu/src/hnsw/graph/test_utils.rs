//! this file is used by test utilities

use std::sync::Arc;

use super::graph_store::GraphPg;
use crate::{
    execution::hawk_main::{BothEyes, LEFT, RIGHT},
    genesis::state_accessor::{
        STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID, STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
    },
    graph_checkpoint::download_genesis_checkpoint_plaintext,
    hnsw::{
        graph::{
            graph_diff::{
                explicit::{ExplicitNeighborhoodDiffer, SortBy},
                jaccard::DetailedJaccardDiffer,
                node_equiv::ensure_node_equivalence,
                run_diff,
            },
            mutation::EdgeType,
            neighborhood::Neighborhood,
            GraphMutation, MutationOp, UpdateEntryPoint,
        },
        vector_store::{VectorStore, VectorStoreMut},
        SortedNeighborhood,
    },
};
use crate::{
    graph_checkpoint::{upload_graph_checkpoint_plaintext, GraphCheckpointState},
    hawkers::{aby3::aby3_store::FhdOps, plaintext_store::PlaintextStore},
    hnsw::GraphMem,
    protocol::shared_iris::GaloisRingSharedIris,
};
use aes_prng::AesRng;
use aws_sdk_s3::Client as S3Client;
use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::{
    iris_db::db::IrisDB,
    vector_id::{HasSerialId, SerialId},
};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::Itertools;
use rand::SeedableRng;
use std::path::Path;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DiffMethod {
    /// Use the detailed Jaccard differ.
    DetailedJaccard,
    /// Use the explicit neighborhood differ, sorted by node index.
    ExplicitByIndex,
    /// Use the explicit neighborhood differ, sorted by Jaccard similarity.
    ExplicitByJaccard,
}

/// Number of secret-shared iris code pairs to persist to Postgres per transaction.
const SECRET_SHARING_PG_TX_SIZE: usize = 100;

pub struct DbContext {
    /// Postgres store to persist data against
    pub store: Store,
    graph_pg: GraphPg<PlaintextStore>,
    s3_client: S3Client,
    bucket: String,
    party_id: usize,
}

impl DbContext {
    pub async fn new(
        url: &str,
        schema: &str,
        s3_client: S3Client,
        bucket: String,
        party_id: usize,
    ) -> Self {
        let client = PostgresClient::new(url, schema, AccessMode::ReadWrite)
            .await
            .unwrap();
        let store = Store::new(&client).await.unwrap();
        let graph_pg = GraphPg::new(&client).await.unwrap();
        Self {
            store,
            graph_pg,
            s3_client,
            bucket,
            party_id,
        }
    }

    /// Convenience constructor for contexts that don't use S3 or checkpointing.
    /// A dummy (unconfigured) S3 client is created; calls to S3 will fail, but
    /// as long as no checkpoint operations are performed this is safe.
    pub async fn new_without_s3(url: &str, schema: &str, party_id: usize) -> Self {
        let s3_client = S3Client::from_conf(aws_sdk_s3::config::Builder::new().build());
        Self::new(url, schema, s3_client, String::new(), party_id).await
    }

    // todo: consider the persistent_state table (last indexed modification id) and the modifications table and the
    // iris table. inserting everything as 0 and NULL can cause problems.
    /// Upload a BothEyes graph to S3 and record it in genesis_graph_checkpoint.
    pub async fn store_checkpoint(
        &self,
        graph: &BothEyes<GraphMem<SerialId>>,
        last_indexed_iris_id: i64,
        last_indexed_modification_id: i64,
        graph_mutation_id: Option<i64>,
        is_archival: bool,
    ) -> Result<()> {
        let state = upload_graph_checkpoint_plaintext(
            &self.bucket,
            self.party_id,
            graph,
            &self.s3_client,
            u32::try_from(last_indexed_iris_id)?,
            last_indexed_modification_id,
            graph_mutation_id,
            is_archival,
        )
        .await?;

        let mut tx = self.graph_pg.tx().await?;
        GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
            &mut tx.tx,
            &state.s3_key,
            state.last_indexed_iris_id as i64,
            state.last_indexed_modification_id,
            state.graph_mutation_id,
            &state.blake3_hash,
            is_archival,
            state.graph_version,
        )
        .await?;
        tx.tx.commit().await?;

        Ok(())
    }

    /// Create a checkpoint from a graph in memory: get required metadata and upload to S3 and update databases.
    pub async fn make_checkpoint_from_graph(
        &self,
        graph: &BothEyes<GraphMem<SerialId>>,
    ) -> Result<()> {
        let graph_mutation_id = self.graph_pg.get_max_hawk_graph_mutation_id().await?;
        let last_indexed_iris_id: i64 = self
            .graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
            .await?
            .unwrap_or(0);
        let last_indexed_modification_id: i64 = self
            .graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_MODIFICATION_ID)
            .await?
            .unwrap_or(0);
        self.store_checkpoint(
            graph,
            last_indexed_iris_id,
            last_indexed_modification_id,
            graph_mutation_id,
            false,
        )
        .await
    }

    /// Extends iris shares table by inserting irises following the current
    /// maximum serial id.
    ///
    /// Returns tuple `(start, end)` giving the first and last serial ids
    /// assigned to the inserted shares.
    pub async fn persist_vector_shares(
        &self,
        shares: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>,
    ) -> Result<(usize, usize)> {
        let mut tx = self.store.tx().await?;

        let start_serial_id = self.store.get_max_serial_id().await.unwrap_or(0) + 1;
        let end_serial_id = start_serial_id + shares.len() - 1;

        for batch in &shares.iter().enumerate().chunks(SECRET_SHARING_PG_TX_SIZE) {
            let iris_refs: Vec<_> = batch
                .map(|(idx, (iris_l, iris_r))| StoredIrisRef {
                    id: (start_serial_id + idx) as i64,
                    left_code: &iris_l.code.coefs,
                    left_mask: &iris_l.mask.coefs,
                    right_code: &iris_r.code.coefs,
                    right_mask: &iris_r.mask.coefs,
                })
                .collect();

            self.store.insert_irises(&mut tx, &iris_refs).await?;
            tx.commit().await?;
            tx = self.store.tx().await?;
        }

        Ok((start_serial_id, end_serial_id))
    }

    /// Reconstruct the current graph state by:
    ///   1. Downloading the latest checkpoint from S3
    ///   2. Applying all hawk_graph_mutations written after that checkpoint
    pub async fn get_both_eyes(&self) -> Result<BothEyes<GraphMem<SerialId>>> {
        let checkpoint_row = self.graph_pg.get_latest_genesis_graph_checkpoint().await?;

        let mut graph: BothEyes<GraphMem<SerialId>> = match checkpoint_row {
            None => [GraphMem::new(), GraphMem::new()],
            Some(ref row) => {
                let state = GraphCheckpointState {
                    s3_key: row.s3_key.clone(),
                    last_indexed_iris_id: row.last_indexed_iris_id as u32,
                    last_indexed_modification_id: row.last_indexed_modification_id,
                    graph_mutation_id: row.graph_mutation_id,
                    blake3_hash: row.blake3_hash.clone(),
                    graph_version: row.graph_version,
                    is_archival: row.is_archival,
                };
                download_genesis_checkpoint_plaintext(&self.s3_client, &self.bucket, &state).await?
            }
        };

        // Replay mutations written after the checkpoint
        let min_mutation_id = checkpoint_row.and_then(|r| r.graph_mutation_id);
        let mutation_rows = self
            .graph_pg
            .get_hawk_graph_mutations_after(min_mutation_id)
            .await?;

        for row in mutation_rows {
            let both_eyes = row.deserialize_mutations()?;
            for m in &both_eyes[LEFT] {
                graph[LEFT].insert_apply(m)?;
            }
            for m in &both_eyes[RIGHT] {
                graph[RIGHT].insert_apply(m)?;
            }
        }

        Ok(graph)
    }

    /// Loads a graph from a file and uploads it to S3 as a checkpoint.
    pub async fn make_new_checkpoint(&self, path: &Path, dbg: bool) -> Result<()> {
        let graph = deserialize_graph(path).await?;
        if dbg {
            println!("loaded graph:");
            println!("{:#?}", graph);
        }
        let graph_mutation_id = self.graph_pg.get_max_hawk_graph_mutation_id().await?;
        let last_indexed_iris_id: i64 = self
            .graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID)
            .await?
            .unwrap_or(0);
        let last_indexed_modification_id: i64 = self
            .graph_pg
            .get_persistent_state(STATE_DOMAIN, STATE_KEY_LAST_INDEXED_MODIFICATION_ID)
            .await?
            .unwrap_or(0);
        self.store_checkpoint(
            &graph,
            last_indexed_iris_id,
            last_indexed_modification_id,
            graph_mutation_id,
            false,
        )
        .await
    }

    /// loads the graph from database to memory, writes it to a file,
    /// loads another graph from the file, and finally verifies that
    /// the loaded graph equals the stored graph.
    pub async fn verify_backup(&self, path: &Path, dbg: bool) -> Result<()> {
        let stored_graph = self.get_both_eyes().await?;
        if dbg {
            println!("storing graph:");
            println!("{:#?}", stored_graph);
        }
        let data = tokio::fs::read(path).await?;
        let loaded_graph: BothEyes<GraphMem<SerialId>> = bincode::deserialize(&data)?;
        if dbg {
            println!("loaded graph:");
            println!("{:#?}", loaded_graph);
        }
        if stored_graph != loaded_graph {
            return Err(eyre::eyre!("Loaded graph does not match stored graph"));
        }
        Ok(())
    }

    /// load a graph from the file and compare it against the database
    pub async fn compare_to_db(
        &self,
        path: &Path,
        diff_method: DiffMethod,
        dbg: bool,
    ) -> Result<()> {
        let db_graph = self.get_both_eyes().await?;
        if dbg {
            println!("graph from database:");
            println!("{:#?}", db_graph);
        }
        let loaded_graph = deserialize_graph(path).await?;
        if dbg {
            println!("graph from file:");
            println!("{:#?}", loaded_graph);
        }

        for side in 0..2 {
            if db_graph[side] == loaded_graph[side] {
                println!(
                    "Verdict: Side {} graphs are identical (including ordering of neighborhoods)",
                    side
                );
            } else {
                let node_equiv_result =
                    ensure_node_equivalence(&db_graph[side], &loaded_graph[side]);
                if let Err(err) = node_equiv_result {
                    eprintln!(
                        "Verdict: Side {} graphs are not node-equivalent;\n Reason: {:#?}",
                        side, err
                    );
                } else {
                    match diff_method {
                        DiffMethod::DetailedJaccard => {
                            let differ = DetailedJaccardDiffer::new(15);
                            let result = run_diff(&db_graph[side], &loaded_graph[side], differ);
                            println!("{result}");
                        }
                        DiffMethod::ExplicitByIndex => {
                            let differ = ExplicitNeighborhoodDiffer::new(SortBy::Index);
                            let result = run_diff(&db_graph[side], &loaded_graph[side], differ);
                            println!("{result}");
                        }
                        DiffMethod::ExplicitByJaccard => {
                            let differ = ExplicitNeighborhoodDiffer::new(SortBy::Jaccard);
                            let result = run_diff(&db_graph[side], &loaded_graph[side], differ);
                            println!("{result}");
                        }
                    };
                }
            }
        }
        Ok(())
    }

    /// Populates S3 with a small test graph and returns it. This is needed because
    /// the test database is initially empty and some data is needed to
    /// test the backup and restore commands.
    /// The graph isn't actually random - the RNG uses the same seed.
    pub async fn store_random_graph(&self) -> Result<BothEyes<GraphMem<SerialId>>> {
        const NUM_RANDOM_IRIS_CODES: usize = 10;
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let mut vector_store = PlaintextStore::<FhdOps>::new();

        let vectors = {
            let mut v = vec![];
            for raw_query in IrisDB::new_random_rng(NUM_RANDOM_IRIS_CODES, rng).db {
                let q = Arc::new(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        // get the distance from point[0] to every other point
        let distances = {
            let mut d = vec![];
            let q = vector_store
                .storage
                .get_vector_by_serial_id(1)
                .unwrap()
                .clone();
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&q, v).await?);
            }
            d
        };

        // Build the graph topology as mutations, then apply to an in-memory graph
        let mut left_graph = GraphMem::<SerialId>::new();

        // Set entry point via InsertNode with SetUnique
        let ep_mutation = MutationOp::AddNode {
            id: vectors[0].serial_id(),
            height: 1,
            update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
        };
        left_graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![ep_mutation],
            })
            .unwrap();

        // Build links between vectors 1-3 and 4-6
        for i in 1..4usize {
            let mut links = SortedNeighborhood::new();
            for j in 4..7usize {
                links
                    .insert_and_trim(&mut vector_store, vectors[j], distances[j], links.len() + 1)
                    .await?;
            }
            let neighbors: Vec<SerialId> = links.edge_ids().iter().map(|v| v.serial_id()).collect();
            let mutations = vec![
                MutationOp::AddNode {
                    id: vectors[i].serial_id(),
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddEdges {
                    base: vectors[i].serial_id(),
                    layer: 0,
                    neighbors,
                    edge_type: EdgeType::Base,
                },
            ];
            left_graph
                .insert_apply(&GraphMutation {
                    seq_no: (i as u64) + 1,
                    ops: mutations,
                })
                .unwrap();
        }

        // Use same graph for both eyes (this is test data)
        let right_graph = left_graph.clone();
        let graph = [left_graph, right_graph];

        self.make_checkpoint_from_graph(&graph).await?;
        Ok(graph)
    }
}

async fn deserialize_graph(path: &Path) -> Result<BothEyes<GraphMem<SerialId>>> {
    let data = tokio::fs::read(path).await?;
    Ok(bincode::deserialize(&data)?)
}
