//! this file is used by test utilities

use std::sync::Arc;

use super::{graph_store::GraphPg, layered_graph::EntryPoint};
use crate::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextVectorRef,
    hnsw::{
        graph::{
            graph_diff::{
                explicit::{ExplicitNeighborhoodDiffer, SortBy},
                jaccard::DetailedJaccardDiffer,
                node_equiv::ensure_node_equivalence,
                run_diff,
            },
            neighborhood::Neighborhood,
        },
        vector_store::{VectorStore, VectorStoreMut},
        SortedNeighborhood,
    },
};
use crate::{
    execution::hawk_main::StoreId, hawkers::plaintext_store::PlaintextStore, hnsw::GraphMem,
    protocol::shared_iris::GaloisRingSharedIris,
};
use aes_prng::AesRng;
use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::iris_db::db::IrisDB;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::Itertools;
use rand::SeedableRng;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

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
}

impl DbContext {
    pub async fn new(url: &str, schema: &str) -> Self {
        let client = PostgresClient::new(url, schema, AccessMode::ReadWrite)
            .await
            .unwrap();
        let store = Store::new(&client).await.unwrap();
        let graph_pg = GraphPg::new(&client).await.unwrap();
        Self { store, graph_pg }
    }

    pub async fn persist_graph_db(
        &self,
        graph: GraphMem<PlaintextVectorRef>,
        side: StoreId,
    ) -> Result<()> {
        let mut graph_tx = self.graph_pg.tx().await?;

        let GraphMem {
            entry_points: entry_point,
            layers,
        } = graph;

        if let Some(EntryPoint { point, layer }) = entry_point.first().cloned() {
            let mut graph_ops = graph_tx.with_graph(side);
            graph_ops.set_entry_point(point, layer).await?;
        }

        for (lc, layer) in layers.into_iter().enumerate() {
            for (idx, (pt, links)) in layer.links.into_iter().enumerate() {
                {
                    let mut graph_ops = graph_tx.with_graph(side);
                    graph_ops.set_links(pt, links, lc).await?;
                }

                if (idx % 1000) == 999 {
                    graph_tx.tx.commit().await?;
                    graph_tx = self.graph_pg.tx().await?;
                }
            }
        }

        graph_tx.tx.commit().await?;

        Ok(())
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

    /// loads a graph from database to memory
    pub async fn load_graph_to_mem(
        &self,
        side: StoreId,
    ) -> Result<GraphMem<PlaintextVectorRef>, eyre::Report> {
        let mut graph_tx = self.graph_pg.tx().await?;
        let mut graph_ops = graph_tx.with_graph(side);

        graph_ops.load_to_mem(self.graph_pg.pool(), 2).await
    }

    /// helper function to get a `BothEyes<GraphMem>`
    pub async fn get_both_eyes(&self) -> Result<BothEyes<GraphMem<PlaintextVectorRef>>> {
        Ok([
            self.load_graph_to_mem(StoreId::Left).await?,
            self.load_graph_to_mem(StoreId::Right).await?,
        ])
    }

    /// loads a graph from database to memory and then writes it to a file
    pub async fn write_graph_to_file(&self, path: &Path, dbg: bool) -> Result<()> {
        let graph = self.get_both_eyes().await?;
        if dbg {
            println!("storing graph:");
            println!("{:#?}", graph);
        }
        serialize_graph(path, &graph).await
    }

    /// loads a graph to memory from a file and then persists it to the database
    pub async fn load_graph_from_file(&self, path: &Path, dbg: bool) -> Result<()> {
        let loaded_graph = deserialize_graph(path).await?;
        if dbg {
            println!("loaded graph:");
            println!("{:#?}", loaded_graph);
        }
        // this order corresponds to the LEFT and RIGHT constants in hawk main.
        // this code assumes that the stored graph could be very large and that a
        // clone could increase the program's runtime unnecessarily.
        let [left, right] = loaded_graph;
        self.persist_graph_db(left, StoreId::Left).await?;
        self.persist_graph_db(right, StoreId::Right).await?;
        Ok(())
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
        serialize_graph(path, &stored_graph).await?;

        let data = tokio::fs::read(path).await?;
        let loaded_graph: BothEyes<GraphMem<PlaintextVectorRef>> = bincode::deserialize(&data)?;
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

    /// populates the database with a small graph. This is needed because
    /// the test database is initially empty and some data is needed to
    /// test the backup and restore commands.
    /// The graph isn't actually random - the RNG uses the same seed.
    pub async fn store_random_graph(&self) -> Result<()> {
        const NUM_RANDOM_IRIS_CODES: usize = 10;
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let mut vector_store = PlaintextStore::new();

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

        let mut tx = self.graph_pg.tx().await?;
        let mut graph_ops = tx.with_graph(StoreId::Left);

        let ep = graph_ops.get_entry_point().await?;
        assert!(ep.is_none());

        // set point[0] as the entry point
        let ep2 = EntryPoint {
            point: vectors[0],
            layer: ep.map(|e| e.1).unwrap_or_default() + 1,
        };
        graph_ops.set_entry_point(ep2.point, ep2.layer).await?;

        // create edges between vectors 1-3 and 4-6
        // imagine vectors 1-3 as layer 1 and vectors 4-6 as layer 2
        // then layers 1 and 2 are fully connected.
        for i in 1..4 {
            let mut links = SortedNeighborhood::new();

            for j in 4..7 {
                links
                    .insert_and_trim(&mut vector_store, vectors[j], distances[j], None)
                    .await?;
            }
            let links = links.edge_ids();
            graph_ops.set_links(vectors[i], links.clone(), 0).await?;
        }

        tx.tx.commit().await?;
        Ok(())
    }
}

async fn serialize_graph(
    path: &Path,
    value: &BothEyes<GraphMem<PlaintextVectorRef>>,
) -> Result<()> {
    let serialized = bincode::serialize(value)?;
    let mut file = File::create(path).await?;
    file.write_all(&serialized).await?;
    file.flush().await?;
    Ok(())
}

async fn deserialize_graph(path: &Path) -> Result<BothEyes<GraphMem<PlaintextVectorRef>>> {
    let data = tokio::fs::read(path).await?;
    Ok(bincode::deserialize(&data)?)
}
