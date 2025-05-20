//! this file is used by test utilities

use std::sync::Arc;

use super::{graph_store::GraphPg, layered_graph::EntryPoint};
use crate::hnsw::{
    vector_store::{VectorStore, VectorStoreMut},
    SortedNeighborhood,
};
use crate::{
    execution::hawk_main::StoreId, hawkers::plaintext_store::PlaintextStore, hnsw::GraphMem,
    protocol::shared_iris::GaloisRingSharedIris,
};
use aes_prng::AesRng;
use bincode;
use eyre::Result;
use iris_mpc_common::iris_db::db::IrisDB;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::Itertools;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Number of secret-shared iris code pairs to persist to Postgres per transaction.
const SECRET_SHARING_PG_TX_SIZE: usize = 100;

pub struct DbContext {
    /// Postgres store to persist data against
    pub store: Store,
    graph_pg: GraphPg<PlaintextStore>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct BothSides {
    left: GraphMem<PlaintextStore>,
    right: GraphMem<PlaintextStore>,
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
        graph: GraphMem<PlaintextStore>,
        side: StoreId,
    ) -> Result<()> {
        let mut graph_tx = self.graph_pg.tx().await?;

        let GraphMem {
            entry_point,
            layers,
        } = graph;

        if let Some(EntryPoint { point, layer }) = entry_point {
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

    pub async fn load_graph_to_mem(
        &self,
        side: StoreId,
    ) -> Result<GraphMem<PlaintextStore>, eyre::Report> {
        let mut graph_tx = self.graph_pg.tx().await?;
        let mut graph_ops = graph_tx.with_graph(side);

        graph_ops.load_to_mem().await
    }

    // loads the graph from the database to memory and then writes it to a file
    pub async fn write_graph_to_file(&self, path: &std::path::Path, dbg: bool) -> Result<()> {
        let left = self.load_graph_to_mem(StoreId::Left).await?;
        let right = self.load_graph_to_mem(StoreId::Right).await?;
        let graph = BothSides { left, right };
        if dbg {
            println!("storing graph:");
            println!("{:#?}", graph);
        }
        let serialized = bincode::serialize(&graph)?;
        let mut file = File::create(path).await?;
        file.write_all(&serialized).await?;
        file.flush().await?;
        Ok(())
    }

    // loads a graph to memory from a file and then persists it to the database
    pub async fn load_graph_from_file(&self, path: &std::path::Path, dbg: bool) -> Result<()> {
        let data = tokio::fs::read(path).await?;
        let graph: BothSides = bincode::deserialize(&data)?;
        if dbg {
            println!("loaded graph:");
            println!("{:#?}", graph);
        }
        self.persist_graph_db(graph.left, StoreId::Left).await?;
        self.persist_graph_db(graph.right, StoreId::Right).await?;
        Ok(())
    }

    pub async fn test_load_store(&self, path: &std::path::Path, dbg: bool) -> Result<()> {
        let left = self.load_graph_to_mem(StoreId::Left).await?;
        let right = self.load_graph_to_mem(StoreId::Right).await?;
        let stored_graph = BothSides { left, right };
        if dbg {
            println!("storing graph:");
            println!("{:#?}", stored_graph);
        }
        let serialized = bincode::serialize(&stored_graph)?;
        let mut file = File::create(path).await?;
        file.write_all(&serialized).await?;
        file.flush().await?;

        let data = tokio::fs::read(path).await?;
        let loaded_graph: BothSides = bincode::deserialize(&data)?;
        if dbg {
            println!("loaded graph:");
            println!("{:#?}", loaded_graph);
        }
        if stored_graph != loaded_graph {
            return Err(eyre::eyre!("Loaded graph does not match stored graph"));
        }
        Ok(())
    }

    // this function was stolen from other test code.
    pub async fn gen_random(&self) -> Result<()> {
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let mut vector_store = PlaintextStore::new();

        let vectors = {
            let mut v = vec![];
            for raw_query in IrisDB::new_random_rng(10, rng).db {
                let q = Arc::new(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        let distances = {
            let mut d = vec![];
            let q = vector_store.points[0].clone();
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&q, v).await?);
            }
            d
        };

        let mut tx = self.graph_pg.tx().await?;
        let mut graph_ops = tx.with_graph(StoreId::Left);

        let ep = graph_ops.get_entry_point().await?;
        assert!(ep.is_none());

        let ep2 = EntryPoint {
            point: vectors[0],
            layer: ep.map(|e| e.1).unwrap_or_default() + 1,
        };

        graph_ops.set_entry_point(ep2.point, ep2.layer).await?;

        let (point3, layer3) = graph_ops.get_entry_point().await?.unwrap();
        let ep3 = EntryPoint {
            point: point3,
            layer: layer3,
        };

        assert_eq!(ep2, ep3);

        for i in 1..4 {
            let mut links = SortedNeighborhood::new();

            for j in 4..7 {
                links
                    .insert(&mut vector_store, vectors[j], distances[j])
                    .await?;
            }
            let links = links.edge_ids();

            graph_ops.set_links(vectors[i], links.clone(), 0).await?;

            let links2 = graph_ops.get_links(&vectors[i], 0).await?;
            assert_eq!(*links, *links2);
        }

        tx.tx.commit().await?;
        Ok(())
    }
}
