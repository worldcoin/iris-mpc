use crate::hawkers::plaintext_store::{PlaintextStore, PointId};
use aes_prng::AesRng;
use bincode;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use rand::{rngs::ThreadRng, SeedableRng};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Serialize, Deserialize)]
pub struct PlaintextHnsw {
    pub searcher: HawkSearcher,
    pub vector:   PlaintextStore,
    pub graph:    GraphMem<PlaintextStore>,
}

impl Default for PlaintextHnsw {
    fn default() -> Self {
        Self {
            searcher: HawkSearcher::default(),
            vector:   PlaintextStore::default(),
            graph:    GraphMem::new(),
        }
    }
}

impl PlaintextHnsw {
    pub fn gen_uniform_random(size: usize) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            let mut hnsw = Self::default();

            for idx in 0..size {
                let raw_query = IrisCode::random_rng(&mut rng);
                let query = hnsw.vector.prepare_query(raw_query.clone());
                let neighbors = hnsw
                    .searcher
                    .search_to_insert(&mut hnsw.vector, &mut hnsw.graph, &query)
                    .await;
                let inserted = hnsw.vector.insert(&query).await;
                hnsw.searcher
                    .insert_from_search_results(
                        &mut hnsw.vector,
                        &mut hnsw.graph,
                        &mut rng,
                        inserted,
                        neighbors,
                    )
                    .await;
                if idx % 10 == 9 {
                    println!("{}", idx + 1);
                }
            }
            hnsw
        })
    }

    pub fn insert(&mut self, iris: IrisCode) -> PointId {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let mut rng = ThreadRng::default();

            let query = self.vector.prepare_query(iris);
            let neighbors = self
                .searcher
                .search_to_insert(&mut self.vector, &mut self.graph, &query)
                .await;
            let inserted = self.vector.insert(&query).await;
            self.searcher
                .insert_from_search_results(
                    &mut self.vector,
                    &mut self.graph,
                    &mut rng,
                    inserted,
                    neighbors,
                )
                .await;
            inserted
        })
    }

    pub fn search(&mut self, query: IrisCode) -> (PointId, f64) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let query = self.vector.prepare_query(query);
            let neighbors = self
                .searcher
                .search_to_insert(&mut self.vector, &mut self.graph, &query)
                .await;
            let (nearest, (dist_num, dist_denom)) = neighbors[0].get_nearest().unwrap();
            (*nearest, (*dist_num as f64) / (*dist_denom as f64))
        })
    }

    pub fn insert_uniform_random(&mut self) -> PointId {
        let mut rng = ThreadRng::default();
        let raw_query = IrisCode::random_rng(&mut rng);

        self.insert(raw_query)
    }

    pub fn write_to_file(&self, filename: &str) -> bincode::Result<()> {
        write_serde_bin(self, filename)
    }

    pub fn read_from_file(filename: &str) -> bincode::Result<Self> {
        read_serde_bin(filename)
    }
}

pub fn gen_uniform_iris_code_array() -> IrisCodeArray {
    let mut rng = ThreadRng::default();
    IrisCodeArray::random_rng(&mut rng)
}

fn write_serde_bin<T: Serialize>(data: &T, filename: &str) -> bincode::Result<()> {
    let file = File::create(filename).map_err(bincode::ErrorKind::Io)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

fn read_serde_bin<T: DeserializeOwned>(filename: &str) -> bincode::Result<T> {
    let file = File::open(filename).map_err(bincode::ErrorKind::Io)?;
    let reader = BufReader::new(file);
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}
