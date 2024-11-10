use crate::hawkers::plaintext_store::PlaintextStore;
use aes_prng::AesRng;
use bincode;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use rand::{rngs::ThreadRng, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

pub fn gen_uniform_iris_code_array() -> IrisCodeArray {
    let mut rng = ThreadRng::default();
    IrisCodeArray::random_rng(&mut rng)
}

pub fn gen_empty_index() -> (PlaintextStore, GraphMem<PlaintextStore>) {
    let vector = PlaintextStore::default();
    let graph = GraphMem::new();

    (vector, graph)
}

pub fn write_serde_bin<T: Serialize>(data: &T, filename: &str) -> bincode::Result<()> {
    let file = File::create(filename).map_err(bincode::ErrorKind::Io)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

pub fn read_serde_bin<T: DeserializeOwned>(filename: &str) -> bincode::Result<T> {
    let file = File::open(filename).map_err(bincode::ErrorKind::Io)?;
    let reader = BufReader::new(file);
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn gen_uniform_random_index(size: usize) -> (PlaintextStore, GraphMem<PlaintextStore>) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let (mut vector, mut graph) = gen_empty_index();
        let searcher = HawkSearcher::default();

        for idx in 0..size {
            let raw_query = IrisCode::random_rng(&mut rng);
            let query = vector.prepare_query(raw_query.clone());
            let neighbors = searcher
                .search_to_insert(&mut vector, &mut graph, &query)
                .await;
            let inserted = vector.insert(&query).await;
            searcher
                .insert_from_search_results(&mut vector, &mut graph, &mut rng, inserted, neighbors)
                .await;
            if idx % 10 == 9 {
                println!("{}", idx + 1);
            }
        }
        (vector, graph)
    })
}

pub fn insert_iris(
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
    iris: IrisCode,
) -> u32 {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();
        let searcher = HawkSearcher::default();

        // let raw_query = IrisCode::random_rng(&mut rng);
        let query = vector.prepare_query(iris);
        let neighbors = searcher.search_to_insert(vector, graph, &query).await;
        let inserted = vector.insert(&query).await;
        searcher
            .insert_from_search_results(vector, graph, &mut rng, inserted, neighbors)
            .await;
        inserted.0
    })
}

pub fn search_iris(
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
    query: IrisCode,
) -> (u32, f64) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let searcher = HawkSearcher::default();
        let query = vector.prepare_query(query);
        let neighbors = searcher.search_to_insert(vector, graph, &query).await;
        let nearest = neighbors[0].get_nearest().unwrap();
        let dist = (nearest.1 .0 as f64) / (nearest.1 .1 as f64);
        (nearest.0 .0, dist)
    })
}

pub fn insert_random(vector: &mut PlaintextStore, graph: &mut GraphMem<PlaintextStore>) -> u32 {
    let mut rng = ThreadRng::default();
    let raw_query = IrisCode::random_rng(&mut rng);

    insert_iris(vector, graph, raw_query)
}
