use crate::hawkers::plaintext_store::PlaintextStore;
use aes_prng::AesRng;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use rand::{rngs::ThreadRng, SeedableRng};

// pub fn gen_zero_iris_code_array_str() -> String {
//     IrisCodeArray::ZERO.to_base64().unwrap()
// }

// pub fn gen_one_iris_code_array_str() -> String {
//     IrisCodeArray::ONES.to_base64().unwrap()
// }

pub fn gen_uniform_iris_code_array() -> IrisCodeArray {
    let mut rng = ThreadRng::default();
    IrisCodeArray::random_rng(&mut rng)
}

pub fn gen_empty_index() -> (PlaintextStore, GraphMem<PlaintextStore>) {
    let vector = PlaintextStore::default();
    let graph = GraphMem::new();

    (vector, graph)
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
