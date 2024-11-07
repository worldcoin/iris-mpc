use aes_prng::AesRng;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use rand::SeedableRng;

const DATABASE_SIZE: usize = 1_000;

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let (_vector, _graph) = rt.block_on(async move {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut vector = PlaintextStore::default();
        let mut graph = GraphMem::new();
        let searcher = HawkSearcher::default();

        for idx in 0..DATABASE_SIZE {
            let raw_query = IrisCode::random_rng(&mut rng);
            let query = vector.prepare_query(raw_query.clone());
            let neighbors = searcher
                .search_to_insert(&mut vector, &mut graph, &query)
                .await;
            let inserted = vector.insert(&query).await;
            searcher
                .insert_from_search_results(&mut vector, &mut graph, &mut rng, inserted, neighbors)
                .await;
            if idx % 100 == 99 {
                println!("{}", idx + 1);
            }
        }
        (vector, graph)
    });
}
