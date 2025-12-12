use std::sync::Arc;

use aes_prng::AesRng;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{GraphMem, HnswSearcher},
};
use rand::SeedableRng;

const DATABASE_SIZE: usize = 1_000;

fn main() -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let (_vector, _graph) = rt.block_on(async move {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut vector = PlaintextStore::new();
        let mut graph = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        for idx in 0..DATABASE_SIZE {
            let raw_query = IrisCode::random_rng(&mut rng);
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            searcher
                .insert(&mut vector, &mut graph, &query, insertion_layer)
                .await?;
            if idx % 100 == 99 {
                println!("{}", idx + 1);
            }
        }
        Ok::<_, eyre::Report>((vector, graph))
    })?;

    Ok(())
}
