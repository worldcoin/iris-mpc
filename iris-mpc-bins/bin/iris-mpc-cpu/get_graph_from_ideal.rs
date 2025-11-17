use std::sync::Arc;
use std::{io::BufReader, path::PathBuf};

use iris_mpc_common::{IrisSerialId, IrisVectorId};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::vector_store::VectorStoreMut;
use iris_mpc_cpu::hnsw::HnswSearcher;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;
use iris_mpc_cpu::{hawkers::ideal_knn_engines::EngineChoice, hnsw::GraphMem};
use rand::seq::IteratorRandom;
use serde_json::Deserializer;

#[tokio::main]
async fn main() {
    let k = 320;
    let echoice = EngineChoice::NaiveFHD;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/store.ndjson");
    let file = std::fs::File::open(path).unwrap();
    let reader = BufReader::new(file);

    let n = 1000;
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .take(n);
    let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();

    assert!(n == irises.len());
    let filepath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.txt");

    let prf_seed = [0u8; 16];

    let mut searcher = HnswSearcher::new(320, 64, 160);
    searcher.params.layer_probability = 0.20;

    let graph = GraphMem::ideal_from_irises(irises.clone(), filepath, &searcher, prf_seed, echoice)
        .unwrap();

    assert!(graph.layers[0].links.len() == n);
    dbg!(graph.layers[1].links.len());
    dbg!(graph.layers[2].links.len());

    for layer in graph.layers.iter() {
        for (_, value) in layer.links.iter() {
            assert_eq!(value.len(), k.min(layer.links.len() - 1));
        }
    }

    let mut store = PlaintextStore::new();
    let irises = irises
        .into_iter()
        .map(|iris| Arc::new(iris))
        .collect::<Vec<_>>();

    for iris in irises.iter() {
        store.insert(&iris).await;
    }

    let mut rng = rand::thread_rng();

    for iris_id in (1..=irises.len()).choose_multiple(&mut rng, 50) {
        let iris_id = IrisVectorId::from_serial_id(iris_id as IrisSerialId);
        let iris = store.storage.get_vector(&iris_id).unwrap();
        let iris_match = iris.get_similar_iris(&mut rng, 0.15);
        let query = Arc::new(iris_match);

        let result = searcher
            .search(&mut store, &graph, &query, 1)
            .await
            .unwrap();

        assert!(result.match_count(&mut store).await.unwrap() >= 1);
    }
}
