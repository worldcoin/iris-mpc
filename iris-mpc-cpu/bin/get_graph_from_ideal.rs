use std::sync::Arc;
use std::{io::BufReader, path::PathBuf};

use iris_mpc_common::{IrisSerialId, IrisVectorId};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::vector_store::VectorStoreMut;
use iris_mpc_cpu::hnsw::HnswSearcher;
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use iris_mpc_cpu::{hawkers::ideal_knn_engines::EngineChoice, hnsw::GraphMem};
use rand::seq::SliceRandom;
use serde_json::Deserializer;

#[tokio::main]
async fn main() {
    let k = 320;
    let echoice = EngineChoice::NaiveFHD;
    let num_threads = 3;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/store.ndjson");
    let file = std::fs::File::open(path).unwrap();
    let reader = BufReader::new(file);

    let n = 6000;
    let layer1_len = 1000;
    let layer2_len = 500;
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .take(n);
    let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();

    assert!(n == irises.len());
    // First layer: 1000 random serial ids from 1 to n
    let mut rng = rand::thread_rng();
    let mut layer1: Vec<IrisVectorId> = (1..=(n as u32))
        .map(|serial_id| IrisVectorId::from_serial_id(serial_id))
        .collect();
    layer1.shuffle(&mut rng);
    let layer1 = layer1.into_iter().take(layer1_len).collect::<Vec<_>>();

    // Second layer: 100 random samples from the first layer
    let mut layer2 = layer1.clone();
    layer2.shuffle(&mut rng);
    let layer2 = layer2.into_iter().take(layer2_len).collect::<Vec<_>>();
    let entry = layer2[0];

    let nodes_for_nonzero_layers = vec![layer1, layer2];
    let entry_point = Some((entry, 2));
    let filepath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.txt");

    let graph = GraphMem::ideal_from_irises(
        &irises,
        entry_point,
        nodes_for_nonzero_layers,
        filepath,
        k,
        echoice,
        num_threads,
    );

    dbg!(graph.layers[0].links.len());
    assert!(graph.layers[0].links.len() == n);
    assert!(graph.layers[1].links.len() == layer1_len);
    assert!(graph.layers[2].links.len() == layer2_len);

    for layer in graph.layers.iter() {
        for value in layer.links.values() {
            assert_eq!(value.len(), k);
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

    for iris_id in 1..=irises.len() {
        let iris_id = IrisVectorId::from_serial_id(iris_id as IrisSerialId);
        let iris = store.storage.get_vector(&iris_id).unwrap();
        let iris_match = iris.get_similar_iris(&mut rng, 0.15);
        let query = Arc::new(iris_match);
        let searcher = HnswSearcher::new(320, 64, 256);

        let result = searcher
            .search(&mut store, &graph, &query, 1)
            .await
            .unwrap();

        assert!(result.match_count(&mut store).await.unwrap() >= 1);
        if iris_id.serial_id() % 100 == 0 {
            dbg!(&iris_id);
        }
    }
}
