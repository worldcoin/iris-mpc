use std::{io::BufReader, path::PathBuf};

use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use iris_mpc_cpu::{hawkers::ideal_knn_engines::EngineChoice, hnsw::GraphMem};
use rand::seq::SliceRandom;
use serde_json::Deserializer;

fn main() {
    let k = 320;
    let echoice = EngineChoice::NaiveFHD;
    let num_threads = 3;
    dbg!(&env!("CARGO_MANIFEST_DIR"));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/store.ndjson");
    dbg!(&path);
    let file = std::fs::File::open(path).unwrap();
    let reader = BufReader::new(file);

    let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();
    let n = irises.len();
    // First layer: 1000 random serial ids from 1 to n
    let mut rng = rand::thread_rng();
    let mut first_layer: Vec<IrisSerialId> = (1..=(n as u32)).collect();
    first_layer.shuffle(&mut rng);
    let first_layer = first_layer.into_iter().take(1000).collect::<Vec<_>>();

    // Second layer: 100 random samples from the first layer
    let mut second_layer = first_layer.clone();
    second_layer.shuffle(&mut rng);
    let second_layer = second_layer.into_iter().take(500).collect::<Vec<_>>();
    let entry = second_layer[0];

    let nodes_for_nonzero_layers = vec![first_layer, second_layer];
    let entry_point = Some((entry, 2));
    let filepath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.txt");

    let graph = GraphMem::ideal_from_irises(
        irises,
        entry_point,
        nodes_for_nonzero_layers,
        filepath,
        k,
        echoice,
        num_threads,
    );

    assert!(graph.layers[0].links.keys().count() == n);
}
