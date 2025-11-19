use eyre::Result;
use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::hawkers::aby3::aby3_store::DistanceFn;
use iris_mpc_cpu::hawkers::plaintext_store::{fraction_ordering, PlaintextStore};
use iris_mpc_cpu::hnsw::searcher::TopLayerSearchMode;
use iris_mpc_cpu::hnsw::{HnswParams, HnswSearcher, VectorStore};
use iris_mpc_cpu::utils::serialization::graph::write_graph_current;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;
use iris_mpc_cpu::{hawkers::ideal_knn_engines::EngineChoice, hnsw::GraphMem};
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::Deserializer;
use std::cmp::Ordering;
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::{io::BufReader, path::PathBuf};

#[derive(Debug, Deserialize)]
struct IdealGraphConfig {
    graph_size: usize,
    irises_path: PathBuf,
    layer0_path: PathBuf,
    searcher: HnswParams,
    prf_seed: [u8; 16],
    echoice: EngineChoice,
    sanity_check: bool,
}

pub fn load_toml<'a, T, P>(path: P) -> Result<T>
where
    T: Deserialize<'a>,
    P: AsRef<Path>,
{
    let text = std::fs::read_to_string(path)?;
    let de = toml::de::Deserializer::new(&text);
    let t = serde_path_to_error::deserialize(de)?;
    Ok(t)
}

/// This binary constructs an idealized HNSW GraphMem
/// i.e, one where all neighborhoods are exact, instead of approximated.
/// It depends on an iris file and a pre-computed file which stores the neighborhoods
/// for layer 0.
/// The resulting graph is optionally tested by ensuring the correctness
/// of a randomly selected node's neighborhoods.
/// It is then serialized in binary format to a specified file.
///
/// For more details on the logic of the construction, consult
/// `GraphMem::ideal_from_irises`

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3);

    let config_path = &args[1];
    let output_path = &args[2];

    let config: IdealGraphConfig = load_toml(config_path).unwrap();
    let file = std::fs::File::open(config.irises_path).unwrap();
    let reader = BufReader::new(file);

    let n = config.graph_size;
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .take(n);
    let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();

    let searcher = HnswSearcher {
        params: config.searcher,
    };

    let graph = GraphMem::ideal_from_irises(
        irises.clone(),
        config.layer0_path.clone(),
        &searcher,
        config.prf_seed.clone(),
        config.echoice.clone(),
    )
    .unwrap();

    if config.sanity_check {
        // Check number of neighbors for all nodes
        for (lc, layer) in graph.layers.iter().enumerate() {
            let expected_nb_size = searcher.params.get_M_max(lc).min(layer.links.len() - 1);
            for (_, value) in layer.links.iter() {
                assert_eq!(value.len(), expected_nb_size);
            }
        }

        assert!(!graph.entry_point.is_empty());

        let last_graph_layer = graph.layers.last().unwrap();

        if let TopLayerSearchMode::LinearScan(layer_cap) = searcher.params.top_layer_mode {
            assert!(graph.layers.len() <= layer_cap);

            // All entry points should exist in the top graph layer as well
            for entry in &graph.entry_point {
                assert!(
                    last_graph_layer.links.contains_key(&entry.point),
                    "Entry point {:?} not found in last graph layer",
                    entry
                );
            }
        }

        let mut entropy_rng = rand::rngs::StdRng::from_entropy();

        // Sample one iris from the highest graph layer and assert that its neighborhood
        // is correct in every layer.

        let sample = last_graph_layer
            .links
            .keys()
            .choose(&mut entropy_rng)
            .cloned()
            .expect("last layer should have at least one key");

        let mut store = PlaintextStore::new();
        store.distance_fn = match config.echoice {
            EngineChoice::NaiveFHD => DistanceFn::Simple,
            EngineChoice::NaiveMinFHD => DistanceFn::MinimalRotation,
        };

        for (i, iris) in irises.into_iter().enumerate() {
            store.insert_with_id(IrisVectorId::from_serial_id((i as u32) + 1), Arc::new(iris));
        }

        let sample_iris = store.storage.get_vector(&sample).cloned().unwrap();

        for lc in 0..graph.layers.len() {
            let neighbors = graph.layers[lc]
                .get_links(&sample)
                .expect(&format!("{}", lc));

            let mut dists = Vec::new();
            for k in graph.layers[lc].links.keys() {
                if *k != sample {
                    let dist = store.eval_distance(&sample_iris, k).await.unwrap();
                    dists.push((k.clone(), dist));
                }
            }

            dists.sort_by(|a, b| fraction_ordering(&a.1, &b.1));
            dists.truncate(neighbors.len());
            let kth_dist = dists.last().unwrap().1;

            let count_greater = neighbors
                .iter()
                .filter(|n| {
                    let d = sample_iris.get_distance_fraction(store.storage.get_vector(n).unwrap());
                    matches!(fraction_ordering(&d, &kth_dist), Ordering::Greater)
                })
                .count();

            assert!(count_greater == 0);
        }
    }

    let mut writer = std::fs::File::create(output_path).unwrap();
    write_graph_current(&mut writer, graph).unwrap();
}
