use clap::Parser;
use eyre::Result;
use iris_mpc_common::VectorId;
use iris_mpc_cpu::hawkers::aby3::aby3_store::{DistanceOps, FhdOps, NhdOps};
use iris_mpc_cpu::hawkers::ideal_knn_engines::{EngineChoice, EngineChoiceInt4};
use iris_mpc_cpu::hawkers::plaintext_deep_id_store::{Int4Vector, PlaintextDeepIDStore};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::GraphMem;
use iris_mpc_cpu::hnsw::{HnswSearcher, VectorStore};
use iris_mpc_cpu::utils::serialization::check_store_kind_unambiguous;
use iris_mpc_cpu::utils::serialization::graph::write_graph_current;
use iris_mpc_cpu::utils::serialization::int4_ndjson::int4_vectors_from_ndjson;
use iris_mpc_cpu::utils::serialization::types::iris_base64::Base64IrisCode;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::Deserializer;
use std::cmp::Ordering;
use std::path::Path;
use std::sync::Arc;
use std::{io::BufReader, path::PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the configuration TOML file.
    #[clap(long)]
    job_spec: PathBuf,

    /// Path where the constructed graph will be written.
    #[clap(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct IdealGraphConfig {
    graph_size: usize,
    layer0_path: PathBuf,
    searcher: HnswSearcher,
    prf_seed: [u8; 16],
    sanity_check: bool,
    #[serde(flatten)]
    store: StoreKindConfig,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StoreKindConfig {
    Iris {
        irises_path: PathBuf,
        echoice: EngineChoice,
    },
    DeepID {
        vectors_path: PathBuf,
        echoice: EngineChoiceInt4,
        threshold: i32,
    },
}

pub fn load_toml<'a, T, P>(path: P) -> Result<T>
where
    T: Deserialize<'a>,
    P: AsRef<Path>,
{
    let text = std::fs::read_to_string(path)?;
    check_store_kind_unambiguous(&text)?;
    let de = toml::de::Deserializer::new(&text);
    let t = serde_path_to_error::deserialize(de)?;
    Ok(t)
}

async fn run_sanity_check_iris<D: DistanceOps>(
    graph: &GraphMem,
    searcher: &HnswSearcher,
    irises: Vec<iris_mpc_common::iris_db::iris::IrisCode>,
    echoice: EngineChoice,
) {
    // Check number of neighbors for all nodes
    for (lc, layer) in graph.layers.iter().enumerate() {
        let expected_nb_size = searcher
            .params
            .get_M_max(lc)
            .min(layer.links.len().saturating_sub(1));
        for (_, value) in layer.links.iter() {
            assert_eq!(value.degree(), expected_nb_size);
        }
    }

    // Check layers are valid for the max graph layer
    assert!(graph.num_layers() <= searcher.max_graph_layer + 1);

    // All entry points should exist in the top graph layer as well
    let last_graph_layer = graph.layers.last().unwrap();
    for entry in &graph.entry_points {
        assert!(
            last_graph_layer.links.contains_key(&entry.point),
            "Entry point {:?} not found in last graph layer",
            entry
        );
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

    let mut store = PlaintextStore::<D>::new();
    store.distance_mode = echoice.distance_mode();

    for (i, iris) in irises.into_iter().enumerate() {
        store.insert_with_id(VectorId::from_serial_id((i as u32) + 1), Arc::new(iris));
    }

    let sample_vid = VectorId::from_serial_id(sample);
    let sample_iris = store.storage.get_vector(&sample_vid).cloned().unwrap();

    for lc in 0..graph.layers.len() {
        let neighbors = graph.layers[lc]
            .get_links(&sample)
            .unwrap_or_else(|| panic!("{}", lc));

        let mut dists = Vec::new();
        for k in graph.layers[lc].links.keys() {
            if *k != sample {
                let k_vid = VectorId::from_serial_id(*k);
                let dist = store.eval_distance(&sample_iris, &k_vid).await.unwrap();
                dists.push((*k, dist));
            }
        }

        if !dists.is_empty() {
            dists.sort_by(|a, b| D::plaintext_ordering(&a.1, &b.1));
            dists.truncate(neighbors.degree());
            let kth_dist = dists.last().unwrap().1;

            let count_greater = neighbors
                .neighbors()
                .iter()
                .filter(|n| {
                    let n_vid = VectorId::from_serial_id(**n);
                    let d = D::plaintext_distance(
                        &sample_iris,
                        store.storage.get_vector(&n_vid).unwrap(),
                        store.distance_mode,
                    );
                    matches!(D::plaintext_ordering(&d, &kth_dist), Ordering::Greater)
                })
                .count();

            assert!(count_greater == 0);
        }
    }
}

async fn run_sanity_check_deep_id(
    graph: &GraphMem,
    searcher: &HnswSearcher,
    vectors: Vec<Int4Vector>,
    threshold: i32,
) {
    for (lc, layer) in graph.layers.iter().enumerate() {
        let expected_nb_size = searcher
            .params
            .get_M_max(lc)
            .min(layer.links.len().saturating_sub(1));
        for (_, value) in layer.links.iter() {
            assert_eq!(value.degree(), expected_nb_size);
        }
    }

    assert!(graph.num_layers() <= searcher.max_graph_layer + 1);

    let last_graph_layer = graph.layers.last().unwrap();
    for entry in &graph.entry_points {
        assert!(
            last_graph_layer.links.contains_key(&entry.point),
            "Entry point {:?} not found in last graph layer",
            entry
        );
    }

    let mut entropy_rng = rand::rngs::StdRng::from_entropy();

    let sample = last_graph_layer
        .links
        .keys()
        .choose(&mut entropy_rng)
        .cloned()
        .expect("last layer should have at least one key");

    let mut store = PlaintextDeepIDStore::new(threshold);
    for (i, v) in vectors.into_iter().enumerate() {
        store.insert_with_id(VectorId::from_serial_id((i as u32) + 1), Arc::new(v));
    }

    let sample_vid = VectorId::from_serial_id(sample);
    let sample_vec = Arc::new(store.storage.get_vector(&sample_vid).cloned().unwrap());

    for lc in 0..graph.layers.len() {
        let neighbors = graph.layers[lc]
            .get_links(&sample)
            .unwrap_or_else(|| panic!("{}", lc));

        let mut dists: Vec<(VectorId, i32)> = Vec::new();
        for k in graph.layers[lc].links.keys() {
            if *k != sample {
                let k_vid = VectorId::from_serial_id(*k);
                let dist = store.eval_distance(&sample_vec, &k_vid).await.unwrap();
                dists.push((k_vid, dist));
            }
        }
        if dists.is_empty() {
            continue;
        }

        // Larger dot = closer. Sort descending by dot, ascending by id on ties.
        dists.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.serial_id().cmp(&b.0.serial_id())));
        dists.truncate(neighbors.degree());
        let kth_dot = dists.last().unwrap().1;

        let count_closer_outside = neighbors
            .neighbors()
            .iter()
            .filter(|n| {
                let n_vid = VectorId::from_serial_id(**n);
                let other = store.storage.get_vector(&n_vid).unwrap();
                matches!(kth_dot.cmp(&sample_vec.dot(other)), Ordering::Greater)
            })
            .count();
        assert!(count_closer_outside == 0);
    }
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
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config: IdealGraphConfig = load_toml(&cli.job_spec)?;
    let searcher = config.searcher;

    let graph = match config.store {
        StoreKindConfig::Iris {
            irises_path,
            echoice,
        } => {
            let file = std::fs::File::open(irises_path).unwrap();
            let reader = BufReader::new(file);
            let n = config.graph_size;
            let stream = Deserializer::from_reader(reader)
                .into_iter::<Base64IrisCode>()
                .take(n);
            let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();

            let graph = GraphMem::ideal_from_irises(
                irises.clone(),
                config.layer0_path.clone(),
                &searcher,
                config.prf_seed,
                echoice,
            )
            .unwrap();

            if config.sanity_check {
                match echoice {
                    EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => {
                        run_sanity_check_iris::<FhdOps>(&graph, &searcher, irises, echoice).await;
                    }
                    EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => {
                        run_sanity_check_iris::<NhdOps>(&graph, &searcher, irises, echoice).await;
                    }
                }
            }

            graph
        }
        StoreKindConfig::DeepID {
            vectors_path,
            echoice,
            threshold,
        } => {
            let n = config.graph_size;
            let vectors = int4_vectors_from_ndjson(&vectors_path, Some(n)).unwrap();

            let graph = GraphMem::ideal_from_int4_vectors(
                vectors.clone(),
                config.layer0_path.clone(),
                &searcher,
                config.prf_seed,
                echoice,
            )?;

            if config.sanity_check {
                run_sanity_check_deep_id(&graph, &searcher, vectors, threshold).await;
            }

            graph
        }
    };

    let mut writer = std::fs::File::create(&cli.output)?;
    write_graph_current(&mut writer, graph)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_iris_config_deserializes() {
        // Mirrors resources/iris-mpc-cpu/ideal_config.toml: N_PARAM_LAYERS = 5,
        // M_limit defaulted by serde, Geometric layer distribution.
        let toml_str = r#"
graph_size = 8
irises_path = "irises.ndjson"
layer0_path = "layer0.txt"
prf_seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
echoice = "NaiveFHD"
sanity_check = false

[searcher]
max_graph_layer = 4
layer_distribution = { Geometric = { layer_probability = 0.25 } }
[searcher.params]
M = [10, 10, 10, 10, 10]
M_max = [10, 10, 10, 10, 10]
M_limit = [12, 12, 12, 12, 12]
ef_constr_search = [320, 320, 320, 320, 320]
ef_constr_insert = [320, 320, 320, 320, 320]
ef_search = [320, 320, 320, 320, 320]
"#;
        let cfg: IdealGraphConfig = toml::from_str(toml_str).expect("iris config deserializes");
        if !matches!(cfg.store, StoreKindConfig::Iris { .. }) {
            panic!("expected Iris variant");
        }
    }

    #[test]
    fn deepid_config_deserializes() {
        let toml_str = r#"
graph_size = 8
vectors_path = "vectors.ndjson"
layer0_path = "layer0.txt"
prf_seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
echoice = "NaiveInt4Dot"
threshold = 1000
sanity_check = false

[searcher]
max_graph_layer = 4
layer_distribution = { Geometric = { layer_probability = 0.25 } }
[searcher.params]
M = [10, 10, 10, 10, 10]
M_max = [10, 10, 10, 10, 10]
M_limit = [12, 12, 12, 12, 12]
ef_constr_search = [320, 320, 320, 320, 320]
ef_constr_insert = [320, 320, 320, 320, 320]
ef_search = [320, 320, 320, 320, 320]
"#;
        let cfg: IdealGraphConfig = toml::from_str(toml_str).expect("deepid config deserializes");
        if !matches!(cfg.store, StoreKindConfig::DeepID { .. }) {
            panic!("expected DeepID variant");
        }
    }
}
