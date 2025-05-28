use aes_prng::AesRng;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::GraphMem,
    py_bindings::{io::save_graph_as_csv, io::write_bin, plaintext_store::to_ndjson_file},
};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // create a folder ./iris-mpc-cpu/data if it is non-existent
    let crate_root = env!("CARGO_MANIFEST_DIR");
    // std::fs::create_dir_all(format!("{crate_root}/data"))?;
    // let mut rng = AesRng::seed_from_u64(0_u64);
    // println!("Generating plaintext store with 5000 irises");
    // let mut store = PlaintextStore::new_random(&mut rng, 1000);
    // println!("Writing store to file");
    // to_ndjson_file(&store, &format!("{crate_root}/data/store.ndjson"))?;

    let searcher = HnswSearcher {
        params: HnswParams::new(320, 256, 256),
    };
    for graph_size in [100, 200, 500, 1000] {
        println!("Generating graph with {} vertices", graph_size);
        let graph = store
            .generate_graph(&mut rng, graph_size, &searcher)
            .await?;
        write_bin(&graph, &format!("{crate_root}/data/graph_{graph_size}.dat"))?;
        save_graph_as_csv(
            &graph,
            0,
            &format!("{crate_root}/data/graph_{graph_size}.csv"),
        )?;
    }
    for graph_size in [100, 200, 500, 1000] {
        let file = File::open(format!("{crate_root}/data/graph_{graph_size}.dat"))?;
        let mut reader = BufReader::new(file);
        // Replace with the actual type used for V in GraphMem<V>
        let graph: GraphMem<PlaintextStore> = bincode::deserialize_from(&mut reader)?;
        println!("{:#?}", graph);
        save_graph_as_csv(
            &graph,
            0,
            &format!("{crate_root}/data/graph_{graph_size}.csv"),
        )?;
    }
    //
    Ok(())
}
