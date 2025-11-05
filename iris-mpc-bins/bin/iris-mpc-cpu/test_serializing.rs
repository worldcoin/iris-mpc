// This 'use' path is based on the 'mod tests' block in your graph.rs file.
// It assumes that your project's src/lib.rs (or src/main.rs if it's a binary-only crate)
// correctly defines the module structure (e.g., `pub mod utils;`).
use iris_mpc_cpu::utils::serialization::graph::{read_graph_from_file, GraphFormat};

// The 'eyre' crate is used in graph.rs, so we use it here too.
// Make sure 'eyre' is a dependency in your Cargo.toml
use eyre::Result;

fn main() -> Result<()> {
    println!("--- HNSW Graph Deserialization Test ---");
    println!("Running test for V0 format...");

    // --- V0 Test ---
    // Hardcode the path as requested.
    // PLEASE REPLACE THIS with the actual path to your V0 file.
    let path_v0 = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/db1M_M96_efConstruction256_graph.bin"
    );

    println!("Attempting to read V0 graph from: {}", path_v0);
    match read_graph_from_file(path_v0, GraphFormat::V0) {
        Ok(graph) => {
            println!("[SUCCESS] V0 graph deserialized.");
            println!("    Layers: {}", graph.layers.len());
            for (i, layer) in graph.layers.iter().enumerate() {
                println!("    Layer {}: {} nodes", i, layer.links.len());
            }
            // Safely check the first layer
            if let Some(first_layer) = graph.layers.first() {
                println!("    Layer 0 Nodes: {}", first_layer.links.len());
            }
            println!("    Entry Points: {}", graph.entry_point.len());
        }
        Err(e) => {
            eprintln!("[FAILURE] Failed to deserialize V0 graph.");
            eprintln!("    Error: {:?}", e);
        }
    }

    println!("\nRunning test for V1 format...");

    // --- V1 Test ---
    // Hardcode the path as requested.
    // PLEASE REPLACE THIS with the actual path to your V1 file.
    // Use the cargo manifest directory to resolve the path relative to the project root.
    let path_v1 = concat!(env!("CARGO_MANIFEST_DIR"), "/graph_65536.dat");

    println!("Attempting to read V1 graph from: {}", path_v1);

    match read_graph_from_file(path_v1, GraphFormat::V1) {
        Ok(graph) => {
            println!("[SUCCESS] V1 graph deserialized.");
            println!("    Layers: {}", graph.layers.len());
            // Safely check the first layer
            if let Some(first_layer) = graph.layers.first() {
                println!("    Layer 0 Nodes: {}", first_layer.links.len());
            }
            println!("    Entry Points: {}", graph.entry_point.len());
            dbg!(graph.entry_point.first().unwrap());
        }
        Err(e) => {
            eprintln!("[FAILURE] Failed to deserialize V1 graph.");
            eprintln!("    Error: {:?}", e);
        }
    }

    println!("\n--- Test Complete ---");
    Ok(())
}
