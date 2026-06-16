//! Verifies that an Int4 neighborhoods results file (body lines only) parses
//! via the same reader `generate_ideal_graph.rs` uses, and that node ids form
//! the contiguous 1..=N sequence the consumer's zip relies on.

use std::io::Write;

use aes_prng::AesRng;
use iris_mpc_cpu::hawkers::ideal_knn_engines::{
    read_knn_results_from_file, EngineChoiceInt4, EngineInt4, KNNResult,
};
use iris_mpc_cpu::hawkers::plaintext_deep_id_store::Int4Vector;
use rand::SeedableRng;
use tempfile::tempdir;

#[test]
fn int4_neighborhoods_file_is_consumable_and_contiguous() {
    let mut rng = AesRng::seed_from_u64(0xD00D);
    let n = 32usize;
    let k = 4usize;
    let vectors: Vec<Int4Vector> = (0..n).map(|_| Int4Vector::random(&mut rng)).collect();

    // Compute neighborhoods exactly as the binary does (engine path).
    let mut engine = EngineInt4::init(EngineChoiceInt4::NaiveInt4Dot, vectors, k, 1);
    let results: Vec<KNNResult> = engine.compute_chunk(n);

    // Write a results file: a header line (ignored by the consumer) + body lines.
    let dir = tempdir().unwrap();
    let path = dir.path().join("layer0.ndjson");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{\"some\":\"header the consumer skips\"}}").unwrap();
        // First, serialize the nodes as u32 values for the test file
        for (i, _) in results.iter().enumerate() {
            let json = serde_json::json!({
                "node": (i + 1) as u32,
                "neighbors": results[i].neighbors.iter().map(|v| v.serial_id()).collect::<Vec<_>>()
            });
            writeln!(f, "{}", json).unwrap();
        }
    }

    // Parse with the SAME reader generate_ideal_graph.rs uses.
    let parsed = read_knn_results_from_file(path).unwrap();
    assert_eq!(parsed.len(), n, "one result per node");

    // Node ids must be the contiguous 1..=N sequence (the consumer zips
    // sorted ids against vectors loaded in file order).
    let mut nodes: Vec<u32> = parsed.iter().map(|r| r.node.serial_id()).collect();
    nodes.sort_unstable();
    let expected: Vec<u32> = (1..=n as u32).collect();
    assert_eq!(nodes, expected, "node ids form 1..=N");

    // Each neighborhood has k entries, all in range and excluding self.
    for r in &parsed {
        assert_eq!(r.neighbors.len(), k);
        for nb in &r.neighbors {
            assert!((1..=n as u32).contains(&nb.serial_id()));
            assert_ne!(*nb, r.node, "self must be excluded");
        }
    }
}
