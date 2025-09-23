use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_common::vector_id::SerialId;
use iris_mpc_cpu::{
    hawkers::naive_knn_plaintext::naive_knn,
    py_bindings::{limited_iterator, plaintext_store::Base64IrisCode},
};
use metrics::IntoF64;
use serde_json::Deserializer;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IrisCodeWithSerialId {
    pub iris_code: IrisCode,
    pub serial_id: SerialId,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of irises to process
    #[arg(long, default_value_t = 1000)]
    num_irises: usize,

    /// Number of threads to use
    #[arg(long, default_value_t = 1)]
    num_threads: usize,
}
#[tokio::main]
async fn main() {
    let args = Args::parse();
    let n_existing_irises = 0;
    let num_irises = args.num_irises;

    let mut path_to_iris_codes = PathBuf::new();
    path_to_iris_codes.push("iris-mpc-cpu/data/store.ndjson".to_owned());

    let file = File::open(path_to_iris_codes.as_path()).unwrap();
    let reader = BufReader::new(file);

    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(2 * n_existing_irises);

    let mut irises: [Vec<IrisCode>; 2] = [Vec::new(), Vec::new()];

    let stream = limited_iterator(stream, Some(num_irises * 2));
    for (idx, json_pt) in stream.enumerate() {
        let iris_code_query = (&json_pt.unwrap()).into();
        let _serial_id = ((idx / 2) + 1 + n_existing_irises) as u32;

        let side = idx % 2;
        irises[side].push(iris_code_query);
    }

    let start = Instant::now();
    naive_knn(irises[0].clone(), args.num_threads);
    let duration = start.elapsed();
    println!(
        "naive_knn took {:?} (per number of pairs)",
        duration.into_f64() / (num_irises as f64) / (num_irises as f64)
    );
}
