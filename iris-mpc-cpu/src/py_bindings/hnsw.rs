use super::plaintext_store::Base64IrisCode;
use crate::{
    hawkers::plaintext_store::{PlaintextStore, PointId},
    hnsw::{GraphMem, HnswSearcher},
};
use iris_mpc_common::iris_db::iris::IrisCode;
use rand::rngs::ThreadRng;
use serde_json::{self, Deserializer};
use std::{fs::File, io::BufReader};

pub fn search(
    query: IrisCode,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
) -> (PointId, f64) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let query = vector.prepare_query(query);
        let neighbors = searcher.search(vector, graph, &query, 1).await;
        let (nearest, (dist_num, dist_denom)) = neighbors.get_nearest().unwrap();
        (*nearest, (*dist_num as f64) / (*dist_denom as f64))
    })
}

// TODO could instead take iterator of IrisCodes to make more flexible
pub fn insert(
    iris: IrisCode,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
) -> PointId {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();

        let query = vector.prepare_query(iris);
        searcher.insert(vector, graph, &query, &mut rng).await
    })
}

pub fn insert_uniform_random(
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
) -> PointId {
    let mut rng = ThreadRng::default();
    let raw_query = IrisCode::random_rng(&mut rng);

    insert(raw_query, searcher, vector, graph)
}

pub fn fill_uniform_random(
    num: usize,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();

        for idx in 0..num {
            let raw_query = IrisCode::random_rng(&mut rng);
            let query = vector.prepare_query(raw_query.clone());
            searcher.insert(vector, graph, &query, &mut rng).await;
            if idx % 100 == 99 {
                println!("{}", idx + 1);
            }
        }
    })
}

pub fn fill_from_ndjson_file(
    filename: &str,
    limit: Option<usize>,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextStore>,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();

        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);

        // Create an iterator over deserialized objects
        let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
        let stream = super::limited_iterator(stream, limit);

        // Iterate over each deserialized object
        for json_pt in stream {
            let raw_query = (&json_pt.unwrap()).into();
            let query = vector.prepare_query(raw_query);
            searcher.insert(vector, graph, &query, &mut rng).await;
        }
    })
}
