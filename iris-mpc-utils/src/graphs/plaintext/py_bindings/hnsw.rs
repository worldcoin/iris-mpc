use super::plaintext_store::Base64IrisCode;
use crate::misc::limited_iterator;
use iris_mpc_common::{iris_db::iris::IrisCode, vector_id::VectorId};
use iris_mpc_cpu::{
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef},
    hnsw::{GraphMem, HnswSearcher},
};
use rand::rngs::ThreadRng;
use serde_json::{self, Deserializer};
use std::{fs::File, io::BufReader, sync::Arc};

pub fn search(
    query: IrisCode,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextVectorRef>,
) -> (VectorId, f64) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let query = Arc::new(query);
        let neighbors = searcher.search(vector, graph, &query, 1).await.unwrap();
        let (nearest, (dist_num, dist_denom)) = neighbors.get_nearest().unwrap();
        (*nearest, (*dist_num as f64) / (*dist_denom as f64))
    })
}

// TODO could instead take iterator of IrisCodes to make more flexible
pub fn insert(
    iris: IrisCode,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextVectorRef>,
) -> VectorId {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();

        let query = Arc::new(iris);
        let insertion_layer = searcher.select_layer_rng(&mut rng).unwrap();
        searcher
            .insert(vector, graph, &query, insertion_layer)
            .await
            .unwrap()
    })
}

pub fn insert_uniform_random(
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextVectorRef>,
) -> VectorId {
    let mut rng = ThreadRng::default();
    let raw_query = IrisCode::random_rng(&mut rng);

    insert(raw_query, searcher, vector, graph)
}

pub fn fill_uniform_random(
    num: usize,
    searcher: &HnswSearcher,
    vector: &mut PlaintextStore,
    graph: &mut GraphMem<PlaintextVectorRef>,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        let mut rng = ThreadRng::default();

        for idx in 0..num {
            let query = Arc::new(IrisCode::random_rng(&mut rng));
            let insertion_layer = searcher.select_layer_rng(&mut rng).unwrap();
            searcher
                .insert(vector, graph, &query, insertion_layer)
                .await
                .unwrap();
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
    graph: &mut GraphMem<PlaintextVectorRef>,
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
        let stream = limited_iterator(stream, limit);

        // Iterate over each deserialized object
        for json_pt in stream {
            let raw_query = (&json_pt.unwrap()).into();
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.select_layer_rng(&mut rng).unwrap();
            searcher
                .insert(vector, graph, &query, insertion_layer)
                .await
                .unwrap();
        }
    })
}
