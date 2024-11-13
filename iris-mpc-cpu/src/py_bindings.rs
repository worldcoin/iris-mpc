pub mod hnsw {
    use super::plaintext_store::Base64IrisCode;
    use crate::hawkers::plaintext_store::{PlaintextStore, PointId};
    use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
    use iris_mpc_common::iris_db::iris::IrisCode;
    use rand::rngs::ThreadRng;
    use serde_json::{self, Deserializer};
    use std::{fs::File, io::BufReader};

    pub fn search(
        query: IrisCode,
        searcher: &HawkSearcher,
        vector: &mut PlaintextStore,
        graph: &mut GraphMem<PlaintextStore>,
    ) -> (PointId, f64) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let query = vector.prepare_query(query);
            let neighbors = searcher.search_to_insert(vector, graph, &query).await;
            let (nearest, (dist_num, dist_denom)) = neighbors[0].get_nearest().unwrap();
            (*nearest, (*dist_num as f64) / (*dist_denom as f64))
        })
    }

    // TODO could instead take iterator of IrisCodes to make more flexible
    pub fn insert(
        iris: IrisCode,
        searcher: &HawkSearcher,
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
            let neighbors = searcher.search_to_insert(vector, graph, &query).await;
            let inserted = vector.insert(&query).await;
            searcher
                .insert_from_search_results(vector, graph, &mut rng, inserted, neighbors)
                .await;
            inserted
        })
    }

    pub fn insert_uniform_random(
        searcher: &HawkSearcher,
        vector: &mut PlaintextStore,
        graph: &mut GraphMem<PlaintextStore>,
    ) -> PointId {
        let mut rng = ThreadRng::default();
        let raw_query = IrisCode::random_rng(&mut rng);

        insert(raw_query, searcher, vector, graph)
    }

    pub fn fill_uniform_random(
        num: usize,
        searcher: &HawkSearcher,
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
                let neighbors = searcher.search_to_insert(vector, graph, &query).await;
                let inserted = vector.insert(&query).await;
                searcher
                    .insert_from_search_results(vector, graph, &mut rng, inserted, neighbors)
                    .await;
                if idx % 100 == 99 {
                    println!("{}", idx + 1);
                }
            }
        })
    }

    pub fn fill_from_ndjson_file(
        filename: &str,
        limit: Option<usize>,
        searcher: &HawkSearcher,
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
                let neighbors = searcher.search_to_insert(vector, graph, &query).await;
                let inserted = vector.insert(&query).await;
                searcher
                    .insert_from_search_results(vector, graph, &mut rng, inserted, neighbors)
                    .await;
            }
        })
    }
}

pub mod plaintext_store {
    use crate::hawkers::plaintext_store::{PlaintextIris, PlaintextPoint, PlaintextStore};
    use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
    use serde::{Deserialize, Serialize};
    use std::{
        fs::File,
        io::{self, BufReader, BufWriter, Write},
    };

    /// Iris code representation using base64 encoding compatible with Open IRIS
    #[derive(Serialize, Deserialize)]
    pub struct Base64IrisCode {
        iris_codes: String,
        mask_codes: String,
    }

    impl From<&IrisCode> for Base64IrisCode {
        fn from(value: &IrisCode) -> Self {
            Self {
                iris_codes: value.code.to_base64().unwrap(),
                mask_codes: value.mask.to_base64().unwrap(),
            }
        }
    }

    impl From<&Base64IrisCode> for IrisCode {
        fn from(value: &Base64IrisCode) -> Self {
            Self {
                code: IrisCodeArray::from_base64(&value.iris_codes).unwrap(),
                mask: IrisCodeArray::from_base64(&value.mask_codes).unwrap(),
            }
        }
    }

    pub fn from_ndjson_file(filename: &str, len: Option<usize>) -> io::Result<PlaintextStore> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        // Create an iterator over deserialized objects
        let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
        let stream = super::limited_iterator(stream, len);

        // Iterate over each deserialized object
        let mut vector = PlaintextStore::default();
        for json_pt in stream {
            let json_pt = json_pt?;
            vector.points.push(PlaintextPoint {
                data:          PlaintextIris((&json_pt).into()),
                is_persistent: true,
            });
        }

        if let Some(num) = len {
            if vector.points.len() != num {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "File {} contains too few entries; number read: {}",
                        filename,
                        vector.points.len()
                    ),
                ));
            }
        }

        Ok(vector)
    }

    pub fn to_ndjson_file(vector: &PlaintextStore, filename: &str) -> std::io::Result<()> {
        // Serialize the objects to the file
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        for pt in &vector.points {
            let json_pt: Base64IrisCode = (&pt.data.0).into();
            serde_json::to_writer(&mut writer, &json_pt)?;
            writer.write_all(b"\n")?; // Write a newline after each JSON object
        }
        writer.flush()?;
        Ok(())
    }
}

pub mod io {
    use anyhow::Result;
    use bincode;
    use serde::{de::DeserializeOwned, Serialize};
    use serde_json;
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
    };

    pub fn write_bin<T: Serialize>(data: &T, filename: &str) -> Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, data)?;
        Ok(())
    }

    pub fn read_bin<T: DeserializeOwned>(filename: &str) -> Result<T> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let data: T = bincode::deserialize_from(reader)?;
        Ok(data)
    }

    pub fn write_json<T: Serialize>(data: &T, filename: &str) -> Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data)?;
        Ok(())
    }

    pub fn read_json<T: DeserializeOwned>(filename: &str) -> Result<T> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let data: T = serde_json::from_reader(reader)?;
        Ok(data)
    }
}

pub fn limited_iterator<I>(iter: I, limit: Option<usize>) -> Box<dyn Iterator<Item = I::Item>>
where
    I: Iterator + 'static,
{
    match limit {
        Some(num) => Box::new(iter.take(num)),
        None => Box::new(iter),
    }
}
