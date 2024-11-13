// use super::{hawk_searcher::PyHawkSearcher, plaintext_store::PyPlaintextStore, graph_store::PyGraphStore, iris_code::PyIrisCode};
// use iris_mpc_cpu::py_bindings::{self, PlaintextHnsw};
// use pyo3::prelude::*;

// #[pyclass]
// #[derive(Clone, Default)]
// pub struct PyHnsw(pub PlaintextHnsw);

// #[pyfunction]
// pub fn hnsw_insert(iris: PyIrisCode, searcher: &PyHawkSearcher, vector: &mut PyPlaintextStore, graph: &mut PyGraphStore) -> u32 {
//     py_bindings::hnsw::insert(iris.0, &searcher.0, &mut vector.0, &mut graph.0).0
// }

// #[pymethods]
// #[allow(non_snake_case)]
// impl PyHnsw {
//     #[new]
//     fn new(searcher: PyHawkSearcher, vector: PyPlaintextStore, graph: PyGraphStore) -> Self {
//         Self(PlaintextHnsw {
//             searcher: searcher.0,
//             vector: vector.0,
//             graph: graph.0
//         })
//     }

//     // #[getter]
//     // fn searcher(&self) -> PyHawkSearcher {
//     //     PyHawkSearcher(self.0.searcher)
//     // }

//     fn fill_uniform_random(&mut self, num: usize) {
//         self.0.fill_uniform_random(num);
//     }

//     fn fill_from_ndjson(&mut self, filename: String, num: usize) {
//         self.0.fill_from_ndjson_file(&filename, num);
//     }

//     fn insert(&mut self, iris: &PyIrisCode) -> u32 {
//         self.0.insert(iris.0.clone()).0
//     }

//     /// Search HNSW index and return nearest ID and its distance from query
//     fn search(&mut self, query: &PyIrisCode) -> (u32, f64) {
//         let (id, dist) = self.0.search(query.0.clone());
//         (id.0, dist)
//     }

//     fn get_iris(&self, id: u32) -> PyIrisCode {
//         self.0.vector.points[id as usize].data.0.clone().into()
//     }

//     fn len(&self) -> usize {
//         self.0.vector.points.len()
//     }
// }
