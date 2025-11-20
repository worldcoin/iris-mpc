use super::{graph_store::PyGraphStore, iris_code::PyIrisCode, plaintext_store::PyPlaintextStore};
use iris_mpc_cpu::{
    hnsw::searcher::{HnswParams, HnswSearcher, LayerDistribution, LayerMode, N_PARAM_LAYERS},
    py_bindings,
    utils::serialization::{read_json, write_json},
};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone)]
pub struct PyHnswSearcher(pub HnswSearcher);

impl Default for PyHnswSearcher {
    fn default() -> Self {
        Self(HnswSearcher::new_with_test_parameters())
    }
}

#[pymethods]
#[allow(non_snake_case)]
impl PyHnswSearcher {
    #[new]
    pub fn new(M: usize, ef_constr: usize, ef_search: usize) -> Self {
        Self::new_standard(ef_constr, ef_search, M)
    }

    #[staticmethod]
    pub fn new_standard(M: usize, ef_constr: usize, ef_search: usize) -> Self {
        Self(HnswSearcher::new_standard(ef_constr, ef_search, M))
    }

    #[staticmethod]
    pub fn new_uniform(M: usize, ef: usize) -> Self {
        let params = HnswParams::new_uniform(ef, M);
        let layer_mode = LayerMode::Standard;
        let layer_distribution = LayerDistribution::new_geometric_from_M(M);
        Self(HnswSearcher {
            params,
            layer_mode,
            layer_distribution,
        })
    }

    /// Construct `HnswSearcher` with fully general parameters, specifying the
    /// values of various parameters used during construction and search at
    /// different levels of the graph hierarchy.
    #[staticmethod]
    pub fn new_general(
        M: [usize; N_PARAM_LAYERS],
        M_max: [usize; N_PARAM_LAYERS],
        M_max_extra: [usize; N_PARAM_LAYERS],
        ef_constr_search: [usize; N_PARAM_LAYERS],
        ef_constr_insert: [usize; N_PARAM_LAYERS],
        ef_search: [usize; N_PARAM_LAYERS],
        layer_probability: f64,
    ) -> Self {
        let params = HnswParams {
            M,
            M_max,
            M_max_extra,
            ef_constr_search,
            ef_constr_insert,
            ef_search,
        };
        let layer_mode = LayerMode::Standard;
        let layer_distribution = LayerDistribution::Geometric { layer_probability };
        Self(HnswSearcher {
            params,
            layer_mode,
            layer_distribution,
        })
    }

    pub fn insert(
        &self,
        iris: PyIrisCode,
        vector: &mut PyPlaintextStore,
        graph: &mut PyGraphStore,
    ) -> u32 {
        let id = py_bindings::hnsw::insert(iris.0, &self.0, &mut vector.0, &mut graph.0);
        id.serial_id()
    }

    pub fn insert_uniform_random(
        &self,
        vector: &mut PyPlaintextStore,
        graph: &mut PyGraphStore,
    ) -> u32 {
        let id = py_bindings::hnsw::insert_uniform_random(&self.0, &mut vector.0, &mut graph.0);
        id.serial_id()
    }

    pub fn fill_uniform_random(
        &self,
        num: usize,
        vector: &mut PyPlaintextStore,
        graph: &mut PyGraphStore,
    ) {
        py_bindings::hnsw::fill_uniform_random(num, &self.0, &mut vector.0, &mut graph.0);
    }

    #[pyo3(signature = (filename, vector, graph, limit=None))]
    pub fn fill_from_ndjson_file(
        &self,
        filename: String,
        vector: &mut PyPlaintextStore,
        graph: &mut PyGraphStore,
        limit: Option<usize>,
    ) {
        py_bindings::hnsw::fill_from_ndjson_file(
            &filename,
            limit,
            &self.0,
            &mut vector.0,
            &mut graph.0,
        );
    }

    /// Search HNSW index and return nearest ID and its distance from query
    pub fn search(
        &mut self,
        query: &PyIrisCode,
        vector: &mut PyPlaintextStore,
        graph: &mut PyGraphStore,
    ) -> (u32, f64) {
        let (id, dist) =
            py_bindings::hnsw::search(query.0.clone(), &self.0, &mut vector.0, &mut graph.0);
        (id.serial_id(), dist)
    }

    #[staticmethod]
    pub fn read_from_json(filename: String) -> PyResult<Self> {
        let result =
            read_json(&filename).map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    pub fn write_to_json(&self, filename: String) -> PyResult<()> {
        write_json(&self.0, &filename).map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
