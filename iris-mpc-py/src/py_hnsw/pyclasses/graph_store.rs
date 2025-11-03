use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextVectorRef,
    hnsw::graph::layered_graph::GraphMem,
    utils::serialization::{read_bin, write_bin},
};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone, Default)]
pub struct PyGraphStore(pub GraphMem<PlaintextVectorRef>);

#[pymethods]
impl PyGraphStore {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    pub fn read_from_bin(filename: String) -> PyResult<Self> {
        let result =
            read_bin(&filename).map_err(|_| PyIOError::new_err("Unable to read from file"))?;

        Ok(Self(result))
    }

    pub fn write_to_bin(&self, filename: String) -> PyResult<()> {
        write_bin(&self.0, &filename).map_err(|_| PyIOError::new_err("Unable to write to file"))
    }

    pub fn get_max_layer(&self) -> u32 {
        self.0.layers.len().try_into().unwrap()
    }

    pub fn get_layer_nodes(&self, layer_index: usize) -> Option<Vec<u32>> {
        self.0
            .layers
            .get(layer_index)
            .map(|layer| layer.links.keys().map(|k| k.serial_id()).collect())
    }

    pub fn get_links(&self, vector_id: u32, layer_index: usize) -> PyResult<Option<Vec<u32>>> {
        let raw_ret =
            self.0.layers[layer_index].get_links(&IrisVectorId::from_serial_id(vector_id));
        Ok(raw_ret.map(|neighborhood| neighborhood.0.iter().map(|nb| nb.serial_id()).collect()))
    }
}
