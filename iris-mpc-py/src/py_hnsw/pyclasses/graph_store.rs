use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::layered_graph::GraphMem, py_bindings,
};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone)]
pub struct PyIrisVectorId(pub IrisVectorId);

#[pymethods]
impl PyIrisVectorId {
    #[staticmethod]
    pub fn from_serial_id(serial_id: usize) -> Self {
        Self(IrisVectorId::from_serial_id(serial_id.try_into().unwrap()))
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct PyGraphStore(pub GraphMem<PlaintextStore>);

#[pymethods]
impl PyGraphStore {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    pub fn read_from_bin(filename: String) -> PyResult<Self> {
        let result = py_bindings::io::read_bin(&filename)
            .map_err(|_| PyIOError::new_err("Unable to read from file"))?;

        Ok(Self(result))
    }

    pub fn write_to_bin(&self, filename: String) -> PyResult<()> {
        py_bindings::io::write_bin(&self.0, &filename)
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }

    pub fn get_links(
        &self,
        vector_id: PyIrisVectorId,
        layer_index: usize,
    ) -> PyResult<Option<Vec<u32>>> {
        let raw_ret = self.0.layers[layer_index].get_links(&vector_id.0);
        Ok(raw_ret.map(|neighborhood| neighborhood.0.iter().map(|nb| nb.serial_id()).collect()))
    }
}
