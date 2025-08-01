use std::sync::Arc;

use super::iris_code::PyIrisCode;
use iris_mpc_cpu::{hawkers::plaintext_store::PlaintextStore, py_bindings};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone, Default)]
pub struct PyPlaintextStore(pub PlaintextStore);

#[pymethods]
impl PyPlaintextStore {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: u32) -> PyIrisCode {
        (*self.0.storage.points[&(id + 1)].1).clone().into()
    }

    pub fn insert(&mut self, iris: PyIrisCode) -> u32 {
        self.0.storage.append(Arc::new(iris.0)).serial_id()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.storage.points.is_empty()
    }

    #[staticmethod]
    #[pyo3(signature = (filename, len=None))]
    pub fn read_from_ndjson(filename: String, len: Option<usize>) -> PyResult<Self> {
        let result = py_bindings::plaintext_store::from_ndjson_file(&filename, len)
            .map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    pub fn write_to_ndjson(&self, filename: String) -> PyResult<()> {
        py_bindings::plaintext_store::to_ndjson_file(&self.0, &filename)
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
