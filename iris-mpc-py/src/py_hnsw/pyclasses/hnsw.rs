use iris_mpc_cpu::py_bindings::PlaintextHnsw;

use pyo3::{exceptions::PyAttributeError, prelude::*};
use super::iris_code::PyIrisCode;

#[pyclass]
pub struct PyHnsw(pub PlaintextHnsw);

#[pymethods]
#[allow(non_snake_case)]
impl PyHnsw {
    #[new]
    fn new(M: usize, ef: usize) -> Self {
        Self(PlaintextHnsw::new(M, ef, ef))
    }

    fn fill_uniform_random(&mut self, num: usize) {
        self.0.fill_uniform_random(num);
    }

    fn insert_uniform_random(&mut self) -> u32 {
        self.0.insert_uniform_random().0
    }

    fn insert(&mut self, iris: &PyIrisCode) -> u32 {
        self.0.insert(iris.0.clone()).0
    }

    /// Search HNSW index and return nearest ID and its distance from query
    fn search(&mut self, query: &PyIrisCode) -> (u32, f64) {
        let (id, dist) = self.0.search(query.0.clone());
        (id.0, dist)
    }

    fn get_iris(&self, id: u32) -> PyIrisCode {
        self.0.vector.points[id as usize].data.0.clone().into()
    }

    fn len(&self) -> usize {
        self.0.vector.points.len()
    }

    fn write_to_file(&self, filename: &str) -> PyResult<()> {
        self.0
            .write_to_file(filename)
            .map_err(|_| PyAttributeError::new_err("Unable to write to file"))
    }

    #[staticmethod]
    fn read_from_file(filename: &str) -> PyResult<Self> {
        let result = PlaintextHnsw::read_from_file(filename)
            .map_err(|_| PyAttributeError::new_err("Unable to read from file"))?;
        Ok(PyHnsw(result))
    }
}
