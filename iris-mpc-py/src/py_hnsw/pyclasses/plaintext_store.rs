use std::{path::Path, sync::Arc};

use super::iris_code::PyIrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, utils::serialization::iris_ndjson::IrisSelection,
};
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
        (self
            .0
            .storage
            .get_vector_by_serial_id(id)
            .unwrap()
            .as_ref()
            .clone())
        .into()
    }

    pub fn eval_distance_to_id(&self, lhs: PyIrisCode, rhs: u32) -> (u16, u16) {
        let iris_rhs = self.get(rhs);
        lhs.get_distance_fraction(iris_rhs)
    }

    pub fn eval_distance(&self, lhs: u32, rhs: u32) -> (u16, u16) {
        let iris_lhs = self.get(lhs);
        self.eval_distance_to_id(iris_lhs, rhs)
    }

    pub fn insert(&mut self, iris: PyIrisCode) -> u32 {
        self.0.storage.append(Arc::new(iris.0)).serial_id()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.storage.get_points().is_empty()
    }

    #[staticmethod]
    #[pyo3(signature = (filename, len=None))]
    pub fn read_from_ndjson(filename: String, len: Option<usize>) -> PyResult<Self> {
        let result =
            PlaintextStore::from_ndjson_file(Path::new(&filename), len, IrisSelection::All)
                .map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    pub fn write_to_ndjson(&self, filename: String) -> PyResult<()> {
        self.0
            .to_ndjson_file(Path::new(&filename))
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
