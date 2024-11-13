use super::iris_code::PyIrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::{PlaintextIris, PlaintextPoint, PlaintextStore},
    py_bindings,
};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone, Default)]
pub struct PyPlaintextStore(pub PlaintextStore);

#[pymethods]
impl PyPlaintextStore {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, id: u32) -> PyIrisCode {
        self.0.points[id as usize].data.0.clone().into()
    }

    fn insert(&mut self, iris: PyIrisCode) -> u32 {
        let new_id = self.0.points.len() as u32;
        self.0.points.push(PlaintextPoint {
            data:          PlaintextIris(iris.0),
            is_persistent: true,
        });
        new_id
    }

    fn len(&self) -> usize {
        self.0.points.len()
    }

    #[staticmethod]
    #[pyo3(signature = (filename, len=None))]
    fn read_from_ndjson(filename: String, len: Option<usize>) -> PyResult<Self> {
        let result = py_bindings::plaintext_store::from_ndjson_file(&filename, len)
            .map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    fn write_to_ndjson(&self, filename: String) -> PyResult<()> {
        py_bindings::plaintext_store::to_ndjson_file(&self.0, &filename)
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
