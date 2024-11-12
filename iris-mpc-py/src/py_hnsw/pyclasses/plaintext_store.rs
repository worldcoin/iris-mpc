
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Default)]
pub struct PyPlaintextStore(pub PlaintextStore);

#[pymethods]
#[allow(non_snake_case)]
impl PyPlaintextStore {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    fn read_from_ndjson(filename: String, len: usize) -> PyResult<Self> {
        let result = PlaintextStore::read_ndjson_file(&filename, len)
            .map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    fn write_to_ndjson(&self, filename: String) -> PyResult<()> {
        self.0
            .write_ndjson_file(&filename)
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
