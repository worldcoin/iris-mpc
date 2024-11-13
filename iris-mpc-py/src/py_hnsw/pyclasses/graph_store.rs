use hawk_pack::graph_store::GraphMem;
use iris_mpc_cpu::{hawkers::plaintext_store::PlaintextStore, py_bindings};
use pyo3::{exceptions::PyIOError, prelude::*};

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
}
