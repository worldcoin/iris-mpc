use iris_mpc_cpu::{py_bindings,hawkers::plaintext_store::PlaintextStore};
use hawk_pack::graph_store::GraphMem;
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Clone, Default)]
pub struct PyGraphStore(pub GraphMem<PlaintextStore>);

#[pymethods]
impl PyGraphStore {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    fn read_from_bin(filename: String) -> PyResult<Self> {
        let result = py_bindings::read_serde_bin(&filename)
            .map_err(|_| PyIOError::new_err("Unable to read from file"))?;
        Ok(Self(result))
    }

    fn write_to_bin(&self, filename: String) -> PyResult<()> {
        py_bindings::write_serde_bin(&self.0, &filename)
            .map_err(|_| PyIOError::new_err("Unable to write to file"))
    }
}
