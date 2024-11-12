use iris_mpc_cpu::py_bindings;
use hawk_pack::hnsw_db::{HawkSearcher, Params};
use pyo3::{exceptions::PyIOError, prelude::*};

#[pyclass]
#[derive(Default)]
pub struct PyHawkSearcher(pub HawkSearcher);

#[pymethods]
#[allow(non_snake_case)]
impl PyHawkSearcher {
    #[new]
    fn new(M: usize, ef_constr: usize, ef_search: usize) -> Self {
        Self::new_standard(ef_constr, ef_search, M)
    }

    #[staticmethod]
    fn new_standard(M: usize, ef_constr: usize, ef_search: usize) -> Self {
        let params = Params::new_standard(ef_constr, ef_search, M);
        Self(HawkSearcher { params })
    }

    #[staticmethod]
    fn new_uniform(M: usize, ef: usize) -> Self {
        let params = Params::new_uniform(ef, M);
        Self(HawkSearcher { params })
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
