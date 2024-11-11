use super::pyclasses::{hnsw::PyHnsw, iris_code::PyIrisCode, iris_code_array::PyIrisCodeArray};
use pyo3::prelude::*;

#[pymodule]
fn iris_mpc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrisCodeArray>()?;
    m.add_class::<PyIrisCode>()?;
    m.add_class::<PyHnsw>()?;
    Ok(())
}
