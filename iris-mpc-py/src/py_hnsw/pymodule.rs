use super::pyclasses::{
    iris_code_array::PyIrisCodeArray,
    iris_code::PyIrisCode,
    hnsw::PyHnsw,
};
use pyo3::prelude::*;

#[pymodule]
fn iris_mpc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrisCodeArray>()?;
    m.add_class::<PyIrisCode>()?;
    m.add_class::<PyHnsw>()?;
    Ok(())
}
