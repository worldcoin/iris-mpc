use super::pyclasses::{
    graph_store::PyGraphStore, hawk_searcher::PyHnswSearcher, iris_code::PyIrisCode,
    iris_code_array::PyIrisCodeArray, plaintext_store::PyPlaintextStore,
};
use pyo3::prelude::*;

#[pymodule]
fn iris_mpc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrisCodeArray>()?;
    m.add_class::<PyIrisCode>()?;
    m.add_class::<PyPlaintextStore>()?;
    m.add_class::<PyGraphStore>()?;
    m.add_class::<PyHnswSearcher>()?;
    Ok(())
}
