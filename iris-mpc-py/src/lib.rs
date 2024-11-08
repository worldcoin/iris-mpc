use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use pyo3::{exceptions::PyAttributeError, prelude::*};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, py_bindings
};
use hawk_pack::graph_store::GraphMem;

#[pyclass]
#[derive(Clone)]
struct PyIrisCodeArray(IrisCodeArray);

#[pymethods]
impl PyIrisCodeArray {

    #[new]
    fn new_py(input: String) -> Self {
        Self::from_base64(input)
    }

    fn to_base64(&self) -> String {
        self.0.to_base64().unwrap()
    }

    #[staticmethod]
    fn from_base64(input: String) -> Self {
        Self(IrisCodeArray::from_base64(&input).unwrap())
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self(IrisCodeArray::ZERO)
    }

    #[staticmethod]
    fn ones() -> Self {
        Self(IrisCodeArray::ONES)
    }

    #[staticmethod]
    fn uniform() -> Self {
        Self(py_bindings::gen_uniform_iris_code_array())
    }
}

impl From<IrisCodeArray> for PyIrisCodeArray {
    fn from(value: IrisCodeArray) -> Self {
        Self(value)
    }
}

#[pyclass]
struct PyIrisCode {
    #[pyo3(get)]
    code: PyIrisCodeArray,

    #[pyo3(get)]
    mask: PyIrisCodeArray,
}

#[pymethods]
impl PyIrisCode {
    #[new]
    fn new_py(code: &PyIrisCodeArray, mask: &PyIrisCodeArray) -> Self {
        Self {
            code: code.clone(),
            mask: mask.clone(),
        }
    }

    #[staticmethod]
    fn from_open_iris_template<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict_obj = obj.call_method0("serialize")
            .map_err(|_| PyAttributeError::new_err("Object has no method 'serialize'"))?;

        // Extract the base64-encoded strings from the dictionary
        let iris_codes_str = dict_obj.get_item("iris_codes")?;
        let mask_codes_str = dict_obj.get_item("mask_codes")?;

        // Convert the base64 strings into MyData instances
        let code: PyIrisCodeArray = iris_codes_str.extract()?;
        let mask: PyIrisCodeArray = mask_codes_str.extract()?;

        // Step 5: Construct and return PyIrisCode
        Ok(PyIrisCode{ code, mask } )
    }
}

impl PyIrisCode {
    fn to_iris_code(&self) -> IrisCode {
        IrisCode { code: self.code.0.clone(), mask: self.mask.0.clone() }
    }
}

impl From<IrisCode> for PyIrisCode {
    fn from(value: IrisCode) -> Self {
        Self {
            code: value.code.into(),
            mask: value.mask.into(),
        }
    }
}

#[pyclass]
struct PyHnsw {
    vector: PlaintextStore,
    graph: GraphMem<PlaintextStore>,
}

#[pymethods]
impl PyHnsw {
    #[new]
    fn new_py() -> Self {
        let (vector, graph) = py_bindings::gen_empty_index();
        PyHnsw {
            vector,
            graph,
        }
    }

    #[staticmethod]
    fn gen_uniform(size: usize) -> PyHnsw {
        let (vector, graph) = py_bindings::gen_uniform_random_index(size);
        PyHnsw {
            vector,
            graph,
        }
    }

    fn insert_uniform_random(&mut self) -> u32 {
        py_bindings::insert_random(&mut self.vector, &mut self.graph)
    }

    fn insert(&mut self, iris: &PyIrisCode) -> u32 {
        // let iris = IrisCode { code: iris.code.0.clone(), mask: iris.mask.0.clone() };
        py_bindings::insert_iris(&mut self.vector, &mut self.graph, iris.to_iris_code())
    }

    /// Search HNSW index and return nearest ID and its distance from query
    fn search(&mut self, query: &PyIrisCode) -> (u32, f64) {
        py_bindings::search_iris(&mut self.vector, &mut self.graph, query.to_iris_code())
    }

    fn get_iris(&self, id: u32) -> PyIrisCode {
        self.vector.points[id as usize].data.0.clone().into()
    }

    fn len(&self) -> usize {
        self.vector.points.len()
    }

    // TODO de/serialize hnsw state
}

#[pymodule]
fn iris_mpc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrisCodeArray>()?;
    m.add_class::<PyIrisCode>()?;
    m.add_class::<PyHnsw>()?;
    Ok(())
}
