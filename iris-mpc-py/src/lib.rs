use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use iris_mpc_cpu::py_bindings::{self, PlaintextHnsw};
use pyo3::{exceptions::PyAttributeError, prelude::*};

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
struct PyIrisCode(IrisCode);

#[pymethods]
impl PyIrisCode {
    #[new]
    fn new_py(code: &PyIrisCodeArray, mask: &PyIrisCodeArray) -> Self {
        Self(IrisCode {
            code: code.0,
            mask: mask.0,
        })
    }

    fn code(&self) -> PyIrisCodeArray {
        PyIrisCodeArray(self.0.code)
    }

    fn mask(&self) -> PyIrisCodeArray {
        PyIrisCodeArray(self.0.mask)
    }

    #[staticmethod]
    fn from_open_iris_template(obj: &Bound<PyAny>) -> PyResult<Self> {
        let dict_obj = obj
            .call_method0("serialize")
            .map_err(|_| PyAttributeError::new_err("Object has no method 'serialize'"))?;

        // Extract the base64-encoded strings from the dictionary
        let iris_codes_str = dict_obj.get_item("iris_codes")?;
        let mask_codes_str = dict_obj.get_item("mask_codes")?;

        // Convert the base64 strings into MyData instances
        let code: PyIrisCodeArray = iris_codes_str.extract()?;
        let mask: PyIrisCodeArray = mask_codes_str.extract()?;

        // Step 5: Construct and return PyIrisCode
        Ok(Self(IrisCode {
            code: code.0,
            mask: mask.0,
        }))
    }
}

impl From<IrisCode> for PyIrisCode {
    fn from(value: IrisCode) -> Self {
        Self(value)
    }
}

#[pyclass]
struct PyHnsw(PlaintextHnsw);

#[pymethods]
impl PyHnsw {
    #[new]
    fn new_py() -> Self {
        Self(PlaintextHnsw::default())
    }

    #[staticmethod]
    fn gen_uniform(size: usize) -> PyHnsw {
        PyHnsw(PlaintextHnsw::gen_uniform_random(size))
    }

    fn insert_uniform_random(&mut self) -> u32 {
        self.0.insert_uniform_random().0
    }

    fn insert(&mut self, iris: &PyIrisCode) -> u32 {
        self.0.insert(iris.0.clone()).0
    }

    /// Search HNSW index and return nearest ID and its distance from query
    fn search(&mut self, query: &PyIrisCode) -> (u32, f64) {
        let (id, dist) = self.0.search(query.0.clone());
        (id.0, dist)
    }

    fn get_iris(&self, id: u32) -> PyIrisCode {
        self.0.vector.points[id as usize].data.0.clone().into()
    }

    fn len(&self) -> usize {
        self.0.vector.points.len()
    }

    fn write_to_file(&self, filename: &str) -> PyResult<()> {
        self.0
            .write_to_file(filename)
            .map_err(|_| PyAttributeError::new_err("Unable to write to file"))
    }

    #[staticmethod]
    fn read_from_file(filename: &str) -> PyResult<Self> {
        let result = PlaintextHnsw::read_from_file(filename)
            .map_err(|_| PyAttributeError::new_err("Unable to read from file"))?;
        Ok(PyHnsw(result))
    }
}

#[pymodule]
fn iris_mpc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrisCodeArray>()?;
    m.add_class::<PyIrisCode>()?;
    m.add_class::<PyHnsw>()?;
    Ok(())
}
