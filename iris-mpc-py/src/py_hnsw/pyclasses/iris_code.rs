use super::iris_code_array::PyIrisCodeArray;
use iris_mpc_common::iris_db::iris::IrisCode;
use pyo3::{prelude::*, types::PyDict};

#[pyclass]
pub struct PyIrisCode(pub IrisCode);

#[pymethods]
impl PyIrisCode {
    #[new]
    fn new(code: &PyIrisCodeArray, mask: &PyIrisCodeArray) -> Self {
        Self(IrisCode {
            code: code.0,
            mask: mask.0,
        })
    }

    #[getter]
    fn code(&self) -> PyIrisCodeArray {
        PyIrisCodeArray(self.0.code)
    }

    #[getter]
    fn mask(&self) -> PyIrisCodeArray {
        PyIrisCodeArray(self.0.mask)
    }

    #[pyo3(signature = (version=None))]
    fn to_open_iris_template_dict<'py>(&self, py: Python<'py>, version: Option<String>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        dict.set_item("iris_codes", self.0.code.to_base64().unwrap())?;
        dict.set_item("iris_masks", self.0.mask.to_base64().unwrap())?;
        dict.set_item("iris_code_version", version)?;

        Ok(dict)
    }

    #[staticmethod]
    fn from_open_iris_template_dict(dict_obj: &Bound<PyDict>) -> PyResult<Self> {
        // Extract base64-encoded iris code arrays
        let iris_codes_str = dict_obj.get_item("iris_codes")?.unwrap();
        let mask_codes_str = dict_obj.get_item("mask_codes")?.unwrap();

        // Convert the base64 strings into PyIrisCodeArrays
        let code: PyIrisCodeArray = iris_codes_str.extract()?;
        let mask: PyIrisCodeArray = mask_codes_str.extract()?;

        // Construct and return PyIrisCode
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
