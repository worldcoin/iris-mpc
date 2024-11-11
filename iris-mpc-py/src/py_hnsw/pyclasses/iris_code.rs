use super::iris_code_array::PyIrisCodeArray;
use iris_mpc_common::iris_db::iris::IrisCode;
use pyo3::{exceptions::PyAttributeError, prelude::*};

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
