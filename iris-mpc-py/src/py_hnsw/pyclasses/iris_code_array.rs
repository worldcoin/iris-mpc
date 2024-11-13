use iris_mpc_common::iris_db::iris::IrisCodeArray;
use pyo3::prelude::*;
use rand::rngs::ThreadRng;

#[pyclass]
#[derive(Clone, Default)]
pub struct PyIrisCodeArray(pub IrisCodeArray);

#[pymethods]
impl PyIrisCodeArray {
    #[new]
    pub fn new(input: String) -> Self {
        Self::from_base64(input)
    }

    pub fn to_base64(&self) -> String {
        self.0.to_base64().unwrap()
    }

    #[staticmethod]
    pub fn from_base64(input: String) -> Self {
        Self(IrisCodeArray::from_base64(&input).unwrap())
    }

    #[staticmethod]
    pub fn zeros() -> Self {
        Self(IrisCodeArray::ZERO)
    }

    #[staticmethod]
    pub fn ones() -> Self {
        Self(IrisCodeArray::ONES)
    }

    #[staticmethod]
    pub fn uniform_random() -> Self {
        let mut rng = ThreadRng::default();
        Self(IrisCodeArray::random_rng(&mut rng))
    }
}

impl From<IrisCodeArray> for PyIrisCodeArray {
    fn from(value: IrisCodeArray) -> Self {
        Self(value)
    }
}
