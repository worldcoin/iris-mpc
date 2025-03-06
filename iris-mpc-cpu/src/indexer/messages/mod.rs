#![allow(dead_code)]

// Event: Raised when an indexation is complete.
#[derive(Clone, Debug)]
pub struct OnIndexationEnd;

// Event: Raised when an indexation error occurs.
#[derive(Clone, Debug)]
pub struct OnIndexationError;

// Event: Raised when an indexation starts.
#[derive(Clone, Debug)]
pub struct OnIndexationStart;

// Event: Raised when an Iris id is ready for processing.
#[derive(Clone, Debug)]
pub struct OnIrisIdPulledFromStore {
    pub(crate) id_of_iris: i64,
}

// Event: Raised when raw Iris data is ready for processing.
#[derive(Clone, Debug)]
pub struct OnIrisDataPulledFromStore {
    pub(crate) id_of_iris: i64,
    pub(crate) code_left: Vec<u16>,
    pub(crate) code_right: Vec<u16>,
    pub(crate) mask_left: Vec<u16>,
    pub(crate) mask_right: Vec<u16>,
}
