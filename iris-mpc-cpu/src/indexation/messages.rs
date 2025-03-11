#![allow(dead_code)]

// Event: raised when indexation is complete.
#[derive(Clone, Debug)]
pub struct OnIndexationEnd;

// Event: raised when an indexation error occurs.
#[derive(Clone, Debug)]
pub struct OnIndexationError;

// Event: raised when indexation starts.
#[derive(Clone, Debug)]
pub struct OnIndexationStart;

// Event: raised when an Iris batch has been processed.
#[derive(Clone, Debug)]
pub struct OnBatchIndexationEnd;

// Event: raised when a new Iris batch for indexation has been produced.
#[derive(Clone, Debug)]
pub struct OnBatchIndexationStart {
    // Range of Iris identifiers to be indexed.
    pub(crate) batch_range: (i64, i64),
}

// Event: raised when a new Iris for indexation has been produced.
#[derive(Clone, Debug)]
pub struct OnBatchElementIndexationStart {
    // Range of Iris identifiers to be indexed.
    pub(crate) id_of_iris: i64,
}

// Event: raised when raw Iris data is ready for processing.
// TODO: use byte slice rather than vecs
#[derive(Clone, Debug, Default)]
pub struct OnIrisDataPulledFromStore {
    // Iris ID, see pgres primary key.
    pub(crate) id_of_iris: i64,

    // Iris code share: left.
    pub(crate) left_code: Vec<u16>,

    // Iris mask share: left.
    pub(crate) left_mask: Vec<u16>,

    // Iris code share: right.
    pub(crate) right_code: Vec<u16>,

    // Iris mask share: right.
    pub(crate) right_mask: Vec<u16>,
}
