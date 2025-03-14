use super::types::IrisGaloisShares;

// Event: raised when genesis indexation process starts.
#[derive(Clone, Debug)]
pub struct OnBegin;

// Event: raised when a new Iris batch for indexation has been produced.
#[derive(Clone, Debug)]
pub struct OnBeginBatch {
    // Range of Iris identifiers to be indexed.
    pub(crate) batch: Vec<i64>,
}

// Event: raised when a new Iris for indexation has been produced.
#[derive(Clone, Debug)]
pub struct OnBeginBatchItem {
    // Range of Iris identifiers to be indexed.
    pub(crate) id_of_iris: i64,
}

// Event: raised when genesis indexation is complete.
#[derive(Clone, Debug)]
pub struct OnEnd;

// Event: raised when an Iris batch has been processed.
#[derive(Clone, Debug)]
pub struct OnEndOfBatch;

// Event: raised when genesis indexation error occurs.
#[derive(Clone, Debug)]
pub struct OnError;

// Event: raised when raw Iris data is ready for processing.
// TODO: use byte slice rather than vecs
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OnBeginOfIrisSharesIndexation {
    // Fetched Iris data.
    pub(crate) shares: OnFetchOfIrisShares,
}

// Event: raised when secret shared iris data has been fetched.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OnFetchOfIrisShares {
    // Iris serial ID.
    pub(crate) serial_id: i64,

    // Party's iris secret shares.
    pub(crate) shares: IrisGaloisShares,
}
