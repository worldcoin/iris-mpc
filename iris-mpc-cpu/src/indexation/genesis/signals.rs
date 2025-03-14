use super::types::IrisGaloisShares;

// Signals that indexation process starts.
#[derive(Clone, Debug)]
pub struct OnBegin;

// Signals that a new Iris batch is ready for processing.
#[derive(Clone, Debug)]
pub struct OnBeginBatch {
    // Set of Iris serial identifiers to be indexed.
    pub(crate) batch: Vec<i64>,
}

// Signals that a new Iris identifier is ready for processing.
#[derive(Clone, Debug)]
pub struct OnBeginBatchItem {
    // Range of Iris identifiers to be indexed.
    pub(crate) id_of_iris: i64,
}

// Signals that Iris shares are ready for indexation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OnBeginIrisSharesIndexation {
    // Iris serial ID.
    pub(crate) serial_id: i64,

    // Party's iris secret shares.
    pub(crate) shares: IrisGaloisShares,
}

// Signals that indexation is complete.
#[derive(Clone, Debug)]
pub struct OnEnd;

// Signals that Iris batch indexation is complete.
#[derive(Clone, Debug)]
pub struct OnEndOfBatch;

// Signals that an indexation error occurred.
#[derive(Clone, Debug)]
pub struct OnError;

// Signals that Iris shares have been fetched from a store.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OnFetchOfIrisShares {
    // Iris serial ID.
    pub(crate) serial_id: i64,

    // Party's iris secret shares.
    pub(crate) shares: IrisGaloisShares,
}
