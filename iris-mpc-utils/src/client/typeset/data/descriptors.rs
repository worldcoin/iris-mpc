use serde::{Deserialize, Serialize};

/// A descriptor over an Iris code cached within an NDJSON file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IrisDescriptor {
    // Ordinal identifer pointing to a row within an NDJSON file.
    index: usize,
}

impl IrisDescriptor {
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

/// A descriptor over a pair of Iris codes cached within a file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IrisPairDescriptor((IrisDescriptor, IrisDescriptor));

impl IrisPairDescriptor {
    pub fn new(left: IrisDescriptor, right: IrisDescriptor) -> Self {
        Self((left, right))
    }

    pub fn new_from_indexes(left: usize, right: usize) -> Self {
        Self::new(IrisDescriptor::new(left), IrisDescriptor::new(right))
    }

    pub fn left(&self) -> &IrisDescriptor {
        &self.0 .0
    }

    pub fn right(&self) -> &IrisDescriptor {
        &self.0 .1
    }
}
