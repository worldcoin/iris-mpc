use serde::{Deserialize, Serialize};
use uuid;

use iris_mpc_common::IrisSerialId;

/// A descriptor over an Iris code cached within an NDJSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Enumeration over uniqueness request references. Applies to: IdentityDeletion ^ Reauthorization ^ ResetUpdate.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UniquenessRequestDescriptor {
    // A serial identifier assigned from either a processed uniqueness result or a user input override.
    IrisSerialId(IrisSerialId),
    // Label to identify request within batch/file scope.
    Label(String),
    // Unique signup id of system request being processed.
    SignupId(uuid::Uuid),
}

impl UniquenessRequestDescriptor {
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Label(label) => Some(label),
            _ => None,
        }
    }
}
