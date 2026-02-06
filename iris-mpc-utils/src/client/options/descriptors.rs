use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

/// A descriptor over an Iris code cached within a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrisDescriptor {
    // Ordinal identifer typically pointing to a row within an NDJSON file.
    index: usize,

    // TODO: Optionally apply noise, rotations, mirroring, etc.
    mutation: Option<()>,
}

impl IrisDescriptor {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            mutation: None,
        }
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

    pub fn indexes(&self) -> (usize, usize) {
        (self.left().index, self.right().index)
    }
}

/// A descriptor over a system Request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UniquenessRequestDescriptor {
    // Label to identify request within batch/file scope.
    Label(String),

    // Iris serial identifer as assigned by remote system.
    SerialId(IrisSerialId),
}

impl UniquenessRequestDescriptor {
    pub fn new_label(label: &str) -> Self {
        Self::Label(label.to_string())
    }

    pub fn new_serial_id(serial_id: IrisSerialId) -> Self {
        Self::SerialId(serial_id)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{IrisDescriptor, IrisPairDescriptor, UniquenessRequestDescriptor};

    pub(crate) const REQUEST_DESCRIPTOR_0: &str = "IdentityDeletion-0";
    pub(crate) const REQUEST_DESCRIPTOR_1: &str = "ResetCheck-0";
    pub(crate) const REQUEST_DESCRIPTOR_2: &str = "ResetUpdate-0";
    pub(crate) const REQUEST_DESCRIPTOR_3: &str = "Reauthorisation-0";
    pub(crate) const REQUEST_DESCRIPTOR_4_00: &str = "Uniqueness-00";
    pub(crate) const REQUEST_DESCRIPTOR_4_01: &str = "Uniqueness-01";
    pub(crate) const REQUEST_DESCRIPTOR_4_02: &str = "Uniqueness-02";
    pub(crate) const REQUEST_DESCRIPTOR_4_10: &str = "Uniqueness-10";
    pub(crate) const REQUEST_DESCRIPTOR_4_11: &str = "Uniqueness-11";
    pub(crate) const REQUEST_DESCRIPTOR_4_12: &str = "Uniqueness-12";

    impl IrisPairDescriptor {
        pub(crate) fn new_0(offset: usize) -> Self {
            Self::new(
                IrisDescriptor::new(offset + 1),
                IrisDescriptor::new(offset + 2),
            )
        }
    }

    impl UniquenessRequestDescriptor {
        pub(crate) fn new_4_00() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_00)
        }

        pub(crate) fn new_4_01() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_01)
        }

        pub(crate) fn new_4_02() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_02)
        }

        pub(crate) fn new_4_10() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_10)
        }

        pub(crate) fn new_4_11() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_11)
        }

        pub(crate) fn new_4_12() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_12)
        }
    }
}
