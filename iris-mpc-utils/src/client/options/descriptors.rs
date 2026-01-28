use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

/// Options over an Iris code.
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

/// Options over a pair of Iris code's.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrisPairDescriptor((IrisDescriptor, IrisDescriptor));

impl IrisPairDescriptor {
    pub fn new(left: IrisDescriptor, right: IrisDescriptor) -> Self {
        Self((left, right))
    }

    pub fn new_from_indexes(left: usize, right: usize) -> Self {
        Self::new(IrisDescriptor::new(left), IrisDescriptor::new(right))
    }
}

/// Options over an associated request descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestDescriptor {
    // Label to identify request within batch/file scope.
    Label(String),

    // Iris serial identifer as assigned by remote system.
    SerialId(IrisSerialId),
}

impl RequestDescriptor {
    pub fn new_label(label: &str) -> Self {
        Self::Label(label.to_string())
    }

    pub fn new_serial_id(serial_id: IrisSerialId) -> Self {
        Self::SerialId(serial_id)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{IrisDescriptor, IrisPairDescriptor, RequestDescriptor};

    pub(crate) const REQUEST_DESCRIPTOR_0: &str = "00-IdentityDeletion";
    pub(crate) const REQUEST_DESCRIPTOR_1: &str = "10-ResetCheck";
    pub(crate) const REQUEST_DESCRIPTOR_2: &str = "20-ResetUpdate";
    pub(crate) const REQUEST_DESCRIPTOR_3: &str = "30-Reauthorisation";
    pub(crate) const REQUEST_DESCRIPTOR_4_0: &str = "40-Uniqueness";
    pub(crate) const REQUEST_DESCRIPTOR_4_1: &str = "41-Uniqueness";
    pub(crate) const REQUEST_DESCRIPTOR_4_2: &str = "42-Uniqueness";

    impl IrisPairDescriptor {
        pub(crate) fn new_0(offset: usize) -> Self {
            Self::new(IrisDescriptor::new(offset), IrisDescriptor::new(offset + 1))
        }
    }

    impl RequestDescriptor {
        pub(crate) fn new_4_0() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_0)
        }

        pub(crate) fn new_4_1() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_1)
        }

        pub(crate) fn new_4_2() -> Self {
            Self::new_label(REQUEST_DESCRIPTOR_4_2)
        }
    }
}
