use std::{fmt::Display, num::ParseIntError, str::FromStr};

use serde::{Deserialize, Serialize};

/// Unique identifier for an immutable pair of iris codes.
#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId {
    pub id: u32,
}

impl Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.id, f)
    }
}

impl FromStr for VectorId {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(VectorId {
            id: FromStr::from_str(s)?,
        })
    }
}

impl From<usize> for VectorId {
    fn from(id: usize) -> Self {
        VectorId { id: id as u32 }
    }
}

impl From<u32> for VectorId {
    fn from(id: u32) -> Self {
        VectorId { id }
    }
}

impl VectorId {
    /// From Serial ID (1-indexed).
    pub fn from_serial_id(id: u32) -> Self {
        VectorId { id }
    }

    /// To Serial ID (1-indexed).
    pub fn serial_id(&self) -> u32 {
        self.id
    }

    /// From index (0-indexed).
    pub fn from_0_index(index: u32) -> Self {
        VectorId { id: index + 1 }
    }

    /// To index (0-indexed).
    pub fn index(&self) -> u32 {
        self.id - 1
    }
}
