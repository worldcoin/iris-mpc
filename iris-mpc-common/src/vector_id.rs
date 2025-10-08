use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr};

// An Iris pair serial identifier.
pub type SerialId = u32;

// An Iris pair version identifier.
pub type VersionId = i16;

/// Unique identifier for an immutable pair of iris codes.
#[derive(
    Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord,
)]
pub struct VectorId {
    id: SerialId,
    version: VersionId,
}

impl Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format as "id" or "id:version".
        if self.version == 0 {
            write!(f, "{}", self.id)
        } else {
            write!(f, "{}:{}", self.id, self.version)
        }
    }
}

impl FromStr for VectorId {
    type Err = eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse as "id" or "id:version".
        let mut parts = s.split(':');

        let id = parts.next().unwrap().parse::<SerialId>()?;

        let version = match parts.next() {
            Some(v) => v.parse::<VersionId>()?,
            None => 0,
        };

        Ok(VectorId { id, version })
    }
}

impl VectorId {
    pub fn new(serial_id: SerialId, version: VersionId) -> Self {
        VectorId {
            id: serial_id,
            version,
        }
    }

    /// From Serial ID (1-indexed).
    pub const fn from_serial_id(id: SerialId) -> Self {
        VectorId { id, version: 0 }
    }

    /// To Serial ID (1-indexed).
    pub fn serial_id(&self) -> SerialId {
        self.id
    }

    /// From index (0-indexed).
    pub fn from_0_index(index: u32) -> Self {
        VectorId {
            id: index + 1,
            version: 0,
        }
    }

    /// To index (0-indexed).
    pub fn index(&self) -> u32 {
        self.id - 1
    }

    /// Get the version number of the iris code for a same serial ID.
    pub fn version_id(&self) -> VersionId {
        self.version
    }

    /// Whether the version of this vector ID matches the other vector ID.
    pub fn version_matches(&self, other_version: VersionId) -> bool {
        self.version == other_version
    }

    /// Return the next version of this vector ID.
    pub fn next_version(self) -> Self {
        VectorId {
            id: self.id,
            version: self.version + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_id() {
        let id = VectorId::from_serial_id(1);
        assert_eq!(id.serial_id(), 1);
        assert_eq!(id.index(), 0);
        assert_eq!(format!("{}", id), "1");
        assert_eq!(VectorId::from_str("1").unwrap(), id);

        let id = VectorId { id: 11, version: 1 };
        assert_eq!(format!("{}", id), "11:1");
        assert_eq!(VectorId::from_str("11:1").unwrap(), id);
    }
}
