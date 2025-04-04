use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr};

/// Unique identifier for an immutable pair of iris codes.
#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId {
    id: u32,
    version: i16,
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

        let id = parts.next().unwrap().parse::<u32>()?;

        let version = match parts.next() {
            Some(v) => v.parse::<i16>()?,
            None => 0,
        };

        Ok(VectorId { id, version })
    }
}

impl VectorId {
    pub fn new(serial_id: u32, version: i16) -> Self {
        VectorId {
            id: serial_id,
            version,
        }
    }

    /// From Serial ID (1-indexed).
    pub fn from_serial_id(id: u32) -> Self {
        VectorId { id, version: 0 }
    }

    /// To Serial ID (1-indexed).
    pub fn serial_id(&self) -> u32 {
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
