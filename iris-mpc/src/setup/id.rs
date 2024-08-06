use std::str::FromStr;

/// An enum representing the party ID
#[derive(std::cmp::Eq, std::cmp::PartialEq, Clone, Copy, Debug)]
#[repr(u8)]
pub enum PartyID {
    /// Party 0
    ID0 = 0,
    /// Party 1
    ID1 = 1,
    /// Party 2
    ID2 = 2,
}

impl PartyID {
    /// get next ID
    pub fn next_id(&self) -> Self {
        match *self {
            PartyID::ID0 => PartyID::ID1,
            PartyID::ID1 => PartyID::ID2,
            PartyID::ID2 => PartyID::ID0,
        }
    }

    /// get previous ID
    pub fn prev_id(&self) -> Self {
        match *self {
            PartyID::ID0 => PartyID::ID2,
            PartyID::ID1 => PartyID::ID0,
            PartyID::ID2 => PartyID::ID1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartyIDError(String);

impl std::error::Error for PartyIDError {}

impl std::fmt::Display for PartyIDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid party ID: {}", &self.0)
    }
}

impl TryFrom<usize> for PartyID {
    type Error = PartyIDError;

    fn try_from(other: usize) -> Result<Self, Self::Error> {
        match other {
            0 => Ok(PartyID::ID0),
            1 => Ok(PartyID::ID1),
            2 => Ok(PartyID::ID2),
            i => Err(PartyIDError(format!("Invalid party ID: {}", i))),
        }
    }
}

impl TryFrom<u8> for PartyID {
    type Error = PartyIDError;

    #[inline(always)]
    fn try_from(other: u8) -> Result<Self, Self::Error> {
        (other as usize).try_into()
    }
}

impl FromStr for PartyID {
    type Err = PartyIDError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<usize>()
            .map_err(|e| PartyIDError(e.to_string()))?
            .try_into()
    }
}

impl From<PartyID> for u8 {
    #[inline(always)]
    fn from(other: PartyID) -> Self {
        other as u8
    }
}

impl From<PartyID> for usize {
    #[inline(always)]
    fn from(other: PartyID) -> Self {
        other as usize
    }
}

impl std::fmt::Display for PartyID {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", *self as usize)
    }
}
