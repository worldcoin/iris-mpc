use std::fmt;

use iris_mpc_cpu::execution::hawk_main::BothEyes;

use crate::types::IrisCodeAndMaskShares;

/// Enumeration over data associated with a request.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestData {
    IdentityDeletion,
    Reauthorisation,
    ResetCheck,
    ResetUpdate,
    Uniqueness {
        shares: BothEyes<IrisCodeAndMaskShares>,
    },
}

#[derive(Debug, Clone)]
pub struct RequestDataUniqueness {
    shares: BothEyes<IrisCodeAndMaskShares>,
}

impl RequestDataUniqueness {
    pub fn shares(&self) -> &BothEyes<IrisCodeAndMaskShares> {
        &self.shares
    }

    pub fn new(shares: BothEyes<IrisCodeAndMaskShares>) -> Self {
        Self { shares }
    }
}

impl fmt::Display for RequestData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion => {
                write!(f, "IdentityDeletion")
            }
            RequestData::Reauthorisation => {
                write!(f, "Reauthorisation")
            }
            RequestData::ResetCheck => {
                write!(f, "ResetCheck")
            }
            RequestData::ResetUpdate => {
                write!(f, "ResetUpdate")
            }
            RequestData::Uniqueness { shares: _ } => {
                write!(f, "Uniqueness")
            }
        }
    }
}
