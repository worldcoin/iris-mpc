use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u128);

impl From<u128> for SessionId {
    fn from(id: u128) -> Self {
        SessionId(id)
    }
}
