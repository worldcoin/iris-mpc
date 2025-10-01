#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
pub struct ConnectionId(pub u32);

impl ConnectionId {
    pub fn new(val: u32) -> Self {
        Self(val)
    }
}

impl From<u32> for ConnectionId {
    fn from(val: u32) -> Self {
        ConnectionId::new(val)
    }
}
