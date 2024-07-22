use crate::setup::id::PartyID;
use std::{
    fmt::{self, Formatter},
    os::unix::net::SocketAddr,
    str::FromStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye {
    Left,
    Right,
}

impl FromStr for Eye {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "left" => Ok(Eye::Left),
            "right" => Ok(Eye::Right),
            _ => Err(format!("Invalid eye: {}", s)),
        }
    }
}

#[derive(Clone)]
pub struct UpgradeServerConfig {
    pub eye:       Eye,
    pub bind_addr: SocketAddr,
    pub db_url:    String,
    pub party_id:  PartyID,
}

impl std::fmt::Debug for UpgradeServerConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpgradeServerConfig")
            .field("eye", &self.eye)
            .field("bind_addr", &self.bind_addr)
            .field("db_url", &"<redacted>")
            .field("party_id", &self.party_id)
            .finish()
    }
}

#[derive(Clone)]
pub struct UpgradeClientConfig {
    pub eye:      Eye,
    pub servers:  [SocketAddr; 3],
    pub db_url:   String,
    pub party_id: PartyID,
    pub start_id: u64,
    pub end_id:   u64,
}

impl std::fmt::Debug for UpgradeClientConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpgradeClientConfig")
            .field("eye", &self.eye)
            .field("servers", &self.servers)
            .field("db_url", &"<redacted>")
            .field("party_id", &self.party_id)
            .field("start_id", &self.start_id)
            .field("end_id", &self.end_id)
            .finish()
    }
}
