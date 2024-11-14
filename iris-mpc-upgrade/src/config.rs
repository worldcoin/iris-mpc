use clap::Parser;
use iris_mpc_common::id::PartyID;
use std::{
    fmt::{self, Display, Formatter},
    net::SocketAddr,
    str::FromStr,
};

pub const BATCH_TIMEOUT_SECONDS: u64 = 60;
pub const BATCH_SUCCESSFUL_ACK: u8 = 1;
pub const FINAL_BATCH_SUCCESSFUL_ACK: u8 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Eye {
    Left  = 0,
    Right = 1,
}

impl Display for Eye {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Eye::Left => write!(f, "left"),
            Eye::Right => write!(f, "right"),
        }
    }
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

#[derive(Clone, Parser)]
pub struct UpgradeServerConfig {
    #[clap(long)]
    pub bind_addr: SocketAddr,

    #[clap(long)]
    pub db_url: String,

    #[clap(long)]
    pub party_id: PartyID,

    #[clap(long)]
    pub eye: Eye,

    #[clap(long)]
    pub environment: String,
}

impl fmt::Debug for UpgradeServerConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpgradeServerConfig")
            .field("bind_addr", &self.bind_addr)
            .field("db_url", &"<redacted>")
            .field("party_id", &self.party_id)
            .field("eye", &self.eye)
            .finish()
    }
}

#[derive(Parser)]
pub struct UpgradeClientConfig {
    #[clap(long, default_value = "localhost:8000")]
    pub server1: String,

    #[clap(long, default_value = "localhost:8001")]
    pub server2: String,

    #[clap(long, default_value = "localhost:8002")]
    pub server3: String,

    #[clap(long)]
    pub db_start: u64,

    #[clap(long)]
    pub db_end: u64,

    #[clap(long)]
    pub party_id: u8,

    #[clap(long)]
    pub batch_size: u64,

    #[clap(long)]
    pub eye: Eye,

    #[clap(long)]
    pub shares_db_url: String,

    #[clap(long)]
    pub masks_db_url: String,

    #[clap(long)]
    pub batch_timeout_secs: Option<u64>,
}

impl fmt::Debug for UpgradeClientConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpgradeClientConfig")
            .field("server1", &self.server1)
            .field("server2", &self.server2)
            .field("server3", &self.server3)
            .field("shares_db_url", &"<redacted>")
            .field("masks_db_url", &"<redacted>")
            .field("db_start", &self.db_start)
            .field("db_end", &self.db_end)
            .field("party_id", &self.party_id)
            .field("eye", &self.eye)
            .finish()
    }
}

#[derive(Parser)]
pub struct ReShareClientConfig {
    #[clap(long, default_value = "http://localhost:8000", env("SERVER_URL"))]
    pub server_url: String,

    #[clap(long)]
    pub db_start: u64,

    #[clap(long)]
    pub db_end: u64,

    #[clap(long)]
    pub party_id: u8,

    #[clap(long)]
    pub other_party_id: u8,

    #[clap(long)]
    pub target_party_id: u8,

    #[clap(long)]
    pub batch_size: u64,

    #[clap(long)]
    pub db_url: String,

    #[clap(long)]
    pub batch_timeout_secs: Option<u64>,

    #[clap(long)]
    pub environment: String,
}

#[derive(Parser)]
pub struct ReShareServerConfig {
    #[clap(long, default_value = "0.0.0.0:8000", env("BIND_ADDR"))]
    pub bind_addr: SocketAddr,

    #[clap(long)]
    pub db_start: u64,

    #[clap(long)]
    pub db_end: u64,

    #[clap(long)]
    pub party_id: u8,

    #[clap(long)]
    pub sender1_party_id: u8,

    #[clap(long)]
    pub sender2_party_id: u8,

    #[clap(long)]
    pub batch_size: u64,

    #[clap(long)]
    pub db_url: String,

    #[clap(long)]
    pub batch_timeout_secs: Option<u64>,

    #[clap(long)]
    pub environment: String,

    #[clap(long, default_value = "10")]
    pub max_buffer_size: usize,
}
