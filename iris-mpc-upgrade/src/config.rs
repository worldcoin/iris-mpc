use clap::{Args, Parser, Subcommand};
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
    Left = 0,
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

    #[clap(long)]
    pub healthcheck_port: usize,
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
    /// The URL of the server to send reshare messages to
    #[clap(long, default_value = "http://localhost:8000", env("SERVER_URL"))]
    pub server_url: String,

    /// The DB index where we start to send Iris codes from (inclusive)
    #[clap(long)]
    pub db_start: u64,

    /// The DB index where we stop to send Iris codes (exclusive)
    #[clap(long)]
    pub db_end: u64,

    /// the 0-indexed party ID of the client party
    #[clap(long)]
    pub party_id: u8,

    /// the 0-indexed party ID of the other client party
    #[clap(long)]
    pub other_party_id: u8,

    /// the 0-indexed party ID of the receiving party
    #[clap(long)]
    pub target_party_id: u8,

    /// The batch size to use when sending reshare messages (i.e., how many iris
    /// code DB entries per message)
    #[clap(long)]
    pub batch_size: u64,

    /// DB connection URL for the reshare client
    #[clap(long)]
    pub db_url: String,

    /// The amount of time to wait before retrying a batch if the server queue
    /// was full, in milliseconds. Does a simple linear backoff strategy
    #[clap(long, default_value = "100")]
    pub retry_backoff_millis: u64,

    /// The environment in which the reshare protocol is being run (mostly used
    /// for the DB schema name)
    #[clap(long)]
    pub environment: String,

    /// The ARN of the KMS key that will be used to derive the common secret
    #[clap(long)]
    pub my_kms_key_arn: String,

    /// The ARN of the KMS key of the other client party that will be used to
    /// derive the common secret
    #[clap(long)]
    pub other_kms_key_arn: String,

    /// The session ID of the reshare protocol run, this will be used to salt
    /// the common secret derived between the two parties
    #[clap(long)]
    pub reshare_run_session_id: String,

    /// The path to the CA root file for the TLS connection
    #[clap(long)]
    pub ca_root_file_path: String,
}

#[derive(Parser)]
pub struct ReShareServerConfig {
    /// The socket to bind the reshare server to
    #[clap(long, default_value = "0.0.0.0:8000", env("BIND_ADDR"))]
    pub bind_addr: SocketAddr,

    /// The 0-indexed party ID of the server party
    #[clap(long)]
    pub party_id: u8,

    /// The 0-indexed party ID of the first client party (order of the two
    /// client parties does not matter)
    #[clap(long)]
    pub sender1_party_id: u8,

    /// The 0-indexed party ID of the second client party (order of the two
    /// client parties does not matter)
    #[clap(long)]
    pub sender2_party_id: u8,

    /// The maximum allowed batch size for reshare messages
    #[clap(long)]
    pub batch_size: u64,

    /// The DB connection URL to store reshared iris codes to
    #[clap(long)]
    pub db_url: String,

    /// The environment in which the reshare protocol is being run (mostly used
    /// for the DB schema name)
    #[clap(long)]
    pub environment: String,

    /// The maximum buffer size for the reshare server (i.e., how many messages
    /// are accepted from one client without receving corresponding messages
    /// from the other client)
    #[clap(long, default_value = "10")]
    pub max_buffer_size: usize,

    #[clap(long, default_value = "3000")]
    pub healthcheck_port: usize,
}

#[derive(Parser)]
pub struct ReRandomizeDbConfig {
    #[clap(subcommand)]
    pub command: ReRandomizeDbSubCommand,
}

#[derive(Subcommand)]
pub enum ReRandomizeDbSubCommand {
    KeyGen(KeyGenConfig),
    RerandomizeDb(ReRandomizeConfig),
    KeyCleanup(KeyCleanupConfig),
    RerandomizeCheck(ReRandomizeCheckConfig),
}

#[derive(Args)]
pub struct ReRandomizeConfig {
    /// The 0-indexed party ID of the server party
    #[clap(long, env = "PARTY_ID")]
    pub party_id: u8,

    /// The DB connection URL to load iris codes from
    #[clap(long, env = "DB_URL_SOURCE")]
    pub source_db_url: String,

    /// The environment in which the rerandomization protocol is being run (used to determine secrets to load from SM)
    #[clap(long, env = "ENVIRONMENT")]
    pub env: String,

    /// The base URL where the tripartite ECDH public keys are hosted
    #[clap(long, env = "PUBLIC_KEY_BASE_URL")]
    pub public_key_base_url: Option<String>,

    /// The name of the S3 bucket where the tripartite ECDH public keys are stored (used for local testing)
    #[clap(long, env = "PUBLIC_KEY_BUCKET_NAME")]
    pub public_key_bucket_name: Option<String>,

    /// The DB connection URL to store rerandomized iris codes to
    #[clap(long, env = "DB_URL_DEST")]
    pub dest_db_url: String,

    #[clap(long, env = "SCHEMA_NAME_SOURCE")]
    pub source_schema_name: String,

    #[clap(long, env = "SCHEMA_NAME_DEST")]
    pub dest_schema_name: String,

    #[clap(long, default_value = "3000", env = "HEALTHCHECK_PORT")]
    pub healthcheck_port: usize,

    #[clap(long, default_value = "8", env = "NUM_TASKS")]
    pub num_tasks: usize,

    #[clap(long, default_value = "1000", env = "CHUNK_SIZE")]
    pub chunk_size: usize,

    #[clap(long, default_value = "1", env = "RANGE_MIN")]
    pub range_min: usize,

    #[clap(long, default_value = "9223372036854775807", env = "RANGE_MAX")]
    pub range_max_inclusive: usize,
}

#[derive(Args)]
pub struct KeyGenConfig {
    /// The 0-indexed party ID of the server party
    #[clap(long, env = "PARTY_ID")]
    pub party_id: u8,

    #[clap(long, env = "ENVIRONMENT")]
    pub env: String,

    #[clap(long, env = "PUBLIC_KEY_BUCKET_NAME")]
    pub public_key_bucket_name: String,
}
#[derive(Args)]
pub struct KeyCleanupConfig {
    /// The 0-indexed party ID of the server party
    #[clap(long, env = "PARTY_ID")]
    pub party_id: u8,

    #[clap(long, env = "ENVIRONMENT")]
    pub env: String,
}

#[derive(Args)]
pub struct ReRandomizeCheckConfig {
    #[clap(long, env = "OLD_DB_URL_PARTY_0")]
    pub old_db_url_party_0: String,

    #[clap(long, env = "OLD_DB_URL_PARTY_1")]
    pub old_db_url_party_1: String,

    #[clap(long, env = "OLD_DB_URL_PARTY_2")]
    pub old_db_url_party_2: String,

    #[clap(long, env = "OLD_SCHEMA_NAME_PARTY_0")]
    pub old_schema_name_party_0: String,

    #[clap(long, env = "OLD_SCHEMA_NAME_PARTY_1")]
    pub old_schema_name_party_1: String,

    #[clap(long, env = "OLD_SCHEMA_NAME_PARTY_2")]
    pub old_schema_name_party_2: String,

    #[clap(long, env = "NEW_DB_URL_PARTY_0")]
    pub new_db_url_party_0: String,

    #[clap(long, env = "NEW_DB_URL_PARTY_1")]
    pub new_db_url_party_1: String,

    #[clap(long, env = "NEW_DB_URL_PARTY_2")]
    pub new_db_url_party_2: String,

    #[clap(long, env = "NEW_SCHEMA_NAME_PARTY_0")]
    pub new_schema_name_party_0: String,

    #[clap(long, env = "NEW_SCHEMA_NAME_PARTY_1")]
    pub new_schema_name_party_1: String,

    #[clap(long, env = "NEW_SCHEMA_NAME_PARTY_2")]
    pub new_schema_name_party_2: String,
}
