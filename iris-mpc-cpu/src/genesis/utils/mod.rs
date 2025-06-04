pub(crate) mod errors;
pub mod logger;

pub use logger::log_error;
pub use logger::log_info;
pub use logger::log_warn;

// Count of MPC protocol parties.
pub const COUNT_OF_MPC_PARTIES: usize = 3;

// Type alias: Identifier of an MPC participant.
pub type PartyId = usize;
