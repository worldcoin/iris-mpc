use rayon::ThreadPoolBuildError;
use thiserror::Error;

/// An Error enum capturing the errors produced by this crate.
#[derive(Error, Debug)]
pub enum Error {
    /// Config Error
    #[error("Invalid Configuration")]
    Config,
    /// Type conversion error
    #[error("Conversion error")]
    Conversion,
    /// Error from the color_eyre crate
    #[error(transparent)]
    Eyre(#[from] eyre::Report),
    /// Invalid party id provided
    #[error("Invalid Party id {0}")]
    Id(usize),
    /// Message size is invalid
    #[error("Message size is invalid")]
    InvalidMessageSize,
    /// Size is invalid
    #[error("Size is invalid")]
    InvalidSize,
    /// A IO error has orccured
    #[error(transparent)]
    IO(#[from] std::io::Error),
    /// JMP verify failed
    #[error("JMP verify failed")]
    JmpVerify,
    /// Mask HW is to small
    #[error("Mask HW is to small")]
    MaskHW,
    /// Not enough triples
    #[error("Not enough triples")]
    NotEnoughTriples,
    /// Invalid number of parties
    #[error("Invalid number of parties {0}")]
    NumParty(usize),
    /// Verify failed
    #[error("Verify failed")]
    Verify,
    #[error(transparent)]
    ThreadPoolBuildError(#[from] ThreadPoolBuildError),
    /// Some other error has occurred.
    #[error("Err: {0}")]
    Other(String),
}

impl From<String> for Error {
    fn from(mes: String) -> Self {
        Self::Other(mes)
    }
}

impl From<&str> for Error {
    fn from(mes: &str) -> Self {
        Self::Other(mes.to_owned())
    }
}
