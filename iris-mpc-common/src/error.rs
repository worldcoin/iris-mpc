use thiserror::Error;

/// An Error enum capturing the errors produced by this crate.
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid party id provided
    #[error("Invalid Party id {0}")]
    Id(usize),
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
