use thiserror::Error;

// Encapsulates a non-exhaustive set of errors raised during testing.
#[derive(Error, Debug)]
pub enum TestError {
    #[error("A test setup error occurred: {0}")]
    SetupError(String),
}
