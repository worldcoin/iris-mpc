use thiserror::Error;

// Encapsulates a non-exhaustive set of errors raised during testing.
#[derive(Error, Debug)]
pub enum TestError {
    #[error("Node process panic: Node={0} :: Error={1}")]
    #[allow(dead_code)]
    NodeProcessPanicError(usize, String),

    #[error("A test setup error occurred: {0}")]
    #[allow(dead_code)]
    SetupError(String),
}

impl From<eyre::Report> for TestError {
    fn from(err: eyre::Report) -> Self {
        TestError::SetupError(err.to_string())
    }
}
