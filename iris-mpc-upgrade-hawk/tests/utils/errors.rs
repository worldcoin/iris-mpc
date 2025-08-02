use thiserror::Error;

// Encapsulates a non-exhaustive set of errors raised during testing.
#[derive(Error, Debug)]
pub enum TestError {
    #[error("Node process panic: Node={0} :: Error={1}")]
    NodePanicError(usize, String),

    #[error("A test setup error occurred: {0}")]
    #[allow(dead_code)]
    SetupError(String),
}
