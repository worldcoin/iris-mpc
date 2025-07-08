use super::inputs::TestInputs;

/// Encapsulates data related to a test's state.
#[derive(Debug, Clone)]
pub struct TestState {
    /// Data encapsulating workflow inputs.
    inputs: TestInputs,
}
