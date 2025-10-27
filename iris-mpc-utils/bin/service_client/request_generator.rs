/// Encapsulates logic for generating SMPC service requests.
pub struct RequestGenerator {
    // Associated generator options.
    options: RequestGeneratorOptions,
}

impl RequestGenerator {
    pub fn new(options: RequestGeneratorOptions) -> Self {
        Self { options }
    }
}

impl RequestGenerator {
    pub fn options(&self) -> &RequestGeneratorOptions {
        &self.options
    }
}

/// Encapsulates options for generating SMPC service requests.
pub struct RequestGeneratorOptions {}
