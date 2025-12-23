use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use toml;

use iris_mpc_common::IrisSerialId;

/// Set of variants over client parameterisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceClientConfig {
    // Batches of single request type
    Kind {
        /// Number of request batches to generate.
        batch_count: usize,

        /// Size of each batch.
        batch_size: usize,

        /// Determines type of requests to be included in each batch.
        batch_kind: String,

        // A known serial identifier that allows response correlation to be bypassed.
        known_iris_serial_id: Option<IrisSerialId>,
    },
}

impl From<PathBuf> for ServiceClientConfig {
    fn from(value: PathBuf) -> Self {
        assert!(value.exists());

        toml::from_str(&fs::read_to_string(value).unwrap()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use iris_mpc_common::helpers::smpc_request;

    use super::ServiceClientConfig;

    impl ServiceClientConfig {
        pub fn new_1() -> Self {
            Self::Kind {
                batch_count: 1,
                batch_size: 1,
                batch_kind: smpc_request::UNIQUENESS_MESSAGE_TYPE.to_string(),
                known_iris_serial_id: None,
            }
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = ServiceClientConfig::new_1();
    }
}
