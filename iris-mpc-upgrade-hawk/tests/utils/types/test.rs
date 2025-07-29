use crate::utils::{store::DatabaseContext, TestInputs};
use eyre::{eyre, Report, Result};

/// HNSW Genesis test with helper functions. Individual tests will wrap this struct in a new type and implement TestRun.
/// DeriveMore can be used to automatically implement Deref and DerefMut to provide access to the helper functions.
#[derive(Default)]
pub struct Test {
    /// Data encapsulating test inputs.
    pub inputs: Option<TestInputs>,

    /// Results of node process execution.
    pub node_results: Option<Vec<Result<(), Report>>>,
}

/// Constructor.
impl Test {
    pub fn new() -> Self {
        Self {
            inputs: None,
            node_results: None,
        }
    }

    pub async fn get_db_contexts(&self) -> Result<Vec<DatabaseContext>> {
        let inputs = self.inputs.as_ref().ok_or(eyre!("inputs not set"))?;
        let inputs = inputs.net_inputs().node_process_inputs();
        let mut dbs = Vec::new();
        for input in inputs {
            dbs.push(DatabaseContext::from_config(input.config()).await?);
        }
        Ok(dbs)
    }
}
